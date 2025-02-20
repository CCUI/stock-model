import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv
from pathlib import Path
import time

# Load environment variables from the correct path
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class UKStockDataCollector:
    def __init__(self):
        self.ftse_symbols = self._get_ftse_symbols()
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        
        # Get NewsAPI key from environment variable
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if not self.news_api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
            
        # Initialize cache for news data
        self.news_cache = {}
        self.last_api_call = 0
        self.api_call_delay = 1  # Delay between API calls in seconds
            
    def _get_ftse_symbols(self):
        """Get list of FTSE stocks"""
        # Only FTSE 100 components for testing
        ftse_url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
        
        symbols = []
        response = requests.get(ftse_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table', {'class': 'wikitable'})
        
        for table in tables:
            rows = table.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    # Get the symbol text and clean it
                    symbol_text = cols[1].text.strip()
                    
                    # Skip if the symbol contains numeric values (likely a price)
                    if any(char.isdigit() for char in symbol_text):
                        continue
                        
                    # Add .L suffix for London Stock Exchange
                    symbol = symbol_text + '.L'
                    symbols.append(symbol)
        
        # Remove duplicates
        unique_symbols = list(set(symbols))
        
        # Ensure we have symbols
        if not unique_symbols:
            raise ValueError("No valid FTSE symbols could be extracted from the webpage")
            
        return unique_symbols
    
    def collect_historical_data(self, end_date, lookback_days=365, max_retries=3, delay=1):
        """Collect historical price and fundamental data for all FTSE stocks"""
        start_date = end_date - timedelta(days=lookback_days)
        
        all_data = {}
        for symbol in self.ftse_symbols:
            for attempt in range(max_retries):
                try:
                    # Get price data
                    stock = yf.Ticker(symbol)
                    hist_data = stock.history(start=start_date, end=end_date)
                    
                    if not hist_data.empty:
                        # Add technical data
                        hist_data['Symbol'] = symbol
                        
                        # Get fundamental data
                        info = stock.info
                        for key in ['marketCap', 'trailingPE', 'priceToBook', 'debtToEquity']:
                            hist_data[key] = info.get(key, 0)  # Use 0 instead of None for missing values
                        
                        # Get news sentiment
                        news = self._get_news_sentiment(symbol)
                        hist_data['news_sentiment'] = news['sentiment_score'].mean() if not news.empty else 0
                        
                        all_data[symbol] = hist_data
                        break  # Successfully got data, break retry loop
                    
                except Exception as e:
                    if 'Too Many Requests' in str(e) and attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                        continue
                    print(f"Error collecting data for {symbol}: {str(e)}")
                    break  # Break on non-rate-limit errors or final attempt
        
        # Ensure we have at least some data before concatenating
        if not all_data:
            raise ValueError("No data could be collected for any symbols")
            
        return pd.concat(all_data.values(), axis=0)
        
    def _get_news_sentiment(self, symbol: str) -> pd.DataFrame:
        """Collect and analyze news sentiment for a given stock"""
        try:
            # Remove .L suffix and get company name
            clean_symbol = symbol.replace('.L', '')
            company_name = self._get_company_name(clean_symbol)
            
            # Check cache first
            if company_name in self.news_cache:
                return self.news_cache[company_name]
            
            # Implement rate limiting
            current_time = time.time()
            if current_time - self.last_api_call < self.api_call_delay:
                time.sleep(self.api_call_delay - (current_time - self.last_api_call))
            
            # Get news from NewsAPI
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f'"{company_name}" AND (stock OR shares OR market)',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10,  # Limit to 10 articles per company
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            self.last_api_call = time.time()
            
            if response.status_code != 200:
                print(f"Error fetching news for {company_name}: {response.status_code}")
                return pd.DataFrame({'sentiment_score': [0.0]})
            
            data = response.json()
            
            if data['totalResults'] == 0:
                return pd.DataFrame({'sentiment_score': [0.0]})
            
            sentiments = []
            for article in data['articles']:
                # Combine title and description for better sentiment analysis
                text = f"{article['title']} {article['description'] or ''}"
                
                # Use FinBERT for sentiment analysis
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                outputs = self.sentiment_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                # Calculate sentiment score (-1 to 1)
                sentiment_score = float(probabilities[0][2].item() - probabilities[0][0].item())
                
                sentiments.append({
                    'date': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                    'sentiment_score': sentiment_score
                })
            
            if not sentiments:
                return pd.DataFrame({'sentiment_score': [0.0]})
            
            # Calculate weighted sentiment (more recent articles have higher weight)
            sentiment_df = pd.DataFrame(sentiments)
            sentiment_df['weight'] = sentiment_df['date'].apply(
                lambda x: 1 / (1 + (datetime.now() - x).days)
            )
            weighted_sentiment = float(
                (sentiment_df['sentiment_score'] * sentiment_df['weight']).sum() 
                / sentiment_df['weight'].sum()
            )
            
            result = pd.DataFrame({'sentiment_score': [weighted_sentiment]})
            
            # Cache the result
            self.news_cache[company_name] = result
            
            return result
            
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {str(e)}")
            return pd.DataFrame({'sentiment_score': [0.0]})
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol using Wikipedia data"""
        try:
            # First check if we already have the mapping
            if hasattr(self, 'symbol_to_name_map'):
                return self.symbol_to_name_map.get(symbol, symbol)
            
            # Create the mapping
            self.symbol_to_name_map = {}
            response = requests.get("https://en.wikipedia.org/wiki/FTSE_100_Index")
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            for table in tables:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        company_name = cols[0].text.strip()
                        symbol_text = cols[1].text.strip()
                        self.symbol_to_name_map[symbol_text] = company_name
            
            return self.symbol_to_name_map.get(symbol, symbol)
            
        except Exception as e:
            print(f"Error getting company name for {symbol}: {str(e)}")
            return symbol