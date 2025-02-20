import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class UKStockDataCollector:
    def __init__(self):
        self.ftse_symbols = self._get_ftse_symbols()
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    
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
                        news = self._get_news_sentiment(symbol, end_date)
                        hist_data['news_sentiment'] = news['sentiment_score'].mean() if not news.empty else 0
                        
                        all_data[symbol] = hist_data
                        break  # Successfully got data, break retry loop
                    
                except Exception as e:
                    if 'Too Many Requests' in str(e) and attempt < max_retries - 1:
                        import time
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                        continue
                    print(f"Error collecting data for {symbol}: {str(e)}")
                    break  # Break on non-rate-limit errors or final attempt
        
        # Ensure we have at least some data before concatenating
        if not all_data:
            raise ValueError("No data could be collected for any symbols")
            
        return pd.concat(all_data.values(), axis=0)
        
        return pd.concat(all_data.values(), axis=0)
    
    def _get_news_sentiment(self, symbol, date):
        """Collect and analyze news sentiment for a given stock"""
        try:
            # Validate API key before making the API call
            if not self.api_key or self.api_key == 'demo':
                raise ValueError("Please set ALPHA_VANTAGE_API_KEY environment variable with a valid API key.")
            
            # Get news articles from Alpha Vantage with error handling
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for bad status codes
            news_data = response.json()
            
            if 'Note' in news_data:  # Check for rate limit message
                raise Exception(f"Rate limit reached for {symbol}: {news_data['Note']}")
            
            if 'feed' not in news_data:
                raise ValueError(f"No news feed data available for {symbol}")
            
            sentiments = []
            for article in news_data['feed']:
                # Convert article date to datetime
                article_date = datetime.strptime(article['time_published'], '%Y%m%dT%H%M%S')
                
                # Only consider articles from the last 7 days
                if (date - article_date).days > 7:
                    continue
                
                # Use FinBERT for sentiment analysis
                inputs = self.tokenizer(article['title'], return_tensors="pt", padding=True, truncation=True)
                outputs = self.sentiment_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                
                # FinBERT output: [negative, neutral, positive]
                sentiment_score = float(probabilities[0][2].item() - probabilities[0][0].item())  # Ensure float type
                
                sentiments.append({
                    'date': article_date,
                    'sentiment_score': sentiment_score  # Range from -1 to 1
                })
            
            if not sentiments:
                return pd.DataFrame({'sentiment_score': [0.0]})  # Return float
                
            # For single article case, return the sentiment directly
            if len(sentiments) == 1:
                return pd.DataFrame({'sentiment_score': [float(sentiments[0]['sentiment_score'])]})  # Ensure float
                
            sentiment_df = pd.DataFrame(sentiments)
            # Weight recent sentiment more heavily
            sentiment_df['weight'] = sentiment_df['date'].apply(lambda x: 1 / (1 + (date - x).days))
            weighted_sentiment = float((sentiment_df['sentiment_score'] * sentiment_df['weight']).sum() / sentiment_df['weight'].sum())  # Ensure float
            
            return pd.DataFrame({'sentiment_score': [weighted_sentiment]})
            
        except requests.exceptions.RequestException as e:
            print(f"Network error getting news sentiment for {symbol}: {str(e)}")
            return pd.DataFrame({'sentiment_score': [0]})
        except ValueError as e:
            print(f"Value error getting news sentiment for {symbol}: {str(e)}")
            return pd.DataFrame({'sentiment_score': [0]})
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {str(e)}")
            return pd.DataFrame({'sentiment_score': [0]})