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
from abc import ABC, abstractmethod
from tqdm import tqdm
import json
from .data_manager import DataManager


class BaseStockDataCollector(ABC):
    def __init__(self, market='UK', include_news_sentiment=True):
        self.market = market.upper()
        self.include_news_sentiment = include_news_sentiment
        self.symbols = self._get_symbols()
        self.data_manager = DataManager(market=self.market)
        
        if include_news_sentiment:
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            
            # Get NewsAPI key from environment variable
            self.news_api_key = os.getenv('NEWS_API_KEY')
            if not self.news_api_key:
                raise ValueError("NEWS_API_KEY not found in environment variables")
                
            # Initialize cache for news data
            self.news_cache = self.data_manager.load_sentiment_data()
            self.last_api_call = 0
            self.api_call_delay = 2  # Increased delay between API calls
            self.max_retries = 3  # Maximum number of retries for API calls
            self.backoff_factor = 2  # Exponential backoff factor
            self.batch_size = 5  # Number of companies to batch in one API call
            
            # Load cached sentiment data from file
            self._load_sentiment_cache()
    
    @abstractmethod
    def _get_symbols(self):
        """Get list of stock symbols for the specific market"""
        pass
    
    @abstractmethod
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol"""
        pass
    
    def _load_historical_cache(self):
        """Load historical data cache from local file"""
        cache_file = Path(__file__).parent.parent / 'data' / 'historical_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Convert the loaded data back to DataFrame format
                    self.historical_cache = {}
                    for symbol, data in cache_data.items():
                        try:
                            df = pd.DataFrame(data['data'])
                            df.index = pd.to_datetime(df.index)
                            self.historical_cache[symbol] = {
                                'data': df,
                                'timestamp': data['timestamp']
                            }
                        except Exception as e:
                            print(f"Error loading historical cache for {symbol}: {str(e)}")
            except Exception as e:
                print(f"Error loading historical cache: {str(e)}")
                self.historical_cache = {}
        else:
            # Create the data directory if it doesn't exist
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.historical_cache = {}
    
    def _save_historical_cache(self):
        """Save historical data cache to local file"""
        cache_file = Path(__file__).parent.parent / 'data' / 'historical_cache.json'
        try:
            # Convert DataFrame values to serializable format
            cache_data = {}
            for symbol, data in self.historical_cache.items():
                try:
                    # Convert DataFrame to records and handle Timestamp objects
                    df_records = data['data'].reset_index()
                    # Convert all Timestamp and datetime64 columns to ISO format strings
                    for col in df_records.columns:
                        if pd.api.types.is_datetime64_any_dtype(df_records[col]):
                            df_records[col] = df_records[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
                    
                    # Convert any remaining Timestamp objects in the data
                    records = df_records.to_dict('records')
                    for record in records:
                        for key, value in record.items():
                            if isinstance(value, (pd.Timestamp, datetime)):
                                record[key] = value.strftime('%Y-%m-%dT%H:%M:%S')
                    
                    cache_data[symbol] = {
                        'data': records,
                        'timestamp': data['timestamp']
                    }
                except Exception as e:
                    print(f"Error saving historical cache for {symbol}: {str(e)}")
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving historical cache: {str(e)}")
    
    def collect_historical_data(self, end_date, lookback_days=365, max_retries=5, delay=2):
        """Collect historical price and fundamental data for stocks"""
        start_date = end_date - timedelta(days=lookback_days)
        
        # Load historical cache
        self.historical_cache = self.data_manager.load_historical_data()
        
        all_data = {}
        progress_bar = tqdm(self.symbols, desc=f'Collecting {self.market} stock data', unit='stock')
        for symbol in progress_bar:
            progress_bar.set_description(f'Processing {symbol}')
            
            # Check cache first
            if symbol in self.historical_cache:
                cached_data = self.historical_cache[symbol]
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=24):  # Cache valid for 24 hours
                    all_data[symbol] = cached_data['data']
                    continue
            
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
                        
                        # Get news sentiment if enabled
                        if self.include_news_sentiment:
                            news = self._get_news_sentiment(symbol)
                            hist_data['news_sentiment'] = news['sentiment_score'].mean() if not news.empty else 0
                        else:
                            hist_data['news_sentiment'] = 0  # Default value when news sentiment is disabled
                        
                        # Update cache
                        self.historical_cache[symbol] = {
                            'data': hist_data,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Save cache after each successful update
                        self._save_historical_cache()
                        
                        all_data[symbol] = hist_data
                        break  # Successfully got data, break retry loop
                    
                except Exception as e:
                    if 'Too Many Requests' in str(e) and attempt < max_retries - 1:
                        # Exponential backoff with increased base delay
                        wait_time = delay * (2 ** attempt)  # Exponential increase in wait time
                        progress_bar.write(f"Rate limit hit for {symbol}, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    progress_bar.write(f"Error collecting data for {symbol}: {str(e)}")
                    break  # Break on non-rate-limit errors or final attempt
                
                # Add delay between successful requests to prevent rate limiting
                time.sleep(delay)
            
            # Save data periodically to prevent memory issues
            if len(all_data) >= self.data_manager.chunk_size:
                self.data_manager.save_historical_data(all_data)
                all_data = {}
        
        # Save any remaining data
        if all_data:
            self.data_manager.save_historical_data(all_data)
        
        # Save sentiment cache if used
        if self.include_news_sentiment:
            self.data_manager.save_sentiment_data(self.news_cache)
        
        # Return combined data from all chunks
        return pd.concat(self.data_manager.load_historical_data().values())
    
    def _load_sentiment_cache(self):
        """Load sentiment cache from local file"""
        cache_file = Path(__file__).parent.parent / 'data' / 'sentiment_cache.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    # Convert the loaded data back to DataFrame format
                    self.news_cache = {k: pd.DataFrame({'sentiment_score': [v['sentiment_score']],
                                                      'timestamp': [v['timestamp']]})
                                      for k, v in cache_data.items()}
            except Exception as e:
                print(f"Error loading sentiment cache: {str(e)}")
                self.news_cache = {}
        else:
            # Create the data directory if it doesn't exist
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.news_cache = {}
    
    def _save_sentiment_cache(self):
        """Save sentiment cache to local file"""
        cache_file = Path(__file__).parent.parent / 'data' / 'sentiment_cache.json'
        try:
            # Convert DataFrame values to serializable format
            cache_data = {}
            for k, v in self.news_cache.items():
                # Ensure sentiment_score is a native Python float
                sentiment_score = float(v['sentiment_score'].iloc[0]) if isinstance(v['sentiment_score'], pd.Series) else float(v['sentiment_score'])
                # Get timestamp, defaulting to current time if not present
                timestamp = v['timestamp'].iloc[0] if isinstance(v['timestamp'], pd.Series) else v.get('timestamp', datetime.now().isoformat())
                
                cache_data[k] = {
                    'sentiment_score': sentiment_score,
                    'timestamp': timestamp
                }
                
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Error saving sentiment cache: {str(e)}")
    
    def _get_news_sentiment(self, symbol: str) -> pd.DataFrame:
        """Collect and analyze news sentiment for a given stock"""
        try:
            company_name = self._get_company_name(symbol)
            
            # Check cache first
            if company_name in self.news_cache:
                cached_data = self.news_cache[company_name]
                # Check if cached data is less than 24 hours old
                if 'timestamp' in cached_data:
                    cache_time = datetime.fromisoformat(cached_data['timestamp'])
                    if datetime.now() - cache_time < timedelta(hours=24):
                        return cached_data
            
            # Implement enhanced rate limiting with retries
            for attempt in range(self.max_retries):
                current_time = time.time()
                if current_time - self.last_api_call < self.api_call_delay:
                    wait_time = self.api_call_delay - (current_time - self.last_api_call)
                    time.sleep(wait_time)
                
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
                
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        wait_time = self.api_call_delay * (self.backoff_factor ** attempt)
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Rate limit exceeded for {company_name} after {self.max_retries} retries")
                else:
                    print(f"Error fetching news for {company_name}: {response.status_code}")
                    break
            
            if response.status_code != 200:
                print(f"Error fetching news for {company_name}: {response.status_code}")
                return pd.DataFrame({'sentiment_score': [0.0], 'timestamp': datetime.now().isoformat()})
            
            data = response.json()
            
            if data['totalResults'] == 0:
                return pd.DataFrame({'sentiment_score': [0.0], 'timestamp': datetime.now().isoformat()})
            
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
                return pd.DataFrame({'sentiment_score': [0.0], 'timestamp': datetime.now().isoformat()})
            
            # Calculate weighted sentiment (more recent articles have higher weight)
            sentiment_df = pd.DataFrame(sentiments)
            sentiment_df['weight'] = sentiment_df['date'].apply(
                lambda x: 1 / (1 + (datetime.now() - x).days)
            )
            weighted_sentiment = float(
                (sentiment_df['sentiment_score'] * sentiment_df['weight']).sum() 
                / sentiment_df['weight'].sum()
            )
            
            result = pd.DataFrame({
                'sentiment_score': [weighted_sentiment],
                'timestamp': datetime.now().isoformat()
            })
            
            # Cache the result
            self.news_cache[company_name] = result
            
            # Save updated cache to file
            self._save_sentiment_cache()
            
            return result
            
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {str(e)}")
            return pd.DataFrame({'sentiment_score': [0.0], 'timestamp': datetime.now().isoformat()})

class UKStockDataCollector(BaseStockDataCollector):
    def _get_symbols(self):
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
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol using Wikipedia data"""
        try:
            # Remove .L suffix for lookup
            clean_symbol = symbol.replace('.L', '')
            
            # First check if we already have the mapping
            if hasattr(self, 'symbol_to_name_map'):
                return self.symbol_to_name_map.get(clean_symbol, clean_symbol)
            
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
            
            return self.symbol_to_name_map.get(clean_symbol, clean_symbol)
            
        except Exception as e:
            print(f"Error getting company name for {symbol}: {str(e)}")
            return symbol

class USStockDataCollector(BaseStockDataCollector):
    def _get_symbols(self):
        """Get list of S&P 500 stocks"""
        sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        
        symbols = []
        response = requests.get(sp500_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        if table:
            rows = table.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 1:
                    symbol = cols[0].text.strip()
                    if symbol and not any(char.isdigit() for char in symbol):
                        symbols.append(symbol)
        
        # Remove duplicates
        unique_symbols = list(set(symbols))
        
        # Ensure we have symbols
        if not unique_symbols:
            raise ValueError("No valid S&P 500 symbols could be extracted from the webpage")
            
        return unique_symbols
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol using Wikipedia data"""
        try:
            # First check if we already have the mapping
            if hasattr(self, 'symbol_to_name_map'):
                return self.symbol_to_name_map.get(symbol, symbol)
            
            # Create the mapping
            self.symbol_to_name_map = {}
            response = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            
            if table:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        symbol_text = cols[0].text.strip()
                        company_name = cols[1].text.strip()
                        self.symbol_to_name_map[symbol_text] = company_name
            
            return self.symbol_to_name_map.get(symbol, symbol)
            
        except Exception as e:
            print(f"Error getting company name for {symbol}: {str(e)}")
            return symbol