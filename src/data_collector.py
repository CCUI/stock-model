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
import logging
from .utils import setup_logging

logger = setup_logging()

class BaseStockDataCollector(ABC):
    def __init__(self, market='UK', include_news_sentiment=True):
        self.market = market.upper()
        logger.info(f"Initializing {self.market} stock data collector")
        
        # Add delay settings for US market
        self.yf_delay = 2.0 if market.upper() == 'US' else 0.5  # Longer delay for US market
        self.batch_size = 10 if market.upper() == 'US' else 25  # Smaller batch for US market
        
        self.include_news_sentiment = include_news_sentiment
        self.symbols = self._get_symbols()
        self.data_manager = DataManager(market=self.market)
        
        # Initialize historical cache from market-specific directory
        self.historical_cache = self.data_manager.load_historical_data()
        
        # Batch processing settings
        self.processing_delay = self.yf_delay  # Use market-specific delay
        
        if include_news_sentiment:
            # Load FinBERT model only if needed
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                'ProsusAI/finbert',
                torch_dtype=torch.float32  # Use float32 instead of float16 for better compatibility
            )
            self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            
            # Initialize news cache
            self.news_cache = self.data_manager.load_sentiment_data()
            
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
            self.daily_api_limit = 100  # NewsAPI free tier limit
            self.last_sentiment_update_file = self.data_manager.market_dir / 'last_sentiment_update.json'  # Number of companies to batch in one API call
    
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
        cache_file = self.data_manager.historical_cache_file
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
        cache_file = self.data_manager.historical_cache_file
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
    
    def _get_fundamental_data(self, symbol):
        """Get fundamental data for a stock"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            fundamentals = {
                'marketCap': info.get('marketCap', 0),
                'trailingPE': info.get('trailingPE', 0),
                'priceToBook': info.get('priceToBook', 0),
                'debtToEquity': info.get('debtToEquity', 0)
            }
            
            # Convert None values to 0
            for key in fundamentals:
                if fundamentals[key] is None:
                    fundamentals[key] = 0
            
            # Convert market cap to millions (£M)
            fundamentals['marketCap'] = fundamentals['marketCap'] / 1_000_000
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            return {
                'marketCap': 0,
                'trailingPE': 0,
                'priceToBook': 0,
                'debtToEquity': 0
            }

    def collect_historical_data(self, end_date, lookback_days=365):
        """Collect historical data in batches"""
        start_date = end_date - timedelta(days=lookback_days)
        all_data = []
        
        # Process symbols in batches
        for i in tqdm(range(0, len(self.symbols), self.batch_size), desc="Collecting data"):
            batch_symbols = self.symbols[i:i + self.batch_size]
            batch_data = []
            
            for symbol in batch_symbols:
                try:
                    # Check cache first
                    cached_data = self.historical_cache.get(symbol, {}).get('data', None)
                    if cached_data is not None:
                        batch_data.append(cached_data)
                        continue
                    
                    # Add delay to avoid rate limits
                    time.sleep(self.yf_delay)
                    
                    # Fetch new data with retry mechanism
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            stock = yf.Ticker(symbol)
                            hist_data = stock.history(start=start_date, end=end_date)
                            
                            if not hist_data.empty:
                                # Add fundamental data
                                fundamentals = self._get_fundamental_data(symbol)
                                for key, value in fundamentals.items():
                                    hist_data[key] = value
                                
                                # Add symbol column
                                hist_data['Symbol'] = symbol
                                
                                # Add news sentiment if enabled
                                if self.include_news_sentiment:
                                    sentiment_data = self._get_news_sentiment(symbol)
                                    hist_data['news_sentiment'] = sentiment_data.get('sentiment', 0)
                                
                                batch_data.append(hist_data)
                                break
                                
                        except Exception as e:
                            if "Rate limit" in str(e) and attempt < max_retries - 1:
                                wait_time = (attempt + 1) * self.yf_delay
                                logger.warning(f"Rate limit hit for {symbol}, waiting {wait_time}s...")
                                time.sleep(wait_time)
                                continue
                            else:
                                raise
                                
                except Exception as e:
                    logger.error(f"Error collecting data for {symbol}: {str(e)}")
            
            # Save batch data
            if batch_data:
                self.data_manager.save_historical_data(
                    {sym: df for sym, df in zip(batch_symbols, batch_data)}
                )
                all_data.extend(batch_data)
            
            # Delay between batches
            time.sleep(self.processing_delay)
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    def _load_sentiment_cache(self):
        """Load sentiment cache from local file"""
        cache_file = self.data_manager.sentiment_cache_file
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
        cache_file = self.data_manager.sentiment_cache_file
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
    
    def _get_next_sentiment_batch(self) -> list:
        """Get the next batch of stocks that need sentiment updates"""
        try:
            # Load last update information
            last_update = {}
            if self.last_sentiment_update_file.exists():
                with open(self.last_sentiment_update_file, 'r') as f:
                    last_update = json.load(f)
            
            # Get all stocks sorted by update time
            stock_updates = []
            for symbol in self.symbols:
                company_name = self._get_company_name(symbol)
                last_update_time = None
                
                if company_name in self.news_cache:
                    cached_data = self.news_cache[company_name]
                    if 'timestamp' in cached_data:
                        try:
                            last_update_time = datetime.fromisoformat(str(cached_data['timestamp'].iloc[0]))
                        except (ValueError, AttributeError):
                            pass
                
                stock_updates.append({
                    'symbol': symbol,
                    'company_name': company_name,
                    'last_update': last_update_time or datetime.min
                })
            
            # Sort by last update time
            stock_updates.sort(key=lambda x: x['last_update'])
            
            # Get the oldest updated stocks up to the daily limit
            return [item['symbol'] for item in stock_updates[:self.daily_api_limit]]
            
        except Exception as e:
            print(f"Error getting next sentiment batch: {str(e)}")
            return []
    
    def _get_news_sentiment(self, symbol):
        """Get news sentiment for a stock"""
        try:
            # Check if this symbol is in the current batch that needs updating
            current_batch = self._get_next_sentiment_batch()
            if symbol not in current_batch and symbol in self.news_cache:
                return self.news_cache[symbol]
            
            # Get company name for better news search
            company_name = self._get_company_name(symbol)
            
            # Ensure we respect rate limits
            current_time = time.time()
            if current_time - self.last_api_call < self.api_call_delay:
                time.sleep(self.api_call_delay - (current_time - self.last_api_call))
            
            # Search for news articles
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f'"{company_name}" OR "{symbol}"',
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 10,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params)
            self.last_api_call = time.time()
            
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                if articles:
                    # Combine title and description for sentiment analysis
                    texts = [
                        f"{article.get('title', '')} {article.get('description', '')}"
                        for article in articles
                    ]
                    
                    # Calculate sentiment scores
                    sentiments = []
                    for text in texts:
                        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                        outputs = self.sentiment_model(**inputs)
                        sentiment = torch.softmax(outputs.logits, dim=1)
                        sentiments.append(sentiment[0][1].item())  # Positive sentiment score
                    
                    # Calculate weighted average sentiment
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    
                    # Store in cache
                    sentiment_data = {
                        'sentiment': avg_sentiment,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.news_cache[symbol] = sentiment_data
                    
                    # Save to cache file
                    self.data_manager.save_sentiment_data(self.news_cache)
                    
                    return sentiment_data
            
            return {'sentiment': 0, 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return {'sentiment': 0, 'timestamp': datetime.now().isoformat()}

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
        try:
            response = requests.get(sp500_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            
            if table:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 1:
                        symbol = cols[0].text.strip()
                        # Skip problematic symbols and clean others
                        if symbol and not any(char.isdigit() for char in symbol):
                            # Remove .B suffix and other problematic characters
                            symbol = symbol.split('.')[0]
                            if symbol:
                                symbols.append(symbol)
            
            # Remove duplicates and problematic symbols
            unique_symbols = list(set(symbols))
            filtered_symbols = [
                sym for sym in unique_symbols 
                if not any(x in sym for x in ['-', '.', '$'])
            ]
            
            if not filtered_symbols:
                raise ValueError("No valid S&P 500 symbols could be extracted")
            
            logger.info(f"Found {len(filtered_symbols)} valid US stock symbols")
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
            return []
    
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