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
from .enhanced_rate_limiter import EnhancedAPIRateLimiter

logger = setup_logging()

class BaseStockDataCollector(ABC):
    def __init__(self, market='UK', include_news_sentiment=True):
        self.market = market.upper()
        logger.info(f"Initializing {self.market} stock data collector")
        
        # Initialize enhanced rate limiter
        self.rate_limiter = EnhancedAPIRateLimiter()
        
        # Configure market-specific settings
        self.batch_size = 10 if market.upper() == 'US' else 25  # Smaller batch for US market
        
        self.include_news_sentiment = include_news_sentiment
        self.symbols = self._get_symbols()
        self.data_manager = DataManager(market=self.market)
        
        # Initialize historical cache from market-specific directory
        self.historical_cache = self.data_manager.load_historical_data()
        
        # Initialize symbol to name mapping
        self.symbol_to_name_map = {}
        
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
            self.daily_api_limit = 100  # NewsAPI free tier limit
            self.last_sentiment_update_file = self.data_manager.market_dir / 'last_sentiment_update.json'
    
    @abstractmethod
    @abstractmethod
    def _fetch_market_data(self):
        """Fetch both symbols and company names from market-specific source"""
        pass

    def _get_symbols(self):
        """Get list of stock symbols for the specific market"""
        try:
            if not self.symbol_to_name_map:
                self._fetch_market_data()
            return list(self.symbol_to_name_map.keys())
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol"""
        try:
            if not self.symbol_to_name_map:
                self._fetch_market_data()
            return self.symbol_to_name_map.get(symbol, symbol)
        except Exception as e:
            logger.error(f"Error getting company name for {symbol}: {str(e)}")
            return symbol
    
    # These methods are no longer needed as we're using DataManager directly
    # The functionality is now handled by self.data_manager.load_historical_data() and
    # self.data_manager.save_historical_data()
    
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
        """Collect historical data in batches with proper rate limiting and parallel processing"""
        start_date = end_date - timedelta(days=lookback_days)
        all_data = []
        
        # First, identify which symbols need to be fetched (not in cache)
        symbols_to_fetch = []
        cached_data_map = {}
        
        for symbol in self.symbols:
            cached_data = self.historical_cache.get(symbol, {}).get('data', None)
            if cached_data is not None:
                cached_data_map[symbol] = cached_data
            else:
                symbols_to_fetch.append(symbol)
        
        logger.info(f"Found {len(cached_data_map)} cached symbols, need to fetch {len(symbols_to_fetch)}")
        
        # Process symbols in optimized batches
        for i in tqdm(range(0, len(symbols_to_fetch), self.batch_size), desc="Collecting data"):
            batch_symbols = symbols_to_fetch[i:i + self.batch_size]
            batch_data = {}
            
            # Step 1: Fetch historical price data for all symbols in batch
            for symbol in batch_symbols:
                try:
                    # Use rate limiter to fetch data with proper throttling
                    def fetch_history():
                        stock = yf.Ticker(symbol)
                        return stock.history(start=start_date, end=end_date)
                    
                    hist_data = self.rate_limiter.execute_yf_request(fetch_history)
                    
                    if not hist_data.empty:
                        # Add symbol column
                        hist_data['Symbol'] = symbol
                        batch_data[symbol] = hist_data
                except Exception as e:
                    logger.error(f"Error collecting price data for {symbol}: {str(e)}")
            
            # Step 2: Fetch fundamental data for all symbols with price data
            for symbol, hist_data in batch_data.items():
                try:
                    # Use rate limiter for fundamental data requests
                    fundamentals = self._get_fundamental_data(symbol)
                    for key, value in fundamentals.items():
                        hist_data[key] = value
                except Exception as e:
                    logger.error(f"Error collecting fundamental data for {symbol}: {str(e)}")
            
            # Step 3: Add sentiment data if enabled (in a separate batch to avoid mixing API calls)
            if self.include_news_sentiment:
                for symbol, hist_data in batch_data.items():
                    try:
                        sentiment_data = self._get_news_sentiment(symbol)
                        hist_data['news_sentiment'] = sentiment_data.get('sentiment', 0)
                    except Exception as e:
                        logger.error(f"Error collecting sentiment data for {symbol}: {str(e)}")
                        hist_data['news_sentiment'] = 0
            
            # Save batch data
            if batch_data:
                self.data_manager.save_historical_data(batch_data)
                all_data.extend(batch_data.values())
        
        # Combine with cached data
        all_data.extend(cached_data_map.values())
        
        return pd.concat(all_data) if all_data else pd.DataFrame()
    
    # These methods are no longer needed as we're using DataManager directly
    # The functionality is now handled by self.data_manager.load_sentiment_data() and
    # self.data_manager.save_sentiment_data()
    
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
            logger.error(f"Error getting next sentiment batch: {str(e)}")
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
            
            # Define the news API request function for rate limiting
            def fetch_news_articles():
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{company_name}" OR "{symbol}"',
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 10,
                    'apiKey': self.news_api_key
                }
                return requests.get(url, params=params)
            
            # Use rate limiter to make the request
            response = self.rate_limiter.execute_news_request(fetch_news_articles)
            
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
    def _fetch_market_data(self):
        """Fetch both symbols and company names for FTSE stocks"""
        try:
            ftse_url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
            response = requests.get(ftse_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table', {'class': 'wikitable'})
            
            for table in tables:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        company_name = cols[0].text.strip()
                        symbol_text = cols[1].text.strip()
                        
                        # Skip if the symbol contains numeric values (likely a price)
                        if any(char.isdigit() for char in symbol_text):
                            continue
                        
                        # Store both with and without .L suffix
                        symbol_with_suffix = symbol_text + '.L'
                        self.symbol_to_name_map[symbol_with_suffix] = company_name
                        self.symbol_to_name_map[symbol_text] = company_name
            
            # Ensure we have symbols
            if not self.symbol_to_name_map:
                raise ValueError("No valid FTSE symbols could be extracted from the webpage")
                
        except Exception as e:
            logger.error(f"Error fetching UK market data: {str(e)}")

class USStockDataCollector(BaseStockDataCollector):
    def _fetch_market_data(self):
        """Fetch both symbols and company names for S&P 500 stocks"""
        try:
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(sp500_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'wikitable'})
            
            if table:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        symbol_text = cols[0].text.strip()
                        company_name = cols[1].text.strip()
                        
                        # Skip problematic symbols and clean others
                        if symbol_text and not any(char.isdigit() for char in symbol_text):
                            # Remove .B suffix and other problematic characters
                            clean_symbol = symbol_text.split('.')[0]
                            if clean_symbol and not any(x in clean_symbol for x in ['-', '.', '$']):
                                self.symbol_to_name_map[clean_symbol] = company_name
            
            # Ensure we have symbols
            if not self.symbol_to_name_map:
                raise ValueError("No valid S&P 500 symbols could be extracted")
                
            logger.info(f"Found {len(self.symbol_to_name_map)} valid US stock symbols")
            
        except Exception as e:
            logger.error(f"Error fetching US market data: {str(e)}")