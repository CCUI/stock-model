import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import os
import json
from typing import Dict, List, Optional, Tuple
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class EnhancedAPIRateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                # Wait until the oldest request is more than 1 minute old
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.requests.append(now)

class BaseStockDataCollector(ABC):
    def __init__(self, market='UK', include_news_sentiment=True):
        self.market = market.upper()
        logger.info(f"Initializing {self.market} stock data collector")
        
        # Initialize enhanced rate limiter
        self.rate_limiter = EnhancedAPIRateLimiter()
        
        # Configure market-specific settings
        self.batch_size = 10 if market.upper() == 'US' else 25  # Smaller batch for US market
        self.yf_delay = 2.0 if market.upper() == 'US' else 0.5
        self.processing_delay = self.yf_delay
        
        # Initialize symbol to name mapping
        self.symbol_to_name_map = {}
        
        # Fetch market data to initialize symbol_to_name_map
        self._fetch_market_data()
        
        self.include_news_sentiment = include_news_sentiment
        self.symbols = self._get_symbols()
        
        # Initialize database manager
        self.db_manager = DatabaseManager(market=self.market)
        
        if include_news_sentiment:
            # Load news API key
            self.news_api_key = os.getenv('NEWS_API_KEY')
            if not self.news_api_key:
                logger.warning("NEWS_API_KEY not found in environment variables")
                self.include_news_sentiment = False
            
            # Initialize news cache
            self.news_cache = self.db_manager.load_sentiment_data()
            self.last_api_call = 0
            self.api_call_delay = 1.0  # Delay between API calls in seconds
            
            # Get NewsAPI key from environment variable
            self.news_api_key = os.getenv('NEWS_API_KEY')
            if not self.news_api_key:
                raise ValueError("NEWS_API_KEY not found in environment variables")
                
            # Initialize cache for news data
            self.news_cache = self.db_manager.load_sentiment_data()
            self.daily_api_limit = 100  # NewsAPI free tier limit
            self.last_sentiment_update_file = self.db_manager.market_dir / 'last_sentiment_update.json'
    
    @abstractmethod
    def _fetch_market_data(self):
        """Fetch both symbols and company names from market-specific source"""
        pass

    def _get_symbols(self):
        """Get list of stock symbols for the specific market"""
        try:
            return list(self.symbol_to_name_map.keys())
        except Exception as e:
            logger.error(f"Error getting symbols: {str(e)}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol"""
        try:
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
        """Collect historical data using parallel processing"""
        try:
            # Ensure end_date is a datetime object
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            elif not isinstance(end_date, datetime):
                raise ValueError(f"Invalid end_date type: {type(end_date)}. Expected datetime or string in YYYY-MM-DD format.")
            
            # Calculate start date
            start_date = end_date - timedelta(days=lookback_days)
            
            # Check if we need to update sentiment data
            if self.include_news_sentiment:
                try:
                    # Load last update information
                    last_update = {}
                    if self.last_sentiment_update_file.exists():
                        with open(self.last_sentiment_update_file, 'r') as f:
                            last_update = json.load(f)
                    
                    # Check if we need to update sentiment (update daily)
                    last_update_time = datetime.fromisoformat(last_update.get('last_update', '2000-01-01'))
                    current_time = datetime.now()
                    
                    # Force update if last update was more than 1 day ago
                    if (current_time - last_update_time).days >= 1:
                        logger.info("Updating sentiment data...")
                        # Get next batch of stocks that need sentiment updates
                        symbols_to_update = self._get_next_sentiment_batch()
                        
                        if not symbols_to_update:
                            logger.info("No symbols need sentiment updates")
                        else:
                            # Update sentiment for each symbol
                            for symbol in symbols_to_update:
                                try:
                                    sentiment_data = self._get_news_sentiment(symbol)
                                    self.news_cache[symbol] = sentiment_data
                                    # Save after each update to prevent data loss
                                    self.db_manager.save_sentiment_data(self.news_cache)
                                    time.sleep(self.api_call_delay)  # Respect rate limits
                                    logger.info(f"Updated sentiment for {symbol}")
                                except Exception as e:
                                    logger.error(f"Error updating sentiment for {symbol}: {str(e)}")
                            
                            # Update last update time
                            last_update['last_update'] = current_time.isoformat()
                            with open(self.last_sentiment_update_file, 'w') as f:
                                json.dump(last_update, f)
                            
                            logger.info("Sentiment data update completed")
                    else:
                        logger.info("Sentiment data is up to date")
                except Exception as e:
                    logger.error(f"Error updating sentiment data: {str(e)}")
            
            # Create batches of symbols
            symbol_batches = [
                self.symbols[i:i+self.batch_size] 
                for i in range(0, len(self.symbols), self.batch_size)
            ]
            
            # Function to process a batch
            def process_batch(batch_symbols):
                batch_data = {}
                for symbol in batch_symbols:
                    try:
                        # Check if we need to fetch new data
                        need_fresh_data = True
                        cached_data = self.db_manager.load_historical_data(symbols=[symbol], start_date=start_date, end_date=end_date)
                        
                        # Only use cache if it contains data for the requested end_date
                        if symbol in cached_data:
                            df = cached_data[symbol]
                            if not df.empty and df.index.max().date() >= end_date.date():
                                # Ensure Symbol column exists
                                df['Symbol'] = symbol
                                batch_data[symbol] = df
                                need_fresh_data = False
                        
                        if need_fresh_data:
                            # Fetch new data - use end_date + 1 day to ensure we get data for end_date
                            # This is because yfinance's end date is exclusive
                            stock = yf.Ticker(symbol)
                            next_day = end_date + timedelta(days=1)
                            hist_data = stock.history(start=start_date, end=next_day)
                            
                            if not hist_data.empty:
                                # Add fundamental data
                                fundamentals = self._get_fundamental_data(symbol)
                                for key, value in fundamentals.items():
                                    hist_data[key] = value
                                
                                # Add symbol column
                                hist_data['Symbol'] = symbol
                                
                                # Add company name column
                                company_name = self._get_company_name(symbol)
                                hist_data['CompanyName'] = company_name
                                
                                # Add news sentiment if enabled
                                if self.include_news_sentiment:
                                    sentiment_data = self._get_news_sentiment(symbol)
                                    hist_data['news_sentiment'] = sentiment_data.get('sentiment', 0)
                                
                                # Ensure all required columns exist
                                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                                for col in required_columns:
                                    if col not in hist_data.columns:
                                        logger.error(f"Missing required column {col} for {symbol}")
                                        raise ValueError(f"Missing required column {col}")
                                
                                batch_data[symbol] = hist_data
                                
                                # Save to database
                                self.db_manager.save_historical_data({symbol: hist_data})
                                
                                # Add delay to respect rate limits
                                time.sleep(self.yf_delay)
                            else:
                                logger.warning(f"No data returned for {symbol}")
                        
                    except Exception as e:
                        logger.error(f"Error collecting data for {symbol}: {str(e)}")
                
                return batch_data
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_batch, symbol_batches))
            
            # Combine results
            all_data = {}
            for batch_result in results:
                all_data.update(batch_result)
            
            # Save to SQLite database
            self.db_manager.save_historical_data(all_data)
            
            # Convert to DataFrame and ensure Symbol column exists
            df_list = []
            for symbol, df in all_data.items():
                if not df.empty:
                    # Ensure Symbol column exists
                    df['Symbol'] = symbol
                    df_list.append(df)
            
            return pd.concat(df_list) if df_list else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in collect_historical_data: {str(e)}")
            return pd.DataFrame()
    
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
            # Force update all stocks if any are older than 1 day
            current_time = datetime.now()
            needs_update = [item for item in stock_updates 
                           if (current_time - item['last_update']).days >= 1]
            
            if needs_update:
                logger.info(f"Found {len(needs_update)} stocks needing sentiment updates")
                return [item['symbol'] for item in needs_update[:self.daily_api_limit]]
            
            logger.info("No stocks need sentiment updates")
            return []
            
        except Exception as e:
            logger.error(f"Error getting next sentiment batch: {str(e)}")
            return []
    
    def _get_news_sentiment(self, symbol):
        """Get news sentiment using TextBlob"""
        try:
            if symbol in self.news_cache:
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
                    
                    # Calculate sentiment scores using TextBlob
                    sentiments = []
                    for text in texts:
                        blob = TextBlob(text)
                        sentiments.append(blob.sentiment.polarity)  # -1 to 1 scale
                    
                    # Calculate weighted average sentiment
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    
                    # Store in cache
                    sentiment_data = {
                        'sentiment': avg_sentiment,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.news_cache[symbol] = sentiment_data
                    
                    # Save to cache file
                    self.db_manager.save_sentiment_data(self.news_cache)
                    
                    return sentiment_data
            
            return {'sentiment': 0, 'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Error getting news sentiment for {symbol}: {str(e)}")
            return {'sentiment': 0, 'timestamp': datetime.now().isoformat()}

    def _detect_anomalies(self, df):
        """Detect and handle anomalies in the data"""
        # Check for price jumps
        returns = df['Close'].pct_change()
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Identify outliers (3 standard deviations)
        outliers = abs(returns - mean_return) > (3 * std_return)
        
        # Replace outliers with interpolated values
        if outliers.any():
            df.loc[outliers, 'Close'] = df['Close'].interpolate(method='linear')
            
            # Recalculate derived columns
            df['Open'] = df['Open'] * (df['Close'] / df['Close'].shift(0))
            df['High'] = df['High'] * (df['Close'] / df['Close'].shift(0))
            df['Low'] = df['Low'] * (df['Close'] / df['Close'].shift(0))
        
        return df

class UKStockDataCollector(BaseStockDataCollector):
    def _fetch_market_data(self):
        """Fetch FTSE 100 symbols and company names"""
        try:
            # FTSE 100 symbols
            ftse100_symbols = [
                'AAF.L', 'AAL.L', 'ABF.L', 'ADM.L', 'AHT.L', 'ANTO.L', 'AUTO.L', 'AV.L', 'AZN.L',
                'BA.L', 'BARC.L', 'BATS.L', 'BKG.L', 'BLND.L', 'BP.L', 'BRBY.L', 'BT-A.L',
                'CCH.L', 'CNA.L', 'CPG.L', 'CRDA.L', 'DGE.L', 'EXPN.L', 'EZJ.L', 'FERG.L',
                'FLTR.L', 'FRES.L', 'GSK.L', 'HIK.L', 'HLMA.L', 'HSBA.L', 'IAG.L', 'IHG.L',
                'III.L', 'IMB.L', 'INF.L', 'ITRK.L', 'ITV.L', 'JD.L', 'KGF.L', 'LAND.L',
                'LGEN.L', 'LLOY.L', 'LSE.L', 'MKS.L', 'MNDI.L', 'MRO.L', 'NG.L', 'NWG.L',
                'NXT.L', 'OCDO.L', 'PHNX.L', 'PRU.L', 'PSN.L', 'PSON.L', 'RB.L', 'RDSA.L',
                'RDSB.L', 'REL.L', 'RIO.L', 'RKT.L', 'RMV.L', 'RR.L', 'RSA.L', 'RTO.L',
                'SBRY.L', 'SDR.L', 'SGE.L', 'SGRO.L', 'SHP.L', 'SKG.L', 'SMDS.L', 'SMIN.L',
                'SMT.L', 'SN.L', 'SSE.L', 'STAN.L', 'STJ.L', 'SVT.L', 'TSCO.L', 'TUI.L',
                'ULVR.L', 'UU.L', 'VOD.L', 'WPP.L', 'WTB.L'
            ]
            
            # Initialize symbol to name mapping
            self.symbol_to_name_map = {}
            
            # Get company names using yfinance
            for symbol in ftse100_symbols:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    company_name = info.get('longName', symbol)
                    self.symbol_to_name_map[symbol] = company_name
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error fetching company name for {symbol}: {str(e)}")
                    self.symbol_to_name_map[symbol] = symbol
            
            logger.info(f"Successfully extracted {len(self.symbol_to_name_map)} UK stock symbols")
            
        except Exception as e:
            logger.error(f"Error fetching UK market data: {str(e)}")
            # Initialize with empty data if fetch fails
            self.symbol_to_name_map = {}

class USStockDataCollector(BaseStockDataCollector):
    def _fetch_market_data(self):
        """Fetch both symbols and company names for S&P 500 stocks"""
        try:
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            response = requests.get(sp500_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            table = soup.find('table', {'id': 'constituents'})
            self.symbol_to_name_map = {}
            
            if table:
                rows = table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        symbol = cols[0].text.strip()
                        company_name = cols[1].text.strip()
                        
                        # Clean up symbol (remove .B, etc.)
                        clean_symbol = symbol.split('.')[0]
                        
                        # Skip problematic symbols
                        if clean_symbol and not any(char in clean_symbol for char in ['-', '$']):
                            self.symbol_to_name_map[clean_symbol] = company_name
            
            logger.info(f"Found {len(self.symbol_to_name_map)} valid US stock symbols")
            return list(self.symbol_to_name_map.keys())
            
        except Exception as e:
            logger.error(f"Error fetching US market data: {str(e)}")
            return []