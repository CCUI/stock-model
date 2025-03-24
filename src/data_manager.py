import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import sqlite3

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, market='UK'):
        self.market = market.upper()
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure market-specific directories exist
        self.market_dir = self.data_dir / self.market.lower()
        self.market_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate files for different data types and chunks in market-specific directory
        self.historical_cache_file = self.market_dir / 'historical_cache.json'
        self.sentiment_cache_file = self.market_dir / 'sentiment_cache.json'
        self.chunk_size = 25  # Reduced chunk size
        self.max_chunks_in_memory = 4  # Limit concurrent chunks in memory
        
    def save_historical_data(self, data_dict):
        """Save historical data in chunks"""
        try:
            # Create directory if it doesn't exist
            self.market_dir.mkdir(parents=True, exist_ok=True)
            
            # Split data into chunks
            chunks = {}
            symbols = list(data_dict.keys())
            
            for i in range(0, len(symbols), self.chunk_size):
                chunk_symbols = symbols[i:i + self.chunk_size]
                chunk_data = {sym: data_dict[sym] for sym in chunk_symbols}
                chunk_file = self.market_dir / f'historical_chunk_{i//self.chunk_size}.json'
                
                # Save chunk
                self._save_chunk(chunk_file, chunk_data)
                chunks[f'chunk_{i//self.chunk_size}'] = chunk_file.name
            
            # Save chunk index
            with open(self.historical_cache_file, 'w') as f:
                json.dump({
                    'chunks': chunks, 
                    'last_updated': datetime.now().isoformat()
                }, f)
                
            logger.info(f"Saved historical data in {len(chunks)} chunks for {self.market} market")
            
        except Exception as e:
            logger.error(f"Error saving historical data: {str(e)}")
    
    def _save_chunk(self, chunk_file, chunk_data):
        """Save a single chunk of data"""
        serialized_chunk = {}
        for symbol, data in chunk_data.items():
            if isinstance(data, pd.DataFrame):
                df = data
                timestamp = datetime.now()
            else:
                df = data['data']
                timestamp = data['timestamp']
            
            # Convert DataFrame to records and handle datetime objects
            df_records = df.reset_index()
            for col in df_records.columns:
                if pd.api.types.is_datetime64_any_dtype(df_records[col]):
                    df_records[col] = df_records[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
                elif isinstance(df_records[col].iloc[0], (pd.Timestamp, datetime)):
                    df_records[col] = df_records[col].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S'))
            
            serialized_chunk[symbol] = {
                'data': df_records.to_dict('records'),
                'timestamp': timestamp.isoformat()
            }
        
        with open(chunk_file, 'w') as f:
            json.dump(serialized_chunk, f)
    
    def load_historical_data(self):
        """Load historical data from chunks"""
        try:
            if not self.historical_cache_file.exists():
                return {}
            
            with open(self.historical_cache_file, 'r') as f:
                index = json.load(f)
            
            data_dict = {}
            for chunk_file in index['chunks'].values():
                chunk_path = self.market_dir / chunk_file
                if chunk_path.exists():
                    with open(chunk_path, 'r') as f:
                        chunk_data = json.load(f)
                        for symbol, data in chunk_data.items():
                            try:
                                df = pd.DataFrame(data['data'])
                                df.set_index('Date', inplace=True)
                                df.index = pd.to_datetime(df.index)
                                # Ensure Symbol column is present
                                df['Symbol'] = symbol
                                # Ensure CompanyName column is present
                                if 'CompanyName' not in df.columns:
                                    company_name = next((col_data for col_name, col_data in df.iloc[0].items() if col_name.startswith('CompanyName')), 'Unknown')
                                    df['CompanyName'] = company_name
                                data_dict[symbol] = {
                                    'data': df,
                                    'timestamp': data['timestamp']
                                }
                            except Exception as e:
                                logger.error(f"Error loading data for {symbol}: {str(e)}")
            
            return data_dict
            
        except Exception as e:
            logger.error(f"Error loading historical data: {str(e)}")
            return {}
    def save_sentiment_data(self, sentiment_dict):
        """Save sentiment data to file"""
        try:
            serialized_data = {}
            for symbol, data in sentiment_dict.items():
                serialized_data[symbol] = {
                    'sentiment': float(data['sentiment']),
                    'timestamp': data['timestamp']
                }
            
            with open(self.sentiment_cache_file, 'w') as f:
                json.dump(serialized_data, f)
                
            logger.info(f"Saved sentiment data for {self.market} market")
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {str(e)}")
    
    def load_sentiment_data(self):
        """Load sentiment data from file"""
        try:
            if not self.sentiment_cache_file.exists():
                return {}
            
            with open(self.sentiment_cache_file, 'r') as f:
                data = json.load(f)
            
            return {
                symbol: {
                    'sentiment': float(item['sentiment']),
                    'timestamp': item['timestamp']
                }
                for symbol, item in data.items()
            }
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {str(e)}")
            return {}

    def update_historical_data(self, data_dict, max_age_days=1):
        """Update historical data incrementally"""
        # Load existing data
        existing_data = self.load_historical_data()
        
        # Current time
        current_time = datetime.now()
        
        # Update only stale or missing data
        updated_data = {}
        for symbol, data in data_dict.items():
            # Check if data exists and is fresh
            if symbol in existing_data:
                timestamp = datetime.fromisoformat(existing_data[symbol]['timestamp'])
                age_days = (current_time - timestamp).days
                
                if age_days <= max_age_days:
                    # Data is fresh, keep existing
                    updated_data[symbol] = existing_data[symbol]
                    continue
            
            # Data is stale or missing, use new data
            updated_data[symbol] = {
                'data': data,
                'timestamp': current_time.isoformat()
            }
        
        # Save updated data
        self._save_data(updated_data)
        
        return updated_data

    def refresh_cache(self):
        """Delete all cache files to force fresh data collection"""
        try:
            # Delete the SQLite database
            db_path = self.market_dir / f'{self.market.lower()}_market.db'
            if db_path.exists():
                db_path.unlink()
                logger.info(f"Deleted SQLite database: {db_path}")

            # Delete the sentiment last update file
            last_sentiment_update_file = self.market_dir / 'last_sentiment_update.json'
            if last_sentiment_update_file.exists():
                last_sentiment_update_file.unlink()
                logger.info(f"Deleted sentiment update file: {last_sentiment_update_file}")
            
            # Delete sentiment cache file
            if self.sentiment_cache_file.exists():
                self.sentiment_cache_file.unlink()
                logger.info(f"Deleted sentiment cache file: {self.sentiment_cache_file}")
                
            # Recreate database
            self.db_manager = DatabaseManager(market=self.market)
            
            logger.info("Cache refreshed. Fresh data will be collected on next run.")
            
        except Exception as e:
            logger.error(f"Error refreshing cache: {str(e)}")
            raise

class DatabaseManager:
    def __init__(self, market='UK'):
        self.market = market.upper()
        self.db_path = Path(__file__).parent.parent / 'data' / f'{market.lower()}_market.db'
        self.conn = sqlite3.connect(str(self.db_path))
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()
        
        # Historical price data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_data (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            market_cap REAL,
            pe_ratio REAL,
            price_to_book REAL,
            debt_to_equity REAL,
            PRIMARY KEY (symbol, date)
        )
        ''')
        
        # Sentiment data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            symbol TEXT,
            date TEXT,
            news_sentiment REAL,
            social_sentiment REAL,
            PRIMARY KEY (symbol, date)
        )
        ''')
        
        self.conn.commit()
    
    def save_historical_data(self, data_dict):
        """Save historical data to database"""
        cursor = self.conn.cursor()
        
        for symbol, df in data_dict.items():
            # Convert DataFrame to records
            records = []
            for date, row in df.iterrows():
                record = (
                    symbol,
                    date.strftime('%Y-%m-%d'),
                    row.get('Open', 0),
                    row.get('High', 0),
                    row.get('Low', 0),
                    row.get('Close', 0),
                    row.get('Volume', 0),
                    row.get('marketCap', 0),
                    row.get('trailingPE', 0),
                    row.get('priceToBook', 0),
                    row.get('debtToEquity', 0)
                )
                records.append(record)
            
            # Insert records
            cursor.executemany('''
            INSERT OR REPLACE INTO historical_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
        
        self.conn.commit()
    
    def load_historical_data(self, symbols=None, start_date=None, end_date=None):
        """Load historical data from database"""
        query = '''
        SELECT * FROM historical_data
        '''
        
        conditions = []
        params = []
        
        if symbols:
            placeholders = ','.join(['?'] * len(symbols))
            conditions.append(f'symbol IN ({placeholders})')
            params.extend(symbols)
        
        if start_date:
            conditions.append('date >= ?')
            params.append(start_date.strftime('%Y-%m-%d'))
        
        if end_date:
            conditions.append('date <= ?')
            params.append(end_date.strftime('%Y-%m-%d'))
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        # Execute query
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        # Convert to DataFrame
        columns = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'volume',
            'market_cap', 'pe_ratio', 'price_to_book', 'debt_to_equity'
        ]
        data = cursor.fetchall()
        
        df = pd.DataFrame(data, columns=columns)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Group by symbol
        result = {}
        for symbol, group in df.groupby('symbol'):
            result[symbol] = group
        
        return result