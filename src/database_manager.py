import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import threading

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, market='UK'):
        self.market = market.upper()
        self.market_dir = Path(__file__).parent.parent / 'data' / market.lower()
        self.market_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.market_dir / f'{market.lower()}_market.db'
        self._local = threading.local()
    
    def _get_connection(self):
        """Get a thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(str(self.db_path))
            # Create tables if they don't exist
            self._create_tables(self._local.connection)
        return self._local.connection
    
    def _create_tables(self, conn):
        """Create database tables"""
        cursor = conn.cursor()
        
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
            news_sentiment REAL,
            company_name TEXT,
            PRIMARY KEY (symbol, date)
        )
        ''')
        
        # Sentiment data
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_data (
            symbol TEXT,
            date TEXT,
            sentiment REAL,
            timestamp TEXT,
            PRIMARY KEY (symbol, date)
        )
        ''')
        
        conn.commit()
    
    def save_historical_data(self, data_dict):
        """Save historical data to database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
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
                    row.get('debtToEquity', 0),
                    row.get('news_sentiment', 0),
                    row.get('CompanyName', 'Unknown')
                )
                records.append(record)
            
            # Insert records
            cursor.executemany('''
            INSERT OR REPLACE INTO historical_data
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)
        
        conn.commit()
    
    def load_historical_data(self, symbols=None, start_date=None, end_date=None):
        """Load historical data from database"""
        conn = self._get_connection()
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
        cursor = conn.cursor()
        cursor.execute(query, params)
        
        # Convert to DataFrame
        columns = [
            'symbol', 'date', 'open', 'high', 'low', 'close', 'volume',
            'market_cap', 'pe_ratio', 'price_to_book', 'debt_to_equity',
            'news_sentiment', 'company_name'
        ]
        data = cursor.fetchall()
        
        if not data:
            return {}
        
        df = pd.DataFrame(data, columns=columns)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Group by symbol
        result = {}
        for symbol, group in df.groupby('symbol'):
            result[symbol] = group
        
        return result
    
    def save_sentiment_data(self, sentiment_dict):
        """Save sentiment data to database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        current_time = datetime.now().isoformat()
        
        for symbol, data in sentiment_dict.items():
            cursor.execute('''
            INSERT OR REPLACE INTO sentiment_data (symbol, date, sentiment, timestamp)
            VALUES (?, ?, ?, ?)
            ''', (
                symbol,
                datetime.now().strftime('%Y-%m-%d'),
                float(data['sentiment']),
                current_time
            ))
        
        conn.commit()
    
    def load_sentiment_data(self):
        """Load sentiment data from database"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
        SELECT symbol, sentiment, timestamp
        FROM sentiment_data
        WHERE date = ?
        ''', (datetime.now().strftime('%Y-%m-%d'),))
        
        data = cursor.fetchall()
        return {
            symbol: {
                'sentiment': float(sentiment),
                'timestamp': timestamp
            }
            for symbol, sentiment, timestamp in data
        } 