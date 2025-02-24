import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

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
        """Save historical data with memory optimization"""
        try:
            # Clear existing chunks to free memory
            for chunk_file in self.market_dir.glob('historical_chunk_*.json'):
                chunk_file.unlink()
            
            chunks = {}
            for i in range(0, len(data_dict), self.chunk_size):
                chunk_symbols = list(data_dict.keys())[i:i + self.chunk_size]
                chunk_data = {sym: data_dict[sym] for sym in chunk_symbols}
                
                # Save chunk immediately and clear from memory
                chunk_file = self.market_dir / f'historical_chunk_{i//self.chunk_size}.json'
                self._save_chunk(chunk_file, chunk_data)
                chunks[f'chunk_{i//self.chunk_size}'] = chunk_file.name
                
                # Clear chunk data from memory
                del chunk_data
            
            # Save index file
            with open(self.historical_cache_file, 'w') as f:
                json.dump({
                    'chunks': chunks,
                    'last_updated': datetime.now().isoformat()
                }, f)
            
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