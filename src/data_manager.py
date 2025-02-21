import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, market='UK'):
        self.market = market.upper()
        self.data_dir = Path(__file__).parent.parent / 'data' / self.market.lower()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate files for different data types and chunks
        self.historical_cache_file = self.data_dir / 'historical_cache.json'
        self.sentiment_cache_file = self.data_dir / 'sentiment_cache.json'
        self.chunk_size = 50  # Number of stocks per chunk
        
    def save_historical_data(self, data_dict):
        """Save historical data in chunks"""
        try:
            # Split data into chunks
            chunks = {}
            symbols = list(data_dict.keys())
            
            for i in range(0, len(symbols), self.chunk_size):
                chunk_symbols = symbols[i:i + self.chunk_size]
                chunk_data = {sym: data_dict[sym] for sym in chunk_symbols}
                chunk_file = self.data_dir / f'historical_chunk_{i//self.chunk_size}.json'
                
                # Convert DataFrame values to serializable format
                serialized_chunk = {}
                for symbol, data in chunk_data.items():
                    df_records = data['data'].reset_index()
                    for col in df_records.columns:
                        if pd.api.types.is_datetime64_any_dtype(df_records[col]):
                            df_records[col] = df_records[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
                    
                    serialized_chunk[symbol] = {
                        'data': df_records.to_dict('records'),
                        'timestamp': data['timestamp']
                    }
                
                # Save chunk
                with open(chunk_file, 'w') as f:
                    json.dump(serialized_chunk, f)
                
                chunks[f'chunk_{i//self.chunk_size}'] = chunk_file.name
            
            # Save chunk index
            with open(self.historical_cache_file, 'w') as f:
                json.dump({'chunks': chunks, 'last_updated': datetime.now().isoformat()}, f)
                
            logger.info(f"Saved historical data in {len(chunks)} chunks for {self.market} market")
            
        except Exception as e:
            logger.error(f"Error saving historical data: {str(e)}")
    
    def load_historical_data(self):
        """Load historical data from chunks"""
        try:
            if not self.historical_cache_file.exists():
                return {}
            
            with open(self.historical_cache_file, 'r') as f:
                index = json.load(f)
            
            data_dict = {}
            for chunk_file in index['chunks'].values():
                chunk_path = self.data_dir / chunk_file
                if chunk_path.exists():
                    with open(chunk_path, 'r') as f:
                        chunk_data = json.load(f)
                        for symbol, data in chunk_data.items():
                            try:
                                df = pd.DataFrame(data['data'])
                                df.set_index('Date', inplace=True)
                                df.index = pd.to_datetime(df.index)
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
        """Save sentiment data"""
        try:
            # Convert DataFrame values to serializable format
            serialized_data = {}
            for symbol, data in sentiment_dict.items():
                df_dict = data.to_dict('records')
                serialized_data[symbol] = {
                    'data': df_dict,
                    'timestamp': datetime.now().isoformat()
                }
            
            with open(self.sentiment_cache_file, 'w') as f:
                json.dump(serialized_data, f)
                
            logger.info(f"Saved sentiment data for {self.market} market")
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {str(e)}")
    
    def load_sentiment_data(self):
        """Load sentiment data"""
        try:
            if not self.sentiment_cache_file.exists():
                return {}
            
            with open(self.sentiment_cache_file, 'r') as f:
                data = json.load(f)
            
            sentiment_dict = {}
            for symbol, item in data.items():
                df = pd.DataFrame(item['data'])
                sentiment_dict[symbol] = df
            
            return sentiment_dict
            
        except Exception as e:
            logger.error(f"Error loading sentiment data: {str(e)}")
            return {} 