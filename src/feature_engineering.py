import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import FEATURE_COLUMNS, TECHNICAL_INDICATORS, FEATURES_TO_SCALE
from .sentiment_processor import SentimentProcessor
import gc
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLUMNS
    
    def generate_features(self, stock_data, chunk_size=1000):
        """Generate features in chunks to reduce memory usage"""
        # Group by symbol
        grouped = stock_data.groupby('Symbol')
        
        # Initialize sentiment processor
        sentiment_processor = SentimentProcessor()
        
        # Process in chunks
        all_features = []
        symbols = list(grouped.groups.keys())
        
        for i in range(0, len(symbols), chunk_size):
            chunk_symbols = symbols[i:i+chunk_size]
            chunk_data = []
            
            for symbol in chunk_symbols:
                try:
                    # Get data for this symbol
                    group = grouped.get_group(symbol)
                    
                    # Calculate features
                    group = self._calculate_technical_indicators(group)
                    group = self._calculate_price_features(group)
                    
                    # Add to chunk
                    chunk_data.append(group)
                    
                except Exception as e:
                    logger.error(f"Error generating features for {symbol}: {str(e)}")
            
            # Process chunk
            if chunk_data:
                chunk_df = pd.concat(chunk_data)
                chunk_df = self._handle_missing_values(chunk_df)
                chunk_df = self._scale_features(chunk_df)
                
                # Add sentiment features
                chunk_df = sentiment_processor.add_sentiment_features(chunk_df)
                
                all_features.append(chunk_df)
                
                # Clear memory
                del chunk_data, chunk_df
                gc.collect()
        
        # Combine results
        return pd.concat(all_features) if all_features else pd.DataFrame()
    
    def _calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            
            # Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Bollinger Bands
            df['BB_MIDDLE'] = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df['BB_UPPER'] = df['BB_MIDDLE'] + (std * 2)
            df['BB_LOWER'] = df['BB_MIDDLE'] - (std * 2)
            
            # Stochastic Oscillator
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df['Stochastic_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
            
            # Williams %R
            df['Williams_R'] = -100 * ((high_max - df['Close']) / (high_max - low_min))
            
            # Chaikin Money Flow
            mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mf_volume = mf_multiplier * df['Volume']
            df['CMF'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
            
            # MFI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi_ratio = positive_flow / negative_flow
            df['MFI'] = 100 - (100 / (1 + mfi_ratio))
            
            # ADX and ATR
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift(1))
            tr3 = abs(df['Low'] - df['Close'].shift(1))
            tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
            df['ADX'] = tr.rolling(14).mean()
            df['ATR'] = tr.rolling(14).mean()
            
            # OBV
            df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def _calculate_price_features(self, df):
        """Calculate price-based features"""
        # VWAP
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['Price_to_VWAP'] = df['Close'] / df['VWAP']
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Returns_5d'] = df['Close'].pct_change(5)
        df['Returns_20d'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility_5d'] = df['Returns'].rolling(5).std()
        df['Volatility_20d'] = df['Returns'].rolling(20).std()
        
        # Target variable (next day return)
        df['Target'] = df['Returns'].shift(-1)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the features"""
        # Forward fill for technical indicators
        df[TECHNICAL_INDICATORS] = df[TECHNICAL_INDICATORS].ffill()
        
        # Fill remaining missing values with median
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _scale_features(self, df):
        """Scale numerical features"""
        # Scale features if they exist in the dataframe
        scale_cols = [col for col in FEATURES_TO_SCALE if col in df.columns]
        if scale_cols:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        
        return df
    
    def _add_candlestick_patterns(self, df):
        # Bullish patterns
        df['bullish_engulfing'] = (df['Open'] > df['Close'].shift(1)) & \
                                 (df['Close'] > df['Open'].shift(1)) & \
                                 (df['Close'] > df['Open']) & \
                                 (df['Open'] < df['Close'].shift(1))
        
        # Bearish patterns
        df['bearish_engulfing'] = (df['Open'] < df['Close'].shift(1)) & \
                                 (df['Close'] < df['Open'].shift(1)) & \
                                 (df['Close'] < df['Open']) & \
                                 (df['Open'] > df['Close'].shift(1))
        
        # Convert to numeric
        df['bullish_pattern'] = df['bullish_engulfing'].astype(int)
        df['bearish_pattern'] = df['bearish_engulfing'].astype(int)
        return df
    
    def _add_market_regime(self, df):
        # Simple regime based on moving averages
        df['regime'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
        
        # Volatility regime
        df['volatility_regime'] = np.where(df['Volatility_20d'] > df['Volatility_20d'].rolling(30).mean(), 1, -1)
        return df
    
    def _add_sector_performance(self, df, symbol, all_data):
        # This would require sector classification data
        # Simplified example:
        sector_symbols = [s for s in all_data['Symbol'].unique() if s != symbol]
        if sector_symbols:
            sector_data = all_data[all_data['Symbol'].isin(sector_symbols)]
            sector_returns = sector_data.groupby(sector_data.index)['Returns'].mean()
            df['sector_relative_return'] = df['Returns'] - sector_returns
        return df
    
    def _detect_market_regime(self, df):
        """Detect market regime (bull, bear, sideways)"""
        # Calculate moving averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Determine trend
        df['trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
        
        # Calculate volatility
        df['volatility'] = df['Returns'].rolling(window=20).std()
        
        # Determine regime
        conditions = [
            (df['trend'] == 1) & (df['volatility'] < df['volatility'].quantile(0.7)),  # Bull
            (df['trend'] == -1) & (df['volatility'] < df['volatility'].quantile(0.7)),  # Bear
            (df['volatility'] >= df['volatility'].quantile(0.7))  # Volatile
        ]
        choices = ['bull', 'bear', 'volatile']
        df['regime'] = np.select(conditions, choices, default='sideways')
        
        # Convert to numeric for model
        df['regime_bull'] = (df['regime'] == 'bull').astype(int)
        df['regime_bear'] = (df['regime'] == 'bear').astype(int)
        df['regime_volatile'] = (df['regime'] == 'volatile').astype(int)
        
        return df