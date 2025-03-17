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
        try:
            # Check if stock_data is empty
            if stock_data.empty:
                logger.error("Empty stock data provided to feature engineering")
                return pd.DataFrame()
                
            # Create a copy of the input DataFrame to avoid SettingWithCopyWarning
            stock_data = stock_data.copy()
            
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            if missing_columns:
                logger.error(f"Missing required columns in stock data: {missing_columns}")
                return pd.DataFrame()
            
            # Group by symbol
            grouped = stock_data.groupby('Symbol')
            
            # Initialize sentiment processor
            sentiment_processor = SentimentProcessor()
            
            # Process in chunks
            all_features = []
            symbols = list(grouped.groups.keys())
            
            logger.info(f"Processing features for {len(symbols)} symbols in chunks of {chunk_size}")
            
            for i in range(0, len(symbols), chunk_size):
                chunk_symbols = symbols[i:i+chunk_size]
                chunk_data = []
                
                for symbol in chunk_symbols:
                    try:
                        # Get data for this symbol
                        group = grouped.get_group(symbol).copy()  # Create a copy to avoid warnings
                        
                        # Skip if not enough data points
                        if len(group) < 20:  # Need at least 20 days for most indicators
                            logger.warning(f"Not enough data points for {symbol} (only {len(group)} points). Skipping.")
                            continue
                        
                        # Calculate features
                        group = self._calculate_technical_indicators(group)
                        group = self._calculate_price_features(group)
                        group = self._add_candlestick_patterns(group)
                        group = self._add_market_regime(group)
                        group = self._add_advanced_features(group)
                        
                        # Add to chunk
                        chunk_data.append(group)
                        
                    except Exception as e:
                        logger.error(f"Error generating features for {symbol}: {str(e)}")
                
                # Process chunk
                if chunk_data:
                    try:
                        chunk_df = pd.concat(chunk_data)
                        chunk_df = self._handle_missing_values(chunk_df)
                        chunk_df = self._scale_features(chunk_df)
                        
                        # Add sentiment features
                        chunk_df = sentiment_processor.add_sentiment_features(chunk_df)
                        
                        all_features.append(chunk_df)
                        
                        # Clear memory
                        del chunk_data, chunk_df
                        gc.collect()
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                
                logger.info(f"Processed {min(i+chunk_size, len(symbols))}/{len(symbols)} symbols")
            
            # Combine results
            if all_features:
                result_df = pd.concat(all_features)
                
                # Ensure all required feature columns exist
                for col in self.feature_cols:
                    if col not in result_df.columns:
                        logger.warning(f"Feature column {col} is missing, adding with default values")
                        result_df[col] = 0.0
                
                logger.info(f"Generated features for {len(result_df)} data points across {len(result_df['Symbol'].unique())} symbols")
                return result_df
            else:
                logger.warning("No features were generated")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error in generate_features: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 0.001)  # Avoid division by zero
            df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df.loc[:, 'MACD'] = exp1 - exp2
            df.loc[:, 'MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df.loc[:, 'MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # Moving Averages
            df.loc[:, 'SMA_5'] = df['Close'].rolling(window=5).mean()
            df.loc[:, 'SMA_10'] = df['Close'].rolling(window=10).mean()
            df.loc[:, 'SMA_20'] = df['Close'].rolling(window=20).mean()
            df.loc[:, 'SMA_50'] = df['Close'].rolling(window=50).mean()
            df.loc[:, 'SMA_200'] = df['Close'].rolling(window=200).mean()
            df.loc[:, 'EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df.loc[:, 'EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df.loc[:, 'EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # Bollinger Bands
            df.loc[:, 'BB_MIDDLE'] = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df.loc[:, 'BB_UPPER'] = df['BB_MIDDLE'] + (std * 2)
            df.loc[:, 'BB_LOWER'] = df['BB_MIDDLE'] - (std * 2)
            df.loc[:, 'BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE'].replace(0, 0.001)  # Avoid division by zero
            df.loc[:, 'BB_PCT'] = (df['Close'] - df['BB_LOWER']) / ((df['BB_UPPER'] - df['BB_LOWER']).replace(0, 0.001))  # Avoid division by zero
            
            # Stochastic Oscillator
            low_min = df['Low'].rolling(window=14).min()
            high_max = df['High'].rolling(window=14).max()
            df.loc[:, 'Stochastic_K'] = 100 * ((df['Close'] - low_min) / ((high_max - low_min).replace(0, 0.001)))  # Avoid division by zero
            df.loc[:, 'Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
            
            # Williams %R
            df.loc[:, 'Williams_R'] = -100 * ((high_max - df['Close']) / ((high_max - low_min).replace(0, 0.001)))  # Avoid division by zero
            
            # Chaikin Money Flow
            mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / ((df['High'] - df['Low']).replace(0, 0.001))  # Avoid division by zero
            mf_volume = mf_multiplier * df['Volume']
            df.loc[:, 'CMF'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum().replace(0, 0.001)  # Avoid division by zero
            
            # MFI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi_ratio = positive_flow / negative_flow.replace(0, 0.001)  # Avoid division by zero
            df.loc[:, 'MFI'] = 100 - (100 / (1 + mfi_ratio))
            
            # ADX and ATR
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift(1))
            tr3 = abs(df['Low'] - df['Close'].shift(1))
            tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
            df.loc[:, 'ADX'] = tr.rolling(14).mean()
            df.loc[:, 'ATR'] = tr.rolling(14).mean()
            
            # OBV
            df.loc[:, 'OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
            
            # Ichimoku Cloud
            high_9 = df['High'].rolling(window=9).max()
            low_9 = df['Low'].rolling(window=9).min()
            df.loc[:, 'Ichimoku_Conversion'] = (high_9 + low_9) / 2
            
            high_26 = df['High'].rolling(window=26).max()
            low_26 = df['Low'].rolling(window=26).min()
            df.loc[:, 'Ichimoku_Base'] = (high_26 + low_26) / 2
            
            df.loc[:, 'Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(26)
            df.loc[:, 'Ichimoku_SpanB'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
            
            # Parabolic SAR
            # Simplified implementation
            df.loc[:, 'SAR'] = df['Close'].shift(1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return df
    
    def _calculate_price_features(self, df):
        """Calculate price-based features"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # VWAP
            df.loc[:, 'VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum().replace(0, 0.001)  # Avoid division by zero
            df.loc[:, 'Price_to_VWAP'] = df['Close'] / df['VWAP'].replace(0, 0.001)  # Avoid division by zero
            
            # Returns
            df.loc[:, 'Returns'] = df['Close'].pct_change()
            df.loc[:, 'Returns_5d'] = df['Close'].pct_change(5)
            df.loc[:, 'Returns_10d'] = df['Close'].pct_change(10)
            df.loc[:, 'Returns_20d'] = df['Close'].pct_change(20)
            df.loc[:, 'Returns_60d'] = df['Close'].pct_change(60)
            
            # Log returns (better for modeling)
            df.loc[:, 'Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1).replace(0, 0.001))  # Avoid division by zero
            df.loc[:, 'Log_Returns_5d'] = np.log(df['Close'] / df['Close'].shift(5).replace(0, 0.001))  # Avoid division by zero
            df.loc[:, 'Log_Returns_20d'] = np.log(df['Close'] / df['Close'].shift(20).replace(0, 0.001))  # Avoid division by zero
            
            # Volatility
            df.loc[:, 'Volatility_5d'] = df['Returns'].rolling(5).std()
            df.loc[:, 'Volatility_10d'] = df['Returns'].rolling(10).std()
            df.loc[:, 'Volatility_20d'] = df['Returns'].rolling(20).std()
            df.loc[:, 'Volatility_60d'] = df['Returns'].rolling(60).std()
            
            # Price ratios
            df.loc[:, 'Price_to_SMA20'] = df['Close'] / df['SMA_20'].replace(0, 0.001)  # Avoid division by zero
            df.loc[:, 'Price_to_SMA50'] = df['Close'] / df['SMA_50'].replace(0, 0.001)  # Avoid division by zero
            df.loc[:, 'Price_to_SMA200'] = df['Close'] / df['SMA_200'].replace(0, 0.001)  # Avoid division by zero
            
            # Moving average crossovers
            df.loc[:, 'SMA_5_10_Crossover'] = (df['SMA_5'] > df['SMA_10']).astype(int)
            df.loc[:, 'SMA_10_20_Crossover'] = (df['SMA_10'] > df['SMA_20']).astype(int)
            df.loc[:, 'SMA_20_50_Crossover'] = (df['SMA_20'] > df['SMA_50']).astype(int)
            df.loc[:, 'SMA_50_200_Crossover'] = (df['SMA_50'] > df['SMA_200']).astype(int)
            
            # Volume features
            df.loc[:, 'Volume_Change'] = df['Volume'].pct_change()
            df.loc[:, 'Volume_MA10'] = df['Volume'].rolling(10).mean()
            df.loc[:, 'Relative_Volume'] = df['Volume'] / df['Volume_MA10'].replace(0, 0.001)  # Avoid division by zero
            
            # Target variable (next day return)
            df.loc[:, 'Target'] = df['Returns'].shift(-1)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating price features: {str(e)}")
            return df
    
    def _add_advanced_features(self, df):
        """Add more advanced features for better prediction"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Momentum indicators
            df.loc[:, 'ROC_5'] = df['Close'].pct_change(5) * 100  # Rate of Change
            df.loc[:, 'ROC_10'] = df['Close'].pct_change(10) * 100
            df.loc[:, 'ROC_20'] = df['Close'].pct_change(20) * 100
            
            # Price acceleration
            df.loc[:, 'Price_Acceleration'] = df['Returns'].diff()
            df.loc[:, 'Price_Acceleration_5d'] = df['Returns_5d'].diff()
            
            # Volatility ratio
            df.loc[:, 'Volatility_Ratio'] = df['Volatility_5d'] / df['Volatility_20d'].replace(0, 0.001)  # Avoid division by zero
            
            # Trend strength
            df.loc[:, 'ADX_Trend_Strength'] = df['ADX'] / 25  # Normalized ADX
            
            # RSI divergence
            df.loc[:, 'RSI_Divergence'] = (df['RSI'] - df['RSI'].shift(5)) * (df['Close'] - df['Close'].shift(5))
            
            # Bollinger Band squeeze
            df.loc[:, 'BB_Squeeze'] = df['BB_WIDTH'] < df['BB_WIDTH'].rolling(20).mean()
            df.loc[:, 'BB_Squeeze'] = df['BB_Squeeze'].astype(int)
            
            # Volume-price relationship
            df.loc[:, 'Volume_Price_Trend'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
            df.loc[:, 'Volume_Price_Trend'] = df['Volume_Price_Trend'].rolling(20).sum()
            
            # Gap features
            df.loc[:, 'Gap_Up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
            df.loc[:, 'Gap_Down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
            df.loc[:, 'Gap_Size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1).replace(0, 0.001)  # Avoid division by zero
            
            # Ichimoku cloud position
            df.loc[:, 'Above_Cloud'] = ((df['Close'] > df['Ichimoku_SpanA']) & 
                                    (df['Close'] > df['Ichimoku_SpanB'])).astype(int)
            df.loc[:, 'Below_Cloud'] = ((df['Close'] < df['Ichimoku_SpanA']) & 
                                    (df['Close'] < df['Ichimoku_SpanB'])).astype(int)
            df.loc[:, 'In_Cloud'] = (~df['Above_Cloud'].astype(bool) & 
                                 ~df['Below_Cloud'].astype(bool)).astype(int)
            
            # Lagged features (previous day's indicators)
            for lag in [1, 2, 3, 5]:
                df.loc[:, f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
                df.loc[:, f'MACD_lag_{lag}'] = df['MACD'].shift(lag)
                df.loc[:, f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
            
            # Seasonal features
            if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
                df.loc[:, 'Day_of_Week'] = df.index.dayofweek
                df.loc[:, 'Month'] = df.index.month
                df.loc[:, 'Quarter'] = df.index.quarter
                
                # Convert to dummy variables
                for day in range(5):  # 0-4 for weekdays
                    df.loc[:, f'Day_{day}'] = (df['Day_of_Week'] == day).astype(int)
                
                for month in range(1, 13):
                    df.loc[:, f'Month_{month}'] = (df['Month'] == month).astype(int)
            
            return df
        except Exception as e:
            logger.error(f"Error adding advanced features: {str(e)}")
            return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the features"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Forward fill for technical indicators
            if set(TECHNICAL_INDICATORS).issubset(df.columns):
                df.loc[:, TECHNICAL_INDICATORS] = df[TECHNICAL_INDICATORS].ffill()
            else:
                missing_indicators = [col for col in TECHNICAL_INDICATORS if col not in df.columns]
                logger.warning(f"Missing technical indicators: {missing_indicators}")
                # Add missing columns with default values
                for col in missing_indicators:
                    df[col] = 0.0
            
            # Fill remaining missing values with median
            for col in self.feature_cols:
                if col in df.columns:
                    if df[col].isna().any():
                        median_val = df[col].median()
                        if pd.isna(median_val):  # If median is also NaN
                            df.loc[:, col] = df[col].fillna(0)
                        else:
                            df.loc[:, col] = df[col].fillna(median_val)
                else:
                    # Add missing column with default value
                    df[col] = 0.0
            
            return df
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    def _scale_features(self, df):
        """Scale numerical features"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Scale features if they exist in the dataframe
            scale_cols = [col for col in FEATURES_TO_SCALE if col in df.columns]
            if scale_cols:
                # Handle infinite values
                for col in scale_cols:
                    df.loc[:, col] = df[col].replace([np.inf, -np.inf], np.nan)
                    df.loc[:, col] = df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0)
                
                # Scale features
                df.loc[:, scale_cols] = self.scaler.fit_transform(df[scale_cols])
            
            return df
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return df
    
    def _add_candlestick_patterns(self, df):
        """Add candlestick pattern recognition features"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Bullish patterns
            df.loc[:, 'bullish_engulfing'] = (df['Open'] > df['Close'].shift(1)) & \
                                         (df['Close'] > df['Open'].shift(1)) & \
                                         (df['Close'] > df['Open']) & \
                                         (df['Open'] < df['Close'].shift(1))
            
            # Bearish patterns
            df.loc[:, 'bearish_engulfing'] = (df['Open'] < df['Close'].shift(1)) & \
                                         (df['Close'] < df['Open'].shift(1)) & \
                                         (df['Close'] < df['Open']) & \
                                         (df['Open'] > df['Close'].shift(1))
            
            # Doji
            df.loc[:, 'doji'] = abs(df['Open'] - df['Close']) <= (0.1 * (df['High'] - df['Low']))
            
            # Hammer
            df.loc[:, 'hammer'] = (((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close']).abs()) & 
                           ((df['Close'] - df['Low']) > (0.6 * (df['High'] - df['Low']))) & 
                           ((df['Open'] - df['Low']) > (0.6 * (df['High'] - df['Low']))))
            
            # Shooting Star
            df.loc[:, 'shooting_star'] = (((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close']).abs()) & 
                                  ((df['High'] - df['Close']) > (0.6 * (df['High'] - df['Low']))) & 
                                  ((df['High'] - df['Open']) > (0.6 * (df['High'] - df['Low']))))
            
            # Convert to numeric
            pattern_cols = ['bullish_engulfing', 'bearish_engulfing', 'doji', 'hammer', 'shooting_star']
            for col in pattern_cols:
                df.loc[:, col] = df[col].astype(int)
            
            return df
        except Exception as e:
            logger.error(f"Error adding candlestick patterns: {str(e)}")
            return df
    
    def _add_market_regime(self, df):
        """Add market regime features"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Simple regime based on moving averages
            df.loc[:, 'regime'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
            
            # Volatility regime
            df.loc[:, 'volatility_regime'] = np.where(df['Volatility_20d'] > df['Volatility_20d'].rolling(30).mean(), 1, -1)
            
            # Trend strength
            df.loc[:, 'trend_strength'] = abs(df['SMA_20'] / df['SMA_50'].replace(0, 0.001) - 1) * 100  # Avoid division by zero
            
            # Market phases
            conditions = [
                (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200']),  # Strong uptrend
                (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] < df['SMA_200']),  # Weak uptrend
                (df['SMA_20'] < df['SMA_50']) & (df['SMA_50'] > df['SMA_200']),  # Weak downtrend
                (df['SMA_20'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200']),  # Strong downtrend
            ]
            choices = [3, 1, -1, -3]
            df.loc[:, 'market_phase'] = np.select(conditions, choices, default=0)
            
            return df
        except Exception as e:
            logger.error(f"Error adding market regime: {str(e)}")
            return df
    
    def _add_sector_performance(self, df, symbol, all_data):
        """Add sector relative performance features"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # This would require sector classification data
            # Simplified example:
            sector_symbols = [s for s in all_data['Symbol'].unique() if s != symbol]
            if sector_symbols:
                sector_data = all_data[all_data['Symbol'].isin(sector_symbols)]
                sector_returns = sector_data.groupby(sector_data.index)['Returns'].mean()
                df.loc[:, 'sector_relative_return'] = df['Returns'] - sector_returns
            return df
        except Exception as e:
            logger.error(f"Error adding sector performance: {str(e)}")
            return df
    
    def _detect_market_regime(self, df):
        """Detect market regime (bull, bear, sideways)"""
        try:
            # Make a copy of the DataFrame to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Calculate moving averages
            df.loc[:, 'SMA_50'] = df['Close'].rolling(window=50).mean()
            df.loc[:, 'SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Determine trend
            df.loc[:, 'trend'] = np.where(df['SMA_50'] > df['SMA_200'], 1, -1)
            
            # Calculate volatility
            df.loc[:, 'volatility'] = df['Returns'].rolling(window=20).std()
            
            # Determine regime
            conditions = [
                (df['trend'] == 1) & (df['volatility'] < df['volatility'].quantile(0.7)),  # Bull
                (df['trend'] == -1) & (df['volatility'] < df['volatility'].quantile(0.7)),  # Bear
                (df['volatility'] >= df['volatility'].quantile(0.7))  # Volatile
            ]
            choices = ['bull', 'bear', 'volatile']
            df.loc[:, 'regime'] = np.select(conditions, choices, default='sideways')
            
            # Convert to numeric for model
            df.loc[:, 'regime_bull'] = (df['regime'] == 'bull').astype(int)
            df.loc[:, 'regime_bear'] = (df['regime'] == 'bear').astype(int)
            df.loc[:, 'regime_volatile'] = (df['regime'] == 'volatile').astype(int)
            
            return df
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return df