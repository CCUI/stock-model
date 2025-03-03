import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import FEATURE_COLUMNS, TECHNICAL_INDICATORS, FEATURES_TO_SCALE

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLUMNS
    
    def generate_features(self, stock_data):
        """Generate technical and fundamental features for stock prediction"""
        features_df = stock_data.copy()
        
        # Initialize sentiment features with default values
        sentiment_features = ['news_sentiment', 'social_sentiment', 'sector_sentiment', 'market_sentiment']
        for col in sentiment_features:
            if col not in features_df.columns:
                features_df[col] = 0.0
        
        # Group by symbol to calculate features for each stock
        grouped = features_df.groupby('Symbol')
        
        all_features = []
        for symbol, group in grouped:
            try:
                # Sort by date
                group = group.sort_index()
                
                # Calculate all technical indicators
                group = self._calculate_technical_indicators(group)
                
                # Calculate price ratios and returns
                group = self._calculate_price_features(group)
                
                # Add to features list
                all_features.append(group)
                
            except Exception as e:
                print(f"Error generating features for {symbol}: {str(e)}")
                continue
        
        features_df = pd.concat(all_features)
        
        # Fill missing values
        features_df = self._handle_missing_values(features_df)
        
        # Scale numerical features
        features_df = self._scale_features(features_df)
        
        # Ensure only defined feature columns are included
        for col in self.feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0  # Add missing columns with default values
        
        return features_df
    
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
            
            # Stochastic Oscillator
            low_min = df['Low'].rolling(14).min()
            high_max = df['High'].rolling(14).max()
            df['Stochastic_K'] = ((df['Close'] - low_min) / (high_max - low_min)) * 100
            df['Stochastic_D'] = df['Stochastic_K'].rolling(3).mean()
            
            # Williams %R
            df['Williams_R'] = ((high_max - df['Close']) / (high_max - low_min)) * -100
            
            # Chaikin Money Flow (CMF)
            mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mf_volume = mf_multiplier * df['Volume']
            df['CMF'] = mf_volume.rolling(20).sum() / df['Volume'].rolling(20).sum()
            
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
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