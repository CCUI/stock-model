import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def generate_features(self, stock_data):
        """Generate technical and fundamental features for stock prediction"""
        features_df = stock_data.copy()
        
        # Group by symbol to calculate features for each stock
        grouped = features_df.groupby('Symbol')
        
        all_features = []
        for symbol, group in grouped:
            try:
                # Sort by date
                group = group.sort_index()
                
                # Technical indicators
                # RSI
                delta = group['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                group['RSI'] = 100 - (100 / (1 + rs))
                
                # MACD
                exp1 = group['Close'].ewm(span=12, adjust=False).mean()
                exp2 = group['Close'].ewm(span=26, adjust=False).mean()
                group['MACD'] = exp1 - exp2
                
                # Money Flow Index
                typical_price = (group['High'] + group['Low'] + group['Close']) / 3
                money_flow = typical_price * group['Volume']
                positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
                negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
                mfi_ratio = positive_flow / negative_flow
                group['MFI'] = 100 - (100 / (1 + mfi_ratio))
                
                # Moving Averages
                group['EMA_20'] = group['Close'].ewm(span=20, adjust=False).mean()
                group['SMA_50'] = group['Close'].rolling(window=50).mean()
                
                # ADX
                tr1 = group['High'] - group['Low']
                tr2 = abs(group['High'] - group['Close'].shift(1))
                tr3 = abs(group['Low'] - group['Close'].shift(1))
                tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
                group['ADX'] = tr.rolling(14).mean()
                
                # ATR
                group['ATR'] = tr.rolling(14).mean()
                
                # Bollinger Bands
                sma = group['Close'].rolling(window=20).mean()
                std = group['Close'].rolling(window=20).std()
                group['BB_UPPER'] = sma + (std * 2)
                group['BB_MIDDLE'] = sma
                group['BB_LOWER'] = sma - (std * 2)
                
                # Volume indicators
                group['OBV'] = (group['Volume'] * (~group['Close'].diff().le(0) * 2 - 1)).cumsum()
                group['VWAP'] = (group['Close'] * group['Volume']).cumsum() / group['Volume'].cumsum()
                
                # Price ratios and returns
                group['Price_to_VWAP'] = group['Close'] / group['VWAP']
                group['Returns'] = group['Close'].pct_change()
                group['Returns_5d'] = group['Close'].pct_change(5)
                group['Returns_20d'] = group['Close'].pct_change(20)
                
                # Volatility features
                group['Volatility_5d'] = group['Returns'].rolling(5).std()
                group['Volatility_20d'] = group['Returns'].rolling(20).std()
                
                # Target variable (next day return)
                group['Target'] = group['Returns'].shift(-1)
                
                all_features.append(group)
                
            except Exception as e:
                print(f"Error generating features for {symbol}: {str(e)}")
                continue
        
        features_df = pd.concat(all_features)
        
        # Fill missing values
        features_df = self._handle_missing_values(features_df)
        
        # Scale numerical features
        features_df = self._scale_features(features_df)
        
        return features_df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the features"""
        # Forward fill for technical indicators
        technical_cols = ['RSI', 'MACD', 'MFI', 'EMA_20', 'SMA_50', 'ADX', 'ATR',
                         'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'OBV', 'VWAP']
        df[technical_cols] = df[technical_cols].fillna(method='ffill')
        
        # Fill remaining missing values with median and convert to float
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Convert all feature columns to float type
        feature_cols = ['RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV',
                       'Returns', 'Returns_5d', 'Returns_20d',
                       'Volatility_5d', 'Volatility_20d',
                       'Price_to_VWAP', 'marketCap', 'trailingPE',
                       'priceToBook', 'debtToEquity', 'news_sentiment']
        
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            df[col] = df[col].astype(float)
        
        return df
    
    def _scale_features(self, df):
        """Scale numerical features"""
        # Columns to scale
        cols_to_scale = ['RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV',
                        'Returns', 'Returns_5d', 'Returns_20d',
                        'Volatility_5d', 'Volatility_20d']
        
        # Scale features
        df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        
        return df