import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
    def train_model(self, features_df):
        """Train the model using historical data"""
        # Prepare features and target
        feature_cols = [
            'RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV',
            'Returns', 'Returns_5d', 'Returns_20d',
            'Volatility_5d', 'Volatility_20d',
            'Price_to_VWAP', 'marketCap', 'trailingPE',
            'priceToBook', 'debtToEquity', 'news_sentiment'
        ]
        
        # Remove rows with missing target values
        features_df = features_df.dropna(subset=['Target'])
        
        # Split features and target
        X = features_df[feature_cols]
        y = features_df['Target']
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train the model
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = dict(zip(feature_cols, self.model.feature_importances_))
        
        return self.model
    
    def get_model_metrics(self, X, y):
        """Calculate model performance metrics"""
        predictions = self.model.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'feature_importance': self.feature_importance
        }
        
        return metrics