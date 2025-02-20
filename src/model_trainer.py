import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self):
        self.base_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.ensemble_models = [
            XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4),
            XGBRegressor(n_estimators=250, learning_rate=0.15, max_depth=8),
            XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, gamma=1)
        ]
        
    def train_model(self, features_df):
        """Train ensemble of models using historical data"""
        # Prepare features and target
        feature_cols = [
            'RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV',
            'Returns', 'Returns_5d', 'Returns_20d',
            'Volatility_5d', 'Volatility_20d',
            'Price_to_VWAP', 'marketCap', 'trailingPE',
            'priceToBook', 'debtToEquity', 'news_sentiment',
            'Price_to_Fib', 'Price_to_Pivot'
        ]
        
        # Remove rows with missing target values
        features_df = features_df.dropna(subset=['Target'])
        
        # Split features and target
        X = features_df[feature_cols]
        y = features_df['Target']
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train ensemble models
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train base model
            self.base_model.fit(X_train, y_train)
            
            # Train ensemble models
            for model in self.ensemble_models:
                model.fit(X_train, y_train)
        
        # Get feature importance from base model
        self.feature_importance = dict(zip(feature_cols, self.base_model.feature_importances_))
        
        return self
    
    def predict(self, X):
        """Make predictions using ensemble average"""
        # Get predictions from all models
        base_pred = self.base_model.predict(X)
        ensemble_preds = [model.predict(X) for model in self.ensemble_models]
        
        # Calculate weighted average (base_model: 0.4, ensemble_models: 0.6)
        weights = [0.4] + [0.2] * len(self.ensemble_models)
        weighted_preds = np.average([base_pred] + ensemble_preds, weights=weights, axis=0)
        
        return weighted_preds
    
    def get_model_metrics(self, X, y):
        """Calculate model performance metrics"""
        predictions = self.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'feature_importance': self.feature_importance
        }
        
        return metrics