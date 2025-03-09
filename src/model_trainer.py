import numpy as np
import pandas as pd
from xgboost import XGBRegressor
# Remove problematic imports
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import logging
from .utils import setup_logging
from .config import FEATURE_COLUMNS
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

logger = setup_logging()

class ModelTrainer:
    def __init__(self):
        # Use only XGBoost since it's already installed
        self.model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Import feature columns from config
        self.feature_cols = FEATURE_COLUMNS
    
    def train_model(self, features_df):
        """Train the model using historical data"""
        try:
            # Use the same feature columns defined in FeatureEngineer
            feature_cols = [col for col in self.feature_cols if col in features_df.columns]
            
            if not feature_cols:
                logger.error("No valid feature columns found in data")
                return None
            
            # Remove rows with missing target values
            features_df = features_df.dropna(subset=['Target'])
            
            # Split features and target
            X = features_df[feature_cols]
            y = features_df['Target']
            
            if len(X) < 100:
                logger.warning(f"Limited training data available: {len(X)} samples")
            
            # Train model
            self.model.fit(X, y)
            
            # Log model metrics
            metrics = self.get_model_metrics(X, y)
            logger.info(f"Model metrics: {metrics}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return None
    
    def get_model_metrics(self, X, y):
        """Evaluate model using time series cross-validation"""
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize metrics
        metrics = {
            'mae': [],
            'rmse': [],
            'r2': [],
            'direction_accuracy': []
        }
        
        # Perform cross-validation
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics['mae'].append(mean_absolute_error(y_test, y_pred))
            metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics['r2'].append(r2_score(y_test, y_pred))
            
            # Direction accuracy (up/down)
            direction_correct = ((y_test > 0) == (y_pred > 0)).mean()
            metrics['direction_accuracy'].append(direction_correct)
        
        # Return average metrics
        return {k: np.mean(v) for k, v in metrics.items()}

    def analyze_feature_importance(self, model, feature_names):
        """Analyze feature importance to understand model decisions"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # For ensemble model
            importances = np.mean([
                m.feature_importances_ for m in model['base_models'].values()
                if hasattr(m, 'feature_importances_')
            ], axis=0)
        
        # Create DataFrame of feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def tune_hyperparameters(self, X, y):
        """Tune model hyperparameters using Bayesian optimization"""
        # Define parameter space
        param_space = {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.3),
            'max_depth': (3, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0)
        }
        
        # Define objective function
        @use_named_args(dimensions=list(param_space.items()))
        def objective(**params):
            model = XGBRegressor(**params, random_state=42)
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Use direction accuracy as metric
                direction_correct = ((y_test > 0) == (y_pred > 0)).mean()
                scores.append(direction_correct)
            
            return -np.mean(scores)  # Negative because we want to maximize
        
        # Run optimization
        result = gp_minimize(objective, list(param_space.values()), n_calls=50, random_state=42)
        
        # Get best parameters
        best_params = {name: value for name, value in zip(param_space.keys(), result.x)}
        
        return best_params

    def ensemble_predict(self, X):
        """Predict using only XGBoost instead of an ensemble"""
        # Original ensemble code might have used multiple models
        # Now we'll just use XGBoost
        return self.model.predict(X)