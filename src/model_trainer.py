import numpy as np
import pandas as pd
from xgboost import XGBRegressor
# Remove problematic imports
# from lightgbm import LGBMRegressor
# from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import logging
from .utils import setup_logging
from .config import FEATURE_COLUMNS
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import VotingRegressor
import joblib
from pathlib import Path
import xgboost as xgb

logger = setup_logging()

class ModelTrainer:
    def __init__(self):
        # Default XGBoost parameters
        self.default_params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42
        }
        
        # Initialize model with default parameters
        self.model = XGBRegressor(**self.default_params)
        
        # Import feature columns from config
        self.feature_cols = FEATURE_COLUMNS
        
        # Path for model persistence
        self.model_dir = Path(__file__).parent.parent / 'models'
        self.model_dir.mkdir(exist_ok=True)
        self.model_path = self.model_dir / 'stock_prediction_model.joblib'
        self.selected_features_path = self.model_dir / 'selected_features.joblib'
        
        # Feature importance threshold
        self.feature_importance_threshold = 0.01
    
    def _load_model(self):
        """Load the trained model from disk"""
        try:
            if self.model_path.exists():
                model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
                return model
            return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    def _save_model(self, model):
        """Save the trained model to disk"""
        try:
            joblib.dump(model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def _select_important_features(self, X, y):
        """Select important features using a feature selector"""
        try:
            # Train a model for feature selection with more trees
            selection_model = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42
            )
            
            # Split data for feature selection
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Train with early stopping using the correct callback syntax
            selection_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[xgb.callback.EarlyStopping(rounds=50, metric='mae')],
                verbose=False
            )
            
            # Get feature importances
            importances = selection_model.feature_importances_
            
            # Calculate importance threshold based on mean and std
            mean_importance = np.mean(importances)
            std_importance = np.std(importances)
            threshold = max(mean_importance - 0.5 * std_importance, self.feature_importance_threshold)
            
            # Select features based on importance
            selected_indices = np.where(importances > threshold)[0]
            
            # Get selected feature names
            selected_features = X.columns[selected_indices].tolist()
            
            # Ensure we have at least some features
            if not selected_features:
                logger.warning("No features selected, using top 10 features by importance")
                top_indices = np.argsort(importances)[-10:]  # Get indices of top 10 features
                selected_features = X.columns[top_indices].tolist()
            
            # Log feature selection results
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Feature selection results:\n{importance_df.head(10)}")
            logger.info(f"Selected {len(selected_features)} features: {selected_features}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {str(e)}")
            # Return all features if selection fails
            return X.columns.tolist()
    
    def _tune_hyperparameters(self, X, y):
        """Tune model hyperparameters using Bayesian optimization"""
        try:
            # Define parameter space
            space = [
                Integer(100, 500, name='n_estimators'),
                Real(0.01, 0.2, name='learning_rate'),
                Integer(3, 10, name='max_depth'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bytree'),
                Real(0, 5, name='gamma'),
                Real(0, 1, name='reg_alpha'),
                Real(0.1, 10, name='reg_lambda')
            ]
            
            # Define objective function
            @use_named_args(space)
            def objective(n_estimators, learning_rate, max_depth, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda):
                model = XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    gamma=gamma,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=42
                )
                
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
            result = gp_minimize(objective, space, n_calls=30, random_state=42)
            
            # Get best parameters
            best_params = {
                'n_estimators': int(result.x[0]),
                'learning_rate': float(result.x[1]),
                'max_depth': int(result.x[2]),
                'subsample': float(result.x[3]),
                'colsample_bytree': float(result.x[4]),
                'gamma': float(result.x[5]),
                'reg_alpha': float(result.x[6]),
                'reg_lambda': float(result.x[7]),
                'random_state': 42
            }
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            return self.default_params
    
    def train_model(self, features_df, tune_hyperparams=True, use_feature_selection=True):
        """Train the model with feature selection and hyperparameter tuning"""
        try:
            # Ensure we have valid feature columns
            feature_cols = [col for col in features_df.columns if col not in ['Symbol', 'CompanyName', 'Target']]
            if not feature_cols:
                logger.error("No valid feature columns found")
                return None
            
            # Remove rows with missing target values
            features_df = features_df.dropna(subset=['Target'])
            
            # Split features and target
            X = features_df[feature_cols]
            y = features_df['Target']
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Try to load existing model first
            try:
                model = self._load_model()
                if model is not None:
                    logger.info("Loaded existing model")
                    return model
            except Exception as e:
                logger.info(f"No existing model found or error loading: {str(e)}")
            
            # Feature selection
            if use_feature_selection:
                try:
                    selected_features = self._select_important_features(X_train, y_train)
                    if selected_features:
                        X_train = X_train[selected_features]
                        X_val = X_val[selected_features]
                        logger.info(f"Selected {len(selected_features)} features: {selected_features}")
                        
                        # Save selected features
                        joblib.dump(selected_features, self.selected_features_path)
                    else:
                        logger.warning("No features selected, using all features")
                except Exception as e:
                    logger.error(f"Error in feature selection: {str(e)}")
                    # Continue with all features if selection fails
            
            # Hyperparameter tuning
            if tune_hyperparams:
                try:
                    best_params = self._tune_hyperparameters(X_train, y_train)
                    logger.info(f"Best hyperparameters: {best_params}")
                except Exception as e:
                    logger.error(f"Error in hyperparameter tuning: {str(e)}")
                    best_params = self.default_params
            else:
                best_params = self.default_params
            
            # Train final model
            try:
                model = XGBRegressor(**best_params)
                
                # Create evaluation set for early stopping
                eval_set = [(X_train, y_train), (X_val, y_val)]
                
                # Train with early stopping
                model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    eval_metric='mae',
                    callbacks=[xgb.callback.EarlyStopping(rounds=50, metric='mae')],
                    verbose=False
                )
                
                # Save model
                self._save_model(model)
                
                # Log metrics
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_mae = mean_absolute_error(y_train, train_pred)
                val_mae = mean_absolute_error(y_val, val_pred)
                
                logger.info(f"Training MAE: {train_mae:.4f}")
                logger.info(f"Validation MAE: {val_mae:.4f}")
                
                # Save feature importance
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                importance_df.to_csv(self.model_dir / 'feature_importance.csv', index=False)
                
                return model
                
            except Exception as e:
                logger.error(f"Error training model: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
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
                m.feature_importances_ for m in model.estimators_
                if hasattr(m, 'feature_importances_')
            ], axis=0)
        
        # Create DataFrame of feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df

    def ensemble_predict(self, X):
        """Make predictions using the model"""
        if self.model is None:
            logger.error("Model is None, cannot make predictions")
            # Return a default prediction (0) to avoid crashing
            return np.zeros(len(X))
        return self.model.predict(X)