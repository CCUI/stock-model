import pandas as pd
import numpy as np
from datetime import datetime
import pandas as pd
from datetime import datetime
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        # Import feature columns from config to ensure consistency
        from .config import FEATURE_COLUMNS
        self.feature_cols = FEATURE_COLUMNS
        
        # Path for model persistence
        self.model_dir = Path(__file__).parent.parent / 'models'
        self.model_dir.mkdir(exist_ok=True)
        self.selected_features_path = self.model_dir / 'selected_features.joblib'
    
    def predict_top_gainers(self, model, features_df, top_n=5, market='UK', confidence_threshold=0.0):
        """Predict top gaining stocks for the next day with confidence scores"""
        try:
            # Check if model is None
            if model is None:
                logger.error("Model is None, cannot make predictions")
                raise ValueError("Model is None. Please ensure the model is properly trained.")
            
            # Load selected features if available
            selected_features = None
            if self.selected_features_path.exists():
                try:
                    selected_features = joblib.load(self.selected_features_path)
                    logger.info(f"Loaded {len(selected_features)} selected features for prediction")
                except Exception as e:
                    logger.warning(f"Could not load selected features: {str(e)}. Using all features.")
                    selected_features = self.feature_cols
            else:
                logger.warning("Selected features file not found. Using all features.")
                selected_features = self.feature_cols
                
            # Filter stocks by market based on symbol pattern
            if market.upper() == 'UK':
                # UK stocks typically have .L suffix
                market_stocks = features_df[features_df['Symbol'].str.endswith('.L')]
            elif market.upper() == 'US':
                # US stocks don't have .L suffix
                market_stocks = features_df[~features_df['Symbol'].str.endswith('.L')]
            else:
                # If market not specified, use all stocks
                market_stocks = features_df
                
            # Get the latest data for each stock
            latest_data = market_stocks.groupby('Symbol').last()
            
            # Create a list to store missing features
            missing_features = []
            
            # Ensure all required features exist
            for col in selected_features:
                if col not in latest_data.columns:
                    missing_features.append(col)
                    latest_data[col] = 0.0
            
            # Log warning if features are missing
            if missing_features:
                logger.warning(f"The following features are missing and were set to 0.0: {', '.join(missing_features)}")
            
            # Handle additional features in the data that aren't in the model
            # This fixes the feature_names mismatch error
            extra_features = [col for col in latest_data.columns if col not in selected_features and col in ['social_sentiment', 'sector_sentiment', 'market_sentiment']]
            if extra_features:
                for col in extra_features:
                    if col in latest_data.columns:
                        # Instead of dropping these columns, keep them for the report
                        # but don't include them in the prediction features
                        pass
            
            # Prepare features for prediction
            X = latest_data[selected_features]
            
            # Validate data before prediction
            if X.isnull().any().any():
                logger.warning("Features contain null values, filling with 0")
                X = X.fillna(0)
            
            # Make predictions
            try:
                predictions = model.predict(X)
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                # Return a default prediction
                logger.warning("Using default predictions (all zeros)")
                predictions = np.zeros(len(X))
            
            # Calculate prediction confidence
            confidence_scores = self._calculate_confidence_scores(model, X, predictions)
            
            # Add predictions to the dataframe
            latest_data['predicted_return'] = predictions
            latest_data['confidence_score'] = confidence_scores
            
            # Filter by confidence threshold if specified
            if confidence_threshold > 0:
                confident_predictions = latest_data[latest_data['confidence_score'] >= confidence_threshold]
                if len(confident_predictions) > 0:
                    logger.info(f"Filtered to {len(confident_predictions)} stocks with confidence >= {confidence_threshold}")
                    latest_data = confident_predictions
                else:
                    logger.warning(f"No stocks met the confidence threshold of {confidence_threshold}. Using all predictions.")
            
            # Sort by predicted return and get top N
            top_gainers = latest_data.sort_values('predicted_return', ascending=False).head(top_n)
            
            # Save predictions for later evaluation
            self._save_predictions(top_gainers, market)
            
            return top_gainers
        except Exception as e:
            error_msg = f"Error in predict_top_gainers: {str(e)}"
            logger.error(error_msg)
            # Return an empty DataFrame with the expected columns instead of raising an exception
            empty_df = pd.DataFrame(columns=['predicted_return', 'confidence_score'])
            logger.warning("Returning empty DataFrame due to error")
            return empty_df
    
    def _calculate_confidence_scores(self, model, X, predictions):
        """Calculate confidence scores for predictions"""
        try:
            # Check if model is None
            if model is None:
                logger.warning("Model is None, using default confidence scores")
                return np.ones(len(X)) * 0.5  # Default confidence of 0.5
                
            # Method 1: Based on feature importance and feature values
            if hasattr(model, 'feature_importances_'):
                # Get feature importances
                importances = model.feature_importances_
                
                # Check if the number of features matches
                if len(importances) != X.shape[1]:
                    logger.warning(f"Feature count mismatch: model has {len(importances)} features, input has {X.shape[1]} features")
                    return np.ones(len(X)) * 0.5  # Default confidence of 0.5
                
                # Normalize feature values
                X_norm = X.copy()
                for col in X.columns:
                    if X[col].std() > 0:
                        X_norm[col] = (X[col] - X[col].mean()) / X[col].std()
                    else:
                        X_norm[col] = 0
                
                # Calculate weighted feature contribution
                weighted_contributions = np.abs(X_norm.values) * importances
                
                # Sum contributions for each prediction
                confidence_scores = np.sum(weighted_contributions, axis=1) / np.sum(importances)
                
                # Normalize to 0-1 range
                if confidence_scores.max() > confidence_scores.min():
                    confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())
                
                return confidence_scores
            
            # Method 2: Based on prediction magnitude and historical volatility
            else:
                # Get historical volatility if available
                if 'Volatility_20d' in X.columns:
                    vol = X['Volatility_20d'].values
                    vol = np.clip(vol, 0.0001, None)  # Avoid division by zero
                else:
                    vol = np.ones(len(X)) * 0.02  # Default 2% volatility
                
                # Calculate confidence based on prediction magnitude relative to volatility
                abs_preds = np.abs(predictions)
                confidence_scores = 1 - np.exp(-abs_preds / vol)
                
                # Normalize to 0-1 range
                if confidence_scores.max() > confidence_scores.min():
                    confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())
                
                return confidence_scores
                
        except Exception as e:
            logger.error(f"Error calculating confidence scores: {str(e)}")
            return np.ones(len(predictions)) * 0.5  # Default confidence of 0.5
    
    def _save_predictions(self, predictions_df, market):
        """Save predictions for later evaluation"""
        try:
            # Check if predictions_df is empty
            if predictions_df.empty:
                logger.warning("Empty predictions DataFrame, skipping save")
                return
                
            # Create a simplified dataframe with just the predictions
            pred_data = predictions_df.reset_index()[['Symbol', 'predicted_return', 'confidence_score']]
            pred_data['date'] = datetime.now().strftime('%Y-%m-%d')
            pred_data['market'] = market
            
            # Save to CSV
            predictions_file = self.model_dir / f'predictions_{market}_{datetime.now().strftime("%Y%m%d")}.csv'
            pred_data.to_csv(predictions_file, index=False)
            logger.info(f"Saved predictions to {predictions_file}")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
    
    def generate_analysis_report(self, predictions, features_df):
        """Generate detailed analysis report for predicted top gainers in JSON format"""
        # Check if predictions is empty
        if predictions.empty:
            logger.warning("Empty predictions DataFrame, returning empty report")
            return []
            
        report = []
        
        for symbol in predictions.index:
            try:
                stock_data = features_df[features_df['Symbol'] == symbol].sort_index()
                
                # Skip if no data for this symbol
                if stock_data.empty:
                    logger.warning(f"No data found for symbol {symbol}")
                    continue
                    
                latest_data = stock_data.iloc[-1]
                
                # Calculate additional metrics
                momentum_score = self._calculate_momentum_score(stock_data)
                trend_strength = self._calculate_trend_strength(stock_data)
                risk_score = self._calculate_risk_score(stock_data)
                
                # Calculate predicted price
                current_price = latest_data['Close']
                predicted_return = predictions.loc[symbol, 'predicted_return']
                predicted_price = current_price * (1 + predicted_return)
                
                # Get confidence score
                confidence_score = predictions.loc[symbol, 'confidence_score'] if 'confidence_score' in predictions.columns else 0.5
                
                # Create JSON structure for each stock
                stock_report = {
                    'symbol': symbol,
                    'company_name': stock_data['CompanyName'].iloc[0] if 'CompanyName' in stock_data.columns else 'Unknown',
                    'overview': {
                        'current_price': float(round(current_price, 2)),
                        'predicted_price': float(round(predicted_price, 2)),
                        'predicted_return_percent': float(round(predicted_return * 100, 2)),
                        'confidence_score': float(round(confidence_score * 100, 2))
                    },
                    'technical_analysis': {
                        'rsi': {
                            'value': float(round(latest_data.get('RSI', 50), 2)),
                            'signal': 'Oversold' if latest_data.get('RSI', 50) < 30 else 'Overbought' if latest_data.get('RSI', 50) > 70 else 'Neutral'
                        },
                        'macd': {
                            'value': float(round(latest_data.get('MACD', 0), 2)),
                            'signal': 'Bullish' if latest_data.get('MACD', 0) > 0 else 'Bearish'
                        },
                        'momentum_score': float(round(momentum_score, 2)),
                        'trend_strength': float(round(trend_strength, 2)),
                        'stochastic_k': float(round(latest_data.get('Stochastic_K', 50), 2)),
                        'stochastic_d': float(round(latest_data.get('Stochastic_D', 50), 2)),
                        'williams_r': float(round(latest_data.get('Williams_R', -50), 2)),
                        'cmf': float(round(latest_data.get('CMF', 0), 2))
                    },
                    'market_regime': {
                        'regime': 'Bullish' if latest_data.get('regime', 0) > 0 else 'Bearish',
                        'volatility_regime': 'High' if latest_data.get('volatility_regime', 0) > 0 else 'Low',
                        'market_phase': self._interpret_market_phase(latest_data.get('market_phase', 0))
                    },
                    'fundamental_analysis': {
                        'market_cap_millions': float(round(latest_data.get('marketCap', 0)/1e6, 2)),
                        'pe_ratio': float(round(latest_data.get('trailingPE', 0), 2)),
                        'price_to_book': float(round(latest_data.get('priceToBook', 0), 2)),
                        'debt_to_equity': float(round(latest_data.get('debtToEquity', 0), 2))
                    },
                    'market_sentiment': {
                        'news_sentiment_score': float(round(latest_data.get('news_sentiment', 0), 2)),
                        'volatility_risk_score': float(round(risk_score, 2))
                    },
                    'recent_performance': {
                        'return_1d': float(round(latest_data.get('Returns', 0) * 100, 2)),
                        'return_5d': float(round(latest_data.get('Returns_5d', 0) * 100, 2)),
                        'return_20d': float(round(latest_data.get('Returns_20d', 0) * 100, 2))
                    },
                    'key_signals': self._get_key_signals(latest_data)
                }
                
                report.append(stock_report)
            except Exception as e:
                logger.error(f"Error generating report for {symbol}: {str(e)}")
        
        return report
    
    def _interpret_market_phase(self, phase_value):
        """Interpret market phase value"""
        if phase_value == 3:
            return "Strong Uptrend"
        elif phase_value == 1:
            return "Weak Uptrend"
        elif phase_value == -1:
            return "Weak Downtrend"
        elif phase_value == -3:
            return "Strong Downtrend"
        else:
            return "Sideways"
    
    def _get_key_signals(self, latest_data):
        """Extract key trading signals from the data"""
        signals = []
        
        # Check for bullish signals
        if latest_data.get('bullish_engulfing', 0) == 1:
            signals.append({"type": "bullish", "name": "Bullish Engulfing Pattern", "strength": 8})
        
        if latest_data.get('hammer', 0) == 1:
            signals.append({"type": "bullish", "name": "Hammer Pattern", "strength": 7})
        
        if latest_data.get('RSI', 50) < 30:
            signals.append({"type": "bullish", "name": "RSI Oversold", "strength": 6})
        
        if latest_data.get('SMA_20_50_Crossover', 0) == 1 and latest_data.get('SMA_20_50_Crossover', 0) != latest_data.get('SMA_20_50_Crossover', 0):
            signals.append({"type": "bullish", "name": "Golden Cross (20/50)", "strength": 9})
        
        # Check for bearish signals
        if latest_data.get('bearish_engulfing', 0) == 1:
            signals.append({"type": "bearish", "name": "Bearish Engulfing Pattern", "strength": 8})
        
        if latest_data.get('shooting_star', 0) == 1:
            signals.append({"type": "bearish", "name": "Shooting Star Pattern", "strength": 7})
        
        if latest_data.get('RSI', 50) > 70:
            signals.append({"type": "bearish", "name": "RSI Overbought", "strength": 6})
        
        # Ichimoku signals
        if latest_data.get('Above_Cloud', 0) == 1:
            signals.append({"type": "bullish", "name": "Price Above Ichimoku Cloud", "strength": 7})
        
        if latest_data.get('Below_Cloud', 0) == 1:
            signals.append({"type": "bearish", "name": "Price Below Ichimoku Cloud", "strength": 7})
        
        # Volume signals
        if latest_data.get('Relative_Volume', 1) > 2:
            signals.append({"type": "neutral", "name": "High Relative Volume", "strength": 5})
        
        # Bollinger Band signals
        if latest_data.get('BB_Squeeze', 0) == 1:
            signals.append({"type": "neutral", "name": "Bollinger Band Squeeze", "strength": 8})
        
        # Sort by strength
        signals.sort(key=lambda x: x["strength"], reverse=True)
        
        return signals[:5]  # Return top 5 signals
    
    def _calculate_momentum_score(self, stock_data):
        """Calculate momentum score based on multiple indicators"""
        try:
            latest = stock_data.iloc[-1]
            
            # Combine RSI, MACD, and MFI signals
            rsi_score = (latest.get('RSI', 50) - 30) / 40  # Normalize RSI
            macd_score = 1 if latest.get('MACD', 0) > 0 else 0
            mfi_score = (latest.get('MFI', 50) - 30) / 40  # Normalize MFI
            
            # Add ROC indicators
            roc_score = 0
            if 'ROC_5' in latest and 'ROC_20' in latest:
                roc_score = (latest['ROC_5'] / 10 + latest['ROC_20'] / 20) / 2
                roc_score = max(-1, min(1, roc_score))  # Clamp to [-1, 1]
            
            # Weight the signals
            momentum_score = (rsi_score * 0.3 + macd_score * 0.2 + mfi_score * 0.2 + roc_score * 0.3) * 10
            return max(0, min(10, momentum_score))  # Scale to 0-10
        except Exception as e:
            logger.error(f"Error calculating momentum score: {str(e)}")
            return 5.0  # Return neutral score on error
    
    def _calculate_trend_strength(self, stock_data):
        """Calculate trend strength based on multiple indicators"""
        try:
            latest = stock_data.iloc[-1]
            
            # Price momentum
            price_score = 0
            if 'Returns_5d' in latest and 'Returns_20d' in latest:
                price_score = (latest['Returns_5d'] * 0.6 + latest['Returns_20d'] * 0.4) * 10
            
            # RSI trend
            rsi_score = 0
            if 'RSI' in latest:
                rsi_score = (latest['RSI'] - 50) / 5  # Convert RSI to -10 to +10 scale
            
            # MACD trend
            macd_score = 0
            if 'MACD' in latest:
                macd_score = 10 if latest['MACD'] > 0 else -10
            
            # Volume trend
            volume_score = 0
            if 'OBV' in latest:
                volume_score = 10 if latest['OBV'] > 0 else -10
            
            # Moving average trend
            ma_score = 0
            if all(col in latest.index for col in ['SMA_20', 'SMA_50']):
                ma_score = 10 if latest['SMA_20'] > latest['SMA_50'] else -10
            
            # Add ADX trend strength if available
            adx_score = 0
            if 'ADX_Trend_Strength' in latest:
                adx_score = latest['ADX_Trend_Strength'] * 10
            
            # Combine scores
            trend_score = (price_score + rsi_score + macd_score + volume_score + ma_score + adx_score) / 6
            
            # Scale to 0-10 range
            return max(0, min(10, trend_score + 5))
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            return 5.0  # Return neutral score on error
    
    def _calculate_risk_score(self, stock_data):
        """Calculate risk score based on volatility and price stability"""
        try:
            latest = stock_data.iloc[-1]
            
            # Normalize volatility measures
            vol_5d_score = 10 - (latest.get('Volatility_5d', 0.01) * 100)  # Lower volatility = better score
            vol_20d_score = 10 - (latest.get('Volatility_20d', 0.01) * 100)
            
            # Consider price stability relative to Bollinger Bands if available
            bb_score = 5.0  # Default neutral score
            if all(col in latest.index for col in ['BB_UPPER', 'BB_LOWER', 'BB_MIDDLE']):
                bb_position = (latest['Close'] - latest['BB_LOWER']) / (latest['BB_UPPER'] - latest['BB_LOWER'])
                bb_score = 10 - abs(bb_position - 0.5) * 20  # Center position = better score
            
            # Add volatility ratio if available
            vol_ratio_score = 5.0
            if 'Volatility_Ratio' in latest:
                vol_ratio_score = 10 - (latest['Volatility_Ratio'] * 5)
                vol_ratio_score = max(0, min(10, vol_ratio_score))
            
            # Combine scores
            risk_score = (vol_5d_score * 0.3 + vol_20d_score * 0.3 + bb_score * 0.2 + vol_ratio_score * 0.2)
            return max(0, min(10, risk_score))  # Scale to 0-10
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {str(e)}")
            return 5.0  # Return neutral score on error
    
    def explain_prediction(self, model, features_df, symbol):
        """Explain prediction for a specific stock using XGBoost's feature importance"""
        try:
            # Check if model is None
            if model is None:
                logger.error("Model is None, cannot explain prediction")
                return {
                    'symbol': symbol,
                    'error': 'Model is None. Please ensure the model is properly trained.'
                }
            
            # Load selected features if available
            selected_features = None
            if self.selected_features_path.exists():
                try:
                    selected_features = joblib.load(self.selected_features_path)
                    logger.info(f"Loaded {len(selected_features)} selected features for explanation")
                except Exception as e:
                    logger.warning(f"Could not load selected features: {str(e)}. Using all features.")
                    selected_features = self.feature_cols
            else:
                logger.warning("Selected features file not found. Using all features.")
                selected_features = self.feature_cols
                
            # Get features for the symbol
            symbol_features = features_df[features_df['Symbol'] == symbol].iloc[-1]
            
            # Ensure all required features exist
            for col in selected_features:
                if col not in symbol_features:
                    logger.warning(f"Feature {col} is missing, adding with default value 0.0")
                    symbol_features[col] = 0.0
            
            # Get feature values
            X = symbol_features[selected_features].values.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Get feature importance from the model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Check if the number of features matches
                if len(importances) != len(selected_features):
                    logger.warning(f"Feature count mismatch: model has {len(importances)} features, selected features has {len(selected_features)} features")
                    return {
                        'symbol': symbol,
                        'prediction': float(prediction),
                        'error': 'Feature count mismatch between model and selected features'
                    }
                
                # Create feature importance dataframe
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': importances,
                    'value': X[0]
                }).sort_values('importance', ascending=False)
                
                # Calculate feature contribution (simplified approach)
                feature_importance['contribution'] = feature_importance['importance'] * feature_importance['value']
                
                # Normalize contributions
                total_contribution = feature_importance['contribution'].abs().sum()
                if total_contribution > 0:
                    feature_importance['contribution_pct'] = feature_importance['contribution'] / total_contribution * 100
                else:
                    feature_importance['contribution_pct'] = 0
                
                # Determine top factors
                top_factors = feature_importance.head(10)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_scores(model, pd.DataFrame([symbol_features[selected_features]]), [prediction])[0]
                
                explanation = {
                    'symbol': symbol,
                    'prediction': float(prediction),
                    'predicted_return_percent': float(round(prediction * 100, 2)),
                    'confidence_score': float(round(confidence_score * 100, 2)),
                    'top_factors': top_factors[['feature', 'importance', 'contribution_pct']].to_dict('records')
                }
                
                return explanation
            else:
                # If model doesn't have feature_importances_ attribute
                return {
                    'symbol': symbol,
                    'prediction': prediction,
                    'message': 'Feature importance not available for this model type'
                }
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return {
                'symbol': symbol,
                'prediction': prediction if 'prediction' in locals() else None,
                'error': str(e)
            }