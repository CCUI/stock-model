import pandas as pd
import numpy as np
from datetime import datetime
import logging
from .config import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self):
        self.feature_cols = FEATURE_COLUMNS
    
    def predict_top_gainers(self, model, features_df, top_n=10, market='UK'):
        """Predict top gaining stocks for tomorrow"""
        try:
            # Get the latest data for each stock
            latest_data = features_df.groupby('Symbol').last()
            
            # Ensure all required features are present
            required_features = model.feature_names_in_
            for feature in required_features:
                if feature not in latest_data.columns:
                    logger.warning(f"Missing feature {feature}, setting to 0")
                    latest_data[feature] = 0.0
            
            # Select only the features used by the model
            X = latest_data[required_features]
            
            # Make predictions
            predictions = model.predict(X)
            
            # Create DataFrame with predictions
            results = pd.DataFrame({
                'Symbol': X.index,
                'Predicted_Return': predictions
            })
            
            # Sort by predicted return and get top N
            top_gainers = results.nlargest(top_n, 'Predicted_Return')
            
            # Add company names if available
            if 'CompanyName' in features_df.columns:
                company_names = features_df.groupby('Symbol')['CompanyName'].first()
                top_gainers = top_gainers.join(company_names)
            
            return top_gainers
            
        except Exception as e:
            logger.error(f"Error in predict_top_gainers: {str(e)}")
            raise Exception(f"Failed to analyze stocks: {str(e)}")
    
    def generate_analysis_report(self, predictions, features_df):
        """Generate detailed analysis report for predicted top gainers in JSON format"""
        try:
            if predictions is None or predictions.empty:
                logger.warning("No predictions available to generate report")
                return []
            
            report = []
            
            for symbol in predictions.index:
                try:
                    # Get data for this symbol
                    stock_data = features_df[features_df['Symbol'] == symbol]
                    
                    if stock_data.empty:
                        logger.warning(f"No data found for symbol {symbol}")
                        continue
                    
                    stock_data = stock_data.sort_index()
                    latest_data = stock_data.iloc[-1]
                    
                    # Calculate additional metrics with error handling
                    try:
                        momentum_score = self._calculate_momentum_score(stock_data)
                    except Exception as e:
                        logger.error(f"Error calculating momentum score for {symbol}: {str(e)}")
                        momentum_score = 5.0  # Default neutral score
                    
                    try:
                        trend_strength = self._calculate_trend_strength(stock_data)
                    except Exception as e:
                        logger.error(f"Error calculating trend strength for {symbol}: {str(e)}")
                        trend_strength = 5.0  # Default neutral score
                    
                    try:
                        risk_score = self._calculate_risk_score(stock_data)
                    except Exception as e:
                        logger.error(f"Error calculating risk score for {symbol}: {str(e)}")
                        risk_score = 5.0  # Default neutral score
                    
                    # Get current price and predicted return with error handling
                    try:
                        current_price = latest_data.get('Close', 0)
                        predicted_return = predictions.loc[symbol, 'Predicted_Return']
                        predicted_price = current_price * (1 + predicted_return)
                    except Exception as e:
                        logger.error(f"Error calculating price predictions for {symbol}: {str(e)}")
                        current_price = 0
                        predicted_price = 0
                        predicted_return = 0
                    
                    # Create JSON structure for each stock with safe value access
                    stock_report = {
                        'symbol': symbol,
                        'company_name': stock_data['CompanyName'].iloc[0] if 'CompanyName' in stock_data.columns else 'Unknown',
                        'overview': {
                            'current_price': float(round(current_price, 2)),
                            'predicted_price': float(round(predicted_price, 2)),
                            'predicted_return_percent': float(round(predicted_return * 100, 2))
                        },
                        'technical_analysis': {
                            'rsi': {
                                'value': float(round(latest_data.get('RSI', 50), 2)),
                                'signal': self._get_rsi_signal(latest_data.get('RSI', 50))
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
                        }
                    }
                    
                    report.append(stock_report)
                    
                except Exception as e:
                    logger.error(f"Error generating report for {symbol}: {str(e)}")
                    continue
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {str(e)}")
            return []
    
    def _calculate_momentum_score(self, stock_data):
        """Calculate momentum score based on multiple indicators"""
        latest = stock_data.iloc[-1]
        
        # Combine RSI, MACD, and MFI signals
        rsi_score = (latest['RSI'] - 30) / 40  # Normalize RSI
        macd_score = 1 if latest['MACD'] > 0 else 0
        mfi_score = (latest['MFI'] - 30) / 40  # Normalize MFI
        
        # Weight the signals
        momentum_score = (rsi_score * 0.4 + macd_score * 0.3 + mfi_score * 0.3) * 10
        return max(0, min(10, momentum_score))  # Scale to 0-10
    
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
            
            # Combine scores
            trend_score = (price_score + rsi_score + macd_score + volume_score + ma_score) / 5
            
            # Scale to 0-10 range
            return max(0, min(10, trend_score + 5))
            
        except Exception as e:
            print(f"Error calculating trend strength: {str(e)}")
            return 5.0  # Return neutral score on error
    
    def _calculate_risk_score(self, stock_data):
        """Calculate risk score based on volatility and price stability"""
        try:
            latest = stock_data.iloc[-1]
            
            # Normalize volatility measures
            vol_5d_score = 10 - (latest['Volatility_5d'] * 100)  # Lower volatility = better score
            vol_20d_score = 10 - (latest['Volatility_20d'] * 100)
            
            # Consider price stability relative to Bollinger Bands if available
            bb_score = 5.0  # Default neutral score
            if all(col in latest.index for col in ['BB_UPPER', 'BB_LOWER', 'BB_MIDDLE']):
                bb_position = (latest['Close'] - latest['BB_LOWER']) / (latest['BB_UPPER'] - latest['BB_LOWER'])
                bb_score = 10 - abs(bb_position - 0.5) * 20  # Center position = better score
            
            # Combine scores
            risk_score = (vol_5d_score * 0.4 + vol_20d_score * 0.4 + bb_score * 0.2)
            return max(0, min(10, risk_score))  # Scale to 0-10
            
        except Exception as e:
            print(f"Error calculating risk score: {str(e)}")
            return 5.0  # Return neutral score on error
    
    def explain_prediction(self, model, features_df, symbol):
        """Explain prediction for a specific stock using feature importance"""
        try:
            # Get features for the symbol
            symbol_data = features_df[features_df['Symbol'] == symbol]
            if symbol_data.empty:
                logger.error(f"No data found for symbol {symbol}")
                return None
                
            symbol_features = symbol_data.iloc[-1]
            
            # Get feature values
            feature_cols = [col for col in self.feature_cols if col in symbol_features.index]
            X = symbol_features[feature_cols].values.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Get feature importance from the model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                logger.warning("Model does not have feature importances")
                return {
                    'symbol': symbol,
                    'prediction': prediction,
                    'factors': []
                }
            
            # Create feature importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances,
                'value': X[0]
            })
            
            # Sort by absolute importance
            importance_df['abs_importance'] = abs(importance_df['importance'])
            importance_df = importance_df.sort_values('abs_importance', ascending=False)
            
            # Get top contributing factors
            top_factors = importance_df.head(5).to_dict('records')
            
            explanation = {
                'symbol': symbol,
                'prediction': float(prediction),
                'factors': top_factors
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction for {symbol}: {str(e)}")
            return None

    def _get_rsi_signal(self, rsi_value):
        """Get RSI signal with error handling"""
        try:
            if rsi_value < 30:
                return 'Oversold'
            elif rsi_value > 70:
                return 'Overbought'
            else:
                return 'Neutral'
        except Exception:
            return 'Neutral'