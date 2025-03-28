import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd
from datetime import datetime

class StockPredictor:
    def __init__(self):
        # Import feature columns from config to ensure consistency
        from .config import FEATURE_COLUMNS
        self.feature_cols = FEATURE_COLUMNS
    
    def predict_top_gainers(self, model, features_df, top_n=5, market='UK'):
        """Predict top gaining stocks for the next day"""
        try:
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
            for col in self.feature_cols:
                if col not in latest_data.columns:
                    missing_features.append(col)
                    latest_data[col] = 0.0
            
            # Log warning if features are missing
            if missing_features:
                print(f"Warning: The following features are missing and were set to 0.0: {', '.join(missing_features)}")
            
            # Handle additional features in the data that aren't in the model
            # This fixes the feature_names mismatch error
            extra_features = [col for col in latest_data.columns if col not in self.feature_cols and col in ['social_sentiment', 'sector_sentiment', 'market_sentiment']]
            if extra_features:
                for col in extra_features:
                    if col in latest_data.columns:
                        # Instead of dropping these columns, keep them for the report
                        # but don't include them in the prediction features
                        pass
            
            # Prepare features for prediction
            X = latest_data[self.feature_cols]
            
            # Validate data before prediction
            if X.isnull().any().any():
                raise ValueError("Features contain null values after preprocessing")
            
            # Make predictions
            predictions = model.predict(X)
            
            # Add predictions to the dataframe
            latest_data['predicted_return'] = predictions
            
            # Sort by predicted return and get top N
            top_gainers = latest_data.sort_values('predicted_return', ascending=False).head(top_n)
            
            return top_gainers
        except Exception as e:
            error_msg = f"Error in predict_top_gainers: {str(e)}"
            print(error_msg)
            raise Exception(f"Failed to analyze stocks: {str(e)}")
    
    def generate_analysis_report(self, predictions, features_df):
        """Generate detailed analysis report for predicted top gainers in JSON format"""
        report = []
        
        for symbol in predictions.index:
            stock_data = features_df[features_df['Symbol'] == symbol].sort_index()
            latest_data = stock_data.iloc[-1]
            
            # Calculate additional metrics
            momentum_score = self._calculate_momentum_score(stock_data)
            trend_strength = self._calculate_trend_strength(stock_data)
            risk_score = self._calculate_risk_score(stock_data)
            
            # Calculate predicted price
            current_price = latest_data['Close']
            predicted_return = predictions.loc[symbol, 'predicted_return']
            predicted_price = current_price * (1 + predicted_return)
            
            # Ensure sentiment data exists
            news_sentiment = float(round(latest_data.get('news_sentiment', 0.0), 2))
            social_sentiment = float(round(latest_data.get('social_sentiment', 0.0), 2))
            sector_sentiment = float(round(latest_data.get('sector_sentiment', 0.0), 2))
            market_sentiment = float(round(latest_data.get('market_sentiment', 0.0), 2))
            
            # Create JSON structure for each stock
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
                        'value': float(round(latest_data['RSI'], 2)),
                        'signal': 'Oversold' if latest_data['RSI'] < 30 else 'Overbought' if latest_data['RSI'] > 70 else 'Neutral'
                    },
                    'macd': {
                        'value': float(round(latest_data['MACD'], 2)),
                        'signal': 'Bullish' if latest_data['MACD'] > 0 else 'Bearish'
                    },
                    'momentum_score': float(round(momentum_score, 2)),
                    'trend_strength': float(round(trend_strength, 2)),
                    'stochastic_k': float(round(latest_data['Stochastic_K'], 2)),
                    'stochastic_d': float(round(latest_data['Stochastic_D'], 2)),
                    'williams_r': float(round(latest_data['Williams_R'], 2)),
                    'cmf': float(round(latest_data['CMF'], 2))
                },
                'fundamental_analysis': {
                    'market_cap_millions': float(round(latest_data['marketCap']/1e6, 2)),
                    'pe_ratio': float(round(latest_data['trailingPE'], 2)),
                    'price_to_book': float(round(latest_data['priceToBook'], 2)),
                    'debt_to_equity': float(round(latest_data['debtToEquity'], 2))
                },
                'market_sentiment': {
                    'news_sentiment_score': news_sentiment,
                    'social_sentiment_score': social_sentiment,
                    'sector_sentiment_score': sector_sentiment,
                    'market_sentiment_score': market_sentiment,
                    'volatility_risk_score': float(round(risk_score, 2))
                },
                'recent_performance': {
                    'return_1d': float(round(latest_data['Returns'] * 100, 2)),
                    'return_5d': float(round(latest_data['Returns_5d'] * 100, 2)),
                    'return_20d': float(round(latest_data['Returns_20d'] * 100, 2))
                }
            }
            
            report.append(stock_report)
        
        return report
    
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
            # Ensure we have at least a week of data
            if len(stock_data) < 5:
                return 5.0  # Return neutral score for insufficient data
                
            latest = stock_data.iloc[-1]
            
            # Get volatility metrics - make sure we don't have NaN or zero values
            vol_5d = latest.get('Volatility_5d', 0.01)
            vol_5d = max(0.0001, vol_5d)  # Ensure value is positive
            
            vol_20d = latest.get('Volatility_20d', 0.01)
            vol_20d = max(0.0001, vol_20d)  # Ensure value is positive
            
            # Calculate volatility scores on a logarithmic scale to better differentiate
            # Higher volatility = higher risk score
            vol_5d_score = min(10, max(1, 5 + 5 * (vol_5d / 0.02)))  # Scale around typical volatility of 2%
            vol_20d_score = min(10, max(1, 5 + 5 * (vol_20d / 0.04)))  # Scale around typical volatility of 4%
            
            # Consider price stability relative to Bollinger Bands if available
            bb_score = 5.0  # Default neutral score
            if all(key in latest for key in ['BB_UPPER', 'BB_LOWER', 'BB_MIDDLE', 'Close']):
                try:
                    bb_width = (latest['BB_UPPER'] - latest['BB_LOWER']) / latest['BB_MIDDLE']
                    bb_width_score = min(10, max(1, 5 * bb_width * 10))  # Wider bands = higher risk
                    
                    bb_position = (latest['Close'] - latest['BB_LOWER']) / (latest['BB_UPPER'] - latest['BB_LOWER'])
                    position_score = min(10, max(1, 10 * abs(bb_position - 0.5) * 2))  # Extreme positions = higher risk
                    
                    bb_score = (bb_width_score * 0.5 + position_score * 0.5)
                except:
                    bb_score = 5.0
            
            # Recent price activity - sharp changes indicate higher risk
            price_change_score = 5.0
            if 'Returns' in latest and 'Returns_5d' in latest:
                recent_change = abs(latest['Returns'] * 100)  # Convert to percentage
                weekly_change = abs(latest['Returns_5d'] * 100)  # Convert to percentage
                price_change_score = min(10, max(1, (recent_change + weekly_change)))
            
            # Combine scores
            risk_score = (vol_5d_score * 0.3 + vol_20d_score * 0.3 + bb_score * 0.2 + price_change_score * 0.2)
            return round(max(1, min(10, risk_score)), 1)  # Scale to 1-10 and round to 1 decimal place
            
        except Exception as e:
            print(f"Error calculating risk score: {str(e)}")
            return 5.0  # Return neutral score on error
    
    def explain_prediction(self, model, features_df, symbol):
        """Explain prediction for a specific stock"""
        try:
            # Get features for the symbol
            symbol_features = features_df[features_df['Symbol'] == symbol].iloc[-1]
            
            # Get feature values
            feature_values = {}
            for feature in self.feature_cols:
                if feature in symbol_features:
                    feature_values[feature] = float(symbol_features[feature])
                else:
                    feature_values[feature] = 0.0
            
            X = pd.DataFrame([feature_values])
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Get feature importance
            try:
                # For tree-based models like XGBoost
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Calculate the effect direction
                    effect = []
                    for i, feature_name in enumerate(self.feature_cols):
                        # Get feature value and determine if it's above average
                        if feature_name in X:
                            feature_val = X[feature_name].iloc[0]
                            # If feature value is high (above average), it contributes in direction of importance
                            # If feature value is low (below average), it contributes opposite of importance
                            effect.append(importances[i] * (1 if feature_val > 0 else -1))
                        else:
                            effect.append(0)
                    
                    # Create a dataframe with features and importance values
                    feature_importance = pd.DataFrame({
                        'feature': self.feature_cols,
                        'importance': importances,
                        'effect': effect
                    }).sort_values('importance', ascending=False)
                    
                    # Determine top factors
                    positive_factors = feature_importance[feature_importance['effect'] > 0].head(3)
                    negative_factors = feature_importance[feature_importance['effect'] < 0].head(3)
                    
                    explanation = {
                        'symbol': symbol,
                        'prediction': float(prediction),
                        'positive_factors': positive_factors[['feature', 'importance']].to_dict('records'),
                        'negative_factors': negative_factors[['feature', 'importance']].to_dict('records')
                    }
                    
                    return explanation
                else:
                    # For other models, return basic prediction without explanation
                    return {
                        'symbol': symbol,
                        'prediction': float(prediction),
                        'note': 'Feature importance not available for this model type'
                    }
            except Exception as e:
                # Fallback to just prediction if feature importance fails
                print(f"Error getting feature importance: {str(e)}")
                return {
                    'symbol': symbol,
                    'prediction': float(prediction),
                    'note': f'Could not generate explanation: {str(e)}'
                }
                
        except Exception as e:
            print(f"Error explaining prediction: {str(e)}")
            return {
                'symbol': symbol,
                'prediction': None,
                'error': str(e)
            }