import pandas as pd
import numpy as np
from datetime import datetime

class StockPredictor:
    def predict_top_gainers(self, model, features_df, top_n=5):
        """Predict top gaining stocks for the next day"""
        # Get the latest data for each stock
        latest_data = features_df.groupby('Symbol').last()
        
        # Prepare features for prediction
        feature_cols = [
            'RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV',
            'Returns', 'Returns_5d', 'Returns_20d',
            'Volatility_5d', 'Volatility_20d',
            'Price_to_VWAP', 'marketCap', 'trailingPE',
            'priceToBook', 'debtToEquity', 'news_sentiment',
            'Price_to_Fib', 'Price_to_Pivot'
        ]
        
        X = latest_data[feature_cols]
        
        # Make predictions
        predictions = model.predict(X)
        
        # Add predictions to the dataframe
        latest_data['predicted_return'] = predictions
        
        # Sort by predicted return and get top N
        top_gainers = latest_data.sort_values('predicted_return', ascending=False).head(top_n)
        
        return top_gainers
    
    def generate_analysis_report(self, predictions, features_df):
        """Generate detailed analysis report for predicted top gainers"""
        report = []
        
        for symbol in predictions.index:
            stock_data = features_df[features_df['Symbol'] == symbol].sort_index()
            latest_data = stock_data.iloc[-1]
            
            # Calculate additional metrics
            momentum_score = self._calculate_momentum_score(stock_data)
            trend_strength = self._calculate_trend_strength(stock_data)
            risk_score = self._calculate_risk_score(stock_data)
            
            # Generate report for each stock
            stock_report = f"\n{symbol} - Predicted Return: {predictions.loc[symbol, 'predicted_return']*100:.2f}%\n"
            stock_report += "=" * 50 + "\n"
            
            # Technical Analysis
            stock_report += "\nTechnical Analysis:\n"
            stock_report += f"- RSI ({latest_data['RSI']:.2f}): {'Oversold' if latest_data['RSI'] < 30 else 'Overbought' if latest_data['RSI'] > 70 else 'Neutral'}\n"
            stock_report += f"- MACD: {'Bullish' if latest_data['MACD'] > 0 else 'Bearish'} momentum\n"
            stock_report += f"- Momentum Score: {momentum_score:.2f}/10\n"
            stock_report += f"- Trend Strength: {trend_strength:.2f}/10\n"
            
            # Fundamental Analysis
            stock_report += "\nFundamental Analysis:\n"
            stock_report += f"- Market Cap: £{latest_data['marketCap']/1e6:.2f}M\n"
            stock_report += f"- P/E Ratio: {latest_data['trailingPE']:.2f}\n"
            stock_report += f"- Price to Book: {latest_data['priceToBook']:.2f}\n"
            stock_report += f"- Debt to Equity: {latest_data['debtToEquity']:.2f}%\n"
            
            # Market Sentiment
            stock_report += "\nMarket Sentiment:\n"
            stock_report += f"- News Sentiment Score: {latest_data['news_sentiment']:.2f}\n"
            stock_report += f"- Volatility Risk Score: {risk_score:.2f}/10\n"
            
            # Recent Performance
            stock_report += "\nRecent Performance:\n"
            stock_report += f"- 1-Day Return: {latest_data['Returns']*100:.2f}%\n"
            stock_report += f"- 5-Day Return: {latest_data['Returns_5d']*100:.2f}%\n"
            stock_report += f"- 20-Day Return: {latest_data['Returns_20d']*100:.2f}%\n"
            
            report.append(stock_report)
        
        return '\n'.join(report)
    
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
        """Calculate trend strength based on moving averages and ADX"""
        latest = stock_data.iloc[-1]
        
        # Check if price is above moving averages
        above_ema20 = latest['Close'] > latest['EMA_20']
        above_sma50 = latest['Close'] > latest['SMA_50']
        
        # Normalize ADX (0-100 scale)
        adx_score = latest['ADX'] / 10
        
        # Combine signals
        trend_score = (above_ema20 * 0.3 + above_sma50 * 0.3 + adx_score * 0.4) * 10
        return max(0, min(10, trend_score))  # Scale to 0-10
    
    def _calculate_risk_score(self, stock_data):
        """Calculate risk score based on volatility and price stability"""
        latest = stock_data.iloc[-1]
        
        # Normalize volatility measures
        vol_5d_score = 10 - (latest['Volatility_5d'] * 100)  # Lower volatility = better score
        vol_20d_score = 10 - (latest['Volatility_20d'] * 100)
        
        # Consider price stability relative to Bollinger Bands
        bb_position = (latest['Close'] - latest['BB_LOWER']) / (latest['BB_UPPER'] - latest['BB_LOWER'])
        bb_score = 10 - abs(bb_position - 0.5) * 20  # Center position = better score
        
        # Combine scores
        risk_score = (vol_5d_score * 0.3 + vol_20d_score * 0.3 + bb_score * 0.4)
        return max(0, min(10, risk_score))  # Scale to 0-10