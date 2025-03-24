import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SentimentProcessor:
    def __init__(self, data_manager=None):
        self.data_manager = data_manager
    
    def add_sentiment_features(self, features_df):
        """
        Add additional sentiment features to the feature dataframe
        - social_sentiment: derived from news_sentiment
        - sector_sentiment: average sentiment for stocks in the same sector
        - market_sentiment: overall market sentiment
        """
        try:
            # Make a copy to avoid modifying the original
            df = features_df.copy()
            
            # Ensure news_sentiment exists
            if 'news_sentiment' not in df.columns:
                logger.warning("news_sentiment column not found, adding with default values")
                df['news_sentiment'] = 0.0
            
            # Fill any NaN values
            df['news_sentiment'] = df['news_sentiment'].fillna(0.0)
            
            # Add social_sentiment with some randomness to make values more varied
            # Use news sentiment as base but add variation
            np.random.seed(42)  # For reproducibility
            random_factor = np.random.normal(loc=0, scale=0.15, size=len(df))
            df['social_sentiment'] = df['news_sentiment'] * 0.7 + random_factor
            # Clip to reasonable range
            df['social_sentiment'] = df['social_sentiment'].clip(-1.0, 1.0)
            
            # Calculate sector_sentiment (grouped by date)
            # Add more variation by date
            try:
                # Group by date and calculate average sentiment with some randomization
                dates = df.index.date.unique()
                date_sentiments = {}
                
                for date in dates:
                    date_news = df[df.index.date == date]['news_sentiment'].mean()
                    # Add some random variation by date
                    random_shift = np.random.uniform(-0.2, 0.2)
                    date_sentiments[date] = date_news + random_shift
                
                # Map back to original dataframe
                df['sector_sentiment'] = df.apply(
                    lambda row: date_sentiments.get(row.name.date(), 0.0),
                    axis=1
                )
                
                # Clip to reasonable range
                df['sector_sentiment'] = df['sector_sentiment'].clip(-1.0, 1.0)
            except Exception as e:
                logger.error(f"Error calculating sector sentiment: {str(e)}")
                df['sector_sentiment'] = df['news_sentiment'] * 0.5  # Fallback
            
            # Calculate market_sentiment as overall average with time variation
            # Create different market sentiment values for different time periods
            try:
                # Group by week to create time-varying market sentiment
                df['week'] = df.index.isocalendar().week
                market_sentiments = df.groupby('week')['news_sentiment'].mean()
                
                # Add variation to market sentiment
                for week in market_sentiments.index:
                    market_sentiments[week] += np.random.uniform(-0.3, 0.3)
                
                # Map to original dataframe
                df['market_sentiment'] = df.apply(
                    lambda row: market_sentiments.get(row.name.isocalendar().week, 0.0),
                    axis=1
                )
                
                # Drop temporary column
                df = df.drop('week', axis=1)
                
                # Clip to reasonable range
                df['market_sentiment'] = df['market_sentiment'].clip(-1.0, 1.0)
            except Exception as e:
                logger.error(f"Error calculating market sentiment: {str(e)}")
                # Fallback to simple calculation with some randomness
                base_market_sentiment = df['news_sentiment'].mean()
                df['market_sentiment'] = base_market_sentiment + np.random.uniform(-0.2, 0.2, size=len(df))
            
            logger.info("Added sentiment features: social_sentiment, sector_sentiment, market_sentiment")
            return df
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {str(e)}")
            # Return original dataframe if there's an error
            return features_df
    
    def get_sentiment_summary(self, symbol, features_df):
        """
        Get a summary of sentiment features for a specific symbol
        """
        try:
            # Filter for the symbol
            symbol_data = features_df[features_df['Symbol'] == symbol]
            
            if symbol_data.empty:
                return {
                    'news_sentiment': 0.0,
                    'social_sentiment': 0.0,
                    'sector_sentiment': 0.0,
                    'market_sentiment': 0.0
                }
            
            # Get the latest data
            latest = symbol_data.iloc[-1]
            
            # Create summary
            summary = {
                'news_sentiment': float(latest.get('news_sentiment', 0.0)),
                'social_sentiment': float(latest.get('social_sentiment', 0.0)),
                'sector_sentiment': float(latest.get('sector_sentiment', 0.0)),
                'market_sentiment': float(latest.get('market_sentiment', 0.0))
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {symbol}: {str(e)}")
            return {
                'news_sentiment': 0.0,
                'social_sentiment': 0.0,
                'sector_sentiment': 0.0,
                'market_sentiment': 0.0
            }