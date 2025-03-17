import pandas as pd
import numpy as np
from datetime import datetime
import logging
import traceback
from textblob import TextBlob

logger = logging.getLogger(__name__)

class SentimentProcessor:
    def __init__(self, data_manager=None):
        self.data_manager = data_manager
        self.sentiment_cache = {}  # Cache for sentiment values
    
    def add_sentiment_features(self, features_df):
        """
        Add additional sentiment features to the feature dataframe
        - news_sentiment: sentiment from news articles
        - social_sentiment: derived from news_sentiment
        - sector_sentiment: average sentiment for stocks in the same sector
        - market_sentiment: overall market sentiment
        """
        try:
            # Check if dataframe is empty
            if features_df.empty:
                logger.warning("Empty dataframe provided to sentiment processor")
                return features_df
                
            # Make a copy to avoid modifying the original
            df = features_df.copy()
            
            # Ensure news_sentiment exists
            if 'news_sentiment' not in df.columns:
                logger.warning("news_sentiment column not found, adding with default values")
                df['news_sentiment'] = 0.0
            
            # Add social_sentiment (currently derived from news_sentiment)
            # In a future version, this could be replaced with actual social media sentiment
            df['social_sentiment'] = df['news_sentiment'] * 0.8  # Slightly adjusted version of news sentiment
            
            # Calculate sector_sentiment (simplified version - in reality would group by sector)
            try:
                # Group by date and calculate average sentiment
                if isinstance(df.index, pd.DatetimeIndex):
                    date_sentiment = df.groupby(df.index.date)['news_sentiment'].mean().reset_index()
                    date_sentiment.columns = ['date', 'avg_sentiment']
                    
                    # Map back to original dataframe
                    df['sector_sentiment'] = df.apply(
                        lambda row: date_sentiment[
                            date_sentiment['date'] == row.name.date()
                        ]['avg_sentiment'].values[0] if len(date_sentiment[
                            date_sentiment['date'] == row.name.date()
                        ]) > 0 else 0.0, 
                        axis=1
                    )
                else:
                    # If index is not datetime, use a simpler approach
                    logger.warning("DataFrame index is not DatetimeIndex, using simplified sector sentiment")
                    df['sector_sentiment'] = df.groupby('Symbol')['news_sentiment'].transform('mean')
            except Exception as e:
                logger.error(f"Error calculating sector sentiment: {str(e)}")
                df['sector_sentiment'] = 0.0
            
            # Calculate market_sentiment (overall average)
            try:
                market_avg_sentiment = df['news_sentiment'].mean()
                if pd.isna(market_avg_sentiment):
                    market_avg_sentiment = 0.0
                df['market_sentiment'] = market_avg_sentiment
            except Exception as e:
                logger.error(f"Error calculating market sentiment: {str(e)}")
                df['market_sentiment'] = 0.0
            
            # Ensure all sentiment columns exist and have valid values
            sentiment_columns = ['news_sentiment', 'social_sentiment', 'sector_sentiment', 'market_sentiment']
            for col in sentiment_columns:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    # Replace NaN values with 0
                    df[col] = df[col].fillna(0.0)
                    # Clip values to range [-1, 1]
                    df[col] = df[col].clip(-1, 1)
            
            logger.info("Added sentiment features: social_sentiment, sector_sentiment, market_sentiment")
            return df
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {str(e)}")
            traceback.print_exc()
            # Return original dataframe if there's an error
            return features_df
    
    def get_sentiment_summary(self, symbol, features_df):
        """
        Get a summary of sentiment features for a specific symbol
        """
        try:
            # Check if dataframe is empty
            if features_df.empty:
                logger.warning(f"Empty dataframe provided for sentiment summary of {symbol}")
                return self._get_default_sentiment()
                
            # Filter for the symbol
            if 'Symbol' in features_df.columns:
                symbol_data = features_df[features_df['Symbol'] == symbol]
            else:
                logger.warning(f"Symbol column not found in dataframe for {symbol}")
                return self._get_default_sentiment()
            
            if symbol_data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return self._get_default_sentiment()
            
            # Get the latest data
            latest = symbol_data.iloc[-1]
            
            # Create summary
            summary = {
                'news_sentiment': float(latest.get('news_sentiment', 0.0)),
                'social_sentiment': float(latest.get('social_sentiment', 0.0)),
                'sector_sentiment': float(latest.get('sector_sentiment', 0.0)),
                'market_sentiment': float(latest.get('market_sentiment', 0.0))
            }
            
            # Ensure values are in valid range
            for key in summary:
                if pd.isna(summary[key]):
                    summary[key] = 0.0
                summary[key] = max(-1.0, min(1.0, summary[key]))  # Clip to [-1, 1]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {symbol}: {str(e)}")
            traceback.print_exc()
            return self._get_default_sentiment()
    
    def _get_default_sentiment(self):
        """Return default sentiment values"""
        return {
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'sector_sentiment': 0.0,
            'market_sentiment': 0.0
        }
    
    def analyze_text_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        Returns a value between -1 (negative) and 1 (positive)
        """
        try:
            if not text or pd.isna(text) or text.strip() == "":
                return 0.0
                
            # Check cache
            if text in self.sentiment_cache:
                return self.sentiment_cache[text]
                
            # Analyze sentiment
            analysis = TextBlob(text)
            sentiment = analysis.sentiment.polarity
            
            # Cache result
            self.sentiment_cache[text] = sentiment
            
            return sentiment
        except Exception as e:
            logger.error(f"Error analyzing text sentiment: {str(e)}")
            return 0.0
    
    def analyze_news_batch(self, news_items):
        """
        Analyze sentiment for a batch of news items
        news_items should be a list of dictionaries with 'title' and 'description' keys
        """
        try:
            if not news_items:
                return 0.0
                
            sentiments = []
            for item in news_items:
                title = item.get('title', '')
                description = item.get('description', '')
                
                # Combine title and description, with title weighted more heavily
                text = f"{title} {title} {description}"
                sentiment = self.analyze_text_sentiment(text)
                sentiments.append(sentiment)
            
            # Return average sentiment
            if sentiments:
                return sum(sentiments) / len(sentiments)
            return 0.0
        except Exception as e:
            logger.error(f"Error analyzing news batch: {str(e)}")
            return 0.0