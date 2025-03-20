import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from src.data_collector import UKStockDataCollector, USStockDataCollector
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import StockPredictor
from src.utils import setup_logging, get_trading_day
from src.prediction_history import PredictionHistoryManager
import argparse
import sys

def main(market='UK', analysis_date=None, include_news_sentiment=True, refresh_cache=False):
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    logger = setup_logging()
    
    # If no date provided, use the last trading day based on the selected market
    if analysis_date is None:
        analysis_date = get_trading_day(market, datetime.now())
    
    logger.info(f"Starting analysis for {market} market on date: {analysis_date}")
    
    try:
        # Initialize components
        if market.upper() == 'UK':
            data_collector = UKStockDataCollector(market='UK', include_news_sentiment=include_news_sentiment)
        elif market.upper() == 'US':
            data_collector = USStockDataCollector(market='US', include_news_sentiment=include_news_sentiment)
        else:
            raise ValueError(f"Unsupported market: {market}. Available markets: UK, US")
        
        # Refresh cache if requested
        if refresh_cache:
            logger.info("Refreshing cache...")
            data_collector.data_manager.refresh_cache()
            logger.info("Cache refresh completed")
            
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer()
        predictor = StockPredictor()
        
        # Collect historical data
        logger.info(f"Collecting historical stock data for {market} market...")
        stock_data = data_collector.collect_historical_data(end_date=analysis_date)
        
        # Generate features
        logger.info("Generating features...")
        features_df = feature_engineer.generate_features(stock_data)
        
        # Train model if needed
        logger.info("Training/updating model...")
        model = model_trainer.train_model(features_df)
        
        # Make predictions
        logger.info("Making predictions...")
        predictions = predictor.predict_top_gainers(model, features_df, top_n=10, market=market)
        
        # Generate detailed analysis report
        logger.info("Generating analysis report...")
        report = predictor.generate_analysis_report(predictions, features_df)
        
        # Save prediction to history
        logger.info("Saving prediction to history...")
        history_manager = PredictionHistoryManager(market=market)
        history_manager.save_prediction(predictions, report, analysis_date)
        
        print(f"\nTop 10 Predicted Gainers for Tomorrow ({market} Market):")
        print("=====================================")
        print(report)
        
        # Get prediction history
        prediction_history = history_manager.get_prediction_history()
        print(f"\nPrediction History (Last {len(prediction_history)} records):")
        print("=====================================")
        for i, record in enumerate(prediction_history):
            print(f"Record {i+1}: {record['date']} - {record['market']} Market")
        
        return predictions, report, prediction_history
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Market Analysis and Prediction')
    parser.add_argument('--market', choices=['UK', 'US'], default='UK', help='Market to analyze (UK or US)')
    parser.add_argument('--date', help='Analysis date (YYYY-MM-DD format)')
    parser.add_argument('--no-sentiment', action='store_true', help='Disable news sentiment analysis')
    parser.add_argument('--refresh-cache', action='store_true', help='Refresh all cache files')
    
    args = parser.parse_args()
    
    # Convert date string to datetime if provided
    analysis_date = None
    if args.date:
        try:
            analysis_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print("Error: Date must be in YYYY-MM-DD format")
            sys.exit(1)
    
    # Run main function with arguments
    main(
        market=args.market,
        analysis_date=analysis_date,
        include_news_sentiment=not args.no_sentiment,
        refresh_cache=args.refresh_cache
    )