import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_collector import UKStockDataCollector, USStockDataCollector
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import StockPredictor
from src.utils import setup_logging, get_trading_day
from src.load_env import load_environment_variables
from src.prediction_history import PredictionHistoryManager
import argparse
import logging
import traceback
import sys

def main(market='UK', analysis_date=None, include_news_sentiment=True):
    # Load environment variables
    load_environment_variables()
    
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
            
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer()
        predictor = StockPredictor()
        
        # Collect historical data
        logger.info(f"Collecting historical stock data for {market} market...")
        stock_data = data_collector.collect_historical_data(end_date=analysis_date)
        
        if stock_data.empty:
            logger.error("No stock data collected. Exiting.")
            return None, None, None
        
        # Generate features
        logger.info("Generating features...")
        features_df = feature_engineer.generate_features(stock_data)
        
        if features_df.empty:
            logger.error("No features generated. Exiting.")
            return None, None, None
        
        # Train model
        logger.info("Training/updating model...")
        model = model_trainer.train_model(features_df, tune_hyperparams=True, use_feature_selection=True)
        
        if model is None:
            logger.error("Model training failed. Exiting.")
            return None, None, None
        
        # Make predictions
        logger.info("Making predictions...")
        top_gainers = predictor.predict_top_gainers(model, features_df, top_n=10, market=market)
        
        if top_gainers.empty:
            logger.error("No predictions generated. Exiting.")
            return None, None, None
        
        # Generate analysis report
        logger.info("Generating analysis report...")
        analysis_report = predictor.generate_analysis_report(top_gainers, features_df)
        
        # Save prediction to history
        logger.info("Saving prediction to history...")
        try:
            history_manager = PredictionHistoryManager(market=market)
            history_manager.save_prediction(top_gainers, analysis_report, analysis_date)
            
            # Get prediction history
            prediction_history = history_manager.get_prediction_history()
        except Exception as e:
            logger.error(f"Error saving prediction history: {str(e)}")
            prediction_history = []
        
        # Print results
        print(f"\nTop {len(top_gainers)} Predicted Gainers for Tomorrow ({market} Market):")
        print("=====================================")
        if analysis_report:
            for stock in analysis_report:
                print(f"{stock['symbol']} - {stock['company_name']}: {stock['overview']['predicted_return_percent']}% (Confidence: {stock['overview']['confidence_score']}%)")
        else:
            print("No detailed report available.")
        
        if prediction_history:
            print(f"\nPrediction History (Last {len(prediction_history)} records):")
            print("=====================================")
            for i, record in enumerate(prediction_history):
                print(f"Record {i+1}: {record['date']} - {record['market']} Market")
        
        return top_gainers, analysis_report, prediction_history
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock Market Analysis Tool')
    parser.add_argument('--market', type=str, choices=['UK', 'US'], default='UK',
                      help='Market to analyze (UK or US)')
    parser.add_argument('--no-news', action='store_true',
                      help='Disable news sentiment analysis')
    parser.add_argument('--date', type=str,
                      help='Analysis date (YYYY-MM-DD format)')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with more verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    analysis_date = None
    if args.date:
        try:
            analysis_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Please use YYYY-MM-DD format.")
            sys.exit(1)
    
    try:
        main(
            market=args.market,
            analysis_date=analysis_date,
            include_news_sentiment=not args.no_news
        )
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)