import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_collector import UKStockDataCollector, USStockDataCollector
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import StockPredictor
from src.utils import setup_logging, get_uk_trading_day
from src.load_env import load_environment_variables

def main(market='UK', analysis_date=None, include_news_sentiment=True):
    # Setup logging
    logger = setup_logging()
    
    # Load environment variables
    load_environment_variables()
    
    # If no date provided, use the last trading day
    if analysis_date is None:
        analysis_date = get_uk_trading_day(datetime.now() - timedelta(days=1))
    
    logger.info(f"Starting analysis for {market} market on date: {analysis_date}")
    
    try:
        # Initialize components
        if market.upper() == 'UK':
            data_collector = UKStockDataCollector(include_news_sentiment=include_news_sentiment)
        elif market.upper() == 'US':
            data_collector = USStockDataCollector(include_news_sentiment=include_news_sentiment)
        else:
            raise ValueError(f"Unsupported market: {market}. Available markets: UK, US")
            
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer()
        predictor = StockPredictor()
        
        # Collect historical data for selected market
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
        predictions = predictor.predict_top_gainers(model, features_df, top_n=5)
        
        # Generate detailed analysis report
        logger.info("Generating analysis report...")
        report = predictor.generate_analysis_report(predictions, features_df)
        
        print(f"\nTop 5 Predicted Gainers for Tomorrow ({market} Market):")
        print("=====================================\n")
        print(report)
        
        return predictions, report
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Market Analysis Tool')
    parser.add_argument('--market', type=str, choices=['UK', 'US'], default='UK',
                        help='Market to analyze (UK or US)')
    parser.add_argument('--no-news', action='store_false', dest='include_news_sentiment',
                        help='Disable news sentiment analysis')
    parser.add_argument('--date', type=str, help='Analysis date (YYYY-MM-DD format)')
    
    args = parser.parse_args()
    
    analysis_date = None
    if args.date:
        analysis_date = datetime.strptime(args.date, '%Y-%m-%d')
    
    main(market=args.market, analysis_date=analysis_date, include_news_sentiment=args.include_news_sentiment)