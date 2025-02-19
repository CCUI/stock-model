import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_collector import UKStockDataCollector
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.predictor import StockPredictor
from src.utils import setup_logging, get_uk_trading_day

def main(analysis_date=None):
    # Setup logging
    logger = setup_logging()
    
    # If no date provided, use the last trading day
    if analysis_date is None:
        analysis_date = get_uk_trading_day(datetime.now() - timedelta(days=1))
    
    logger.info(f"Starting analysis for date: {analysis_date}")
    
    try:
        # Initialize components
        data_collector = UKStockDataCollector()
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer()
        predictor = StockPredictor()
        
        # Collect historical data for all UK stocks
        logger.info("Collecting historical stock data...")
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
        
        print("\nTop 5 Predicted Gainers for Tomorrow:")
        print("=====================================\n")
        print(report)
        
        return predictions, report
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()