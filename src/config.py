# Feature configuration
FEATURE_COLUMNS = [
    # Technical Indicators
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MFI', 'ADX', 'ATR', 'OBV',
    'Stochastic_K', 'Stochastic_D', 'Williams_R', 'CMF',
    
    # Price and Volume Features
    'Returns', 'Returns_5d', 'Returns_10d', 'Returns_20d', 'Returns_60d',
    'Log_Returns', 'Log_Returns_5d', 'Log_Returns_20d',
    'Volatility_5d', 'Volatility_10d', 'Volatility_20d', 'Volatility_60d',
    'Price_to_VWAP', 'Price_to_SMA20', 'Price_to_SMA50', 'Price_to_SMA200',
    
    # Moving Average Crossovers
    'SMA_5_10_Crossover', 'SMA_10_20_Crossover', 'SMA_20_50_Crossover', 'SMA_50_200_Crossover',
    
    # Volume Features
    'Volume_Change', 'Relative_Volume',
    
    # Advanced Features
    'ROC_5', 'ROC_10', 'ROC_20',
    'Price_Acceleration', 'Price_Acceleration_5d',
    'Volatility_Ratio', 'ADX_Trend_Strength',
    'RSI_Divergence', 'BB_Squeeze', 'Volume_Price_Trend',
    'Gap_Up', 'Gap_Down', 'Gap_Size',
    
    # Candlestick Patterns
    'bullish_engulfing', 'bearish_engulfing', 'doji', 'hammer', 'shooting_star',
    
    # Market Regime
    'regime', 'volatility_regime', 'trend_strength', 'market_phase',
    'regime_bull', 'regime_bear', 'regime_volatile',
    
    # Ichimoku Cloud
    'Above_Cloud', 'Below_Cloud', 'In_Cloud',
    
    # Lagged Features
    'RSI_lag_1', 'MACD_lag_1', 'Returns_lag_1',
    'RSI_lag_2', 'MACD_lag_2', 'Returns_lag_2',
    'RSI_lag_3', 'MACD_lag_3', 'Returns_lag_3',
    'RSI_lag_5', 'MACD_lag_5', 'Returns_lag_5',
    
    # Seasonal Features (will be dynamically created)
    'Day_0', 'Day_1', 'Day_2', 'Day_3', 'Day_4',
    'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
    'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12',
    
    # Fundamental Features
    'marketCap', 'trailingPE', 'priceToBook', 'debtToEquity',
    
    # Sentiment Features
    'news_sentiment', 'social_sentiment', 'sector_sentiment', 'market_sentiment'
]

# Technical indicators to be forward filled
TECHNICAL_INDICATORS = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MFI', 'ADX', 'ATR', 'OBV', 'VWAP',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 
    'EMA_5', 'EMA_10', 'EMA_20',
    'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_WIDTH', 'BB_PCT',
    'Stochastic_K', 'Stochastic_D', 'Williams_R', 'CMF',
    'Ichimoku_Conversion', 'Ichimoku_Base', 'Ichimoku_SpanA', 'Ichimoku_SpanB',
    'SAR'
]

# Features to be scaled
FEATURES_TO_SCALE = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'MFI', 
    'ADX', 'ATR', 'OBV', 'CMF',
    'Returns', 'Returns_5d', 'Returns_10d', 'Returns_20d', 'Returns_60d',
    'Log_Returns', 'Log_Returns_5d', 'Log_Returns_20d',
    'Volatility_5d', 'Volatility_10d', 'Volatility_20d', 'Volatility_60d',
    'ROC_5', 'ROC_10', 'ROC_20',
    'Price_Acceleration', 'Price_Acceleration_5d', 
    'Volatility_Ratio', 'Volume_Price_Trend',
    'Gap_Size', 'trend_strength',
    'news_sentiment', 'social_sentiment', 'sector_sentiment', 'market_sentiment'
]

# Additional technical indicators (not used as features but needed for analysis)
ANALYSIS_INDICATORS = [
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_5', 'EMA_10', 'EMA_20',
    'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'BB_WIDTH',
    'Ichimoku_SpanA', 'Ichimoku_SpanB'
]

# Seasonal features (will be added dynamically)
SEASONAL_FEATURES = [
    'Day_of_Week', 'Month', 'Quarter',
    'Day_0', 'Day_1', 'Day_2', 'Day_3', 'Day_4',
    'Month_1', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6',
    'Month_7', 'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12'
]

# Model configuration
MODEL_CONFIG = {
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_child_weight': 1,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42
    }
}

# Hyperparameter tuning configuration
HYPERPARAMETER_TUNING = {
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
}

# Prediction configuration
PREDICTION_CONFIG = {
    'confidence_threshold': 0.6,  # Minimum confidence score to consider a prediction valid
    'top_n_predictions': 10,      # Number of top predictions to return
    'min_features_required': 0.8  # Minimum percentage of features required for prediction
}

# File paths
FILE_PATHS = {
    'model_dir': 'models',
    'data_dir': 'data',
    'log_dir': 'logs',
    'prediction_history': 'data/prediction_history.csv',
    'analysis_reports': 'reports'
}

# API rate limits (calls per minute)
API_RATE_LIMITS = {
    'alpha_vantage': 5,
    'yahoo_finance': 100,
    'news_api': 10,
    'twitter_api': 15
}

# Error handling
ERROR_CONFIG = {
    'max_retries': 3,
    'retry_delay': 5,  # seconds
    'log_errors': True
}