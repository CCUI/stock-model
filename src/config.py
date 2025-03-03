# Feature configuration
FEATURE_COLUMNS = [
    # Technical Indicators
    'RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV',
    'Stochastic_K', 'Stochastic_D', 'Williams_R', 'CMF',
    # Price and Volume Features
    'Returns', 'Returns_5d', 'Returns_20d',
    'Volatility_5d', 'Volatility_20d',
    'Price_to_VWAP',
    # Fundamental Features
    'marketCap', 'trailingPE', 'priceToBook', 'debtToEquity',
    # Sentiment Features
    'news_sentiment', 'social_sentiment', 'sector_sentiment',
    'market_sentiment'
]

# Technical indicators to be forward filled
TECHNICAL_INDICATORS = [
    'RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV', 'VWAP',
    'SMA_20', 'SMA_50', 'EMA_20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER',
    'Stochastic_K', 'Stochastic_D', 'Williams_R', 'CMF'
]

# Features to be scaled
FEATURES_TO_SCALE = [
    'ADX', 'ATR', 'OBV', 'CMF',
    'Returns', 'Returns_5d', 'Returns_20d',
    'Volatility_5d', 'Volatility_20d',
    'social_sentiment', 'sector_sentiment', 'market_sentiment'
]

# Additional technical indicators (not used as features but needed for analysis)
ANALYSIS_INDICATORS = [
    'SMA_20', 'SMA_50', 'EMA_20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'
]