# Feature configuration
FEATURE_COLUMNS = [
    'RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV',
    'Returns', 'Returns_5d', 'Returns_20d',
    'Volatility_5d', 'Volatility_20d',
    'Price_to_VWAP', 'marketCap', 'trailingPE',
    'priceToBook', 'debtToEquity', 'news_sentiment'
]

# Technical indicators to be forward filled
TECHNICAL_INDICATORS = [
    'RSI', 'MACD', 'MFI', 'ADX', 'ATR', 'OBV', 'VWAP',
    'SMA_20', 'SMA_50', 'EMA_20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'
]

# Features to be scaled
FEATURES_TO_SCALE = [
    'ADX', 'ATR', 'OBV',
    'Returns', 'Returns_5d', 'Returns_20d',
    'Volatility_5d', 'Volatility_20d'
]

# Additional technical indicators (not used as features but needed for analysis)
ANALYSIS_INDICATORS = [
    'SMA_20', 'SMA_50', 'EMA_20', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'
] 