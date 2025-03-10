import logging
from datetime import datetime, timedelta
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_uk_trading_day(date):
    """Get the current UK trading day or last trading day if current day is weekend or current day's data isn't available yet"""
    current_time = datetime.now()
    
    # If current day is a weekday, use it only if it's after market close (4:30 PM UK time)
    if date.weekday() <= 4:  # 0-4 are Monday to Friday
        # If it's today and before market close, use previous trading day
        if date.date() == current_time.date() and current_time.hour < 16:  # Before 4 PM
            # Go back to the previous day
            date = date - timedelta(days=1)
            # If it's a weekend, adjust further
            while date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                date = date - timedelta(days=1)
        return date
    
    # If it's a weekend, adjust to the last weekday
    while date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
        date = date - timedelta(days=1)
    
    # TODO: Add UK holiday calendar check if needed
    return date

def get_us_trading_day(date):
    """Get the current US trading day or last trading day if current day is weekend or current day's data isn't available yet"""
    current_time = datetime.now()
    
    # If current day is a weekday, use it only if it's after market close (4:00 PM EST)
    if date.weekday() <= 4:  # 0-4 are Monday to Friday
        # If it's today and before market close, use previous trading day
        if date.date() == current_time.date() and current_time.hour < 16:  # Before 4 PM
            # Go back to the previous day
            date = date - timedelta(days=1)
            # If it's a weekend, adjust further
            while date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                date = date - timedelta(days=1)
        return date
    
    # If it's a weekend, adjust to the last weekday
    while date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
        date = date - timedelta(days=1)
    
    # TODO: Add US holiday calendar check if needed
    return date

def get_trading_day(market, date):
    """Get the current trading day for the specified market or last trading day if current day is weekend"""
    if market.upper() == 'UK':
        return get_uk_trading_day(date)
    elif market.upper() == 'US':
        return get_us_trading_day(date)
    else:
        # Default to UK trading day if market not recognized
        return get_uk_trading_day(date)

def cleanup_old_data(market_dir: Path, max_age_days: int = 30):
    """Clean up old data files"""
    current_time = datetime.now()
    
    # Clean up old chunk files
    for chunk_file in market_dir.glob('historical_chunk_*.json'):
        if (current_time - datetime.fromtimestamp(chunk_file.stat().st_mtime)).days > max_age_days:
            chunk_file.unlink()
    
    # Clean up old sentiment data
    sentiment_file = market_dir / 'sentiment_cache.json'
    if sentiment_file.exists() and (current_time - datetime.fromtimestamp(sentiment_file.stat().st_mtime)).days > max_age_days:
        sentiment_file.unlink()