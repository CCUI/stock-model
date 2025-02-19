import logging
from datetime import datetime, timedelta

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_uk_trading_day(date):
    """Get the last UK trading day from a given date"""
    # Adjust for weekends
    while date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
        date = date - timedelta(days=1)
    
    # TODO: Add UK holiday calendar check if needed
    return date