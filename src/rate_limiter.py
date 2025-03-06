from datetime import datetime, timedelta
from requests_ratelimiter import LimiterSession
from pyrate_limiter import Duration, RequestRate, Limiter
import time
import logging

logger = logging.getLogger(__name__)

class APIRateLimiter:
    def __init__(self):
        # YFinance rate limits (2000 requests per hour per IP)
        self.yf_limiter = Limiter(
            RequestRate(2000, Duration.HOUR),  # 2000 requests per hour
            RequestRate(30, Duration.MINUTE)   # 30 requests per minute for safety
        )
        
        # NewsAPI rate limits (100 requests per day for free tier)
        self.news_limiter = Limiter(
            RequestRate(100, Duration.DAY),    # 100 requests per day
            RequestRate(5, Duration.MINUTE)    # 5 requests per minute for safety
        )
        
        # Wikipedia rate limits (conservative limits to be polite)
        self.wiki_limiter = Limiter(
            RequestRate(200, Duration.HOUR),   # 200 requests per hour
            RequestRate(10, Duration.MINUTE)   # 10 requests per minute
        )
        
        # Create rate-limited sessions
        self.yf_session = LimiterSession(per_second=1)  # Max 1 request per second
        self.news_session = LimiterSession(per_second=0.5)  # Max 1 request per 2 seconds
        self.wiki_session = LimiterSession(per_second=0.2)  # Max 1 request per 5 seconds
        
        # Initialize last request timestamps
        self.last_yf_request = datetime.min
        self.last_news_request = datetime.min
        self.last_wiki_request = datetime.min
        
        # Backoff settings
        self.max_retries = 3
        self.base_backoff = 2  # Base backoff in seconds
    
    def _apply_backoff(self, attempt):
        """Apply exponential backoff"""
        backoff = self.base_backoff * (2 ** attempt)
        time.sleep(backoff)
    
    def execute_yf_request(self, func, *args, **kwargs):
        """Execute YFinance API request with rate limiting"""
        for attempt in range(self.max_retries):
            try:
                with self.yf_limiter.ratelimit('yfinance', delay=True):
                    current_time = datetime.now()
                    # Ensure minimum delay between requests
                    time_since_last = (current_time - self.last_yf_request).total_seconds()
                    if time_since_last < 1:
                        time.sleep(1 - time_since_last)
                    
                    result = func(*args, **kwargs)
                    self.last_yf_request = datetime.now()
                    return result
            except Exception as e:
                if "Rate limit" in str(e) and attempt < self.max_retries - 1:
                    logger.warning(f"YFinance rate limit hit, attempt {attempt + 1}/{self.max_retries}")
                    self._apply_backoff(attempt)
                    continue
                raise
    
    def execute_news_request(self, func, *args, **kwargs):
        """Execute NewsAPI request with rate limiting"""
        for attempt in range(self.max_retries):
            try:
                with self.news_limiter.ratelimit('newsapi', delay=True):
                    current_time = datetime.now()
                    time_since_last = (current_time - self.last_news_request).total_seconds()
                    if time_since_last < 2:
                        time.sleep(2 - time_since_last)
                    
                    result = func(*args, **kwargs)
                    self.last_news_request = datetime.now()
                    return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"NewsAPI request failed, attempt {attempt + 1}/{self.max_retries}")
                    self._apply_backoff(attempt)
                    continue
                raise
    
    def execute_wiki_request(self, func, *args, **kwargs):
        """Execute Wikipedia request with rate limiting"""
        for attempt in range(self.max_retries):
            try:
                with self.wiki_limiter.ratelimit('wikipedia', delay=True):
                    current_time = datetime.now()
                    time_since_last = (current_time - self.last_wiki_request).total_seconds()
                    if time_since_last < 5:
                        time.sleep(5 - time_since_last)
                    
                    result = func(*args, **kwargs)
                    self.last_wiki_request = datetime.now()
                    return result
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Wikipedia request failed, attempt {attempt + 1}/{self.max_retries}")
                    self._apply_backoff(attempt)
                    continue
                raise