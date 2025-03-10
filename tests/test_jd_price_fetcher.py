import unittest
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collector import UKStockDataCollector
from src.utils import setup_logging

logger = setup_logging()

class TestJDPriceFetcher(unittest.TestCase):
    def setUp(self):
        # Initialize the UK stock data collector
        self.collector = UKStockDataCollector(include_news_sentiment=False)
        
        # JD Sports Fashion PLC ticker symbol with UK suffix
        self.jd_symbol = 'JD.L'
        
    def test_fetch_jd_current_price(self):
        """Test fetching the current price for JD Sports Fashion PLC"""
        try:
            # Use yfinance directly to get the current price
            ticker = yf.Ticker(self.jd_symbol)
            
            # Get the latest market data
            market_data = ticker.history(period='1d')
            
            # Check if we got any data
            self.assertFalse(market_data.empty, "No market data returned for JD.L")
            
            # Get the latest closing price
            latest_close = market_data['Close'].iloc[-1]
            
            # Print the results
            print(f"\nJD Sports Fashion PLC (JD.L) Current Price Information:")
            print(f"Price: £{latest_close:.2f}")
            
            # Get additional information
            info = ticker.info
            
            # Print company name and additional information if available
            if info:
                company_name = info.get('longName', 'JD Sports Fashion PLC')
                day_high = info.get('dayHigh', 'N/A')
                day_low = info.get('dayLow', 'N/A')
                volume = info.get('volume', 'N/A')
                market_cap = info.get('marketCap', 'N/A')
                
                # Format market cap to millions
                if isinstance(market_cap, (int, float)) and market_cap > 0:
                    market_cap = f"£{market_cap / 1_000_000:.2f}M"
                
                print(f"Company: {company_name}")
                print(f"Day Range: £{day_low} - £{day_high}")
                print(f"Volume: {volume}")
                print(f"Market Cap: {market_cap}")
            
            # Verify the price is a positive number
            self.assertGreater(latest_close, 0, "Price should be greater than zero")
            
            return latest_close
            
        except Exception as e:
            self.fail(f"Error fetching JD.L price: {str(e)}")
    
    def test_company_name_mapping(self):
        """Test that the company name mapping works for JD symbol"""
        # Get the company name from the collector's mapping
        company_name = self.collector._get_company_name(self.jd_symbol)
        
        # Verify we got a non-empty string
        self.assertIsInstance(company_name, str)
        self.assertTrue(len(company_name) > 0)
        
        print(f"\nCompany name for {self.jd_symbol}: {company_name}")

if __name__ == '__main__':
    # Run the tests
    unittest.main()