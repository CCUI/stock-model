import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class UKStockDataCollector:
    def __init__(self):
        self.ftse_symbols = self._get_ftse_symbols()
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
    
    def _get_ftse_symbols(self):
        """Get list of FTSE stocks"""
        # Only FTSE 100 components for testing
        ftse_url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
        
        symbols = []
        response = requests.get(ftse_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table', {'class': 'wikitable'})
        
        for table in tables:
            rows = table.find_all('tr')[1:]
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    # Get the symbol text and clean it
                    symbol_text = cols[1].text.strip()
                    
                    # Skip if the symbol contains numeric values (likely a price)
                    if any(char.isdigit() for char in symbol_text):
                        continue
                        
                    # Add .L suffix for London Stock Exchange
                    symbol = symbol_text + '.L'
                    symbols.append(symbol)
        
        # Remove duplicates
        unique_symbols = list(set(symbols))
        
        # Ensure we have symbols
        if not unique_symbols:
            raise ValueError("No valid FTSE symbols could be extracted from the webpage")
            
        return unique_symbols
    
    def collect_historical_data(self, end_date, lookback_days=365, max_retries=3, delay=1):
        """Collect historical price and fundamental data for all FTSE stocks"""
        start_date = end_date - timedelta(days=lookback_days)
        
        all_data = {}
        for symbol in self.ftse_symbols:
            for attempt in range(max_retries):
                try:
                    # Get price data
                    stock = yf.Ticker(symbol)
                    hist_data = stock.history(start=start_date, end=end_date)
                    
                    if not hist_data.empty:
                        # Add technical data
                        hist_data['Symbol'] = symbol
                        
                        # Get fundamental data
                        info = stock.info
                        for key in ['marketCap', 'trailingPE', 'priceToBook', 'debtToEquity']:
                            hist_data[key] = info.get(key, 0)  # Use 0 instead of None for missing values
                        
                        # Get news sentiment
                        news = self._get_news_sentiment(symbol, end_date)
                        hist_data['news_sentiment'] = news['sentiment_score'].mean() if not news.empty else 0
                        
                        all_data[symbol] = hist_data
                        break  # Successfully got data, break retry loop
                    
                except Exception as e:
                    if 'Too Many Requests' in str(e) and attempt < max_retries - 1:
                        import time
                        time.sleep(delay * (attempt + 1))  # Exponential backoff
                        continue
                    print(f"Error collecting data for {symbol}: {str(e)}")
                    break  # Break on non-rate-limit errors or final attempt
        
        # Ensure we have at least some data before concatenating
        if not all_data:
            raise ValueError("No data could be collected for any symbols")
            
        return pd.concat(all_data.values(), axis=0)
        
        return pd.concat(all_data.values(), axis=0)
    
    def _get_news_sentiment(self, symbol, date):
        """Collect and analyze news sentiment for a given stock"""
        try:
            # Get news articles from Alpha Vantage
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
            response = requests.get(url)
            news_data = response.json()
            
            if 'feed' not in news_data:
                return pd.DataFrame()
            
            sentiments = []
            for article in news_data['feed']:
                # Use FinBERT for sentiment analysis
                inputs = self.tokenizer(article['title'], return_tensors="pt", padding=True, truncation=True)
                outputs = self.sentiment_model(**inputs)
                sentiment_score = torch.softmax(outputs.logits, dim=1)
                
                sentiments.append({
                    'date': article['time_published'],
                    'sentiment_score': sentiment_score[0][1].item()  # Positive sentiment score
                })
            
            return pd.DataFrame(sentiments)
            
        except Exception as e:
            print(f"Error getting news sentiment for {symbol}: {str(e)}")
            return pd.DataFrame()