import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import torch
import os
from src.data_collector import UKStockDataCollector

class TestUKStockDataCollector(unittest.TestCase):
    def setUp(self):
        # Mock environment variable for API key
        self.mock_api_key = 'test_api_key'
        with patch.dict(os.environ, {'ALPHA_VANTAGE_API_KEY': self.mock_api_key}):
            self.collector = UKStockDataCollector()
        self.test_date = datetime(2024, 1, 1)
        self.test_symbol = 'TSCO.L'  # Tesco PLC

    def test_news_sentiment_api_call(self):
        """Test that the Alpha Vantage API is called with correct parameters"""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'feed': []}
            mock_get.return_value = mock_response

            self.collector._get_news_sentiment(self.test_symbol, self.test_date)

            # Verify API call
            mock_get.assert_called_once()
            call_args = mock_get.call_args[0][0]
            self.assertIn('TSCO.L', call_args)
            self.assertIn('NEWS_SENTIMENT', call_args)
            self.assertIn(self.mock_api_key, call_args)

    def test_news_sentiment_calculation(self):
        """Test sentiment calculation with mock news data"""
        mock_news_data = {
            'feed': [
                {
                    'title': 'Positive news about the company',
                    'time_published': self.test_date.strftime('%Y%m%dT%H%M%S')
                }
            ]
        }

        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_news_data
            mock_get.return_value = mock_response

            # Mock FinBERT model outputs and torch.softmax
            mock_outputs = MagicMock()
            mock_outputs.logits = torch.tensor([[0.1, 0.3, 0.6]])  # Raw logits
            mock_probabilities = torch.tensor([[0.2, 0.3, 0.7]])  # Normalized probabilities
            
            with patch.object(self.collector.sentiment_model, '__call__', return_value=mock_outputs):
                with patch('torch.softmax', return_value=mock_probabilities):
                    result = self.collector._get_news_sentiment(self.test_symbol, self.test_date)

                    # Verify result format
                    self.assertIsInstance(result, pd.DataFrame)
                    self.assertIn('sentiment_score', result.columns)
                    self.assertEqual(len(result), 1)

                    # Calculate expected sentiment (positive - negative)
                    sentiment_score = 0.7 - 0.2  # pos - neg = 0.5
                    self.assertAlmostEqual(float(result['sentiment_score'].iloc[0]), sentiment_score, places=2)

    def test_empty_news_feed(self):
        """Test handling of empty news feed"""
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.return_value = {'feed': []}
            mock_get.return_value = mock_response

            result = self.collector._get_news_sentiment(self.test_symbol, self.test_date)

            # Verify default sentiment score
            self.assertEqual(result['sentiment_score'].iloc[0], 0)

    def test_api_error_handling(self):
        """Test handling of API errors"""
        with patch('requests.get') as mock_get:
            # Mock API error response with specific error message
            mock_get.side_effect = Exception('API Error: Invalid API key or rate limit exceeded')

            # Call the method and verify error handling
            result = self.collector._get_news_sentiment(self.test_symbol, self.test_date)

            # Verify API was called with correct parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args[0][0]
            self.assertIn(self.test_symbol, call_args)
            self.assertIn('NEWS_SENTIMENT', call_args)
            self.assertIn(self.mock_api_key, call_args)

            # Verify error handling returns default sentiment
            self.assertIsInstance(result, pd.DataFrame)
            self.assertIn('sentiment_score', result.columns)
            self.assertEqual(len(result), 1)
            self.assertEqual(result['sentiment_score'].iloc[0], 0)

if __name__ == '__main__':
    unittest.main()