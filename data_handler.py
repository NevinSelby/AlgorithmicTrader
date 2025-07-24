import yfinance as yf
import pandas as pd
import backtrader as bt
from datetime import datetime


class DataHandler:
    """
    Responsible for fetching and providing financial data.
    """
    
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize with stock ticker and date range.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (str or datetime): Start date for data fetch
            end_date (str or datetime): End date for data fetch
        """
        self.ticker = ticker.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    
    def fetch_data(self):
        """
        Uses yfinance to download historical OHLCV data.
        
        Returns:
            pandas.DataFrame: Historical stock data with OHLCV columns
            
        Raises:
            Exception: If data fetch fails or ticker is invalid
        """
        try:
            # Download data using yfinance
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=self.start_date, end=self.end_date)
            
            if self.data.empty:
                raise Exception(f"No data found for ticker {self.ticker}")
            
            # Ensure column names are consistent
            self.data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Remove unnecessary columns for backtrader
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            print(f"Successfully fetched {len(self.data)} days of data for {self.ticker}")
            return self.data
            
        except Exception as e:
            raise Exception(f"Error fetching data for {self.ticker}: {str(e)}")
    
    def get_feed(self):
        """
        Converts pandas DataFrame to backtrader PandasData feed.
        
        Returns:
            backtrader.feeds.PandasData: Data feed for backtrader
            
        Raises:
            Exception: If data hasn't been fetched yet
        """
        if self.data is None:
            raise Exception("Data not fetched yet. Call fetch_data() first.")
        
        try:
            # Create backtrader data feed
            data_feed = bt.feeds.PandasData(
                dataname=self.data,
                datetime=None,  # Use index as datetime
                open=0,
                high=1, 
                low=2,
                close=3,
                volume=4,
                openinterest=None
            )
            
            return data_feed
            
        except Exception as e:
            raise Exception(f"Error creating data feed: {str(e)}")
    
    def get_data_info(self):
        """
        Returns basic information about the fetched data.
        
        Returns:
            dict: Information about the data
        """
        if self.data is None:
            return {"status": "No data fetched"}
        
        return {
            "ticker": self.ticker,
            "start_date": self.data.index.min().strftime('%Y-%m-%d'),
            "end_date": self.data.index.max().strftime('%Y-%m-%d'),
            "total_days": len(self.data),
            "latest_close": self.data['Close'].iloc[-1]
        } 