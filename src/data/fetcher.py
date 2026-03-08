"""
Data Fetcher Module

Fetches gold prices and macroeconomic data from multiple sources:
- Yahoo Finance (gold futures, ETFs, VIX)
- FRED API (inflation, interest rates, USD index)
- World Gold Council (supply/demand data)

Author: Essabri ali rayan
Version: 1.0
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import logging
import requests
from io import StringIO
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldDataFetcher:
    """
    Primary class for fetching gold-related financial data.
    
    Attributes:
        cache_dir: Directory for caching downloaded data
        fred_api_key: API key for FRED (Federal Reserve Economic Data)
    """
    
    # Symbol mappings
    GOLD_SYMBOLS = {
        'futures': 'GC=F',           # COMEX Gold Futures
        'etf_gld': 'GLD',            # SPDR Gold Shares ETF
        'etf_iau': 'IAU',            # iShares Gold Trust
        'spot': 'XAUUSD',            # Spot gold (approximated via forex)
    }
    
    MACRO_SYMBOLS = {
        'vix': '^VIX',               # Volatility Index
        'dxy': 'DX-Y.NYB',           # US Dollar Index (approximate)
        'sp500': '^GSPC',            # S&P 500 for correlation
        'ten_year': '^TNX',          # 10-Year Treasury Yield
    }
    
    FRED_SERIES = {
        'cpi': 'CPIAUCSL',           # Consumer Price Index
        'real_rates': 'REAINTRATREARAT10Y',  # 10-Year Real Interest Rate
        'nominal_rates': 'DGS10',    # 10-Year Nominal Rate
        'dxy_fred': 'DTWEXBGS',      # Trade Weighted US Dollar Index
        'gold_monetary': 'WGC-GOLD_MONETARY',  # Central bank holdings
    }
    
    def __init__(self, cache_dir: str = './cache', fred_api_key: Optional[str] = None):
        """
        Initialize the data fetcher.
        
        Args:
            cache_dir: Path to cache directory
            fred_api_key: FRED API key for macro data (optional)
        """
        self.cache_dir = cache_dir
        self.fred_api_key = fred_api_key
        self._session = requests.Session()
        self._cache: Dict[str, pd.DataFrame] = {}
        
        logger.info("GoldDataFetcher initialized")
    
    def fetch_gold_prices(
        self,
        symbol: str = 'GC=F',
        period: str = '10y',
        interval: str = '1d',
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical gold price data from Yahoo Finance.
        
        Args:
            symbol: Ticker symbol (default: GC=F for gold futures)
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            auto_adjust: Adjust for splits and dividends
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        try:
            logger.info(f"Fetching gold prices for {symbol}, period={period}, interval={interval}")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=auto_adjust)
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Standardize column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            df.index.name = 'date'
            
            # Add metadata
            df['symbol'] = symbol
            
            logger.info(f"Fetched {len(df)} rows of price data")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching gold prices: {e}")
            raise
    
    def fetch_multiple_assets(
        self,
        symbols: List[str],
        period: str = '10y',
        price_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Fetch and combine multiple asset prices into a single DataFrame.
        
        Args:
            symbols: List of ticker symbols
            period: Data period
            price_column: Which price column to use ('open', 'high', 'low', 'close')
            
        Returns:
            DataFrame with one column per asset
        """
        data = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_gold_prices(symbol, period=period)
                col_name = symbol.replace('=', '_').replace('^', '')
                data[col_name] = df[price_column]
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                continue
        
        if not data:
            raise ValueError("No data fetched for any symbol")
        
        combined = pd.DataFrame(data)
        combined.index.name = 'date'
        
        return combined
    
    def fetch_macro_data(
        self,
        series_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch macroeconomic data from FRED API.
        
        Note: Requires FRED API key set during initialization or as environment variable.
        
        Args:
            series_ids: List of FRED series IDs (default: key indicators)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with macroeconomic indicators
        """
        if series_ids is None:
            series_ids = ['CPIAUCSL', 'DGS10', 'DTWEXBGS', 'VIXCLS']
        
        if not self.fred_api_key:
            logger.warning("No FRED API key provided. Returning empty DataFrame.")
            return pd.DataFrame()
        
        try:
            from fredapi import Fred
            fred = Fred(api_key=self.fred_api_key)
            
            data = {}
            for series_id in series_ids:
                try:
                    series = fred.get_series(
                        series_id,
                        observation_start=start_date,
                        observation_end=end_date
                    )
                    data[series_id] = series
                    logger.info(f"Fetched FRED series: {series_id}")
                except Exception as e:
                    logger.warning(f"Failed to fetch {series_id}: {e}")
            
            df = pd.DataFrame(data)
            df.index.name = 'date'
            
            return df
            
        except ImportError:
            logger.error("fredapi not installed. Install with: pip install fredapi")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching macro data: {e}")
            return pd.DataFrame()
    
    def fetch_vix(self, period: str = '10y') -> pd.DataFrame:
        """
        Fetch VIX (volatility/fear index) data.
        
        Args:
            period: Data period
            
        Returns:
            DataFrame with VIX data
        """
        return self.fetch_gold_prices('^VIX', period=period)
    
    def fetch_dxy(self, period: str = '10y') -> pd.DataFrame:
        """
        Fetch US Dollar Index (DXY) data.
        
        Note: Uses UUP (Invesco DB USD Index Bullish Fund) as proxy if DXY not available.
        
        Args:
            period: Data period
            
        Returns:
            DataFrame with DXY data
        """
        try:
            # Try DXY first
            return self.fetch_gold_prices('DX-Y.NYB', period=period)
        except:
            # Fallback to UUP ETF
            logger.info("Falling back to UUP ETF for USD strength")
            return self.fetch_gold_prices('UUP', period=period)
    
    def fetch_gold_fundamentals(self) -> Dict:
        """
        Fetch fundamental data about gold (current price, 52-week range, etc.).
        
        Returns:
            Dictionary with fundamental metrics
        """
        try:
            ticker = yf.Ticker('GC=F')
            info = ticker.info
            
            fundamentals = {
                'current_price': info.get('regularMarketPrice'),
                'previous_close': info.get('regularMarketPreviousClose'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'average_volume': info.get('averageVolume'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange'),
                'last_updated': datetime.now().isoformat()
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals: {e}")
            return {}
    
    def fetch_all_gold_data(
        self,
        period: str = '10y',
        include_macro: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch comprehensive gold dataset including prices and related assets.
        
        Args:
            period: Data period
            include_macro: Whether to include FRED macro data
            
        Returns:
            Dictionary with DataFrames for each data category
        """
        data = {}
        
        # Core gold prices
        data['gold_futures'] = self.fetch_gold_prices('GC=F', period=period)
        
        # Gold ETFs
        try:
            data['gld_etf'] = self.fetch_gold_prices('GLD', period=period)
        except:
            pass
        
        # Related markets
        try:
            data['vix'] = self.fetch_vix(period=period)
        except:
            pass
            
        try:
            data['dxy'] = self.fetch_dxy(period=period)
        except:
            pass
        
        try:
            data['sp500'] = self.fetch_gold_prices('^GSPC', period=period)
        except:
            pass
        
        try:
            data['treasury_10y'] = self.fetch_gold_prices('^TNX', period=period)
        except:
            pass
        
        # Macro data
        if include_macro and self.fred_api_key:
            data['macro'] = self.fetch_macro_data(period=period)
        
        logger.info(f"Fetched {len(data)} data categories")
        return data
    
    def get_current_gold_price(self, symbol: str = 'GC=F') -> float:
        """
        Get the current gold price (last available close).
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Current price as float
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='5d', interval='1d')
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return 0.0
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    def fetch_world_gold_council_data(self) -> Optional[pd.DataFrame]:
        """
        Attempt to fetch World Gold Council supply/demand data.
        
        Note: WGC data often requires API access or manual download.
        This method provides a placeholder for integration.
        
        Returns:
            DataFrame with supply/demand data if available
        """
        logger.info("World Gold Council integration placeholder")
        return None


def fetch_gold_data_simple(
    days: int = 2520,  # ~10 years
    symbols: List[str] = ['GC=F']
) -> pd.DataFrame:
    """
    Simple function to fetch gold price data with minimal setup.
    
    Args:
        days: Number of trading days to fetch
        symbols: List of symbols to fetch
        
    Returns:
        DataFrame with price data
    """
    fetcher = GoldDataFetcher()
    period = f"{int(days/252)}y" if days > 252 else f"{days}d"
    
    if len(symbols) == 1:
        return fetcher.fetch_gold_prices(symbols[0], period=period)
    else:
        return fetcher.fetch_multiple_assets(symbols, period=period)


if __name__ == "__main__":
    # Example usage
    fetcher = GoldDataFetcher()
    
    # Fetch gold futures
    gold_data = fetcher.fetch_gold_prices('GC=F', period='1y')
    print(f"\nGold Futures (GC=F) - Last 5 days:")
    print(gold_data.tail())
    
    # Get current price
    current = fetcher.get_current_gold_price()
    print(f"\nCurrent Gold Price: ${current:.2f}")
    
    # Fetch VIX
    vix_data = fetcher.fetch_vix(period='1y')
    print(f"\nVIX - Last 5 days:")
    print(vix_data.tail())