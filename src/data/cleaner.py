"""
Data Cleaner Module

Handles data quality issues including:
- Missing values (interpolation, forward-fill)
- Outlier detection and treatment
- Duplicate removal
- Data validation
- Time series alignment

Author: Essabri ali rayan
Version: 1.3
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union, Tuple
from scipy import stats
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Professional data cleaning pipeline for financial time series.
    
    Handles common issues in gold price data including missing values,
    outliers, and non-trading day gaps.
    """
    
    def __init__(
        self,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 3.0,
        max_gap_days: int = 5
    ):
        """
        Initialize cleaner with configuration parameters.
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'mad')
            outlier_threshold: Threshold for outlier detection
            max_gap_days: Maximum allowed gap in trading days for interpolation
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.max_gap_days = max_gap_days
        
        logger.info(f"DataCleaner initialized: method={outlier_method}, threshold={outlier_threshold}")
    
    def clean_price_data(
        self,
        df: pd.DataFrame,
        price_columns: Optional[List[str]] = None,
        handle_missing: str = 'interpolate',
        handle_outliers: str = 'winsorize',
        add_flags: bool = True
    ) -> pd.DataFrame:
        """
        Main cleaning pipeline for price data.
        
        Args:
            df: Input DataFrame with price data
            price_columns: Columns to clean (default: all numeric columns)
            handle_missing: Method for missing values ('interpolate', 'ffill', 'bfill', 'drop')
            handle_outliers: Method for outliers ('winsorize', 'remove', 'flag', 'none')
            add_flags: Whether to add quality indicator columns
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Ensure datetime index
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean.index = pd.to_datetime(df_clean.index)
        
        # Sort by date
        df_clean = df_clean.sort_index()
        
        # Identify price columns
        if price_columns is None:
            price_columns = self._identify_price_columns(df_clean)
        
        logger.info(f"Cleaning {len(price_columns)} price columns: {price_columns}")
        
        # Add quality flags if requested
        if add_flags:
            df_clean['_data_quality_score'] = 1.0
            df_clean['_is_interpolated'] = False
            df_clean['_is_outlier'] = False
        
        # Handle missing values
        df_clean = self._handle_missing_values(
            df_clean, price_columns, method=handle_missing, add_flags=add_flags
        )
        
        # Handle outliers
        if handle_outliers != 'none':
            df_clean = self._handle_outliers(
                df_clean, price_columns, method=handle_outliers, add_flags=add_flags
            )
        
        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Validate data
        df_clean = self._validate_data(df_clean, price_columns)
        
        logger.info(f"Cleaning complete. Final shape: {df_clean.shape}")
        return df_clean
    
    def _identify_price_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify columns likely containing price data."""
        price_keywords = ['open', 'high', 'low', 'close', 'price', 'adj', 'volume']
        price_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in price_keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    price_cols.append(col)
        
        # If no price columns found, use all numeric columns
        if not price_cols:
            price_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return [c for c in price_cols if not c.startswith('_')]
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'interpolate',
        add_flags: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing values in time series data.
        
        Strategy:
        1. Check for gaps larger than max_gap_days
        2. Interpolate small gaps
        3. Forward fill remaining gaps
        4. Flag all interpolated values
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df_result.columns:
                continue
                
            missing_before = df_result[col].isna().sum()
            
            if missing_before == 0:
                continue
            
            # Calculate gaps
            mask = df_result[col].isna()
            
            if method == 'interpolate':
                # Time-based interpolation for time series
                df_result[col] = df_result[col].interpolate(method='time', limit=self.max_gap_days)
                
                # Flag interpolated values
                if add_flags and '_is_interpolated' in df_result.columns:
                    interpolated_mask = df_result[col].notna() & mask
                    df_result.loc[interpolated_mask, '_is_interpolated'] = True
                    
            elif method == 'ffill':
                df_result[col] = df_result[col].fillna(method='ffill', limit=self.max_gap_days)
            elif method == 'bfill':
                df_result[col] = df_result[col].fillna(method='bfill', limit=self.max_gap_days)
            elif method == 'drop':
                df_result = df_result.dropna(subset=[col])
            
            # Forward fill any remaining NaNs at the beginning
            df_result[col] = df_result[col].fillna(method='ffill')
            
            # Backward fill any remaining NaNs at the end
            df_result[col] = df_result[col].fillna(method='bfill')
            
            missing_after = df_result[col].isna().sum()
            logger.info(f"Column {col}: Filled {missing_before - missing_after} missing values")
        
        return df_result
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'winsorize',
        add_flags: bool = True
    ) -> pd.DataFrame:
        """
        Detect and handle outliers in price data.
        
        Methods:
        - winsorize: Cap at percentiles (default 1% and 99%)
        - remove: Drop outlier rows
        - flag: Only mark as outliers
        """
        df_result = df.copy()
        
        for col in columns:
            if col not in df_result.columns:
                continue
            
            # Calculate returns for outlier detection
            returns = df_result[col].pct_change().dropna()
            
            if len(returns) < 10:
                continue
            
            # Detect outliers based on method
            if self.outlier_method == 'iqr':
                Q1 = returns.quantile(0.25)
                Q3 = returns.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                outlier_mask = (returns < lower_bound) | (returns > upper_bound)
                
            elif self.outlier_method == 'zscore':
                z_scores = np.abs(stats.zscore(returns))
                outlier_mask = z_scores > self.outlier_threshold
                
            elif self.outlier_method == 'mad':
                median = returns.median()
                mad = np.median(np.abs(returns - median))
                modified_z = 0.6745 * (returns - median) / mad
                outlier_mask = np.abs(modified_z) > self.outlier_threshold
            
            outlier_indices = returns[outlier_mask].index
            
            if len(outlier_indices) == 0:
                continue
            
            logger.info(f"Column {col}: Found {len(outlier_indices)} outliers ({self.outlier_method})")
            
            # Handle outliers
            if method == 'winsorize':
                # Cap extreme returns at 99th/1st percentile
                lower_cap = returns.quantile(0.01)
                upper_cap = returns.quantile(0.99)
                
                for idx in outlier_indices:
                    if idx in df_result.index:
                        prev_idx = df_result.index[df_result.index.get_loc(idx) - 1] if df_result.index.get_loc(idx) > 0 else idx
                        prev_price = df_result.loc[prev_idx, col]
                        actual_return = returns.loc[idx]
                        
                        # Cap the return
                        capped_return = np.clip(actual_return, lower_cap, upper_cap)
                        df_result.loc[idx, col] = prev_price * (1 + capped_return)
                        
            elif method == 'remove':
                df_result = df_result.drop(outlier_indices)
                
            elif method == 'flag':
                pass  # Just flag, don't modify
            
            # Add outlier flags
            if add_flags and '_is_outlier' in df_result.columns:
                df_result.loc[outlier_indices, '_is_outlier'] = True
        
        return df_result
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate index entries, keeping last occurrence."""
        if df.index.duplicated().any():
            n_dups = df.index.duplicated().sum()
            df = df[~df.index.duplicated(keep='last')]
            logger.info(f"Removed {n_dups} duplicate index entries")
        return df
    
    def _validate_data(self, df: pd.DataFrame, price_columns: List[str]) -> pd.DataFrame:
        """
        Validate data integrity and calculate quality scores.
        """
        # Check for negative prices
        for col in price_columns:
            if col in df.columns:
                negative_mask = df[col] < 0
                if negative_mask.any():
                    logger.warning(f"Negative values found in {col}: {negative_mask.sum()} rows")
                    df.loc[negative_mask, col] = np.nan
                    df[col] = df[col].fillna(method='ffill')
        
        # Check for zero prices (suspicious)
        for col in price_columns:
            if col in df.columns:
                zero_mask = df[col] == 0
                if zero_mask.any():
                    logger.warning(f"Zero values found in {col}: {zero_mask.sum()} rows")
                    df.loc[zero_mask, col] = np.nan
                    df[col] = df[col].fillna(method='ffill')
        
        # Calculate data quality score
        if '_data_quality_score' in df.columns:
            for col in price_columns:
                if col in df.columns:
                    # Score based on missing data and outliers
                    missing_pct = df[col].isna().mean()
                    outlier_penalty = 0
                    if '_is_outlier' in df.columns:
                        outlier_penalty = df['_is_outlier'].mean() * 0.5
                    
                    quality = 1.0 - missing_pct - outlier_penalty
                    df['_data_quality_score'] = df['_data_quality_score'] * quality
        
        return df
    
    def align_time_series(
        self,
        data_dict: dict,
        freq: str = 'B',  # Business days
        method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Align multiple time series to common dates.
        
        Args:
            data_dict: Dictionary of DataFrames to align
            freq: Target frequency ('B'=business days, 'D'=daily)
            method: Fill method for missing dates
            
        Returns:
            Combined DataFrame with aligned dates
        """
        # Find common date range
        start_dates = [df.index.min() for df in data_dict.values() if not df.empty]
        end_dates = [df.index.max() for df in data_dict.values() if not df.empty]
        
        if not start_dates or not end_dates:
            return pd.DataFrame()
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Create common index
        common_index = pd.date_range(start=common_start, end=common_end, freq=freq)
        
        aligned_data = {}
        
        for name, df in data_dict.items():
            if df.empty:
                continue
            
            # Reindex to common dates
            df_aligned = df.reindex(common_index)
            
            # Forward fill missing values
            if method == 'ffill':
                df_aligned = df_aligned.fillna(method='ffill')
            elif method == 'interpolate':
                df_aligned = df_aligned.interpolate(method='time')
            
            # Rename columns to include source
            for col in df_aligned.columns:
                if not col.startswith('_'):
                    aligned_data[f"{name}_{col}"] = df_aligned[col]
        
        result = pd.DataFrame(aligned_data, index=common_index)
        result.index.name = 'date'
        
        return result
    
    def resample_ohlc(
        self,
        df: pd.DataFrame,
        freq: str = 'W',  # Weekly
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Resample price data to lower frequency (weekly/monthly).
        
        Args:
            df: DataFrame with OHLC data
            freq: Target frequency ('W'=weekly, 'M'=monthly)
            price_col: Primary price column name
            
        Returns:
            Resampled DataFrame
        """
        ohlc_dict = {}
        
        # Map standard column names
        col_mapping = {
            'open': [c for c in df.columns if 'open' in c.lower()][:1],
            'high': [c for c in df.columns if 'high' in c.lower()][:1],
            'low': [c for c in df.columns if 'low' in c.lower()][:1],
            'close': [c for c in df.columns if 'close' in c.lower() or c == price_col][:1],
        }
        
        if col_mapping['open']:
            ohlc_dict['open'] = df[col_mapping['open'][0]].resample(freq).first()
        if col_mapping['high']:
            ohlc_dict['high'] = df[col_mapping['high'][0]].resample(freq).max()
        if col_mapping['low']:
            ohlc_dict['low'] = df[col_mapping['low'][0]].resample(freq).min()
        if col_mapping['close']:
            ohlc_dict['close'] = df[col_mapping['close'][0]].resample(freq).last()
        
        # Volume if available
        vol_cols = [c for c in df.columns if 'volume' in c.lower()]
        if vol_cols:
            ohlc_dict['volume'] = df[vol_cols[0]].resample(freq).sum()
        
        result = pd.DataFrame(ohlc_dict)
        result.index.name = 'date'
        
        return result
    
    def detect_data_gaps(self, df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
        """
        Detect and report gaps in time series data.
        
        Returns:
            DataFrame with gap information
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return pd.DataFrame()
        
        # Calculate differences between consecutive dates
        date_diffs = df.index.to_series().diff().dropna()
        
        # Business days threshold (allow weekends)
        threshold = pd.Timedelta(days=max_gap)
        
        gaps = date_diffs[date_diffs > threshold]
        
        if len(gaps) == 0:
            logger.info("No significant data gaps detected")
            return pd.DataFrame()
        
        gap_info = pd.DataFrame({
            'gap_start': df.index[df.index.get_loc(gap.name) - 1] if df.index.get_loc(gap.name) > 0 else None,
            'gap_end': gap.name,
            'gap_days': gap.days
        } for gap in gaps)
        
        logger.warning(f"Detected {len(gap_info)} data gaps > {max_gap} days")
        return gap_info


def clean_gold_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function for quick gold data cleaning.
    
    Args:
        df: Raw gold price DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataCleaner()
    return cleaner.clean_price_data(df)


if __name__ == "__main__":
    # Example usage
    from fetcher import GoldDataFetcher
    
    fetcher = GoldDataFetcher()
    raw_data = fetcher.fetch_gold_prices('GC=F', period='1y')
    
    print("Raw data sample:")
    print(raw_data.head())
    print(f"\nRaw data shape: {raw_data.shape}")
    print(f"Missing values: {raw_data.isna().sum().sum()}")
    
    # Clean the data
    cleaner = DataCleaner()
    clean_data = cleaner.clean_price_data(raw_data)
    
    print("\nCleaned data sample:")
    print(clean_data.head())
    print(f"\nClean data shape: {clean_data.shape}")
    print(f"Quality flags added: {[c for c in clean_data.columns if c.startswith('_')]}")