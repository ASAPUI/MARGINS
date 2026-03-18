"""
Feature Engineering Module

Transforms raw price data into model-ready features for Monte Carlo simulation.
Creates technical indicators, regime features, and macro factors.

Author: Essabri ali rayan
Version: 1.3
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Professional feature engineering pipeline for gold price prediction.
    
    Generates:
    - Return-based features (log returns, volatility)
    - Technical indicators (moving averages, RSI, MACD)
    - Mean reversion features
    - Macro correlation features
    - Regime detection features
    """
    
    def __init__(self, target_col: str = 'close'):
        """
        Initialize feature engineer.
        
        Args:
            target_col: Primary price column to engineer features from
        """
        self.target_col = target_col
        self.feature_names: List[str] = []
        
        logger.info(f"FeatureEngineer initialized with target: {target_col}")
    
    def create_all_features(
        self,
        df: pd.DataFrame,
        include_technical: bool = True,
        include_macro: bool = True,
        include_regime: bool = True
    ) -> pd.DataFrame:
        """
        Generate complete feature set for modeling.
        
        Args:
            df: Cleaned price DataFrame
            include_technical: Add technical indicators
            include_macro: Add macro-derived features (requires macro columns)
            include_regime: Add regime detection features
            
        Returns:
            DataFrame with all engineered features
        """
        df_features = df.copy()
        
        # Ensure we have the target column
        if self.target_col not in df_features.columns:
            # Try to find close price
            close_cols = [c for c in df_features.columns if 'close' in c.lower()]
            if close_cols:
                self.target_col = close_cols[0]
            else:
                raise ValueError(f"Target column {self.target_col} not found in data")
        
        logger.info(f"Creating features from {self.target_col}")
        
        # 1. Basic return features (ALWAYS)
        df_features = self._add_return_features(df_features)
        
        # 2. Volatility features (ALWAYS)
        df_features = self._add_volatility_features(df_features)
        
        # 3. Technical indicators
        if include_technical:
            df_features = self._add_technical_indicators(df_features)
        
        # 4. Mean reversion features
        df_features = self._add_mean_reversion_features(df_features)
        
        # 5. Macro features (if macro data present)
        if include_macro:
            df_features = self._add_macro_features(df_features)
        
        # 6. Regime features
        if include_regime:
            df_features = self._add_regime_features(df_features)
        
        # 7. Temporal features
        df_features = self._add_temporal_features(df_features)
        
        # Store feature names (excluding original columns)
        original_cols = set(df.columns)
        self.feature_names = [c for c in df_features.columns if c not in original_cols]
        
        logger.info(f"Created {len(self.feature_names)} new features")
        
        # Drop rows with NaN in critical features
        critical_features = ['daily_return', 'volatility_30d']
        df_features = df_features.dropna(subset=[f for f in critical_features if f in df_features.columns])
        
        return df_features
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics."""
        price = df[self.target_col]
        
        # Daily log returns (primary modeling target)
        df['daily_return'] = np.log(price / price.shift(1))
        
        # Simple returns
        df['simple_return'] = price.pct_change()
        
        # Cumulative returns (various horizons)
        for window in [5, 10, 20, 60]:
            df[f'return_{window}d'] = np.log(price / price.shift(window))
        
        # Rolling Sharpe-like ratio (return/volatility)
        rolling_ret = df['daily_return'].rolling(20).mean() * 252
        rolling_vol = df['daily_return'].rolling(20).std() * np.sqrt(252)
        df['sharpe_20d'] = rolling_ret / rolling_vol.replace(0, np.nan)
        
        logger.debug("Added return features")
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility and risk metrics."""
        returns = df['daily_return']
        
        # Rolling volatility (various windows)
        for window in [10, 20, 30, 60, 90]:
            df[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (using high-low)
        if 'high' in df.columns and 'low' in df.columns:
            log_hl = np.log(df['high'] / df['low'])
            df['parkinson_vol'] = np.sqrt((log_hl**2 / (4 * np.log(2))).rolling(20).mean()) * np.sqrt(252)
        
        # Garman-Klass volatility (more efficient)
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            log_hl = np.log(df['high'] / df['low'])
            log_co = np.log(df['close'] / df['open'])
            gk = 0.5 * log_hl**2 - (2*np.log(2)-1) * log_co**2
            df['garman_klass_vol'] = np.sqrt(gk.rolling(20).mean()) * np.sqrt(252)
        
        # Volatility of volatility (regime indicator)
        df['vol_of_vol'] = df['volatility_30d'].rolling(20).std()
        
        # Volatility trend (increasing/decreasing)
        df['vol_trend'] = df['volatility_30d'].diff(5)
        
        logger.debug("Added volatility features")
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators."""
        price = df[self.target_col]
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = price.rolling(window).mean()
            df[f'ema_{window}'] = price.ewm(span=window, adjust=False).mean()
        
        # Moving average distances (mean reversion signals)
        for window in [20, 50, 200]:
            df[f'dist_sma_{window}'] = (price - df[f'sma_{window}']) / df[f'sma_{window}']
        
        # RSI (Relative Strength Index)
        delta = price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # RSI categories
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        
        # MACD
        ema_12 = price.ewm(span=12, adjust=False).mean()
        ema_26 = price.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = price.rolling(20).mean()
        std_20 = price.rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        df['bb_position'] = (price - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Price momentum
        for window in [5, 10, 20]:
            df[f'momentum_{window}d'] = price - price.shift(window)
            df[f'momentum_pct_{window}d'] = (price / price.shift(window) - 1) * 100
        
        logger.debug("Added technical indicators")
        return df
    
    def _add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features specific to mean reversion models."""
        price = df[self.target_col]
        
        # Long-term mean (exponential decay weighted)
        df['long_term_mean'] = price.ewm(span=252, adjust=False).mean()
        
        # Mean reversion gap (distance from long-term mean)
        df['mean_reversion_gap'] = price - df['long_term_mean']
        df['mean_reversion_gap_pct'] = df['mean_reversion_gap'] / df['long_term_mean']
        
        # Half-life of mean reversion (Ornstein-Uhlenbeck parameter proxy)
        # Using AR(1) regression on prices
        lagged_price = price.shift(1)
        delta_price = price.diff()
        
        # Rolling half-life estimation (simplified)
        window = 60
        # Regression: ΔP_t = α + β*P_{t-1} + ε
        # Half-life = -ln(2)/κ where κ ≈ -β
        
        # Approximate using correlation
        df['price_autocorr_1'] = price.rolling(window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
        )
        
        # Mean reversion strength indicator
        df['mr_strength'] = -np.log(df['price_autocorr_1'].clip(-0.99, 0.99))
        
        # Z-score relative to rolling window
        for window in [20, 60]:
            rolling_mean = price.rolling(window).mean()
            rolling_std = price.rolling(window).std()
            df[f'zscore_{window}d'] = (price - rolling_mean) / rolling_std.replace(0, np.nan)
        
        logger.debug("Added mean reversion features")
        return df
    
    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features derived from macroeconomic data."""
        # Gold/USD ratio (if DXY available)
        dxy_cols = [c for c in df.columns if 'dxy' in c.lower() or 'usd' in c.lower()]
        if dxy_cols:
            dxy_col = dxy_cols[0]
            df['gold_usd_ratio'] = df[self.target_col] / df[dxy_col]
            df['gold_usd_corr_20d'] = df[self.target_col].rolling(20).corr(df[dxy_col])
        
        # Real interest rate proxy (if rates data available)
        rate_cols = [c for c in df.columns if 'rate' in c.lower() or 'tnx' in c.lower()]
        if rate_cols:
            rate_col = rate_cols[0]
            df['rates_change_5d'] = df[rate_col].diff(5)
            df['gold_rates_corr_20d'] = df[self.target_col].rolling(20).corr(df[rate_col])
        
        # VIX correlation (fear index)
        vix_cols = [c for c in df.columns if 'vix' in c.lower()]
        if vix_cols:
            vix_col = vix_cols[0]
            df['gold_vix_corr_20d'] = df[self.target_col].rolling(20).corr(df[vix_col])
            df['vix_level'] = df[vix_col]
            df['vix_high'] = (df[vix_col] > 20).astype(int)  # High fear threshold
        
        # Inflation expectations (if CPI data available)
        cpi_cols = [c for c in df.columns if 'cpi' in c.lower()]
        if cpi_cols:
            cpi_col = cpi_cols[0]
            df['inflation_yoy'] = df[cpi_col].pct_change(252) * 100
        
        logger.debug("Added macro features")
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features for detecting market regimes."""
        returns = df['daily_return']
        
        # Volatility regime
        vol_30 = df['volatility_30d']
        vol_percentile = vol_30.rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) if len(x) > 1 else 50, 
            raw=False
        )
        df['vol_regime'] = pd.cut(vol_percentile, bins=[0, 33, 66, 100], labels=[0, 1, 2])
        
        # Trend regime (based on moving average slope)
        sma_slope = df['sma_20'].diff(5)
        df['trend_regime'] = np.where(sma_slope > 0, 1, -1)
        df['trend_regime'] = np.where(np.abs(sma_slope) < sma_slope.rolling(20).std()*0.5, 0, df['trend_regime'])
        
        # Combined regime
        df['combined_regime'] = df['vol_regime'].astype(str) + '_' + df['trend_regime'].astype(str)
        
        # Crisis indicator (high vol + high correlation breakdown)
        if 'gold_vix_corr_20d' in df.columns:
            df['crisis_indicator'] = ((df['volatility_30d'] > df['volatility_30d'].quantile(0.8)) & 
                                     (df['vix_level'] > 25)).astype(int)
        
        # Kurtosis (tail risk indicator)
        df['return_kurtosis_30d'] = returns.rolling(30).kurt()
        
        # Skewness (asymmetry indicator)
        df['return_skew_30d'] = returns.rolling(30).skew()
        
        logger.debug("Added regime features")
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calendar-based features."""
        if isinstance(df.index, pd.DatetimeIndex):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['year'] = df.index.year
            
            # Seasonality indicators
            df['is_jan'] = (df.index.month == 1).astype(int)  # January effect
            df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
            
            # Days since start
            df['days_from_start'] = (df.index - df.index[0]).days
        
        logger.debug("Added temporal features")
        return df
    
    def get_model_inputs(
        self,
        df: pd.DataFrame,
        model_type: str = 'gbm',
        lookback: int = 30
    ) -> Dict[str, np.ndarray]:
        """
        Prepare feature matrix for specific model types.
        
        Args:
            df: DataFrame with all features
            model_type: 'gbm', 'ou', 'heston', 'merton', 'regime'
            lookback: Days of history to include
            
        Returns:
            Dictionary with model inputs
        """
        # Get latest data
        recent_data = df.iloc[-lookback:]
        
        inputs = {
            'current_price': df[self.target_col].iloc[-1],
            'returns': recent_data['daily_return'].dropna().values,
            'volatility': recent_data['volatility_30d'].iloc[-1] if 'volatility_30d' in recent_data.columns else 0.15,
        }
        
        if model_type == 'ou':
            inputs['long_term_mean'] = recent_data['long_term_mean'].iloc[-1] if 'long_term_mean' in recent_data.columns else inputs['current_price']
            inputs['mean_reversion_speed'] = recent_data['mr_strength'].iloc[-1] if 'mr_strength' in recent_data.columns else 0.1
            
        elif model_type == 'heston':
            inputs['variance'] = inputs['volatility'] ** 2
            inputs['long_term_variance'] = (recent_data['volatility_30d'].mean() ** 2) if 'volatility_30d' in recent_data.columns else 0.0225
            
        elif model_type == 'merton':
            # Estimate jump parameters from kurtosis
            if 'return_kurtosis_30d' in recent_data.columns:
                kurt = recent_data['return_kurtosis_30d'].iloc[-1]
                inputs['jump_intensity'] = max(0, (kurt - 3) / 10)  # Approximate
            else:
                inputs['jump_intensity'] = 0.1
            inputs['jump_size_mean'] = recent_data['daily_return'].mean()
            inputs['jump_size_std'] = recent_data['daily_return'].std() * 2
            
        elif model_type == 'regime':
            if 'combined_regime' in recent_data.columns:
                inputs['current_regime'] = recent_data['combined_regime'].iloc[-1]
            else:
                inputs['current_regime'] = '1_0'
        
        return inputs
    
    def select_features_for_model(
        self,
        df: pd.DataFrame,
        model_type: str = 'gbm'
    ) -> pd.DataFrame:
        """
        Select relevant feature subset for specific model.
        
        Args:
            df: Full feature DataFrame
            model_type: Type of stochastic model
            
        Returns:
            DataFrame with selected features
        """
        base_features = ['daily_return', 'volatility_30d', 'volatility_60d']
        
        model_features = {
            'gbm': base_features,
            'ou': base_features + ['mean_reversion_gap', 'mr_strength', 'long_term_mean'],
            'heston': base_features + ['vol_of_vol', 'vol_trend'],
            'merton': base_features + ['return_kurtosis_30d', 'return_skew_30d'],
            'regime': base_features + ['vol_regime', 'trend_regime', 'crisis_indicator']
        }
        
        selected = model_features.get(model_type, base_features)
        available = [f for f in selected if f in df.columns]
        
        return df[available]


def engineer_features(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Convenience function for quick feature engineering.
    
    Args:
        df: Cleaned price DataFrame
        price_col: Name of price column
        
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer(target_col=price_col)
    return engineer.create_all_features(df)
def sequence_builder(feature_matrix: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences for LSTM training from feature matrix.
    
    Creates sliding windows of features for LSTM input and corresponding
    targets (next-step log returns).
    
    Args:
        feature_matrix: Array of shape (N, F) where N is number of timesteps
                       and F is number of features. First feature (column 0)
                       should be log returns.
        window: Size of the lookback window (default 20 trading days)
        
    Returns:
        X: Array of shape (N-window, window, F) - input sequences
        y: Array of shape (N-window,) - next log-return targets
        
    Raises:
        ValueError: If N <= window or feature_matrix is invalid
    """
    if not isinstance(feature_matrix, np.ndarray):
        feature_matrix = np.array(feature_matrix)
        
    if feature_matrix.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {feature_matrix.shape}")
        
    N, F = feature_matrix.shape
    
    if N <= window:
        raise ValueError(f"Need more than {window} timesteps, got {N}")
    
    X = []
    y = []
    
    for i in range(window, N):
        X.append(feature_matrix[i-window:i])
        # Target is the log return at time i (first feature, index 0)
        y.append(feature_matrix[i, 0])
    
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Example usage
    import sys
    from fetcher import GoldDataFetcher
    from cleaner import DataCleaner
    
    # Fetch and clean data
    fetcher = GoldDataFetcher()
    raw_data = fetcher.fetch_gold_prices('GC=F', period='2y')
    
    cleaner = DataCleaner()
    clean_data = cleaner.clean_price_data(raw_data)
    
    # Engineer features
    engineer = FeatureEngineer(target_col='close')
    features_df = engineer.create_all_features(clean_data)
    
    print(f"\nOriginal columns: {len(clean_data.columns)}")
    print(f"Total columns after engineering: {len(features_df.columns)}")
    print(f"\nNew features created: {len(engineer.feature_names)}")
    print(f"\nSample features: {engineer.feature_names[:10]}")
    
    print("\nFeature statistics:")
    print(features_df[['daily_return', 'volatility_30d', 'rsi_14', 'mean_reversion_gap_pct']].describe())
    
    # Get model inputs
    inputs = engineer.get_model_inputs(features_df, model_type='ou')
    print(f"\nModel inputs for OU process: {list(inputs.keys())}")