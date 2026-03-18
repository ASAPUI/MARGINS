"""
Walk-Forward Backtesting Engine

Rigorous backtesting methodology ensuring no lookahead bias.
Uses rolling windows for training and out-of-sample testing.
Author : Essabri Ali Rayan
Version : 1.3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class BacktestResult:
    """Container for backtest results."""
    model_name: str
    predictions: np.ndarray
    actuals: np.ndarray
    dates: pd.DatetimeIndex
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    metrics: Dict
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.errors = self.predictions - self.actuals
        self.abs_errors = np.abs(self.errors)
        self.squared_errors = self.errors ** 2


class WalkForwardTester:
    """
    Walk-forward analysis for time series models.
    
    Never uses future data to predict the past - critical for valid backtests.
    """
    
    def __init__(
        self,
        train_window: int = 252,  # 1 year
        test_window: int = 30,    # 1 month
        step_size: int = 30,      # Roll forward 1 month at a time
        min_train_size: int = 126 # Minimum 6 months training
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.results: List[BacktestResult] = []
    
    def run(
        self,
        model_class,
        prices: pd.Series,
        model_params: Optional[Dict] = None,
        feature_fn: Optional[Callable] = None
    ) -> List[BacktestResult]:
        """
        Execute walk-forward backtest.
        
        Parameters
        ----------
        model_class : class
            Model class to instantiate (GBM, OU, etc.)
        prices : pd.Series
            Historical price series with DatetimeIndex
        model_params : dict, optional
            Fixed parameters for model (if not calibrating)
        feature_fn : callable, optional
            Function to extract features from price history
        
        Returns
        -------
        List[BacktestResult]
            Results for each walk-forward window
        """
        self.results = []
        n = len(prices)
        
        # Generate walk-forward windows
        windows = self._generate_windows(n)
        
        for train_idx, test_idx in windows:
            # Split data
            train_prices = prices.iloc[train_idx]
            test_prices = prices.iloc[test_idx]
            
            # Calculate returns for calibration
            train_returns = np.log(train_prices / train_prices.shift(1)).dropna()
            
            # Instantiate and calibrate model
            model = model_class()
            if hasattr(model, 'calibrate'):
                model.calibrate(train_returns.values)
            
            # Override with fixed params if provided
            if model_params:
                for key, value in model_params.items():
                    setattr(model, key, value)
            
            # Run simulation
            S0 = train_prices.iloc[-1]
            n_days = len(test_prices)
            n_sims = 10000
            
            paths = model.simulate(S0, n_days, n_sims)
            predictions = np.mean(paths, axis=0)  # Expected price path
            
            # Store result
            result = BacktestResult(
                model_name=model.name,
                predictions=predictions,
                actuals=test_prices.values,
                dates=test_prices.index,
                train_start=train_prices.index[0],
                train_end=train_prices.index[-1],
                test_start=test_prices.index[0],
                test_end=test_prices.index[-1],
                metrics=self._calculate_metrics(predictions, test_prices.values)
            )
            
            self.results.append(result)
        
        return self.results
    
    def _generate_windows(self, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index pairs for walk-forward."""
        windows = []
        start = self.train_window
        
        while start + self.test_window <= n:
            train_idx = np.arange(start - self.train_window, start)
            test_idx = np.arange(start, min(start + self.test_window, n))
            windows.append((train_idx, test_idx))
            start += self.step_size
        
        return windows
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict:
        """Calculate performance metrics for a single window."""
        errors = predictions - actuals
        
        # Avoid division by zero in MAPE
        mask = actuals != 0
        mape = np.mean(np.abs(errors[mask] / actuals[mask])) * 100 if np.any(mask) else np.inf
        
        # Directional accuracy
        if len(predictions) > 1:
            pred_dir = np.diff(predictions) > 0
            actual_dir = np.diff(actuals) > 0
            dir_accuracy = np.mean(pred_dir == actual_dir) * 100
        else:
            dir_accuracy = 0.0
        
        return {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mae': float(np.mean(np.abs(errors))),
            'mape': float(mape),
            'bias': float(np.mean(errors)),
            'directional_accuracy': float(dir_accuracy),
            'max_error': float(np.max(np.abs(errors))),
            'final_price_error': float(abs(predictions[-1] - actuals[-1]))
        }
    
    def aggregate_metrics(self) -> Dict:
        """Aggregate metrics across all windows."""
        if not self.results:
            return {}
        
        all_metrics = {}
        metric_keys = self.results[0].metrics.keys()
        
        for key in metric_keys:
            values = [r.metrics[key] for r in self.results]
            all_metrics[f'{key}_mean'] = float(np.mean(values))
            all_metrics[f'{key}_std'] = float(np.std(values))
            all_metrics[f'{key}_min'] = float(np.min(values))
            all_metrics[f'{key}_max'] = float(np.max(values))
        
        return all_metrics
    
    def get_predictions_df(self) -> pd.DataFrame:
        """Get all predictions as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        all_preds = []
        for result in self.results:
            df = pd.DataFrame({
                'date': result.dates,
                'predicted': result.predictions,
                'actual': result.actuals,
                'error': result.errors,
                'model': result.model_name
            })
            all_preds.append(df)
        
        return pd.concat(all_preds, ignore_index=True)


def run_backtest(
    models: Dict[str, object],
    prices: pd.Series,
    train_window: int = 252,
    test_window: int = 30
) -> Dict[str, List[BacktestResult]]:
    """
    Run backtest on multiple models.
    
    Parameters
    ----------
    models : dict
        Dictionary mapping model names to model classes
    prices : pd.Series
        Price series
    train_window : int
        Training window size
    test_window : int
        Testing window size
    
    Returns
    -------
    dict
        Results for each model
    """
    results = {}
    
    for name, model_class in models.items():
        print(f"Backtesting {name}...")
        tester = WalkForwardTester(train_window, test_window)
        results[name] = tester.run(model_class, prices)
    
    return results