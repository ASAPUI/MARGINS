"""
Performance Metrics and Statistics Module

Calculates backtesting metrics, accuracy measures, and performance ratios
for Monte Carlo gold price predictions.
author:Essabri Ali Rayan
Version:1.3
"""

import numpy as np
from typing import Dict, Tuple, List, Union
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else lambda f: f

@jit(nopython=True, cache=True)
def _calculate_max_drawdown_numba(prices: np.ndarray) -> float:
    """
    Numba-optimized max drawdown calculation.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series
    
    Returns
    -------
    float
        Maximum drawdown as a percentage
    """
    peak = prices[0]
    max_dd = 0.0
    
    for price in prices:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        if drawdown > max_dd:
            max_dd = drawdown
    
    return max_dd


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Measures risk-adjusted return. Higher is better.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (daily)
    risk_free_rate : float, default 0.02
        Annual risk-free rate
    periods_per_year : int, default 252
        Number of trading periods per year
    
    Returns
    -------
    float
        Annualized Sharpe ratio
    
    Example
    -------
    >>> returns = np.random.normal(0.0005, 0.01, 252)
    >>> sharpe = calculate_sharpe_ratio(returns)
    >>> print(f"Sharpe Ratio: {sharpe:.2f}")
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = np.mean(excess_returns) / np.std(returns)
    return sharpe * np.sqrt(periods_per_year)


def calculate_max_drawdown(prices: np.ndarray) -> float:
    """
    Calculate maximum drawdown from peak to trough.
    
    Parameters
    ----------
    prices : np.ndarray
        Price series (can be 1D or 2D array of paths)
    
    Returns
    -------
    float
        Maximum drawdown as a percentage (0 to 1)
    
    Example
    -------
    >>> prices = np.array([100, 110, 105, 95, 100, 120])
    >>> mdd = calculate_max_drawdown(prices)
    >>> print(f"Max Drawdown: {mdd*100:.1f}%")
    """
    if prices.ndim == 2:
        # For multiple paths, calculate average max drawdown
        drawdowns = []
        for path in prices:
            dd = _calculate_max_drawdown_numba(path)
            drawdowns.append(dd)
        return float(np.mean(drawdowns))
    else:
        return _calculate_max_drawdown_numba(prices)


def calculate_rmse(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Penalizes large errors more than MAE.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    actual : np.ndarray
        Actual values
    
    Returns
    -------
    float
        RMSE value
    """
    return float(np.sqrt(np.mean((predictions - actual) ** 2)))


def calculate_mae(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Average absolute difference between predictions and actuals.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    actual : np.ndarray
        Actual values
    
    Returns
    -------
    float
        MAE value
    """
    return float(np.mean(np.abs(predictions - actual)))


def calculate_mape(predictions: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Scale-independent error measure. Target < 5% is good.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    actual : np.ndarray
        Actual values
    
    Returns
    -------
    float
        MAPE as percentage
    """
    # Avoid division by zero
    mask = actual != 0
    if not np.any(mask):
        return float('inf')
    
    return float(np.mean(np.abs((predictions[mask] - actual[mask]) / actual[mask])) * 100)


def calculate_directional_accuracy(
    predictions: np.ndarray,
    actual: np.ndarray
) -> float:
    """
    Calculate directional accuracy (up/down prediction accuracy).
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted price series
    actual : np.ndarray
        Actual price series
    
    Returns
    -------
    float
        Percentage of correct directional calls (0 to 100)
    """
    if len(predictions) < 2 or len(actual) < 2:
        return 0.0
    
    pred_direction = np.diff(predictions) > 0
    actual_direction = np.diff(actual) > 0
    
    correct = np.sum(pred_direction == actual_direction)
    return float(correct / len(pred_direction) * 100)


def calculate_coverage(
    actual: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> float:
    """
    Calculate confidence interval coverage.
    
    Percentage of actual values that fall within predicted confidence interval.
    Should be close to the confidence level (e.g., ~95% for 95% CI).
    
    Parameters
    ----------
    actual : np.ndarray
        Actual values
    lower_bound : np.ndarray
        Lower confidence bound
    upper_bound : np.ndarray
        Upper confidence bound
    
    Returns
    -------
    float
        Coverage percentage
    """
    within_bounds = (actual >= lower_bound) & (actual <= upper_bound)
    return float(np.mean(within_bounds) * 100)


def calculate_all_metrics(
    predictions: np.ndarray,
    actual: np.ndarray,
    confidence_lower: np.ndarray = None,
    confidence_upper: np.ndarray = None,
    returns: np.ndarray = None
) -> Dict:
    """
    Calculate all performance metrics at once.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values
    actual : np.ndarray
        Actual values
    confidence_lower : np.ndarray, optional
        Lower confidence bound for coverage calculation
    confidence_upper : np.ndarray, optional
        Upper confidence bound for coverage calculation
    returns : np.ndarray, optional
        Returns series for Sharpe ratio calculation
    
    Returns
    -------
    Dict
        Dictionary containing all metrics
    """
    metrics = {
        'rmse': calculate_rmse(predictions, actual),
        'mae': calculate_mae(predictions, actual),
        'mape': calculate_mape(predictions, actual),
        'directional_accuracy': calculate_directional_accuracy(predictions, actual),
        'bias': float(np.mean(predictions - actual)),
        'correlation': float(np.corrcoef(predictions, actual)[0, 1]) if len(predictions) > 1 else 0.0
    }
    
    if confidence_lower is not None and confidence_upper is not None:
        metrics['coverage'] = calculate_coverage(actual, confidence_lower, confidence_upper)
    
    if returns is not None:
        metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns)
        metrics['volatility'] = float(np.std(returns) * np.sqrt(252))  # Annualized
        metrics['annualized_return'] = float(np.mean(returns) * 252)
    
    return metrics


def calculate_model_comparison(
    models_results: Dict[str, Dict]
) -> Dict:
    """
    Compare multiple models across metrics.
    
    Parameters
    ----------
    models_results : Dict[str, Dict]
        Dictionary mapping model name to metrics dictionary
    
    Returns
    -------
    Dict
        Comparison summary with rankings
    """
    comparison = {
        'rankings': {},
        'best_model': {},
        'all_models': models_results
    }
    
    # Define metrics where lower is better
    lower_is_better = ['rmse', 'mae', 'mape']
    # Define metrics where higher is better
    higher_is_better = ['directional_accuracy', 'coverage', 'sharpe_ratio', 'correlation']
    
    all_metrics = set()
    for model_metrics in models_results.values():
        all_metrics.update(model_metrics.keys())
    
    for metric in all_metrics:
        values = {name: results.get(metric, float('inf') if metric in lower_is_better else float('-inf')) 
                  for name, results in models_results.items()}
        
        if metric in lower_is_better:
            ranked = sorted(values.items(), key=lambda x: x[1])
        else:
            ranked = sorted(values.items(), key=lambda x: x[1], reverse=True)
        
        comparison['rankings'][metric] = ranked
        if ranked:
            comparison['best_model'][metric] = ranked[0][0]
    
    return comparison


def print_metrics_report(metrics: Dict, model_name: str = "Model"):
    """
    Print a formatted metrics report.
    
    Parameters
    ----------
    metrics : Dict
        Metrics dictionary
    model_name : str
        Name of the model for the report header
    """
    print(f"\n{'='*50}")
    print(f"Performance Report: {model_name}")
    print(f"{'='*50}")
    
    if 'rmse' in metrics:
        print(f"RMSE:                 {metrics['rmse']:.4f}")
    if 'mae' in metrics:
        print(f"MAE:                  {metrics['mae']:.4f}")
    if 'mape' in metrics:
        print(f"MAPE:                 {metrics['mape']:.2f}%")
    if 'directional_accuracy' in metrics:
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
    if 'coverage' in metrics:
        print(f"CI Coverage:          {metrics['coverage']:.1f}%")
    if 'sharpe_ratio' in metrics:
        print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
    if 'correlation' in metrics:
        print(f"Correlation:          {metrics['correlation']:.4f}")
    if 'bias' in metrics:
        print(f"Bias:                 {metrics['bias']:.4f}")
    
    print(f"{'='*50}\n")