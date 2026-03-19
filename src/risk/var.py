"""
Value at Risk (VaR) and Conditional Value at Risk (CVaR) Module

Calculates risk metrics from Monte Carlo simulation paths.
Optimized with Numba JIT compilation for performance.
author:Essabri Ali Rayan
version:1.3
"""

import numpy as npt
from typing import Tuple, Dict, Union
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        return args[0] if args and callable(args[0]) else lambda f: f

@jit(nopython=True, cache=True)
def _calculate_var_numba(returns: np.ndarray, confidence_level: float) -> float:
    """
    Numba-optimized VaR calculation.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns (can be negative for losses)
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    float
        Value at Risk (positive number representing loss)
    """
    sorted_returns = np.sort(returns)
    index = int((1.0 - confidence_level) * len(sorted_returns))
    return -sorted_returns[index]


@jit(nopython=True, cache=True)
def _calculate_cvar_numba(returns: np.ndarray, confidence_level: float) -> float:
    """
    Numba-optimized CVaR (Expected Shortfall) calculation.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of returns
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    float
        Conditional Value at Risk (average of worst losses)
    """
    sorted_returns = np.sort(returns)
    cutoff_index = int((1.0 - confidence_level) * len(sorted_returns))
    
    if cutoff_index == 0:
        return -sorted_returns[0]
    
    worst_returns = sorted_returns[:cutoff_index]
    return -np.mean(worst_returns)


def calculate_var(
    paths: np.ndarray, 
    confidence_level: float = 0.95,
    initial_price: float = None
) -> float:
    """
    Calculate Value at Risk from simulated price paths.
    
    VaR represents the maximum loss at a given confidence level.
    For example, 95% VaR of $100 means there's a 5% chance of losing 
    more than $100.
    
    Parameters
    ----------
    paths : np.ndarray
        Simulated price paths (n_simulations x n_days)
    confidence_level : float, default 0.95
        Confidence level for VaR calculation
    initial_price : float, optional
        Starting price. If None, uses first column of paths
    
    Returns
    -------
    float
        Value at Risk (absolute dollar amount)
    
    Example
    -------
    >>> paths = np.random.lognormal(0, 0.1, (10000, 30)) * 2000
    >>> var_95 = calculate_var(paths, 0.95)
    >>> print(f"95% VaR: ${var_95:.2f}")
    """
    if initial_price is None:
        initial_price = paths[:, 0].mean()
    
    final_prices = paths[:, -1]
    returns = (final_prices - initial_price) / initial_price
    
    var = _calculate_var_numba(returns, confidence_level)
    return var * initial_price


def calculate_cvar(
    paths: np.ndarray, 
    confidence_level: float = 0.95,
    initial_price: float = None
) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).
    
    CVaR is the average loss in the worst (1-confidence_level)% of cases.
    More conservative than VaR as it considers the tail beyond VaR.
    
    Parameters
    ----------
    paths : np.ndarray
        Simulated price paths (n_simulations x n_days)
    confidence_level : float, default 0.95
        Confidence level for CVaR calculation
    initial_price : float, optional
        Starting price. If None, uses first column of paths
    
    Returns
    -------
    float
        Conditional Value at Risk (absolute dollar amount)
    
    Example
    -------
    >>> paths = np.random.lognormal(0, 0.1, (10000, 30)) * 2000
    >>> cvar_95 = calculate_cvar(paths, 0.95)
    >>> print(f"95% CVaR: ${cvar_95:.2f}")
    """
    if initial_price is None:
        initial_price = paths[:, 0].mean()
    
    final_prices = paths[:, -1]
    returns = (final_prices - initial_price) / initial_price
    
    cvar = _calculate_cvar_numba(returns, confidence_level)
    return cvar * initial_price


def calculate_confidence_intervals(
    paths: np.ndarray,
    confidence_levels: list = [0.90, 0.95]
) -> Dict[float, Tuple[float, float]]:
    """
    Calculate confidence intervals for final prices.
    
    Parameters
    ----------
    paths : np.ndarray
        Simulated price paths
    confidence_levels : list
        List of confidence levels to calculate
    
    Returns
    -------
    Dict[float, Tuple[float, float]]
        Dictionary mapping confidence level to (lower, upper) bounds
    
    Example
    -------
    >>> paths = np.random.lognormal(0, 0.1, (10000, 30)) * 2000
    >>> cis = calculate_confidence_intervals(paths, [0.90, 0.95])
    >>> print(f"90% CI: ${cis[0.90][0]:.2f} - ${cis[0.90][1]:.2f}")
    """
    final_prices = paths[:, -1]
    intervals = {}
    
    for cl in confidence_levels:
        alpha = 1 - cl
        lower = np.percentile(final_prices, alpha/2 * 100)
        upper = np.percentile(final_prices, (1 - alpha/2) * 100)
        intervals[cl] = (lower, upper)
    
    return intervals


def calculate_risk_metrics(
    paths: np.ndarray,
    initial_price: float = None,
    confidence_levels: list = [0.90, 0.95, 0.99]
) -> Dict:
    """
    Calculate comprehensive risk metrics from simulation paths.
    
    Parameters
    ----------
    paths : np.ndarray
        Simulated price paths
    initial_price : float, optional
        Starting price
    confidence_levels : list
        Confidence levels for VaR/CVaR calculations
    
    Returns
    -------
    Dict
        Dictionary containing all risk metrics
    """
    if initial_price is None:
        initial_price = float(paths[:, 0].mean())
    
    final_prices = paths[:, -1]
    
    metrics = {
        'initial_price': initial_price,
        'expected_price': float(np.mean(final_prices)),
        'price_std': float(np.std(final_prices)),
        'var': {},
        'cvar': {},
        'confidence_intervals': calculate_confidence_intervals(paths, confidence_levels),
        'probability_of_gain': float(np.mean(final_prices > initial_price)),
        'probability_of_loss': float(np.mean(final_prices < initial_price)),
        'best_case': float(np.max(final_prices)),
        'worst_case': float(np.min(final_prices)),
        'median_price': float(np.median(final_prices))
    }
    
    # Calculate VaR and CVaR for each confidence level
    for cl in confidence_levels:
        metrics['var'][cl] = calculate_var(paths, cl, initial_price)
        metrics['cvar'][cl] = calculate_cvar(paths, cl, initial_price)
    
    return metrics


def calculate_probability_above_target(
    paths: np.ndarray,
    target_price: float
) -> float:
    """
    Calculate probability of price exceeding target level.
    
    Parameters
    ----------
    paths : np.ndarray
        Simulated price paths
    target_price : float
        Target price level
    
    Returns
    -------
    float
        Probability (0 to 1) of reaching target
    """
    final_prices = paths[:, -1]
    return float(np.mean(final_prices >= target_price))


def calculate_probability_below_target(
    paths: np.ndarray,
    target_price: float
) -> float:
    """
    Calculate probability of price falling below target level.
    
    Parameters
    ----------
    paths : np.ndarray
        Simulated price paths
    target_price : float
        Target price level
    
    Returns
    -------
    float
        Probability (0 to 1) of falling below target
    """
    final_prices = paths[:, -1]
    return float(np.mean(final_prices <= target_price))