"""
src/portfolio/__init__.py — MARGINS Portfolio Mode
Public API exports for the portfolio module.
"""

from .universe import fetch_universe, get_current_prices, summarize_universe
from .correlation import compute_cholesky, get_correlation_matrix, summarize_correlation
from .simulator import calibrate_models, simulate_portfolio, compute_portfolio_values, GBMModel

__all__ = [
    # universe
    "fetch_universe",
    "get_current_prices",
    "summarize_universe",
    # correlation
    "compute_cholesky",
    "get_correlation_matrix",
    "summarize_correlation",
    # simulator
    "GBMModel",
    "calibrate_models",
    "simulate_portfolio",
    "compute_portfolio_values",
]
