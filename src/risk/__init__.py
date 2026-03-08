"""
Risk Package

Comprehensive risk analysis for Monte Carlo gold price prediction.

Modules:
- var: Value at Risk (VaR) and Conditional VaR (CVaR) calculations
- metrics: Risk-adjusted performance metrics (Sharpe, Sortino, etc.)

Usage:
    from risk import VaRCalculator, RiskMetricsCalculator, quick_var
    
    # VaR calculation
    calc = VaRCalculator(confidence_level=0.95, holding_period=1)
    var_result = calc.historical_var(returns, portfolio_value=100000)
    cvar_result = calc.calculate_cvar(returns, portfolio_value=100000)
    
    # Risk metrics
    metrics_calc = RiskMetricsCalculator()
    metrics = metrics_calc.calculate_all_metrics(returns, prices)
"""

from .var import (
    VaRCalculator,
    VaRResult,
    CVaRResult,
    quick_var
)

from .metrics import (
    RiskMetricsCalculator,
    RiskMetrics,
    calculate_risk_report
)

__version__ = '1.0.0'

__all__ = [
    # VaR
    'VaRCalculator',
    'VaRResult',
    'CVaRResult',
    'quick_var',
    
    # Metrics
    'RiskMetricsCalculator',
    'RiskMetrics',
    'calculate_risk_report',
]