# src/macro/__init__.py
"""
Macro intelligence integration for gold-option.
Connects WorldMonitor API live signals to stochastic model parameters.
Author:Essabri Ali Rayan
Version : 1.3
"""

from .bridge import MacroBridge, MacroSignal
from .adjuster import ParameterAdjuster, ModelParameters
from .signals import AnomalyEvent, RiskTier

__all__ = [
    "MacroBridge",
    "MacroSignal", 
    "ParameterAdjuster",
    "ModelParameters",
    "AnomalyEvent",
    "RiskTier",
]