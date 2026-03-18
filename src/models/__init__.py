"""
Stochastic Models Package

Implementation of 5 stochastic models for gold price prediction:

1. GBM (Geometric Brownian Motion) - Baseline model
2. Ornstein-Uhlenbeck - Mean reversion model (most important for gold)
3. Merton Jump Diffusion - Crisis/jump model
4. Heston - Stochastic volatility model
5. Regime Switching - Markov chain model

Usage:
    from models import (
        GeometricBrownianMotion,
        OrnsteinUhlenbeckModel,
        MertonJumpModel,
        HestonModel,
        RegimeSwitchingModel,
        create_model
    )

    model = create_model('ou', historical_data=prices)
    paths = model.simulate(S0=2000, n_steps=30, n_paths=1000)
Author: Essabri Ali Rayan
Version : 1.3
"""
import numpy as np
from typing import Optional

from .gbm import GeometricBrownianMotion, GBMParameters, ModelParameters, create_gbm_model
from .mean_reversion import OrnsteinUhlenbeckModel, OUParameters, create_ou_model
from .jump_diffusion import MertonJumpModel, MertonParameters, create_merton_model
from .heston import HestonModel, HestonParameters, create_heston_model
from .regime_switching import (
    RegimeSwitchingModel,
    RegimeSwitchingParameters,
    RegimeParameters,
    Regime,
    create_regime_model
)

__version__ = '1.2.0'

__all__ = [
    # GBM
    'GeometricBrownianMotion',
    'GBMParameters',
    'ModelParameters',
    'create_gbm_model',

    # Ornstein-Uhlenbeck
    'OrnsteinUhlenbeckModel',
    'OUParameters',
    'create_ou_model',

    # Merton Jump
    'MertonJumpModel',
    'MertonParameters',
    'create_merton_model',

    # Heston
    'HestonModel',
    'HestonParameters',
    'create_heston_model',

    # Regime Switching
    'RegimeSwitchingModel',
    'RegimeSwitchingParameters',
    'RegimeParameters',
    'Regime',
    'create_regime_model',
]


MODEL_REGISTRY = {
    'gbm':             (GeometricBrownianMotion, create_gbm_model),
    'ou':              (OrnsteinUhlenbeckModel,  create_ou_model),
    'mean_reversion':  (OrnsteinUhlenbeckModel,  create_ou_model),
    'merton':          (MertonJumpModel,          create_merton_model),
    'jump_diffusion':  (MertonJumpModel,          create_merton_model),
    'heston':          (HestonModel,              create_heston_model),
    'regime':          (RegimeSwitchingModel,     create_regime_model),
    'regime_switching':(RegimeSwitchingModel,     create_regime_model),
}


def create_model(
    model_type: str,
    historical_data: Optional[np.ndarray] = None,
    **kwargs
):
    """
    Factory function to create any model by type.

    Args:
        model_type: 'gbm', 'ou', 'merton', 'heston', or 'regime'
        historical_data: Historical prices or returns for calibration
        **kwargs: Additional model parameters

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model_type is not recognized
    """
    model_type = model_type.lower().strip()

    if model_type not in MODEL_REGISTRY:
        available = ', '.join(sorted(set(MODEL_REGISTRY.keys())))
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")

    _, factory_func = MODEL_REGISTRY[model_type]

    if historical_data is not None:
        return factory_func(historical_data, **kwargs)
    else:
        return factory_func(**kwargs)


def list_available_models() -> list:
    """Return list of available model types."""
    return sorted(set(MODEL_REGISTRY.keys()))