"""
Heston Stochastic Volatility Model

Models volatility as a random process for more realistic dynamics.
Captures volatility clustering and smile effects seen in gold options.

Formulas:
dS = mu * S * dt + sqrt(v) * S * dW1
dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2

corr(dW1, dW2) = rho

Author: Essabri ali rayan
Version: 1.0
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HestonParameters:
    """Parameters for Heston stochastic volatility model."""
    mu: float = 0.05  # Price drift
    kappa: float = 2.0  # Mean reversion speed of variance
    theta: float = 0.04  # Long-term variance (20% vol)
    xi: float = 0.3  # Volatility of volatility
    rho: float = -0.7  # Correlation between price and variance shocks
    v0: float = 0.04  # Initial variance
    
    def validate(self):
        """Validate parameters (Feller condition: 2*kappa*theta > xi^2)."""
        if self.kappa <= 0:
            raise ValueError(f"Mean reversion speed must be positive")
        if self.theta <= 0:
            raise ValueError(f"Long-term variance must be positive")
        if self.xi <= 0:
            raise ValueError(f"Vol of vol must be positive")
        if not (-1 <= self.rho <= 1):
            raise ValueError(f"Correlation must be in [-1, 1]")
        
        # Feller condition check (ensures variance stays positive)
        feller = 2 * self.kappa * self.theta > self.xi**2
        if not feller:
            logger.warning(f"Feller condition not satisfied: 2*kappa*theta ({2*self.kappa*self.theta:.4f}) <= xi^2 ({self.xi**2:.4f})")
        
        return self


class HestonModel:
    """
    Heston stochastic volatility model.
    
    Models both price and variance as stochastic processes,
    capturing volatility clustering and leverage effects.
    
    Attributes:
        params: HestonParameters instance
        dt: Time step size
    """
    
    def __init__(self, params: Optional[HestonParameters] = None, dt: float = 1/252):
        """
        Initialize Heston model.
        
        Args:
            params: Model parameters
            dt: Time step in years
        """
        self.params = params or HestonParameters()
        self.params.validate()
        self.dt = dt
        
        logger.info(f"Heston initialized: kappa={self.params.kappa:.2f}, theta={self.params.theta:.4f}, xi={self.params.xi:.2f}")
    
    def calibrate(self, prices: np.ndarray) -> HestonParameters:
        """
        Simple calibration from price series.
        
        Estimates parameters from realized volatility series.
        
        Args:
            prices: Historical price series
            
        Returns:
            Calibrated parameters
        """
        import pandas as pd
        
        # Calculate returns and realized variance
        returns = np.diff(np.log(prices))
        
        # Rolling realized variance (20-day window)
        window = 20
        realized_var = pd.Series(returns**2).rolling(window).mean() * 252
        realized_var = realized_var.dropna().values
        
        if len(realized_var) < 50:
            logger.warning("Not enough data for Heston calibration, using defaults")
            return self.params
        
        # Calibrate variance process parameters
        # Simple moment matching
        theta = np.mean(realized_var)
        kappa = 2.0  # Fixed based on typical gold behavior
        
        # Vol of vol from variance of variance
        xi = np.std(np.diff(realized_var)) / np.sqrt(self.dt)
        xi = np.clip(xi, 0.1, 1.0)  # Reasonable bounds
        
        # Correlation from returns vs variance changes
        var_changes = np.diff(realized_var)
        if len(var_changes) == len(returns):
            rho = np.corrcoef(returns[1:], var_changes)[0, 1]
            rho = np.clip(rho, -0.9, 0.9)
        else:
            rho = -0.7  # Default leverage effect
        
        # Price drift
        mu = np.mean(returns) * 252
        
        # Initial variance
        v0 = realized_var[-1] if len(realized_var) > 0 else theta
        
        self.params = HestonParameters(
            mu=mu,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            v0=v0
        )
        
        logger.info(f"Heston calibrated: theta={theta:.4f}, xi={xi:.2f}, rho={rho:.2f}")
        return self.params
    
    def step(
        self,
        S: np.ndarray,
        v: np.ndarray,
        Z1: np.ndarray,
        Z2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single step simulation for both price and variance.
        
        Uses full truncation scheme to ensure positive variance.
        
        Args:
            S: Current prices
            v: Current variances
            Z1: Random normal for price
            Z2: Random normal for variance
            
        Returns:
            Tuple of (next_prices, next_variances)
        """
        mu, kappa, theta, xi, rho = (
            self.params.mu,
            self.params.kappa,
            self.params.theta,
            self.params.xi,
            self.params.rho
        )
        dt = self.dt
        
        # Ensure positive variance for calculations
        v_pos = np.maximum(v, 0)
        
        # Correlate random variables
        # Z1 = Z_price, Z2 = Z_variance
        # Z_price = rho * Z_variance + sqrt(1-rho^2) * Z_independent
        Z_price = rho * Z2 + np.sqrt(1 - rho**2) * Z1
        Z_var = Z2
        
        # Update variance (full truncation scheme)
        dv = kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * Z_var
        v_next = v + dv
        v_next = np.maximum(v_next, 0)  # Keep variance non-negative
        
        # Update price
        v_next_pos = np.maximum(v_next, 0)
        dS = (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * Z_price
        S_next = S * np.exp(dS)
        
        return S_next, v_next
    
    def simulate(
        self,
        S0: float,
        n_steps: int,
        n_paths: int = 1000,
        random_seed: Optional[int] = None,
        return_variance: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate price paths with stochastic volatility.
        
        Args:
            S0: Initial price
            n_steps: Number of time steps
            n_paths: Number of paths
            random_seed: Random seed
            return_variance: Also return variance paths
            
        Returns:
            Price paths (and variance paths if requested)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize
        S = np.zeros((n_paths, n_steps))
        v = np.zeros((n_paths, n_steps))
        
        S[:, 0] = S0
        v[:, 0] = self.params.v0
        
        # Generate random numbers
        Z1 = np.random.standard_normal((n_paths, n_steps - 1))  # For price
        Z2 = np.random.standard_normal((n_paths, n_steps - 1))  # For variance
        
        # Simulate
        for t in range(1, n_steps):
            S[:, t], v[:, t] = self.step(S[:, t-1], v[:, t-1], Z1[:, t-1], Z2[:, t-1])
        
        logger.info(f"Heston simulation complete: {n_paths} paths, {n_steps} steps")
        
        if return_variance:
            return S, v
        return S
    
    def get_volatility_term_structure(
        self,
        maturities: np.ndarray
    ) -> np.ndarray:
        """
        Calculate implied volatility term structure.
        
        Args:
            maturities: Array of maturities in years
            
        Returns:
            Implied volatilities
        """
        kappa, theta, v0 = self.params.kappa, self.params.theta, self.params.v0
        
        # Expected variance at time t
        expected_var = theta + (v0 - theta) * np.exp(-kappa * maturities)
        
        # Implied volatility (square root of expected variance)
        implied_vol = np.sqrt(expected_var)
        
        return implied_vol
    
    def get_statistics(
        self,
        paths: np.ndarray,
        variance_paths: Optional[np.ndarray] = None,
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate statistics including volatility analysis.
        
        Args:
            paths: Price paths
            variance_paths: Variance paths (optional)
            confidence: Confidence level
            
        Returns:
            Dictionary with statistics
        """
        final_prices = paths[:, -1]
        
        stats_dict = {
            'mean_final': np.mean(final_prices),
            'std_final': np.std(final_prices),
            'median_final': np.median(final_prices),
            f'ci_lower_{int(confidence*100)}': np.percentile(final_prices, (1-confidence)/2 * 100),
            f'ci_upper_{int(confidence*100)}': np.percentile(final_prices, (1+confidence)/2 * 100),
            'avg_realized_vol': np.mean([np.std(np.diff(np.log(p))) * np.sqrt(252) for p in paths])
        }
        
        if variance_paths is not None:
            stats_dict['avg_var'] = np.mean(variance_paths[:, -1])
            stats_dict['avg_vol'] = np.mean(np.sqrt(variance_paths[:, -1]))
        
        return stats_dict


def create_heston_model(
    historical_prices: Optional[np.ndarray] = None,
    **kwargs
) -> HestonModel:
    """
    Factory function for Heston model.
    """
    if historical_prices is not None:
        model = HestonModel()
        model.calibrate(historical_prices)
        
        # Override with any provided parameters
        for key, value in kwargs.items():
            if hasattr(model.params, key):
                setattr(model.params, key, value)
    else:
        params = HestonParameters(**kwargs)
        model = HestonModel(params)
    
    return model


if __name__ == "__main__":
    print("Testing Heston Stochastic Volatility Model")
    
    # Create model
    heston = HestonModel(
        HestonParameters(
            mu=0.04,
            kappa=3.0,
            theta=0.04,   # 20% long-term vol
            xi=0.4,       # High vol of vol
            rho=-0.7,     # Leverage effect
            v0=0.09       # Start with 30% vol (elevated)
        )
    )
    
    S0 = 2000.0
    n_days = 60
    n_paths = 1000
    
    # Simulate with variance tracking
    paths, var_paths = heston.simulate(S0, n_days, n_paths, random_seed=42, return_variance=True)
    
    stats = heston.get_statistics(paths, var_paths)
    
    print(f"\nSimulation Results:")
    print(f"Initial Price: ${S0:.2f}")
    print(f"Expected Final: ${stats['mean_final']:.2f}")
    print(f"Avg Realized Vol: {stats['avg_realized_vol']:.2%}")
    print(f"Final Avg Variance: {stats['avg_var']:.4f}")
    print(f"Final Avg Vol: {stats['avg_vol']:.2%}")
    
    # Term structure
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    vol_ts = heston.get_volatility_term_structure(maturities)
    print(f"\nVolatility Term Structure:")
    for t, vol in zip(maturities, vol_ts):
        print(f"  {t:.2f}Y: {vol:.2%}")