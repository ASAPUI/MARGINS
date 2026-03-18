"""
Geometric Brownian Motion (GBM) Model

The foundational stochastic model for asset pricing.
Used as baseline for comparison with more advanced models.

Formula: dS = mu * S * dt + sigma * S * dW

Author: Essabri Ali rayan
Version: 1.0
"""

import numpy as np
from typing import Optional, Tuple, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GBMParameters:
    """Parameters for Geometric Brownian Motion."""
    mu: float = 0.05    # Drift (annualized expected return)
    sigma: float = 0.15  # Volatility (annualized)

    def validate(self):
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")
        return self


@dataclass
class ModelParameters:
    """Macro-adjusted model parameters produced by ParameterAdjuster."""
    mu_adjusted: float = 0.05
    sigma_adjusted: float = 0.15
    lambda_boost: float = 1.0
    regime_crisis_prior: float = 0.2


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion model for gold price simulation.
    """

    def __init__(self, params: Optional[GBMParameters] = None, dt: float = 1/252):
        self.params = params or GBMParameters()
        self.params.validate()
        self.dt = dt
        logger.info(f"GBM initialized: mu={self.params.mu:.4f}, sigma={self.params.sigma:.4f}")

    def calibrate(self, returns: np.ndarray) -> GBMParameters:
        mu = np.mean(returns) * 252
        sigma = np.std(returns) * np.sqrt(252)
        self.params = GBMParameters(mu=mu, sigma=sigma)
        logger.info(f"GBM calibrated: mu={mu:.4f}, sigma={sigma:.4f}")
        return self.params

    def step(self, S: np.ndarray, Z: np.ndarray) -> np.ndarray:
        mu, sigma = self.params.mu, self.params.sigma
        dt = self.dt
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        return S * np.exp(drift + diffusion)

    def simulate(
        self,
        S0: float,
        n_steps: int,
        n_paths: int = 1000,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        if random_seed is not None:
            np.random.seed(random_seed)
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0
        Z = np.random.standard_normal((n_paths, n_steps - 1))
        for t in range(1, n_steps):
            paths[:, t] = self.step(paths[:, t-1], Z[:, t-1])
        logger.info(f"GBM simulation complete: {n_paths} paths, {n_steps} steps")
        return paths

    def simulate_with_macro(
        self,
        S0: float,
        n_steps: int,
        n_paths: int,
        macro_params: Optional[ModelParameters] = None,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Run simulation with optional macro-adjusted parameters.
        Temporarily applies WorldMonitor-derived mu/sigma, then restores originals.
        """
        original_mu = self.params.mu
        original_sigma = self.params.sigma

        if macro_params is not None:
            self.params.mu = macro_params.mu_adjusted
            self.params.sigma = macro_params.sigma_adjusted
            logger.info(
                f"GBM using macro-adjusted params: "
                f"mu={macro_params.mu_adjusted:.6f}, sigma={macro_params.sigma_adjusted:.4f}"
            )

        try:
            paths = self.simulate(S0, n_steps, n_paths, random_seed)
        finally:
            self.params.mu = original_mu
            self.params.sigma = original_sigma

        return paths

    def analytical_solution(
        self,
        S0: float,
        T: float,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.linspace(0, T, n_points)
        mu, sigma = self.params.mu, self.params.sigma
        mean_path = S0 * np.exp(mu * times)
        std_path = mean_path * np.sqrt(np.exp(sigma**2 * times) - 1)
        return times, mean_path, std_path

    def get_statistics(self, paths: np.ndarray, confidence: float = 0.95) -> Dict:
        final_prices = paths[:, -1]
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        stats = {
            'mean_final': np.mean(final_prices),
            'std_final': np.std(final_prices),
            'median_final': np.median(final_prices),
            'min_final': np.min(final_prices),
            'max_final': np.max(final_prices),
            f'ci_lower_{int(confidence*100)}': np.percentile(final_prices, lower_percentile),
            f'ci_upper_{int(confidence*100)}': np.percentile(final_prices, upper_percentile),
            'expected_return': np.mean(final_prices) / paths[0, 0] - 1,
            'volatility_realized': np.std(final_prices / paths[:, 0] - 1)
        }
        return stats


def create_gbm_model(
    historical_returns: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
    sigma: Optional[float] = None
) -> GeometricBrownianMotion:
    """Factory function to create and optionally calibrate a GBM model."""
    if historical_returns is not None:
        model = GeometricBrownianMotion()
        model.calibrate(historical_returns)
        if mu is not None:
            model.params.mu = mu
        if sigma is not None:
            model.params.sigma = sigma
    else:
        params = GBMParameters(mu=mu or 0.05, sigma=sigma or 0.15)
        model = GeometricBrownianMotion(params)
    return model


if __name__ == "__main__":
    print("Testing Geometric Brownian Motion Model")

    gbm = GeometricBrownianMotion()
    S0 = 2000.0
    n_days = 30
    n_paths = 1000

    paths = gbm.simulate(S0, n_days, n_paths, random_seed=42)
    stats = gbm.get_statistics(paths)

    print(f"\nSimulation Results ({n_paths} paths, {n_days} days):")
    print(f"Initial Price: ${S0:.2f}")
    print(f"Expected Final Price: ${stats['mean_final']:.2f}")
    print(f"Std Dev: ${stats['std_final']:.2f}")
    print(f"95% CI: [${stats['ci_lower_95']:.2f}, ${stats['ci_upper_95']:.2f}]")
    print(f"Expected Return: {stats['expected_return']*100:.2f}%")

    macro = ModelParameters(mu_adjusted=0.03, sigma_adjusted=0.22)
    macro_paths = gbm.simulate_with_macro(S0, n_days, n_paths, macro_params=macro, random_seed=42)
    macro_stats = gbm.get_statistics(macro_paths)
    print(f"\nMacro-adjusted Expected Final Price: ${macro_stats['mean_final']:.2f}")