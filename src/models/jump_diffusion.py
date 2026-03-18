"""
Merton Jump Diffusion Model

Extends GBM with jumps to capture sudden price movements during crises.
Useful for modeling tail risk and crisis scenarios in gold prices.

Formula: dS = mu*S*dt + sigma*S*dW + J*S*dN

Author: Essabri Ali rayan
Version: 1.0
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy import stats as scipy_stats
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MertonParameters:
    """Parameters for Merton Jump Diffusion model."""
    mu: float = 0.05
    sigma: float = 0.15
    lambda_jump: float = 2.0
    mu_jump: float = -0.05
    sigma_jump: float = 0.10

    def validate(self):
        if self.sigma <= 0:
            raise ValueError("Volatility must be positive")
        if self.lambda_jump < 0:
            raise ValueError("Jump intensity must be non-negative")
        if self.sigma_jump <= 0:
            raise ValueError("Jump volatility must be positive")
        return self


@dataclass
class ModelParameters:
    """Macro-adjusted model parameters produced by ParameterAdjuster."""
    mu_adjusted: float = 0.05
    sigma_adjusted: float = 0.15
    lambda_boost: float = 1.0          # Multiplier on lambda_jump
    mu_j_adjusted: Optional[float] = None   # Override jump mean (None = keep calibrated)
    sigma_j_adjusted: Optional[float] = None  # Override jump sigma (None = keep calibrated)
    regime_crisis_prior: float = 0.2


class MertonJumpModel:
    """
    Merton Jump Diffusion model for gold price simulation.

    Captures sudden price movements (jumps) that occur during
    geopolitical crises, central bank interventions, or market shocks.
    """

    def __init__(self, params: Optional[MertonParameters] = None, dt: float = 1/252):
        self.params = params or MertonParameters()
        self.params.validate()
        self.dt = dt
        logger.info(
            f"Merton initialized: mu={self.params.mu:.4f}, "
            f"sigma={self.params.sigma:.4f}, lambda={self.params.lambda_jump:.2f}"
        )

    def calibrate(self, returns: np.ndarray, threshold: float = 3.0) -> MertonParameters:
        z_scores = np.abs(scipy_stats.zscore(returns))
        jump_mask = z_scores > threshold

        n_jumps = np.sum(jump_mask)
        n_total = len(returns)

        lambda_jump = (n_jumps / n_total) / self.dt

        diffusion_returns = returns[~jump_mask]
        mu = np.mean(diffusion_returns) * 252
        sigma = np.std(diffusion_returns) * np.sqrt(252)

        if n_jumps > 5:
            jump_returns = returns[jump_mask]
            mu_jump = np.mean(jump_returns)
            sigma_jump = np.std(jump_returns)
        else:
            mu_jump = -0.05
            sigma_jump = 0.10

        self.params = MertonParameters(
            mu=mu,
            sigma=sigma,
            lambda_jump=lambda_jump,
            mu_jump=mu_jump,
            sigma_jump=sigma_jump
        )
        logger.info(f"Merton calibrated: lambda={lambda_jump:.2f}, mu_jump={mu_jump:.4f}")
        return self.params

    def step(
        self,
        S: np.ndarray,
        Z_diffusion: np.ndarray,
        Z_jump: np.ndarray,
        jump_occurs: np.ndarray
    ) -> np.ndarray:
        mu, sigma = self.params.mu, self.params.sigma
        mu_jump, sigma_jump = self.params.mu_jump, self.params.sigma_jump
        dt = self.dt

        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z_diffusion

        jump_size = np.exp(mu_jump + sigma_jump * Z_jump) - 1
        jump_component = jump_occurs * jump_size

        return S * np.exp(drift + diffusion) * (1 + jump_component)

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

        Z_diffusion = np.random.standard_normal((n_paths, n_steps - 1))
        Z_jump = np.random.standard_normal((n_paths, n_steps - 1))

        jump_prob = 1 - np.exp(-self.params.lambda_jump * self.dt)
        jump_occurs = np.random.random((n_paths, n_steps - 1)) < jump_prob

        for t in range(1, n_steps):
            paths[:, t] = self.step(
                paths[:, t-1],
                Z_diffusion[:, t-1],
                Z_jump[:, t-1],
                jump_occurs[:, t-1]
            )

        logger.info(
            f"Merton simulation complete: {n_paths} paths, "
            f"{n_steps} steps, {np.sum(jump_occurs)} jumps"
        )
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
        Run Merton simulation with macro-adjusted jump parameters.

        Critical integration point — jump models are most sensitive to macro shocks.
        Applies WorldMonitor-derived lambda boost, mu/sigma adjustments,
        then restores original params after the run.
        """
        # Stash originals
        orig_mu = self.params.mu
        orig_sigma = self.params.sigma
        orig_lambda = self.params.lambda_jump
        orig_mu_jump = self.params.mu_jump
        orig_sigma_jump = self.params.sigma_jump

        if macro_params is not None:
            self.params.mu = macro_params.mu_adjusted
            self.params.sigma = macro_params.sigma_adjusted
            self.params.lambda_jump = orig_lambda * macro_params.lambda_boost
            if macro_params.mu_j_adjusted is not None:
                self.params.mu_jump = macro_params.mu_j_adjusted
            if macro_params.sigma_j_adjusted is not None:
                self.params.sigma_jump = macro_params.sigma_j_adjusted
            logger.info(
                f"Merton macro-adjusted: lambda={self.params.lambda_jump:.4f}, "
                f"mu_j={self.params.mu_jump:.4f}, sigma_j={self.params.sigma_jump:.4f}"
            )

        try:
            paths = self.simulate(S0, n_steps, n_paths, random_seed)
        finally:
            # Always restore
            self.params.mu = orig_mu
            self.params.sigma = orig_sigma
            self.params.lambda_jump = orig_lambda
            self.params.mu_jump = orig_mu_jump
            self.params.sigma_jump = orig_sigma_jump

        return paths

    def simulate_crisis_scenario(
        self,
        S0: float,
        n_steps: int,
        crisis_intensity: float = 5.0,
        n_paths: int = 1000,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        original_lambda = self.params.lambda_jump
        self.params.lambda_jump *= crisis_intensity
        try:
            paths = self.simulate(S0, n_steps, n_paths, random_seed)
        finally:
            self.params.lambda_jump = original_lambda
        return paths

    def get_statistics(self, paths: np.ndarray, confidence: float = 0.95) -> Dict:
        final_prices = paths[:, -1]
        returns = np.log(final_prices / paths[:, 0])

        alpha = 1 - confidence

        statistics = {
            'mean_final': np.mean(final_prices),
            'std_final': np.std(final_prices),
            'median_final': np.median(final_prices),
            f'ci_lower_{int(confidence*100)}': np.percentile(final_prices, alpha/2 * 100),
            f'ci_upper_{int(confidence*100)}': np.percentile(final_prices, (1-alpha/2) * 100),
            'expected_return': np.mean(returns),
            'return_volatility': np.std(returns),
            'skewness': scipy_stats.skew(final_prices),
            'kurtosis': scipy_stats.kurtosis(final_prices),
            'prob_loss_10pct': np.mean(returns < -0.10),
            'prob_gain_10pct': np.mean(returns > 0.10),
            'max_drawdown_mean': np.mean(
                np.min(paths / np.maximum.accumulate(paths, axis=1) - 1, axis=1)
            )
        }

        return statistics


def create_merton_model(
    historical_returns: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    lambda_jump: Optional[float] = None,
    mu_jump: Optional[float] = None,
    sigma_jump: Optional[float] = None
) -> MertonJumpModel:
    """Factory function for Merton model."""
    if historical_returns is not None:
        model = MertonJumpModel()
        model.calibrate(historical_returns)
        if mu is not None:
            model.params.mu = mu
        if sigma is not None:
            model.params.sigma = sigma
        if lambda_jump is not None:
            model.params.lambda_jump = lambda_jump
        if mu_jump is not None:
            model.params.mu_jump = mu_jump
        if sigma_jump is not None:
            model.params.sigma_jump = sigma_jump
    else:
        params = MertonParameters(
            mu=mu or 0.05,
            sigma=sigma or 0.15,
            lambda_jump=lambda_jump or 2.0,
            mu_jump=mu_jump or -0.05,
            sigma_jump=sigma_jump or 0.10
        )
        model = MertonJumpModel(params)

    return model


if __name__ == "__main__":
    print("Testing Merton Jump Diffusion Model")

    merton = MertonJumpModel(
        MertonParameters(
            mu=0.03,
            sigma=0.12,
            lambda_jump=3.0,
            mu_jump=-0.03,
            sigma_jump=0.15
        )
    )

    S0 = 2000.0
    n_days = 60
    n_paths = 1000

    paths_normal = merton.simulate(S0, n_days, n_paths, random_seed=42)
    stats_normal = merton.get_statistics(paths_normal)

    paths_crisis = merton.simulate_crisis_scenario(
        S0, n_days, crisis_intensity=3.0, n_paths=n_paths, random_seed=42
    )
    stats_crisis = merton.get_statistics(paths_crisis)

    print(f"\nNormal Scenario:")
    print(f"Expected Final: ${stats_normal['mean_final']:.2f}")
    print(f"Volatility: {stats_normal['return_volatility']:.2%}")
    print(f"P(Loss > 10%): {stats_normal['prob_loss_10pct']:.2%}")

    print(f"\nCrisis Scenario (3x jump intensity):")
    print(f"Expected Final: ${stats_crisis['mean_final']:.2f}")
    print(f"Volatility: {stats_crisis['return_volatility']:.2%}")
    print(f"P(Loss > 10%): {stats_crisis['prob_loss_10pct']:.2%}")

    macro = ModelParameters(mu_adjusted=0.02, sigma_adjusted=0.20, lambda_boost=2.5)
    paths_macro = merton.simulate_with_macro(S0, n_days, n_paths, macro_params=macro, random_seed=42)
    stats_macro = merton.get_statistics(paths_macro)
    print(f"\nMacro-adjusted Scenario:")
    print(f"Expected Final: ${stats_macro['mean_final']:.2f}")
    print(f"P(Loss > 10%): {stats_macro['prob_loss_10pct']:.2%}")