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
from scipy import stats
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MertonParameters:
    """Parameters for Merton Jump Diffusion model."""
    mu: float = 0.05  # Drift (annualized)
    sigma: float = 0.15  # Diffusion volatility
    lambda_jump: float = 2.0  # Jump intensity (jumps per year)
    mu_jump: float = -0.05  # Mean jump size (log-normal)
    sigma_jump: float = 0.10  # Jump size volatility
    
    def validate(self):
        """Validate parameter values."""
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive")
        if self.lambda_jump < 0:
            raise ValueError(f"Jump intensity must be non-negative")
        if self.sigma_jump <= 0:
            raise ValueError(f"Jump volatility must be positive")
        return self


class MertonJumpModel:
    """
    Merton Jump Diffusion model for gold price simulation.
    
    Captures sudden price movements (jumps) that occur during
    geopolitical crises, central bank interventions, or market shocks.
    
    Attributes:
        params: MertonParameters instance
        dt: Time step size (in years)
    """
    
    def __init__(self, params: Optional[MertonParameters] = None, dt: float = 1/252):
        """
        Initialize Merton Jump model.
        
        Args:
            params: Model parameters
            dt: Time step in years
        """
        self.params = params or MertonParameters()
        self.params.validate()
        self.dt = dt
        
        logger.info(f"Merton initialized: mu={self.params.mu:.4f}, sigma={self.params.sigma:.4f}, lambda={self.params.lambda_jump:.2f}")
    
    def calibrate(self, returns: np.ndarray, threshold: float = 3.0) -> MertonParameters:
        """
        Calibrate parameters from historical returns.
        
        Uses method of moments to separate diffusion from jumps.
        
        Args:
            returns: Array of log returns
            threshold: Z-score threshold for identifying jumps
            
        Returns:
            Calibrated parameters
        """
        # Identify jumps as extreme returns
        z_scores = np.abs(stats.zscore(returns))
        jump_mask = z_scores > threshold
        
        n_jumps = np.sum(jump_mask)
        n_total = len(returns)
        
        # Jump intensity (annualized)
        lambda_jump = (n_jumps / n_total) / self.dt
        
        # Diffusion parameters (non-jump returns)
        diffusion_returns = returns[~jump_mask]
        mu = np.mean(diffusion_returns) * 252
        sigma = np.std(diffusion_returns) * np.sqrt(252)
        
        # Jump size parameters (from jump returns)
        if n_jumps > 5:
            jump_returns = returns[jump_mask]
            mu_jump = np.mean(jump_returns)
            sigma_jump = np.std(jump_returns)
        else:
            # Default values if not enough jumps detected
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
    
    def step(self, S: np.ndarray, Z_diffusion: np.ndarray, Z_jump: np.ndarray, jump_occurs: np.ndarray) -> np.ndarray:
        """
        Single step simulation with jumps.
        
        Args:
            S: Current price(s)
            Z_diffusion: Standard normal for diffusion
            Z_jump: Standard normal for jump size
            jump_occurs: Boolean array indicating if jump occurs
            
        Returns:
            Next price(s)
        """
        mu, sigma = self.params.mu, self.params.sigma
        mu_jump, sigma_jump = self.params.mu_jump, self.params.sigma_jump
        dt = self.dt
        
        # Diffusion component (GBM)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z_diffusion
        
        # Jump component
        # Jump size: log(J) ~ N(mu_jump, sigma_jump^2)
        jump_size = np.exp(mu_jump + sigma_jump * Z_jump) - 1
        jump_component = jump_occurs * jump_size
        
        # Combined: S_{t+1} = S_t * exp(diffusion) * (1 + jump)
        S_next = S * np.exp(drift + diffusion) * (1 + jump_component)
        
        return S_next
    
    def simulate(
        self,
        S0: float,
        n_steps: int,
        n_paths: int = 1000,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate multiple price paths with jumps.
        
        Args:
            S0: Initial price
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            random_seed: Random seed
            
        Returns:
            Array of shape (n_paths, n_steps) with simulated prices
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0
        
        # Pre-generate random numbers
        Z_diffusion = np.random.standard_normal((n_paths, n_steps - 1))
        Z_jump = np.random.standard_normal((n_paths, n_steps - 1))
        
        # Poisson process for jumps
        # P(jump in dt) = 1 - exp(-lambda * dt) ≈ lambda * dt for small dt
        jump_prob = 1 - np.exp(-self.params.lambda_jump * self.dt)
        jump_occurs = np.random.random((n_paths, n_steps - 1)) < jump_prob
        
        # Simulate
        for t in range(1, n_steps):
            paths[:, t] = self.step(
                paths[:, t-1],
                Z_diffusion[:, t-1],
                Z_jump[:, t-1],
                jump_occurs[:, t-1]
            )
        
        n_jumps_total = np.sum(jump_occurs)
        logger.info(f"Merton simulation complete: {n_paths} paths, {n_steps} steps, {n_jumps_total} jumps")
        return paths
    
    def simulate_crisis_scenario(
        self,
        S0: float,
        n_steps: int,
        crisis_intensity: float = 5.0,
        n_paths: int = 1000,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate crisis scenario with elevated jump intensity.
        
        Args:
            S0: Initial price
            n_steps: Number of steps
            crisis_intensity: Multiplier for jump intensity
            n_paths: Number of paths
            random_seed: Random seed
            
        Returns:
            Simulated paths
        """
        # Temporarily increase jump intensity
        original_lambda = self.params.lambda_jump
        self.params.lambda_jump *= crisis_intensity
        
        paths = self.simulate(S0, n_steps, n_paths, random_seed)
        
        # Restore original parameter
        self.params.lambda_jump = original_lambda
        
        return paths
    
    def get_statistics(self, paths: np.ndarray, confidence: float = 0.95) -> Dict:
        """
        Calculate statistics including jump analysis.
        
        Args:
            paths: Simulated paths
            confidence: Confidence level
            
        Returns:
            Dictionary with statistics
        """
        final_prices = paths[:, -1]
        returns = np.log(final_prices / paths[:, 0])
        
        alpha = 1 - confidence
        
        stats = {
            'mean_final': np.mean(final_prices),
            'std_final': np.std(final_prices),
            'median_final': np.median(final_prices),
            f'ci_lower_{int(confidence*100)}': np.percentile(final_prices, alpha/2 * 100),
            f'ci_upper_{int(confidence*100)}': np.percentile(final_prices, (1-alpha/2) * 100),
            'expected_return': np.mean(returns),
            'return_volatility': np.std(returns),
            'skewness': scipy_stats.skew(final_prices),
            'kurtosis': stats.kurtosis(final_prices),
            'prob_loss_10pct': np.mean(returns < -0.10),
            'prob_gain_10pct': np.mean(returns > 0.10),
            'max_drawdown_mean': np.mean(np.min(paths / np.maximum.accumulate(paths, axis=1) - 1, axis=1))
        }
        
        return stats


def create_merton_model(
    historical_returns: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    lambda_jump: Optional[float] = None,
    mu_jump: Optional[float] = None,
    sigma_jump: Optional[float] = None
) -> MertonJumpModel:
    """
    Factory function for Merton model.
    """
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
    
    # Create model with crisis-like parameters
    merton = MertonJumpModel(
        MertonParameters(
            mu=0.03,
            sigma=0.12,
            lambda_jump=3.0,  # 3 jumps per year on average
            mu_jump=-0.03,    # Slight negative bias
            sigma_jump=0.15   # Large jump volatility
        )
    )
    
    S0 = 2000.0
    n_days = 60
    n_paths = 1000
    
    # Normal simulation
    paths_normal = merton.simulate(S0, n_days, n_paths, random_seed=42)
    stats_normal = merton.get_statistics(paths_normal)
    
    # Crisis simulation
    paths_crisis = merton.simulate_crisis_scenario(S0, n_days, crisis_intensity=3.0, n_paths=n_paths, random_seed=42)
    stats_crisis = merton.get_statistics(paths_crisis)
    
    print(f"\nNormal Scenario:")
    print(f"Expected Final: ${stats_normal['mean_final']:.2f}")
    print(f"Volatility: {stats_normal['return_volatility']:.2%}")
    print(f"P(Loss > 10%): {stats_normal['prob_loss_10pct']:.2%}")
    
    print(f"\nCrisis Scenario (3x jump intensity):")
    print(f"Expected Final: ${stats_crisis['mean_final']:.2f}")
    print(f"Volatility: {stats_crisis['return_volatility']:.2%}")
    print(f"P(Loss > 10%): {stats_crisis['prob_loss_10pct']:.2%}")