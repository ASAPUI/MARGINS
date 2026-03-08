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
    mu: float = 0.05  # Drift (annualized expected return)
    sigma: float = 0.15  # Volatility (annualized)
    
    def validate(self):
        """Validate parameter values."""
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")
        return self


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion model for gold price simulation.
    
    This is the simplest stochastic model, serving as baseline for
    comparison with more sophisticated models.
    
    Attributes:
        params: GBMParameters instance with model parameters
        dt: Time step size (in years)
    """
    
    def __init__(self, params: Optional[GBMParameters] = None, dt: float = 1/252):
        """
        Initialize GBM model.
        
        Args:
            params: Model parameters (uses defaults if None)
            dt: Time step in years (default: 1/252 for daily)
        """
        self.params = params or GBMParameters()
        self.params.validate()
        self.dt = dt
        
        logger.info(f"GBM initialized: mu={self.params.mu:.4f}, sigma={self.params.sigma:.4f}")
    
    def calibrate(self, returns: np.ndarray) -> GBMParameters:
        """
        Calibrate parameters from historical returns.
        
        Args:
            returns: Array of log returns
            
        Returns:
            Calibrated parameters
        """
        # Annualized drift from mean return
        mu = np.mean(returns) * 252
        
        # Annualized volatility
        sigma = np.std(returns) * np.sqrt(252)
        
        self.params = GBMParameters(mu=mu, sigma=sigma)
        logger.info(f"GBM calibrated: mu={mu:.4f}, sigma={sigma:.4f}")
        return self.params
    
    def step(self, S: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Single step simulation using Euler-Maruyama discretization.
        
        Args:
            S: Current price(s)
            Z: Standard normal random variable(s)
            
        Returns:
            Next price(s)
        """
        mu, sigma = self.params.mu, self.params.sigma
        dt = self.dt
        
        # GBM discretization: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        S_next = S * np.exp(drift + diffusion)
        
        return S_next
    
    def simulate(
        self,
        S0: float,
        n_steps: int,
        n_paths: int = 1000,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate multiple price paths.
        
        Args:
            S0: Initial price
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            random_seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_paths, n_steps) with simulated prices
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize paths
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0
        
        # Generate random numbers
        Z = np.random.standard_normal((n_paths, n_steps - 1))
        
        # Simulate
        for t in range(1, n_steps):
            paths[:, t] = self.step(paths[:, t-1], Z[:, t-1])
        
        logger.info(f"GBM simulation complete: {n_paths} paths, {n_steps} steps")
        return paths
    
    def analytical_solution(
        self,
        S0: float,
        T: float,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate analytical solution statistics.
        
        Args:
            S0: Initial price
            T: Time horizon in years
            n_points: Number of time points
            
        Returns:
            Tuple of (times, mean_path, std_path)
        """
        times = np.linspace(0, T, n_points)
        mu, sigma = self.params.mu, self.params.sigma
        
        # E[S_t] = S0 * exp(mu * t)
        mean_path = S0 * np.exp(mu * times)
        
        # Std[S_t] = S0 * exp(mu * t) * sqrt(exp(sigma^2 * t) - 1)
        std_path = mean_path * np.sqrt(np.exp(sigma**2 * times) - 1)
        
        return times, mean_path, std_path
    
    def get_statistics(self, paths: np.ndarray, confidence: float = 0.95) -> Dict:
        """
        Calculate statistics from simulated paths.
        
        Args:
            paths: Simulated paths array (n_paths, n_steps)
            confidence: Confidence level for intervals
            
        Returns:
            Dictionary with statistics
        """
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
    """
    Factory function to create GBM model.
    
    Args:
        historical_returns: Returns for calibration
        mu: Manual drift override
        sigma: Manual volatility override
        
    Returns:
        Configured GBM model
    """
    if historical_returns is not None:
        model = GeometricBrownianMotion()
        model.calibrate(historical_returns)
        
        # Override if specified
        if mu is not None:
            model.params.mu = mu
        if sigma is not None:
            model.params.sigma = sigma
    else:
        params = GBMParameters(mu=mu or 0.05, sigma=sigma or 0.15)
        model = GeometricBrownianMotion(params)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Testing Geometric Brownian Motion Model")
    
    # Create model with default parameters
    gbm = GeometricBrownianMotion()
    
    # Simulate paths
    S0 = 2000.0  # Initial gold price
    n_days = 30
    n_paths = 1000
    
    paths = gbm.simulate(S0, n_days, n_paths, random_seed=42)
    
    # Calculate statistics
    stats = gbm.get_statistics(paths)
    
    print(f"\nSimulation Results ({n_paths} paths, {n_days} days):")
    print(f"Initial Price: ${S0:.2f}")
    print(f"Expected Final Price: ${stats['mean_final']:.2f}")
    print(f"Std Dev: ${stats['std_final']:.2f}")
    print(f"95% CI: [${stats['ci_lower_95']:.2f}, ${stats['ci_upper_95']:.2f}]")
    print(f"Expected Return: {stats['expected_return']*100:.2f}%")