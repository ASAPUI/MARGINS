"""
Ornstein-Uhlenbeck (Mean Reversion) Model

The most important model for gold price prediction.
Captures gold's tendency to revert to long-term equilibrium.

Formula: dS = kappa * (theta - S) * dt + sigma * dW

Author: Essabri Ali rayan
Version: 1.3
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy import stats
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OUParameters:
    """Parameters for Ornstein-Uhlenbeck process."""
    kappa: float = 2.0  # Speed of mean reversion
    theta: float = 1800.0  # Long-term mean (equilibrium price)
    sigma: float = 100.0  # Volatility (absolute, not percentage)
    
    def validate(self):
        """Validate parameter values."""
        if self.kappa <= 0:
            raise ValueError(f"Mean reversion speed must be positive, got {self.kappa}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")
        return self


class OrnsteinUhlenbeckModel:
    """
    Ornstein-Uhlenbeck model for gold price simulation with mean reversion.
    
    This is the most appropriate single-factor model for gold due to
    gold's documented tendency to revert to long-term fair value.
    
    Attributes:
        params: OUParameters instance
        dt: Time step size (in years)
    """
    
    def __init__(self, params: Optional[OUParameters] = None, dt: float = 1/252):
        """
        Initialize OU model.
        
        Args:
            params: Model parameters (uses defaults if None)
            dt: Time step in years (default: 1/252 for daily)
        """
        self.params = params or OUParameters()
        self.params.validate()
        self.dt = dt
        
        logger.info(f"OU initialized: kappa={self.params.kappa:.4f}, theta={self.params.theta:.2f}, sigma={self.params.sigma:.2f}")
    
        def calibrate(self, prices: np.ndarray) -> OUParameters:
        """
        Calibrate parameters from historical prices using MLE/OLS on AR(1) representation.
        
        Discrete-time OU equivalent: S[t] = theta*(1 - e^(-kappa*dt)) + S[t-1]*e^(-kappa*dt) + noise
        We regress S[t] on S[t-1] to get slope = e^(-kappa*dt) and intercept = theta*(1 - slope).
        
        Args:
            prices: Array of historical prices (levels)
            
        Returns:
            Calibrated parameters
        """
        # Prepare data: S[t] vs S[t-1]
        S_t = prices[1:]   # Current prices
        S_tm1 = prices[:-1]  # Lagged prices
        
        # OLS regression: S[t] = intercept + slope * S[t-1] + error
        # Using numpy lstsq for consistency with existing code style
        X = np.column_stack([np.ones(len(S_tm1)), S_tm1])
        
        # Solve for [intercept, slope]
        beta = np.linalg.lstsq(X, S_t, rcond=None)[0]
        intercept, slope = beta[0], beta[1]
        
        # Extract OU parameters from AR(1) coefficients
        # slope = exp(-kappa * dt)  =>  kappa = -ln(slope) / dt = -ln(slope) * 252
        if slope <= 0:
            # If slope is non-positive, process is not mean-reverting (explosive or non-stationary)
            # Default to slow mean reversion
            kappa = 0.5
        else:
            kappa = -np.log(slope) / self.dt  # self.dt = 1/252, so this equals -np.log(slope) * 252
            
        # Ensure kappa is positive (mean reversion requires kappa > 0)
        if kappa <= 0:
            kappa = 0.5
            
        # theta = intercept / (1 - slope)
        # As slope -> 1 (random walk), theta -> infinity, so handle near-unit-root carefully
        if abs(1 - slope) < 1e-6:
            # Near unit root - use mean of prices as theta
            theta = np.mean(prices)
        else:
            theta = intercept / (1 - slope)
        
        # sigma = std(residuals) * sqrt(252)
        residuals = S_t - (intercept + slope * S_tm1)
        sigma = np.std(residuals) * np.sqrt(252)
        
        # Ensure reasonable bounds
        kappa = np.clip(kappa, 0.1, 10.0)
        sigma = max(sigma, 1.0)
        theta = max(theta, 0.01)  # Ensure positive theta
        
        self.params = OUParameters(kappa=kappa, theta=theta, sigma=sigma)
        logger.info(f"OU calibrated: kappa={kappa:.4f}, theta={theta:.2f}, sigma={sigma:.2f}")
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
        kappa, theta, sigma = self.params.kappa, self.params.theta, self.params.sigma
        dt = self.dt
        
        # OU discretization
        drift = kappa * (theta - S) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        S_next = S + drift + diffusion
        
        # Ensure non-negative prices
        S_next = np.maximum(S_next, 0.01)
        
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
        
        logger.info(f"OU simulation complete: {n_paths} paths, {n_steps} steps")
        return paths
    
    def analytical_solution(
        self,
        S0: float,
        T: float,
        n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate analytical solution statistics.
        
        For OU process:
        E[S_t] = theta + (S0 - theta) * exp(-kappa * t)
        Var[S_t] = sigma^2 / (2*kappa) * (1 - exp(-2*kappa * t))
        
        Args:
            S0: Initial price
            T: Time horizon in years
            n_points: Number of time points
            
        Returns:
            Tuple of (times, mean_path, std_path)
        """
        times = np.linspace(0, T, n_points)
        kappa, theta, sigma = self.params.kappa, self.params.theta, self.params.sigma
        
        # Mean
        mean_path = theta + (S0 - theta) * np.exp(-kappa * times)
        
        # Standard deviation
        variance = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * times))
        std_path = np.sqrt(variance)
        
        return times, mean_path, std_path
    
    def half_life(self) -> float:
        """
        Calculate half-life of mean reversion.
        
        Returns:
            Half-life in years
        """
        return np.log(2) / self.params.kappa
    
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
        
        # Calculate probability of reverting toward theta
        S0 = paths[0, 0]
        theta = self.params.theta
        
        if S0 > theta:
            prob_revert = np.mean(final_prices < S0)
        else:
            prob_revert = np.mean(final_prices > S0)
        
        stats = {
            'mean_final': np.mean(final_prices),
            'std_final': np.std(final_prices),
            'median_final': np.median(final_prices),
            'min_final': np.min(final_prices),
            'max_final': np.max(final_prices),
            f'ci_lower_{int(confidence*100)}': np.percentile(final_prices, lower_percentile),
            f'ci_upper_{int(confidence*100)}': np.percentile(final_prices, upper_percentile),
            'prob_revert_to_mean': prob_revert,
            'half_life_years': self.half_life(),
            'distance_to_equilibrium': (S0 - theta) / theta
        }
        
        return stats


def create_ou_model(
    historical_prices: Optional[np.ndarray] = None,
    kappa: Optional[float] = None,
    theta: Optional[float] = None,
    sigma: Optional[float] = None
) -> OrnsteinUhlenbeckModel:
    """
    Factory function to create OU model.
    
    Args:
        historical_prices: Prices for calibration
        kappa: Manual mean reversion speed override
        theta: Manual long-term mean override
        sigma: Manual volatility override
        
    Returns:
        Configured OU model
    """
    if historical_prices is not None:
        model = OrnsteinUhlenbeckModel()
        model.calibrate(historical_prices)
        
        # Override if specified
        if kappa is not None:
            model.params.kappa = kappa
        if theta is not None:
            model.params.theta = theta
        if sigma is not None:
            model.params.sigma = sigma
    else:
        params = OUParameters(
            kappa=kappa or 2.0,
            theta=theta or 1800.0,
            sigma=sigma or 100.0
        )
        model = OrnsteinUhlenbeckModel(params)
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Testing Ornstein-Uhlenbeck Model")
    
    # Create model
    ou = OrnsteinUhlenbeckModel(
        OUParameters(kappa=3.0, theta=2000.0, sigma=80.0)
    )
    
    print(f"Half-life: {ou.half_life():.2f} years")
    
    # Simulate
    S0 = 2200.0  # Above equilibrium
    n_days = 90
    n_paths = 1000
    
    paths = ou.simulate(S0, n_days, n_paths, random_seed=42)
    
    # Statistics
    stats = ou.get_statistics(paths)
    
    print(f"\nSimulation Results ({n_paths} paths, {n_days} days):")
    print(f"Initial Price: ${S0:.2f}")
    print(f"Equilibrium (theta): ${ou.params.theta:.2f}")
    print(f"Expected Final Price: ${stats['mean_final']:.2f}")
    print(f"Prob. of Reverting: {stats['prob_revert_to_mean']:.2%}")
    print(f"95% CI: [${stats['ci_lower_95']:.2f}, ${stats['ci_upper_95']:.2f}]")