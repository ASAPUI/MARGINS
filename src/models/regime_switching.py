"""
Regime Switching Model (Markov Chain)

Models different market states (calm vs crisis) with distinct parameters.
Captures structural changes in gold market behavior.

Author: Essabri ali rayan
Version: 1.3
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from scipy import stats
from enum import IntEnum
import logging
from dataclasses import dataclass
try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
logger = logging.getLogger(__name__)


class Regime(IntEnum):
    """Market regime states."""
    CALM = 0
    CRISIS = 1
    BUBBLE = 2  # Optional third state


@dataclass
class RegimeParameters:
    """Parameters for a single regime."""
    mu: float  # Drift
    sigma: float  # Volatility
    label: str  # Regime name


@dataclass
class RegimeSwitchingParameters:
    """Parameters for regime switching model."""
    regimes: Dict[Regime, RegimeParameters]
    transition_matrix: np.ndarray  # P[i,j] = prob of switching from i to j
    
    def validate(self):
        """Validate parameters."""
        # Check transition matrix is stochastic
        if not np.allclose(np.sum(self.transition_matrix, axis=1), 1.0):
            raise ValueError("Transition matrix rows must sum to 1")
        
        # Check all regimes have parameters
        for regime in Regime:
            if regime in self.regimes:
                if self.regimes[regime].sigma <= 0:
                    raise ValueError(f"Volatility must be positive for {regime}")
        
        return self
    
    def get_initial_probs(self) -> np.ndarray:
        """Get stationary distribution as initial probabilities."""
        # Solve for stationary distribution
        n = len(self.regimes)
        A = np.vstack([self.transition_matrix.T - np.eye(n), np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1
        
        try:
            stationary = np.linalg.lstsq(A, b, rcond=None)[0]
            return np.maximum(stationary, 0)  # Ensure non-negative
        except:
            # Fallback to uniform
            return np.ones(n) / n


class RegimeSwitchingModel:
    """
    Markov Regime Switching model for gold prices.
    
    Captures different market behaviors:
    - Calm regime: Low volatility, moderate drift
    - Crisis regime: High volatility, positive drift (safe haven)
    - Optional Bubble regime: High volatility, high positive drift
    
    Attributes:
        params: RegimeSwitchingParameters
        dt: Time step
    """
    
    def __init__(self, params: Optional[RegimeSwitchingParameters] = None, dt: float = 1/252):
        """
        Initialize regime switching model.
        
        Args:
            params: Model parameters
            dt: Time step in years
        """
        if params is None:
            # Default two-regime model
            regimes = {
                Regime.CALM: RegimeParameters(mu=0.03, sigma=0.12, label="Calm"),
                Regime.CRISIS: RegimeParameters(mu=0.08, sigma=0.25, label="Crisis")
            }
            # Transition: calm->crisis (5% prob), crisis->calm (20% prob)
            transition = np.array([
                [0.95, 0.05],  # From calm
                [0.20, 0.80]   # From crisis
            ])
            params = RegimeSwitchingParameters(regimes, transition)
        
        self.params = params
        self.params.validate()
        self.dt = dt
        
        logger.info(f"Regime model initialized with {len(params.regimes)} regimes")
    
        def calibrate(self, prices: np.ndarray, n_regimes: int = 2) -> RegimeSwitchingParameters:
        """
        Calibrate using HMM (hmmlearn) if available, else fall back to threshold method.
        """
        # Try hmmlearn first
        if _HMM_AVAILABLE:
            try:
                # Compute log returns
                log_returns = np.diff(np.log(prices))
                
                # Fit Gaussian HMM with 3 regimes
                model_hmm = GaussianHMM(
                    n_components=3, 
                    covariance_type='full', 
                    n_iter=100, 
                    random_state=42
                )
                model_hmm.fit(log_returns.reshape(-1, 1))
                
                # Store fitted parameters as requested
                self.means_ = model_hmm.means_.flatten()  # Daily drift per regime
                self.sigmas_ = np.sqrt([model_hmm.covars_[i][0][0] for i in range(3)])  # Daily vol per regime
                self.transmat_ = model_hmm.transmat_  # Transition matrix
                self.start_prob_ = model_hmm.startprob_  # Initial probabilities
                
                # Annualize parameters: mu * 252, sigma * sqrt(252)
                mus_annual = self.means_ * 252
                sigmas_annual = self.sigmas_ * np.sqrt(252)
                
                # Sort regimes by volatility (low to high) to map to CALM, CRISIS, BUBBLE
                order = np.argsort(sigmas_annual)
                regime_keys = [Regime.CALM, Regime.CRISIS, Regime.BUBBLE]
                
                regimes = {}
                for i, idx in enumerate(order):
                    key = regime_keys[i]
                    regimes[key] = RegimeParameters(
                        mu=float(mus_annual[idx]),
                        sigma=float(sigmas_annual[idx]),
                        label=key.name.capitalize()
                    )
                
                # Reorder transition matrix to match regime ordering
                transmat_ordered = self.transmat_[order, :][:, order]
                
                self.params = RegimeSwitchingParameters(regimes, transmat_ordered)
                logger.info(f"Regime model calibrated with HMM: "
                           f"calm_vol={regimes[Regime.CALM].sigma:.2%}, "
                           f"crisis_vol={regimes[Regime.CRISIS].sigma:.2%}")
                return self.params
                
            except Exception:
                # Silent fallback to threshold method
                pass
        
        # Fallback: Original threshold-based calibration
        import pandas as pd

        returns = np.diff(np.log(prices))

        # Calculate rolling volatility
        window = 20
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna().values

        if len(rolling_vol) < 50:
            logger.warning("Not enough data for regime calibration")
            return self.params

        # Identify regimes using volatility percentiles
        low_thresh = np.percentile(rolling_vol, 40)
        high_thresh = np.percentile(rolling_vol, 80)

        calm_mask = rolling_vol <= low_thresh
        crisis_mask = rolling_vol >= high_thresh

        # Calculate regime parameters
        calm_returns = returns[window-1:][calm_mask[:len(returns[window-1:])]]
        crisis_returns = returns[window-1:][crisis_mask[:len(returns[window-1:])]]

        regimes = {}
        if len(calm_returns) > 10:
            regimes[Regime.CALM] = RegimeParameters(
                mu=np.mean(calm_returns) * 252,
                sigma=np.std(calm_returns) * np.sqrt(252),
                label="Calm"
            )
        else:
            regimes[Regime.CALM] = RegimeParameters(mu=0.03, sigma=0.12, label="Calm")

        if len(crisis_returns) > 10:
            regimes[Regime.CRISIS] = RegimeParameters(
                mu=np.mean(crisis_returns) * 252,
                sigma=np.std(crisis_returns) * np.sqrt(252),
                label="Crisis"
            )
        else:
            regimes[Regime.CRISIS] = RegimeParameters(mu=0.08, sigma=0.25, label="Crisis")

        # Estimate transition matrix
        # Simple frequency-based estimation
        regime_series = np.where(rolling_vol > (low_thresh + high_thresh)/2, 1, 0)

        n_trans = len(regime_series) - 1
        trans_counts = np.zeros((2, 2))

        for i in range(n_trans):
            from_reg = int(regime_series[i])
            to_reg = int(regime_series[i+1])
            trans_counts[from_reg, to_reg] += 1

        # Normalize to probabilities
        trans_probs = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        trans_probs = np.nan_to_num(trans_probs, nan=0.5) # Handle division by zero

        # Ensure valid probabilities
        for i in range(2):
            if trans_probs[i].sum() == 0:
                trans_probs[i] = [0.9, 0.1] if i == 0 else [0.2, 0.8]
            trans_probs[i] = trans_probs[i] / trans_probs[i].sum()

        self.params = RegimeSwitchingParameters(regimes, trans_probs)
        logger.info(f"Regime model calibrated: calm_vol={regimes[Regime.CALM].sigma:.2%}, crisis_vol={regimes[Regime.CRISIS].sigma:.2%}")

        return self.params

    
    def step(
        self,
        S: np.ndarray,
        regime: np.ndarray,
        Z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single step simulation.
        
        Args:
            S: Current prices
            regime: Current regime for each path
            Z: Random normal variables
            
        Returns:
            Tuple of (next_prices, next_regimes)
        """
        dt = self.dt
        n_paths = len(S)
        
        # Get parameters for current regimes
        mu = np.array([self.params.regimes[r].mu for r in regime])
        sigma = np.array([self.params.regimes[r].sigma for r in regime])
        
        # GBM step with regime-specific parameters
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        S_next = S * np.exp(drift + diffusion)
        
        # Regime switching
        regime_next = np.zeros(n_paths, dtype=int)
        for i in range(n_paths):
            current_reg = regime[i]
            probs = self.params.transition_matrix[current_reg]
            regime_next[i] = np.random.choice(len(probs), p=probs)
        
        return S_next, regime_next
    
    def simulate(
        self,
        S0: float,
        n_steps: int,
        n_paths: int = 1000,
        initial_regime: Optional[Regime] = None,
        random_seed: Optional[int] = None,
        return_regimes: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate with regime switching.
        
        Args:
            S0: Initial price
            n_steps: Number of steps
            n_paths: Number of paths
            initial_regime: Starting regime (random if None)
            random_seed: Random seed
            return_regimes: Return regime history
            
        Returns:
            Price paths (and regime paths if requested)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize
        S = np.zeros((n_paths, n_steps))
        regimes = np.zeros((n_paths, n_steps), dtype=int)
        
        S[:, 0] = S0
        
        # Initialize regimes
        if initial_regime is not None:
            regimes[:, 0] = initial_regime
        else:
            initial_probs = self.params.get_initial_probs()
            regimes[:, 0] = np.random.choice(len(initial_probs), size=n_paths, p=initial_probs)
        
        # Generate random numbers
        Z = np.random.standard_normal((n_paths, n_steps - 1))
        
        # Simulate
        for t in range(1, n_steps):
            S[:, t], regimes[:, t] = self.step(S[:, t-1], regimes[:, t-1], Z[:, t-1])
        
        logger.info(f"Regime simulation complete: {n_paths} paths, {n_steps} steps")
        
        if return_regimes:
            return S, regimes
        return S
    
    def simulate_stress_scenario(
        self,
        S0: float,
        n_steps: int,
        n_paths: int = 1000,
        random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate stress scenario starting in crisis regime.
        
        Args:
            S0: Initial price
            n_steps: Number of steps
            n_paths: Number of paths
            random_seed: Random seed
            
        Returns:
            Simulated paths
        """
        return self.simulate(S0, n_steps, n_paths, initial_regime=Regime.CRISIS, random_seed=random_seed)
    
    def get_regime_statistics(
        self,
        regime_paths: np.ndarray
    ) -> Dict:
        """
        Calculate regime occupancy statistics.
        
        Args:
            regime_paths: Array of regime sequences
            
        Returns:
            Dictionary with regime statistics
        """
        n_paths, n_steps = regime_paths.shape
        
        stats = {}
        for regime in self.params.regimes.keys():
            regime_mask = regime_paths == regime
            occupancy = np.mean(regime_mask)
            avg_duration = np.mean([np.sum(path == regime) for path in regime_paths]) / n_steps
            
            stats[f'{self.params.regimes[regime].label}_occupancy'] = occupancy
            stats[f'{self.params.regimes[regime].label}_avg_duration'] = avg_duration
        
        return stats
    
    def get_statistics(
        self,
        paths: np.ndarray,
        regime_paths: Optional[np.ndarray] = None,
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate comprehensive statistics.
        
        Args:
            paths: Price paths
            regime_paths: Regime paths (optional)
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
            'prob_increase': np.mean(final_prices > paths[:, 0])
        }
        
        if regime_paths is not None:
            regime_stats = self.get_regime_statistics(regime_paths)
            stats_dict.update(regime_stats)
        
        return stats_dict


def create_regime_model(
    historical_prices: Optional[np.ndarray] = None,
    calm_mu: float = 0.03,
    calm_sigma: float = 0.12,
    crisis_mu: float = 0.08,
    crisis_sigma: float = 0.25,
    p_calm_to_crisis: float = 0.05,
    p_crisis_to_calm: float = 0.20
) -> RegimeSwitchingModel:
    """
    Factory function for regime switching model.
    """
    if historical_prices is not None:
        model = RegimeSwitchingModel()
        model.calibrate(historical_prices)
    else:
        regimes = {
            Regime.CALM: RegimeParameters(mu=calm_mu, sigma=calm_sigma, label="Calm"),
            Regime.CRISIS: RegimeParameters(mu=crisis_mu, sigma=crisis_sigma, label="Crisis")
        }
        transition = np.array([
            [1 - p_calm_to_crisis, p_calm_to_crisis],
            [p_crisis_to_calm, 1 - p_crisis_to_calm]
        ])
        params = RegimeSwitchingParameters(regimes, transition)
        model = RegimeSwitchingModel(params)
    
    return model


if __name__ == "__main__":
    print("Testing Regime Switching Model")
    
    # Create model
    regime_model = create_regime_model(
        calm_mu=0.02,
        calm_sigma=0.10,
        crisis_mu=0.10,
        crisis_sigma=0.30,
        p_calm_to_crisis=0.03,
        p_crisis_to_calm=0.15
    )
    
    print(f"Transition Matrix:")
    print(regime_model.params.transition_matrix)
    
    S0 = 2000.0
    n_days = 252  # 1 year
    n_paths = 1000
    
    # Normal simulation
    paths, regimes = regime_model.simulate(S0, n_days, n_paths, random_seed=42, return_regimes=True)
    stats = regime_model.get_statistics(paths, regimes)
    
    print(f"\nSimulation Results ({n_paths} paths, {n_days} days):")
    print(f"Expected Final: ${stats['mean_final']:.2f}")
    print(f"Std Dev: ${stats['std_final']:.2f}")
    print(f"95% CI: [${stats['ci_lower_95']:.2f}, ${stats['ci_upper_95']:.2f}]")
    print(f"\nRegime Statistics:")
    print(f"Calm occupancy: {stats.get('Calm_occupancy', 0):.2%}")
    print(f"Crisis occupancy: {stats.get('Crisis_occupancy', 0):.2%}")
    
    # Stress test
    stress_paths = regime_model.simulate_stress_scenario(S0, n_days, n_paths, random_seed=42)
    stress_stats = regime_model.get_statistics(stress_paths)
    print(f"\nStress Scenario (starting in crisis):")
    print(f"Expected Final: ${stress_stats['mean_final']:.2f}")
    print(f"Std Dev: ${stress_stats['std_final']:.2f}")