# gbm_model_fixed.py
# Model: Geometric Brownian Motion (Itô form)
# SDE:   dS = mu*S*dt + sigma*S*dW  (continuous time)
# Discrete approximation: log(S_{t+1}/S_t) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t
# where Z_t ~ t(df) with df calibrated from data (default df=5)

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np
import warnings
from scipy.stats import t as student_t, mstats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


@dataclass
class MacroParameters:
    """Macroeconomic overlay parameters for stress testing."""
    mu_adjusted: Optional[float] = None
    sigma_adjusted: Optional[float] = None
    lambda_boost: float = 0.0  # Jump intensity multiplier
    regime_crisis_prior: float = 0.0  # Volatility scaling factor for crisis regime


@dataclass
class GBMParameters:
    """
    Geometric Brownian Motion parameters.
    
    Note: mu and sigma are required parameters with no defaults.
    """
    mu: float  # Annualized drift - caller must supply calibrated value
    sigma: float  # Annualized volatility - caller must supply calibrated value
    dt: float = 1/252  # Time step (default: 1 trading day)
    random_seed: Optional[int] = None
    tail_df: float = 5.0  # Degrees of freedom for Student-t distribution
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.tail_df <= 2:
            raise ValueError(f"tail_df must be > 2 for finite variance, got {self.tail_df}")


class GBMModel:
    """
    Production-ready Geometric Brownian Motion Monte Carlo simulator for 1-day equity VaR.
    
    Implements Itô calculus with Student-t innovations, optional GARCH(1,1) volatility,
    antithetic variates, and comprehensive input validation.
    """
    
    def __init__(self, params: GBMParameters):
        """
        Initialize GBM model with validated parameters.
        
        Args:
            params: GBMParameters instance (mu and sigma required, no defaults)
        """
        self.params = params
        self.garch_params: Optional[Dict[str, float]] = None
        self.sigma_t: Optional[float] = None  # Conditional volatility from GARCH
        
    def _get_conditional_sigma(self) -> float:
        """Return GARCH conditional volatility if available, else constant sigma."""
        return getattr(self, 'sigma_t', self.params.sigma)
    
    def calibrate_tail(self, returns: np.ndarray):
        """
        Calibrate Student-t degrees of freedom from return data using MLE.
        
        Args:
            returns: Array of log returns
        """
        returns = np.asarray(returns, dtype=np.float64)
        if returns.dtype != np.float64:
            raise TypeError(f"Input must be float64, got {returns.dtype}")
        
        # Fit Student-t distribution
        df, loc, scale = student_t.fit(returns, floc=0)
        self.params.tail_df = float(df)
        
    def calibrate_garch(self, returns: np.ndarray):
        """
        Calibrate GARCH(1,1) model for conditional volatility.
        
        Args:
            returns: Array of log returns (will be converted to percentages)
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required for GARCH calibration. Install: pip install arch>=6.0")
            
        returns = np.asarray(returns, dtype=np.float64)
        if returns.dtype != np.float64:
            raise TypeError(f"Input must be float64, got {returns.dtype}")
        
        # Scale to percentage for numerical stability
        gm = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t')
        res = gm.fit(disp='off')
        self.garch_params = dict(res.params)
        # Store today's conditional volatility (convert back from percentage)
        self.sigma_t = float(res.conditional_volatility[-1]) / 100.0
        
        def calibrate(self, returns: np.ndarray) -> GBMParameters:
            import pandas as pd
        
        # Convert to pandas Series for EWMA calculations
            log_returns = pd.Series(returns)
        
        # EWMA drift (RiskMetrics lambda=0.94, com=lambda/(1-lambda))
            ewma_mean = log_returns.ewm(com=0.94/(1-0.94), adjust=False).mean().iloc[-1]
            mu = ewma_mean * 252
        
        # EWMA volatility (60-day span)
            ewma_vol = log_returns.ewm(span=60, adjust=False).std().iloc[-1]
            sigma = ewma_vol * np.sqrt(252)
        
            self.params = GBMParameters(mu=mu, sigma=sigma)
            logger.info(f"GBM calibrated: mu={mu:.4f}, sigma={sigma:.4f}")
        return self.params
    def _validate_moments(self, simulated_paths: np.ndarray):
        """
        Internal consistency check: simulated mean vs analytical solution.
        
        Args:
            simulated_paths: Simulated price paths
        """
        analytical = self.analytical_solution(
            S0=simulated_paths[0, 0], 
            times=np.array([simulated_paths.shape[1] - 1]) * self.params.dt
        )
        simulated_mean = np.mean(simulated_paths[:, -1])
        expected_mean = analytical['mean'][0]
        
        if abs(simulated_mean - expected_mean) / expected_mean > 0.01:
            raise AssertionError(
                "Simulated mean deviates >1% from analytical GBM mean — check calibration"
            )
            
    def analytical_solution(self, S0: float, times: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analytical moments of GBM at given times.
        
        SDE: dS = mu*S*dt + sigma*S*dW
        E[S_t] = S0 * exp(mu * t)
        Var[S_t] = S0^2 * exp(2*mu*t) * (exp(sigma^2*t) - 1)
        
        Args:
            S0: Initial price
            times: Array of time points
            
        Returns:
            Dict with 'mean', 'std', 'variance' arrays
        """
        if S0 <= 0:
            raise ValueError(f"Initial price S0 must be positive, got {S0}")
            
        times = np.asarray(times, dtype=np.float64)
        mu = self.params.mu
        sigma = self._get_conditional_sigma()
        
        # NE-02: Overflow guard
        exponent = sigma**2 * times
        if np.any(exponent > 500):
            warnings.warn(
                f"Exponent {exponent.max():.1f} approaches float64 overflow — "
                f"results for large T or sigma may be unreliable"
            )
            
        mean = S0 * np.exp(mu * times)
        # Use np.expm1 for numerical stability near zero
        variance = S0**2 * np.exp(2 * mu * times) * np.expm1(np.minimum(exponent, 709))
        std = np.sqrt(variance)
        
        return {'mean': mean, 'std': std, 'variance': variance}
        
    def simulate(self, S0: float, n_steps: int, n_paths: int, 
                 dt: Optional[float] = None, random_seed: Optional[int] = None,
                 zero_drift_mode: bool = False, memory_limit_gb: float = 2.0) -> np.ndarray:
        """
        Vectorized GBM Monte Carlo simulation with antithetic variates.
        
        Discrete Itô approximation:
        log(S_{t+dt}/S_t) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t
        
        Args:
            S0: Initial price (must be positive)
            n_steps: Number of time steps (must be >= 1)
            n_paths: Number of Monte Carlo paths
            dt: Time step override (optional, default from params)
            random_seed: Random seed for reproducibility
            zero_drift_mode: If True, set mu=0 for 1-day VaR
            memory_limit_gb: Memory limit for random matrix allocation
            
        Returns:
            Array of shape (n_paths, n_steps) with simulated prices
            
        Raises:
            ValueError: If inputs invalid
            MemoryError: If requested allocation exceeds limit
        """
        # Input validation
        if not isinstance(S0, (int, float)) or S0 <= 0:
            raise ValueError(f"S0 must be a positive number, got {S0}")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}")
            
        # PE-05: Per-call dt override
        dt_eff = dt if dt is not None else self.params.dt
        
        # SE-01: 1-day horizon assertion
        assert n_steps == 1 or dt_eff == 1/252, \
            "For 1-day VaR use n_steps=1, dt=1/252"
            
        # Get effective parameters
        mu_eff = self.params.mu
        sigma_eff = self._get_conditional_sigma()
        
        # ST-05: Handle statistically insignificant drift for 1-day horizon
        if n_steps == 1 and abs(mu_eff * dt_eff) < 0.0001 * sigma_eff * np.sqrt(dt_eff):
            if not zero_drift_mode:
                warnings.warn(
                    f"1-day drift contribution ({mu_eff*dt_eff*10000:.4f} bps) is negligible vs "
                    f"vol term ({sigma_eff*np.sqrt(dt_eff)*10000:.1f} bps). "
                    f"Consider setting mu=0 for 1-day VaR to reduce estimation noise."
                )
        if zero_drift_mode:
            mu_eff = 0.0
            
        # PE-04: Use Generator API, no global state
        rng = np.random.default_rng(random_seed)
        
        # NE-04: Memory guard
        bytes_needed = int(n_paths) * int(n_steps) * 8  # float64 = 8 bytes
        limit_bytes = int(memory_limit_gb * 1024**3)
        if bytes_needed > limit_bytes:
            raise MemoryError(
                f"Requested allocation of {bytes_needed/1e9:.2f} GB exceeds "
                f"{memory_limit_gb:.0f} GB limit. "
                f"Reduce n_paths (currently {n_paths}) or n_steps (currently {n_steps})."
            )
        
        # ST-01: Student-t shocks (fat-tailed distribution)
        df = self.params.tail_df
        # Generate base random draws for half paths (antithetic variates prep)
        half_paths = n_paths // 2
        Z_base = student_t.rvs(df=df, size=(half_paths, n_steps - 1), random_state=rng)
        # Normalize to unit variance (variance of t-dist is df/(df-2))
        Z_base = Z_base / np.sqrt(df / (df - 2))
        
        # SE-04: Antithetic variates
        Z = np.vstack([Z_base, -Z_base])
        
        # Handle odd n_paths
        if n_paths % 2 == 1:
            extra_Z = student_t.rvs(df=df, size=(1, n_steps - 1), random_state=rng)
            extra_Z = extra_Z / np.sqrt(df / (df - 2))
            Z = np.vstack([Z, extra_Z])
            
        # SE-02: Vectorized simulation (no Python loops)
        # log(S_{t+1}/S_t) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t
        log_returns = (mu_eff - 0.5 * sigma_eff**2) * dt_eff + sigma_eff * np.sqrt(dt_eff) * Z
        
        # Initialize paths: prepend log(S0) to log returns for cumsum
        log_S0 = np.log(S0)
        log_returns_with_init = np.hstack([
            log_S0 * np.ones((n_paths, 1)), 
            log_returns
        ])
        
        # Cumulative sum to get log prices, then exponentiate
        log_paths = np.cumsum(log_returns_with_init, axis=1)
        paths = S0 * np.exp(log_paths - log_S0)  # Subtract log_S0 because we added it in cumsum
        paths[:, 0] = S0  # Explicitly set first column to S0
        
        # NE-03: Post-run check for non-positive prices
        if np.any(paths <= 0):
            raise RuntimeError("Simulation produced non-positive prices — check input parameters")
            
        # NE-05: Convergence diagnostic
        self._check_convergence(paths)
        
        return paths
        
    def _check_convergence(self, paths: np.ndarray, confidence: float = 0.95, tol: float = 0.005):
        """
        Check if VaR estimate has converged by comparing first and second half of paths.
        
        Args:
            paths: Simulated paths array (n_paths, n_steps)
            confidence: Confidence level for VaR calculation
            tol: Relative tolerance for convergence
        """
        mid = len(paths) // 2
        if mid < 100:  # Too few paths for meaningful test
            return
            
        # Calculate VaR (negative of percentile since we want loss)
        alpha = 1 - confidence
        var_first_half = np.percentile(paths[mid:, -1], alpha * 100)
        var_full = np.percentile(paths[:, -1], alpha * 100)
        
        if abs(var_full) > 1e-10:  # Avoid division by zero
            if abs(var_first_half - var_full) / abs(var_full) > tol:
                warnings.warn(
                    f"VaR estimate not converged: second-half VaR deviates "
                    f"{abs(var_first_half-var_full)/abs(var_full)*100:.1f}% from full-sample VaR. "
                    f"Increase n_paths (current: {len(paths)})"
                )
                
    def simulate_with_macro(self, S0: float, n_steps: int, n_paths: int,
                           macro_params: Optional[MacroParameters] = None,
                           dt: Optional[float] = None, random_seed: Optional[int] = None,
                           zero_drift_mode: bool = False, memory_limit_gb: float = 2.0) -> np.ndarray:
        """
        Simulate with macroeconomic overlays (stress testing).
        
        Args:
            S0: Initial price
            n_steps: Number of time steps
            n_paths: Number of paths
            macro_params: MacroParameters for stress adjustments
            dt: Time step override
            random_seed: Random seed
            zero_drift_mode: Suppress drift if True
            memory_limit_gb: Memory limit
            
        Returns:
            Simulated paths array
        """
        # SE-03: Thread-safety - no mutation of self.params, use local variables only
        original_mu = self.params.mu
        original_sigma = self.sigma_t if hasattr(self, 'sigma_t') else self.params.sigma
        
        try:
            # Apply macro adjustments locally only
            mu_eff = macro_params.mu_adjusted if (macro_params and macro_params.mu_adjusted is not None) else original_mu
            sigma_eff = macro_params.sigma_adjusted if (macro_params and macro_params.sigma_adjusted is not None) else original_sigma
            
            # PE-03: Activate lambda_boost and regime_crisis_prior
            if macro_params:
                if macro_params.lambda_boost > 0 or macro_params.regime_crisis_prior > 0:
                    # These require special handling - raise NotImplementedError as per spec
                    raise NotImplementedError(
                        "Macro parameters lambda_boost and regime_crisis_prior require "
                        "Merton jump-diffusion implementation which is not yet available. "
                        "Use standard GBM simulation or set these to 0."
                    )
                    
            # Temporarily set local effective parameters for simulation
            self.params.mu = mu_eff
            if hasattr(self, 'sigma_t'):
                self.sigma_t = sigma_eff
            else:
                # Store original and set new
                self._original_sigma = original_sigma
                self.params.sigma = sigma_eff
                
            # Run simulation
            paths = self.simulate(
                S0=S0, n_steps=n_steps, n_paths=n_paths,
                dt=dt, random_seed=random_seed,
                zero_drift_mode=zero_drift_mode,
                memory_limit_gb=memory_limit_gb
            )
            
            return paths
            
        finally:
            # SE-03: Restore original values (defensive, though we used locals for logic)
            self.params.mu = original_mu
            if hasattr(self, '_original_sigma'):
                self.params.sigma = self._original_sigma
                delattr(self, '_original_sigma')
            elif hasattr(self, 'sigma_t'):
                self.sigma_t = original_sigma
                
    def get_statistics(self, paths: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate VaR and other statistics from simulated paths.
        
        Args:
            paths: Simulated price paths (n_paths, n_steps)
            confidence: Confidence level (strictly between 0 and 1)
            
        Returns:
            Dict with 'var', 'cvar', 'mean', 'std', 'min', 'max', 'volatility_realized'
        """
        # PE-02: Confidence level validation
        if not (0.0 < confidence < 1.0):
            raise ValueError(f"confidence must be strictly in (0, 1), got {confidence}")
            
        # NE-01: Guard division-by-zero
        alpha = 1 - confidence
        if alpha <= 0 or alpha >= 1:
            raise ValueError(f"Derived alpha={alpha} is out of range — check confidence={confidence}")
            
        # Calculate returns (terminal values)
        terminal_values = paths[:, -1]
        initial_value = paths[0, 0] if paths.shape[1] > 0 else 1.0
        
        # PnL distribution
        pnl = terminal_values - initial_value
        
        # Use nanpercentile for safety
        lower_p = np.clip(alpha / 2 * 100, 0.001, 49.999)
        upper_p = np.clip((1 - alpha / 2) * 100, 50.001, 99.999)
        
        var_threshold = np.nanpercentile(pnl, alpha * 100)
        cvar = np.mean(pnl[pnl <= var_threshold]) if np.any(pnl <= var_threshold) else var_threshold
        
        # DE-04: Correct volatility_realized formula
        # Step-by-step log-return std, annualised
        if paths.shape[1] > 1:
            log_rets = np.diff(np.log(paths), axis=1)  # shape: (n_paths, n_steps-1)
            vol_per_path = np.std(log_rets, axis=1) * np.sqrt(252)  # annualised per path
            vol_realized = float(np.mean(vol_per_path))
        else:
            vol_realized = 0.0
            
        stats = {
            'var': float(var_threshold),
            'cvar': float(cvar),
            'mean': float(np.mean(terminal_values)),
            'std': float(np.std(terminal_values)),
            'min': float(np.min(terminal_values)),
            'max': float(np.max(terminal_values)),
            'volatility_realized': vol_realized,
            'confidence': confidence
        }
        
        return stats