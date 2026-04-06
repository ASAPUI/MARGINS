# src/macro/adjuster.py
"""
ParameterAdjuster: Translate WorldMonitor signals into model parameters.
Implements the adjustment formulas from the integration specification.
author:Essabri Ali Rayan
Version :1.3
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import logging

from .signals import MacroSignal


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelParameters:
    """
    Stochastic model parameters with macro adjustments applied.
    
    Contains both base (historical) and adjusted values for transparency.
    """
    # GBM/OU shared
    mu_base: float
    mu_adjusted: float
    sigma_base: float
    sigma_adjusted: float
    
    # OU specific
    theta_base: Optional[float] = None
    theta_adjusted: Optional[float] = None
    kappa: Optional[float] = None  # Unchanged by macro
    
    # Merton jump
    lambda_base: Optional[float] = None
    lambda_adjusted: Optional[float] = None
    mu_j_base: Optional[float] = None
    mu_j_adjusted: Optional[float] = None
    sigma_j_base: Optional[float] = None
    sigma_j_adjusted: Optional[float] = None
    
    # Heston
    theta_v_base: Optional[float] = None
    theta_v_adjusted: Optional[float] = None
    xi_base: Optional[float] = None
    xi_adjusted: Optional[float] = None
    kappa_v: Optional[float] = None  # Unchanged
    rho: Optional[float] = None      # Unchanged
    
    # Regime
    p_crisis: float = 0.1  # Prior probability of crisis regime
    p_calm_to_crisis: float = 0.05  # Transition probability
    
    # Metadata
    adjustment_factors: Dict[str, float] = None
    
    def __post_init__(self):
        if self.adjustment_factors is None:
            object.__setattr__(self, 'adjustment_factors', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize adjustments for output/debugging."""
        return {
            "drift": {
                "base": round(self.mu_base, 6),
                "adjusted": round(self.mu_adjusted, 6),
                "delta": round(self.mu_adjusted - self.mu_base, 6)
            },
            "volatility": {
                "base": round(self.sigma_base, 6),
                "adjusted": round(self.sigma_adjusted, 6),
                "multiplier": round(self.sigma_adjusted / self.sigma_base, 4) 
                    if self.sigma_base > 0 else 1.0
            },
            "jump_intensity": {
                "base": self.lambda_base,
                "adjusted": self.lambda_adjusted,
                "boost_factor": (self.lambda_adjusted / self.lambda_base) 
                    if self.lambda_base and self.lambda_base > 0 else 1.0
            } if self.lambda_base else None,
            "crisis_probability": round(self.p_crisis, 4),
            "adjustment_summary": self.adjustment_factors
        }


class ParameterAdjuster:
    """
    Translates MacroSignal into model parameter deltas.
    
    Implements the formula specifications from Section 3.2 of the 
    integration plan with hard caps to prevent extreme distortion.
    """
    
    # Hard caps to prevent parameter explosion
    SIGMA_MAX_MULTIPLIER = 3.0
    LAMBDA_MAX_MULTIPLIER = 10.0
    MAX_CRISIS_PROB = 0.9
    MIN_CRISIS_PROB = 0.05
    
    def __init__(self, signal: MacroSignal):
        self.signal = signal
        self._adjustment_log: Dict[str, float] = {}
    
    def _calculate_sigma_multiplier(self) -> float:
        """
        Volatility boost based on CII top-5 average.
        Formula: sigma *= 1 + (CII_avg - 50) / 100 * 0.4
        """
        cii_avg = self.signal.cii_top5_avg
        if cii_avg <= 50:
            return 1.0
            
        raw_multiplier = 1.0 + ((cii_avg - 50) / 100.0) * 0.4
        capped = min(raw_multiplier, self.SIGMA_MAX_MULTIPLIER)
        
        self._adjustment_log['sigma_multiplier'] = round(capped, 4)
        self._adjustment_log['cii_avg_input'] = round(cii_avg, 2)
        
        return capped
    
    def _calculate_mu_shift(self) -> float:
        """
        Drift adjustment based on:
        1. Geopolitical factor (max CII)
        2. Real rates (rising real rates = lower gold drift)
        3. DXY (stronger dollar = lower gold drift)
        
        Formula: mu_shift = -0.001 * max(0, CII_max - 70) 
                          - 0.3 * real_rate_delta 
                          - 0.4 * dxy_delta
        """
        cii_max = self.signal.cii_max
        geo_shift = 0.0
        if cii_max > 70:
            geo_shift = -0.001 * (cii_max - 70)
            self._adjustment_log['geo_shift'] = round(geo_shift, 6)
        
        # Financial market adjustments (coefficients from empirical regression)
        rate_shift = -0.3 * self.signal.real_rate_delta
        dxy_shift = -0.4 * self.signal.dxy_delta
        
        total_shift = geo_shift + rate_shift + dxy_shift
        
        self._adjustment_log['mu_shift'] = round(total_shift, 6)
        self._adjustment_log['cii_max_input'] = round(cii_max, 2)
        self._adjustment_log['real_rate_delta'] = round(self.signal.real_rate_delta, 4)
        self._adjustment_log['dxy_delta'] = round(self.signal.dxy_delta, 4)
        self._adjustment_log['rate_contribution'] = round(rate_shift, 6)
        self._adjustment_log['dxy_contribution'] = round(dxy_shift, 6)

        return total_shift
    
    def _calculate_lambda_boost(self, base_lambda: float) -> float:
        """
        Jump intensity boost based on anomaly counts.
        Formula: 
        - z >= 2.0: lambda *= 1 + 0.3 * count
        - z >= 3.0: lambda *= 1 + 0.6 * count (cumulative)
        """
        if base_lambda <= 0:
            return 0.0
            
        high_count = self.signal.high_anomaly_count
        critical_count = self.signal.critical_anomaly_count
        
        # Critical anomalies count toward both tiers
        boost_factor = 1.0
        
        if high_count > 0:
            boost_factor += 0.3 * high_count
        if critical_count > 0:
            boost_factor += 0.6 * critical_count
            
        capped_boost = min(boost_factor, self.LAMBDA_MAX_MULTIPLIER)
        
        self._adjustment_log['lambda_boost_factor'] = round(capped_boost, 4)
        self._adjustment_log['high_anomalies'] = high_count
        self._adjustment_log['critical_anomalies'] = critical_count
        
        return base_lambda * capped_boost
    
    def _calculate_crisis_probability(self) -> float:
        """
        Regime prior for crisis state.
        Formula: P(crisis) = min(0.9, hotspots * 0.08)
        """
        hotspots = self.signal.active_hotspot_count
        prob = min(hotspots * 0.08, self.MAX_CRISIS_PROB)
        prob = max(prob, self.MIN_CRISIS_PROB)  # Minimum baseline risk
        
        self._adjustment_log['hotspot_count'] = hotspots
        self._adjustment_log['crisis_probability'] = round(prob, 4)
        
        return prob
    
    def adjust_gbm(self, mu: float, sigma: float) -> ModelParameters:
        """
        Adjust GBM parameters.
        
        Args:
            mu: Historical drift (annualized)
            sigma: Historical volatility (annualized)
        """
        sigma_mult = self._calculate_sigma_multiplier()
        mu_shift = self._calculate_mu_shift()
        
        return ModelParameters(
            mu_base=mu,
            mu_adjusted=mu + mu_shift,
            sigma_base=sigma,
            sigma_adjusted=sigma * sigma_mult,
            p_crisis=self._calculate_crisis_probability(),
            adjustment_factors=self._adjustment_log.copy()
        )
    
    def adjust_ou(
        self, 
        mu: float, 
        sigma: float, 
        theta: float,
        kappa: float
    ) -> ModelParameters:
        """
        Adjust Ornstein-Uhlenbeck parameters.
        
        Theta (long-term mean) pulled higher if CII_max > 75 (flight-to-safety).
        """
        sigma_mult = self._calculate_sigma_multiplier()
        mu_shift = self._calculate_mu_shift()
        
        # Flight to safety: gold's long-term mean increases during crisis
        theta_shift = 0.0
        if self.signal.cii_max > 75:
            theta_shift = theta * 0.05  # 5% increase in long-term mean
            self._adjustment_log['theta_shift'] = round(theta_shift, 4)
            self._adjustment_log['flight_to_safety'] = True
        
        return ModelParameters(
            mu_base=mu,
            mu_adjusted=mu + mu_shift,
            sigma_base=sigma,
            sigma_adjusted=sigma * sigma_mult,
            theta_base=theta,
            theta_adjusted=theta + theta_shift,
            kappa=kappa,  # Unchanged
            p_crisis=self._calculate_crisis_probability(),
            adjustment_factors=self._adjustment_log.copy()
        )
    
    def adjust_merton(
        self,
        mu: float,
        sigma: float,
        lambda_jump: float,
        mu_j: float,
        sigma_j: float
    ) -> ModelParameters:
        """
        Adjust Merton Jump Diffusion parameters.
        
        Highest impact integration - jumps are most sensitive to macro shocks.
        """
        sigma_mult = self._calculate_sigma_multiplier()
        mu_shift = self._calculate_mu_shift()
        
        # Jump intensity boosted by anomalie
        lambda_adj = self._calculate_lambda_boost(lambda_jump)
        
        # Jump direction: negative mean if extreme instability (crisis drops),
        # positive if very stable (mean reversion up)
        mu_j_shift = 0.0
        if self.signal.cii_max > 80:
            mu_j_shift = -0.02  # Crisis jumps tend down initially
            self._adjustment_log['mu_j_shift'] = 'crisis_negative'
        elif self.signal.cii_top5_avg < 30:
            mu_j_shift = 0.01   # Stable times, upward reversion jumps
            self._adjustment_log['mu_j_shift'] = 'stable_positive'
        
        # Jump volatility scales with intensity
        sigma_j_mult = min(1.0 + (lambda_adj/lambda_jump - 1.0) * 0.5, 2.0) \
            if lambda_jump > 0 else 1.0
        
        return ModelParameters(
            mu_base=mu,
            mu_adjusted=mu + mu_shift,
            sigma_base=sigma,
            sigma_adjusted=sigma * sigma_mult,
            lambda_base=lambda_jump,
            lambda_adjusted=lambda_adj,
            mu_j_base=mu_j,
            mu_j_adjusted=mu_j + mu_j_shift,
            sigma_j_base=sigma_j,
            sigma_j_adjusted=sigma_j * sigma_j_mult,
            p_crisis=self._calculate_crisis_probability(),
            adjustment_factors=self._adjustment_log.copy()
        )
    
    def adjust_heston(
        self,
        mu: float,
        v0: float,
        theta_v: float,
        kappa_v: float,
        xi: float,
        rho: float
    ) -> ModelParameters:
        """
        Adjust Heston stochastic volatility parameters.
        
        Long-run variance (theta_v) increased by sigma_multiplier squared.
        Vol-of-vol (xi) boosted when critical anomalies present.
        """
        sigma_mult = self._calculate_sigma_multiplier()
        mu_shift = self._calculate_mu_shift()
        
        # Variance increases with square of vol multiplier
        theta_v_mult = sigma_mult ** 2
        
        # Vol-of-vol spikes during critical events
        xi_mult = 1.0
        if self.signal.critical_anomaly_count > 0:
            xi_mult = 1.0 + 0.2 * self.signal.critical_anomaly_count
            xi_mult = min(xi_mult, 2.0)
            self._adjustment_log['xi_boost'] = 'critical_anomaly'
        
        return ModelParameters(
            mu_base=mu,
            mu_adjusted=mu + mu_shift,
            sigma_base=v0**0.5,  # Convert variance to vol for consistency
            sigma_adjusted=(v0 * sigma_mult)**0.5,
            theta_v_base=theta_v,
            theta_v_adjusted=theta_v * theta_v_mult,
            xi_base=xi,
            xi_adjusted=xi * xi_mult,
            kappa_v=kappa_v,  # Unchanged
            rho=rho,          # Unchanged
            p_crisis=self._calculate_crisis_probability(),
            adjustment_factors=self._adjustment_log.copy()
        )
    
    def adjust_regime(
        self,
        mu_calm: float,
        sigma_calm: float,
        mu_crisis: float,
        sigma_crisis: float,
        p_calm_to_crisis: float = 0.05,
        p_crisis_to_calm: float = 0.3
    ) -> ModelParameters:
        """
        Adjust Regime Switching parameters.
        
        Prior probability set from hotspot count.
        Transition probability P(calm->crisis) increased during elevated CII.
        """
        p_crisis = self._calculate_crisis_probability()
        
        # Increase transition probability to crisis during instability
        p_transition_boost = 1.0
        if self.signal.cii_top5_avg > 60:
            p_transition_boost = 1.0 + (self.signal.cii_top5_avg - 60) / 100
            p_transition_boost = min(p_transition_boost, 3.0)
        
        p_calm_to_crisis_adj = min(
            p_calm_to_crisis * p_transition_boost,
            0.5  # Cap at 50% per period
        )
        
        self._adjustment_log['p_calm_to_crisis_base'] = round(p_calm_to_crisis, 4)
        self._adjustment_log['p_calm_to_crisis_adj'] = round(p_calm_to_crisis_adj, 4)
        self._adjustment_log['p_crisis_to_calm'] = round(p_crisis_to_calm, 4)
        
        # Return base params (crisis params remain historically calibrated)
        # but with adjusted transition and prior
        return ModelParameters(
            mu_base=mu_calm,
            mu_adjusted=mu_calm,  # Regime params don't shift, transitions do
            sigma_base=sigma_calm,
            sigma_adjusted=sigma_calm,
            p_crisis=p_crisis,
            p_calm_to_crisis=p_calm_to_crisis_adj,
            adjustment_factors=self._adjustment_log.copy()
        )