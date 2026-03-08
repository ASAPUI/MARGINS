"""
Model Aggregator Module

Combines results from multiple stochastic models using various aggregation methods:
- Simple averaging
- Weighted by historical performance
- Bayesian model averaging
- Ensemble methods

Author: Essabri ali rayan
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelWeight:
    """Weight configuration for a single model."""
    name: str
    weight: float = 1.0
    performance_score: Optional[float] = None
    recent_mape: Optional[float] = None
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")


@dataclass
class AggregatedResults:
    """Results from aggregating multiple models."""
    aggregated_paths: np.ndarray
    model_weights: Dict[str, float]
    individual_results: Dict[str, Any]
    statistics: Dict[str, float]
    aggregation_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with all model predictions."""
        data = {
            'Aggregated': self.aggregated_paths.mean(axis=0),
            'Aggregated_Std': self.aggregated_paths.std(axis=0),
        }
        
        for name, result in self.individual_results.items():
            if hasattr(result, 'paths'):
                data[f'{name}_Mean'] = result.paths.mean(axis=0)
                data[f'{name}_Std'] = result.paths.std(axis=0)
        
        return pd.DataFrame(data)


class ModelAggregator:
    """
    Aggregates predictions from multiple stochastic models.
    
    Supports various aggregation methods:
    - equal_weight: Simple average of all models
    - performance_weighted: Weight by historical accuracy
    - bayesian: Bayesian model averaging
    - best_single: Select best performing model
    - trimmed_mean: Remove outliers before averaging
    """
    
    def __init__(self, method: str = 'equal_weight'):
        """
        Initialize aggregator.
        
        Args:
            method: Aggregation method to use
        """
        self.method = method
        self.valid_methods = [
            'equal_weight',
            'performance_weighted',
            'bayesian',
            'best_single',
            'trimmed_mean',
            'median'
        ]
        
        if method not in self.valid_methods:
            raise ValueError(f"Unknown method {method}. Valid: {self.valid_methods}")
        
        logger.info(f"ModelAggregator initialized with method: {method}")
    
    def aggregate(self, results: Dict[str, Any], custom_weights: Optional[Dict[str, float]] = None) -> AggregatedResults:
        """
        Aggregate simulation results from multiple models.
        
        Args:
            results: Dictionary of {model_name: simulation_result}
            custom_weights: Optional custom weights for each model
            
        Returns:
            AggregatedResults container
        """
        if not results:
            raise ValueError("No results to aggregate")
        
        # Extract paths from results
        paths_dict = {}
        for name, result in results.items():
            if hasattr(result, 'paths'):
                paths_dict[name] = result.paths
            elif isinstance(result, np.ndarray):
                paths_dict[name] = result
            else:
                logger.warning(f"Skipping {name}: no paths found")
        
        if not paths_dict:
            raise ValueError("No valid paths found in results")
        
        # Calculate weights
        if custom_weights:
            weights = custom_weights
        else:
            weights = self._calculate_weights(results)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Aggregate paths
        aggregated = self._aggregate_paths(paths_dict, weights)
        
        # Calculate statistics
        stats = self._calculate_aggregated_statistics(aggregated)
        
        return AggregatedResults(
            aggregated_paths=aggregated,
            model_weights=weights,
            individual_results=results,
            statistics=stats,
            aggregation_method=self.method
        )
    
    def _calculate_weights(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate model weights based on aggregation method."""
        n_models = len(results)
        
        if self.method == 'equal_weight':
            return {name: 1.0/n_models for name in results.keys()}
        
        elif self.method == 'performance_weighted':
            weights = {}
            for name, result in results.items():
                if hasattr(result, 'statistics') and 'mape' in result.statistics:
                    mape = result.statistics['mape']
                    weights[name] = 1.0 / (mape + 0.01)
                else:
                    weights[name] = 1.0
            return weights
        
        elif self.method == 'bayesian':
            return self._calculate_bayesian_weights(results)
        
        elif self.method == 'best_single':
            best_model = None
            best_score = -np.inf
            
            for name, result in results.items():
                score = getattr(result, 'performance_score', 0)
                if score > best_score:
                    best_score = score
                    best_model = name
            
            weights = {name: 0.0 for name in results.keys()}
            if best_model:
                weights[best_model] = 1.0
            return weights
        
        elif self.method in ['trimmed_mean', 'median']:
            return {name: 1.0/n_models for name in results.keys()}
        
        else:
            return {name: 1.0/n_models for name in results.keys()}
    
    def _calculate_bayesian_weights(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate Bayesian model averaging weights."""
        bics = {}
        
        for name, result in results.items():
            if hasattr(result, 'log_likelihood'):
                log_lik = result.log_likelihood
                n_params = getattr(result, 'n_params', 3)
                n_obs = getattr(result, 'n_obs', 252)
                bics[name] = n_params * np.log(n_obs) - 2 * log_lik
            else:
                if hasattr(result, 'paths'):
                    variance = np.var(result.paths[:, -1])
                    bics[name] = np.log(variance + 1)
                else:
                    bics[name] = 1.0
        
        min_bic = min(bics.values())
        delta_bics = {k: v - min_bic for k, v in bics.items()}
        exp_terms = {k: np.exp(-0.5 * v) for k, v in delta_bics.items()}
        total = sum(exp_terms.values())
        
        return {k: v/total for k, v in exp_terms.items()}
    
    def _aggregate_paths(self, paths_dict: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
        """Aggregate paths using specified method."""
        all_models = list(paths_dict.keys())
        n_paths_per_model = list(paths_dict.values())[0].shape[0]
        n_steps = list(paths_dict.values())[0].shape[1]
        
        if self.method == 'median':
            stacked = np.stack([paths_dict[m] for m in all_models], axis=0)
            return np.median(stacked, axis=0)
        
        elif self.method == 'trimmed_mean':
            stacked = np.stack([paths_dict[m] for m in all_models], axis=0)
            sorted_paths = np.sort(stacked, axis=0)
            trim = max(1, int(len(all_models) * 0.2))
            trimmed = sorted_paths[trim:-trim] if len(all_models) > 2 else sorted_paths
            return np.mean(trimmed, axis=0)
        
        else:
            weighted_sum = np.zeros((n_paths_per_model, n_steps))
            for model_name, paths in paths_dict.items():
                weight = weights.get(model_name, 1.0/len(all_models))
                weighted_sum += weight * paths
            
            return weighted_sum
    
    def _calculate_aggregated_statistics(self, aggregated_paths: np.ndarray) -> Dict[str, float]:
        """Calculate statistics for aggregated paths."""
        final_prices = aggregated_paths[:, -1]
        
        return {
            'mean_final': float(np.mean(final_prices)),
            'std_final': float(np.std(final_prices)),
            'median_final': float(np.median(final_prices)),
            'min_final': float(np.min(final_prices)),
            'max_final': float(np.max(final_prices)),
            'ci_lower_95': float(np.percentile(final_prices, 2.5)),
            'ci_upper_95': float(np.percentile(final_prices, 97.5)),
            'ci_lower_90': float(np.percentile(final_prices, 5)),
            'ci_upper_90': float(np.percentile(final_prices, 95)),
            'prob_gain': float(np.mean(final_prices > aggregated_paths[0, 0])),
            'skewness': float(stats.skew(final_prices)),
            'kurtosis': float(stats.kurtosis(final_prices)),
        }
    
    def compare_models(self, results: Dict[str, Any], actual_price: Optional[float] = None) -> pd.DataFrame:
        """Create comparison table of all models."""
        comparisons = []
        
        for name, result in results.items():
            row = {'Model': name}
            
            if hasattr(result, 'statistics'):
                stats = result.statistics
                row['Mean_Final'] = stats.get('mean_final', np.nan)
                row['Std_Final'] = stats.get('std_final', np.nan)
                row['CI_Lower'] = stats.get('ci_lower_95', np.nan)
                row['CI_Upper'] = stats.get('ci_upper_95', np.nan)
                row['Prob_Gain'] = stats.get('prob_gain', np.nan)
            
            if actual_price and hasattr(result, 'statistics'):
                mean_pred = result.statistics.get('mean_final', 0)
                row['Prediction_Error'] = mean_pred - actual_price
                row['MAPE'] = abs(mean_pred - actual_price) / actual_price * 100
            
            comparisons.append(row)
        
        return pd.DataFrame(comparisons)
    
    def forecast_distribution(self, results: Dict[str, Any], horizons: List[int] = [7, 30, 90, 252]) -> pd.DataFrame:
        """Generate forecast distribution at multiple horizons."""
        forecasts = []
        
        for name, result in results.items():
            if not hasattr(result, 'paths'):
                continue
            
            paths = result.paths
            for h in horizons:
                if h < paths.shape[1]:
                    prices = paths[:, h]
                    forecasts.append({
                        'Model': name,
                        'Horizon_Days': h,
                        'Mean': np.mean(prices),
                        'Std': np.std(prices),
                        'P05': np.percentile(prices, 5),
                        'P25': np.percentile(prices, 25),
                        'P50': np.percentile(prices, 50),
                        'P75': np.percentile(prices, 75),
                        'P95': np.percentile(prices, 95),
                    })
        
        return pd.DataFrame(forecasts)


def create_ensemble(results: Dict[str, Any], method: str = 'equal_weight') -> AggregatedResults:
    """Quick function to create ensemble from results."""
    aggregator = ModelAggregator(method=method)
    return aggregator.aggregate(results)


if __name__ == "__main__":
    print("Model Aggregator Module")