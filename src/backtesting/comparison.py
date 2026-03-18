"""
Model Comparison and Ranking System

Compares multiple stochastic models across various performance metrics.
Generates leaderboards and statistical significance tests.
Author : Essabri Ali Rayan
Version : 1.3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy import stats
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class ModelScore:
    """Score container for a single model."""
    model_name: str
    overall_rank: int
    scores: Dict[str, float]
    wins: int
    rank_sum: float


class ModelComparison:
    """
    Comprehensive model comparison using multiple metrics.
    
    Ranks models on each metric and calculates overall scores.
    """
    
    # Metrics where lower values are better
    LOWER_IS_BETTER = ['rmse', 'mae', 'mape', 'bias', 'max_error', 'final_price_error']
    
    # Metrics where higher values are better  
    HIGHER_IS_BETTER = ['directional_accuracy', 'sharpe_ratio', 'correlation', 'coverage']
    
    def __init__(self, results: Dict[str, List]):
        """
        Initialize with backtest results.
        
        Parameters
        ----------
        results : dict
            Dictionary mapping model names to lists of BacktestResult
        """
        self.results = results
        self.models = list(results.keys())
        self.metrics_df = self._compile_metrics()
        self.ranks_df = self._calculate_ranks()
        self.scores = self._calculate_overall_scores()
    
    def _compile_metrics(self) -> pd.DataFrame:
        """Compile all metrics into DataFrame."""
        records = []
        
        for model_name, backtest_results in self.results.items():
            # Aggregate metrics across all windows
            metrics = {}
            for result in backtest_results:
                for key, value in result.metrics.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            
            # Calculate means
            row = {'model': model_name}
            for key, values in metrics.items():
                row[f'{key}_mean'] = np.mean(values)
                row[f'{key}_std'] = np.std(values)
            
            records.append(row)
        
        return pd.DataFrame(records).set_index('model')
    
    def _calculate_ranks(self) -> pd.DataFrame:
        """Calculate ranks for each metric."""
        ranks = pd.DataFrame(index=self.models)
        
        metric_cols = [c for c in self.metrics_df.columns if '_mean' in c]
        
        for col in metric_cols:
            metric_name = col.replace('_mean', '')
            
            if metric_name in self.LOWER_IS_BETTER:
                # Rank 1 = best (lowest)
                ranks[f'{metric_name}_rank'] = self.metrics_df[col].rank()
            elif metric_name in self.HIGHER_IS_BETTER:
                # Rank 1 = best (highest)
                ranks[f'{metric_name}_rank'] = self.metrics_df[col].rank(ascending=False)
            else:
                # Default: lower is better
                ranks[f'{metric_name}_rank'] = self.metrics_df[col].rank()
        
        return ranks
    
    def _calculate_overall_scores(self) -> Dict[str, ModelScore]:
        """Calculate overall model scores."""
        scores = {}
        
        # Sum of ranks (lower is better)
        rank_sums = self.ranks_df.sum(axis=1)
        
        # Count wins (rank 1)
        wins = (self.ranks_df == 1).sum(axis=1)
        
        # Overall ranking
        overall_ranks = rank_sums.rank()
        
        for model in self.models:
            scores[model] = ModelScore(
                model_name=model,
                overall_rank=int(overall_ranks[model]),
                scores=self.metrics_df.loc[model].to_dict(),
                wins=int(wins[model]),
                rank_sum=float(rank_sums[model])
            )
        
        return scores
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get ranked leaderboard of models."""
        data = []
        for model, score in sorted(self.scores.items(), key=lambda x: x[1].overall_rank):
            row = {
                'Rank': score.overall_rank,
                'Model': model,
                'Wins': score.wins,
                'Avg_Rank': round(score.rank_sum / len(self.ranks_df.columns), 2),
                'RMSE': round(score.scores.get('rmse_mean', 0), 4),
                'MAPE': round(score.scores.get('mape_mean', 0), 2),
                'Dir_Acc': round(score.scores.get('directional_accuracy_mean', 0), 1)
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def statistical_test(
        self,
        metric: str = 'rmse',
        confidence: float = 0.95
    ) -> pd.DataFrame:
        """
        Perform pairwise t-tests between models.
        
        Parameters
        ----------
        metric : str
            Metric to test
        confidence : float
            Confidence level
        
        Returns
        -------
        pd.DataFrame
            Matrix of p-values
        """
        alpha = 1 - confidence
        models = self.models
        n = len(models)
        p_values = pd.DataFrame(np.ones((n, n)), index=models, columns=models)
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i >= j:
                    continue
                
                # Get metric values for both models
                vals1 = [r.metrics[metric] for r in self.results[model1]]
                vals2 = [r.metrics[metric] for r in self.results[model2]]
                
                # Paired t-test
                if len(vals1) == len(vals2):
                    _, p_val = stats.ttest_rel(vals1, vals2)
                    p_values.loc[model1, model2] = p_val
                    p_values.loc[model2, model1] = p_val
        
        return p_values
    
    def best_model(self, metric: Optional[str] = None) -> str:
        """
        Get best model name.
        
        Parameters
        ----------
        metric : str, optional
            Specific metric to use. If None, uses overall ranking.
        
        Returns
        -------
        str
            Name of best model
        """
        if metric is None:
            # Use overall ranking
            return min(self.scores.items(), key=lambda x: x[1].overall_rank)[0]
        else:
            # Use specific metric
            col = f'{metric}_mean'
            if col in self.metrics_df.columns:
                if metric in self.LOWER_IS_BETTER:
                    return self.metrics_df[col].idxmin()
                else:
                    return self.metrics_df[col].idxmax()
            return None
    
    def plot_comparison(self, metric: str = 'rmse', save_path: Optional[str] = None):
        """Plot boxplot comparison of models."""
        data = []
        labels = []
        
        for model_name, backtest_results in self.results.items():
            values = [r.metrics[metric] for r in backtest_results]
            data.append(values)
            labels.append(model_name)
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=labels)
        plt.title(f'Model Comparison: {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def generate_report(self) -> str:
        """Generate text report of comparison."""
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL COMPARISON REPORT")
        lines.append("=" * 60)
        lines.append("")
        
        # Leaderboard
        lines.append("LEADERBOARD:")
        lines.append("-" * 60)
        lb = self.get_leaderboard()
        lines.append(lb.to_string(index=False))
        lines.append("")
        
        # Best model
        best = self.best_model()
        lines.append(f"OVERALL BEST MODEL: {best}")
        lines.append("")
        
        # Metric-specific winners
        lines.append("METRIC-SPECIFIC WINNERS:")
        lines.append("-" * 60)
        for metric in ['rmse', 'mae', 'mape', 'directional_accuracy']:
            winner = self.best_model(metric)
            lines.append(f"  {metric.upper()}: {winner}")
        lines.append("")
        
        return "\n".join(lines)


def compare_models(
    backtest_results: Dict[str, List],
    confidence: float = 0.95
) -> ModelComparison:
    """
    Convenience function to create ModelComparison.
    
    Parameters
    ----------
    backtest_results : dict
        Results from run_backtest()
    confidence : float
        Confidence level for statistical tests
    
    Returns
    -------
    ModelComparison
        Comparison object
    """
    return ModelComparison(backtest_results)