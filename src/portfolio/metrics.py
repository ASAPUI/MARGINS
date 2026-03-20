"""
metrics.py — MARGINS Portfolio Mode
Portfolio-level risk metrics computed from Monte Carlo simulation paths.

Metrics implemented (per spec §3.3):
  - Portfolio VaR 95%       : percentile(W_T, 5)
  - Portfolio CVaR 95%      : mean(W_T[W_T ≤ VaR])
  - Expected Return         : (mean(W_T) / W_0) - 1
  - Sharpe Ratio            : E[R_p] / σ(R_p) · √252
  - Diversification Ratio   : Σ(wᵢ·σᵢ) / σ_portfolio
  - Max Drawdown            : min over paths of (W_t - peak_t) / peak_t
  - Correlation Benefit     : 1 - (portfolio VaR / sum of individual VaRs)
  - Probability of Gain     : P(W_T > W_0)

Design decisions:
  - All metrics computed on terminal portfolio values W[:, -1]
  - Max drawdown computed path-by-path using np.maximum.accumulate (vectorized)
  - Sharpe annualized assuming daily returns (× √252)
  - Returns a typed dict for easy JSON serialization in CLI output
"""

import numpy as np
import pandas as pd


def compute_portfolio_metrics(
    asset_paths: np.ndarray,
    weights: np.ndarray,
    S0_vec: np.ndarray,
    horizon_days: int | None = None,
    confidence: float = 0.95,
) -> dict:
    """
    Compute all portfolio-level risk metrics from simulated asset paths.

    Parameters
    ----------
    asset_paths : np.ndarray, shape (N, K, T+1)
        Simulated price paths for all assets from simulate_portfolio().
    weights : np.ndarray, shape (N,)
        Portfolio weights. Must sum to 1. wᵢ ≥ 0.
    S0_vec : np.ndarray, shape (N,)
        Initial prices for each asset (= asset_paths[:, :, 0] row means).
    horizon_days : int, optional
        Simulation horizon in trading days. Used for annualizing Sharpe.
        If None, inferred from paths shape.
    confidence : float
        VaR/CVaR confidence level. Default 0.95 (95th percentile).

    Returns
    -------
    metrics : dict
        {
          'portfolio_var_95'     : float  — 5th percentile portfolio value ($)
          'portfolio_cvar_95'    : float  — mean of worst 5% portfolio values ($)
          'expected_return'      : float  — expected return as % over horizon
          'sharpe'               : float  — annualized Sharpe ratio
          'diversification_ratio': float  — weighted vol / portfolio vol
          'avg_max_drawdown'     : float  — average max drawdown across paths (%)
          'prob_gain'            : float  — probability of positive return (%)
          'correlation_benefit'  : float  — fraction of risk removed by correlation
          'portfolio_vol'        : float  — annualized portfolio return std (%)
          'expected_value'       : float  — mean terminal portfolio value ($)
          'initial_value'        : float  — initial portfolio value ($)
          'n_paths'              : int    — number of simulated paths
          'n_steps'              : int    — number of time steps
        }
    """
    N, K, T_plus_1 = asset_paths.shape
    T = T_plus_1 - 1

    if horizon_days is None:
        horizon_days = T

    weights = np.asarray(weights, dtype=float)
    assert np.isclose(weights.sum(), 1.0, atol=1e-6), (
        f"Weights must sum to 1.0, got {weights.sum():.6f}"
    )

    # ── Portfolio value paths: W[k, t] = Σᵢ wᵢ · S[i, k, t] ─────────────
    # einsum 'i,ikt->kt': weighted sum over assets
    W = np.einsum("i,ikt->kt", weights, asset_paths)   # shape (K, T+1)

    W0 = W[:, 0]    # initial portfolio value per path (should be ~same for all paths)
    W_T = W[:, -1]  # terminal portfolio value, shape (K,)

    W0_scalar = float(np.mean(W0))   # scalar initial value for reporting

    # ── Returns ──────────────────────────────────────────────────────────
    ret = (W_T / W0) - 1                           # per-path return over horizon

    # ── VaR and CVaR ─────────────────────────────────────────────────────
    alpha = 1.0 - confidence                       # 0.05 for 95% confidence
    var_threshold = np.percentile(W_T, alpha * 100)  # 5th percentile value
    tail_mask = W_T <= var_threshold
    cvar = float(np.mean(W_T[tail_mask])) if tail_mask.any() else float(var_threshold)

    # ── Sharpe Ratio (annualized) ─────────────────────────────────────────
    ret_mean = float(np.mean(ret))
    ret_std = float(np.std(ret))
    # Annualize: multiply by √(252/T) since we have horizon-period returns
    ann_factor = np.sqrt(252 / horizon_days)
    sharpe = (ret_mean / ret_std * ann_factor) if ret_std > 0 else 0.0

    # ── Portfolio annualized volatility ──────────────────────────────────
    port_vol_ann = ret_std * ann_factor * 100  # in %

    # ── Max Drawdown (per path, then averaged) ────────────────────────────
    # peak[k, t] = max(W[k, 0:t+1]) — running peak
    peak = np.maximum.accumulate(W, axis=1)          # shape (K, T+1)
    drawdown = (W - peak) / np.where(peak > 0, peak, 1.0)  # shape (K, T+1)
    max_dd_per_path = np.min(drawdown, axis=1)       # worst drawdown per path
    avg_max_drawdown = float(np.mean(max_dd_per_path) * 100)  # in %

    # ── Diversification Ratio ─────────────────────────────────────────────
    # σ_portfolio = std of log-returns of portfolio
    # Σ(wᵢ·σᵢ) = weighted sum of individual asset log-return stds
    port_log_ret_std = float(np.std(np.log(W_T / W0)))

    ind_log_ret_stds = np.array([
        np.std(np.log(asset_paths[i, :, -1] / S0_vec[i]))
        for i in range(N)
    ])
    weighted_vol = float(np.dot(weights, ind_log_ret_stds))
    div_ratio = weighted_vol / port_log_ret_std if port_log_ret_std > 0 else 1.0

    # ── Correlation Benefit ───────────────────────────────────────────────
    # Individual 5th-percentile VaRs
    ind_vars = []
    for i in range(N):
        ind_terminal = asset_paths[i, :, -1]
        ind_var = float(np.percentile(ind_terminal, alpha * 100))
        # Weight the individual VaR contribution
        ind_vars.append(weights[i] * ind_var)
    sum_ind_var = float(np.sum(ind_vars))

    # Correlation benefit = 1 - (portfolio VaR / sum of weighted individual VaRs)
    if sum_ind_var > 0 and var_threshold > 0:
        corr_benefit = 1.0 - (float(var_threshold) / sum_ind_var)
    else:
        corr_benefit = 0.0

    return {
        # Core risk metrics
        "portfolio_var_95":      float(var_threshold),
        "portfolio_cvar_95":     float(cvar),
        "expected_return":       float(ret_mean * 100),      # %
        "sharpe":                float(sharpe),
        "diversification_ratio": float(div_ratio),
        "avg_max_drawdown":      float(avg_max_drawdown),    # %
        "prob_gain":             float(np.mean(ret > 0) * 100),  # %
        "correlation_benefit":   float(corr_benefit),
        # Supplementary
        "portfolio_vol":         float(port_vol_ann),        # % annualized
        "expected_value":        float(np.mean(W_T)),
        "initial_value":         float(W0_scalar),
        "n_paths":               int(K),
        "n_steps":               int(T),
    }


def compute_scenario_table(
    asset_paths: np.ndarray,
    weights: np.ndarray,
    percentiles: tuple = (10, 50, 90),
) -> dict:
    """
    Compute bear / base / bull scenario portfolio values.

    Parameters
    ----------
    asset_paths : np.ndarray, shape (N, K, T+1)
    weights : np.ndarray, shape (N,)
    percentiles : tuple
        (bear_pct, base_pct, bull_pct). Default (10, 50, 90).

    Returns
    -------
    scenarios : dict with keys 'bear', 'base', 'bull'
        Each value is the portfolio terminal value at that percentile.
    """
    W = np.einsum("i,ikt->kt", weights, asset_paths)
    W_T = W[:, -1]
    labels = ["bear", "base", "bull"]
    return {
        label: float(np.percentile(W_T, pct))
        for label, pct in zip(labels, percentiles)
    }


def compute_per_asset_metrics(
    asset_paths: np.ndarray,
    tickers: list[str],
    S0_vec: np.ndarray,
    confidence: float = 0.95,
) -> list[dict]:
    """
    Compute per-asset expected return and VaR for reporting.

    Returns
    -------
    list of dicts, one per asset:
        {ticker, expected_return_pct, var_95, prob_gain_pct}
    """
    alpha = 1.0 - confidence
    results = []
    for i, ticker in enumerate(tickers):
        terminal = asset_paths[i, :, -1]
        ret = (terminal / S0_vec[i]) - 1
        results.append({
            "ticker":              ticker,
            "expected_return_pct": float(np.mean(ret) * 100),
            "var_95":              float(np.percentile(terminal, alpha * 100)),
            "prob_gain_pct":       float(np.mean(ret > 0) * 100),
        })
    return results


def print_metrics_report(
    metrics: dict,
    weights: np.ndarray,
    tickers: list[str],
    scenarios: dict | None = None,
    per_asset: list[dict] | None = None,
) -> None:
    """Pretty-print a full portfolio metrics report to stdout."""
    print("\n" + "═" * 58)
    print("  MARGINS — Portfolio Risk Report")
    print("═" * 58)

    print(f"\n  Initial Portfolio Value : ${metrics['initial_value']:>12,.2f}")
    print(f"  Expected Value (mean)   : ${metrics['expected_value']:>12,.2f}")
    print(f"  Expected Return         : {metrics['expected_return']:>+10.2f}%")
    print(f"  Portfolio Volatility    : {metrics['portfolio_vol']:>10.2f}% p.a.")

    print(f"\n  ── Risk Metrics ─────────────────────────────")
    print(f"  VaR 95%  (worst 5%)    : ${metrics['portfolio_var_95']:>12,.2f}")
    print(f"  CVaR 95% (tail mean)   : ${metrics['portfolio_cvar_95']:>12,.2f}")
    print(f"  Sharpe Ratio           : {metrics['sharpe']:>12.3f}")
    print(f"  Avg Max Drawdown       : {metrics['avg_max_drawdown']:>+10.2f}%")
    print(f"  Prob of Gain           : {metrics['prob_gain']:>10.1f}%")

    print(f"\n  ── Diversification ──────────────────────────")
    print(f"  Diversification Ratio  : {metrics['diversification_ratio']:>12.3f}")
    print(f"  Correlation Benefit    : {metrics['correlation_benefit']*100:>10.2f}%")

    if scenarios:
        print(f"\n  ── Scenarios ─────────────────────────────── ")
        print(f"  Bear (10th pct)        : ${scenarios['bear']:>12,.2f}")
        print(f"  Base (50th pct)        : ${scenarios['base']:>12,.2f}")
        print(f"  Bull (90th pct)        : ${scenarios['bull']:>12,.2f}")

    print(f"\n  ── Allocations ──────────────────────────────")
    for ticker, w in zip(tickers, weights):
        print(f"  {ticker:12s}           : {w*100:>8.1f}%")

    if per_asset:
        print(f"\n  ── Per-Asset Summary ────────────────────────")
        print(f"  {'Ticker':12s} {'E[R]':>8s} {'VaR95':>12s} {'P(gain)':>9s}")
        for a in per_asset:
            print(
                f"  {a['ticker']:12s} "
                f"{a['expected_return_pct']:>+7.2f}% "
                f"${a['var_95']:>11,.2f} "
                f"{a['prob_gain_pct']:>8.1f}%"
            )

    print("\n" + "═" * 58 + "\n")
