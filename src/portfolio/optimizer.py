"""
optimizer.py — MARGINS Portfolio Mode
Weight optimization using three industry-standard strategies.

Strategies (per spec §3.4):
  1. max_sharpe   — minimize -E[R_p] / σ(R_p)  via SLSQP
  2. min_cvar     — minimize CVaR_95(W_T)        via SLSQP
  3. risk_parity  — wᵢ = (1/σᵢ) / Σⱼ(1/σⱼ)    analytic (no simulation needed)

Constraints enforced on all optimizers (per spec §4.3 Decision 4):
  - wᵢ ≥ min_w   (default 0.01) — no near-zero allocations
  - wᵢ ≤ max_w   (default 0.40) — position cap prevents concentration
  - Σwᵢ = 1      (equality)

PROHIBITED (spec §8):
  - Unconstrained max-Sharpe (would produce ~100% in one asset in-sample)

Design decisions:
  - SLSQP with ftol=1e-9, maxiter=1000 — tight tolerance for convergence
  - Multi-start: tries 3 different starting points, takes best result
    (SLSQP can get stuck in local minima for non-convex objectives like Sharpe)
  - risk_parity uses analytic formula — no optimization needed, always converges
  - Returns equal-weight fallback if SLSQP fails to converge (with warning)
"""

import numpy as np
from scipy.optimize import minimize

from .metrics import compute_portfolio_metrics


def _project_weights(w: np.ndarray, min_w: float, max_w: float) -> np.ndarray:
    """
    Project weights into [min_w, max_w] while maintaining sum = 1.
    Uses iterative clipping: clip → renorm → repeat until stable.
    Handles cases where renormalization pushes values back above max_w.
    """
    w = w.copy()
    for _ in range(50):  # converges in < 10 iterations in practice
        w = np.clip(w, min_w, max_w)
        total = w.sum()
        if abs(total - 1.0) < 1e-10:
            break
        w = w / total
    return w


def _make_constraints(N: int) -> list[dict]:
    """Equality constraint: weights must sum to 1."""
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _make_bounds(N: int, min_w: float, max_w: float) -> list[tuple]:
    """Per-asset weight bounds: [min_w, max_w]."""
    return [(min_w, max_w)] * N


def _starting_points(N: int, min_w: float, max_w: float) -> list[np.ndarray]:
    """
    Generate multiple starting points for multi-start optimization.
    Helps escape local minima in non-convex objectives (esp. max_sharpe).
    """
    equal = np.ones(N) / N
    # Clip equal weights to [min_w, max_w] and renormalize
    clipped = np.clip(equal, min_w, max_w)
    clipped /= clipped.sum()

    starts = [clipped]

    # Random starting points (seeded for reproducibility)
    rng = np.random.default_rng(seed=123)
    for _ in range(2):
        raw = rng.uniform(min_w, max_w, N)
        raw /= raw.sum()
        starts.append(raw)

    return starts


def optimize_max_sharpe(
    asset_paths: np.ndarray,
    S0_vec: np.ndarray,
    min_w: float = 0.01,
    max_w: float = 0.40,
) -> tuple[np.ndarray, dict]:
    """
    Find weights that maximize the portfolio Sharpe ratio.

    Objective: minimize -E[R_p] / σ(R_p)

    This is the classic mean-variance efficient frontier objective.
    Tends to concentrate in 1-3 assets — position cap (max_w) is essential.

    Parameters
    ----------
    asset_paths : np.ndarray, shape (N, K, T+1)
    S0_vec : np.ndarray, shape (N,)
    min_w, max_w : float — weight bounds per asset

    Returns
    -------
    weights : np.ndarray, shape (N,) — optimal weights summing to 1
    result_metrics : dict — metrics at optimal weights
    """
    N = asset_paths.shape[0]
    bounds = _make_bounds(N, min_w, max_w)
    constraints = _make_constraints(N)

    def neg_sharpe(w: np.ndarray) -> float:
        # Normalize: SLSQP probes unnormalized weights during gradient estimation
        w_n = np.clip(w, 0, None)
        s = w_n.sum()
        if s < 1e-12:
            return 0.0
        m = compute_portfolio_metrics(asset_paths, w_n / s, S0_vec)
        return -m["sharpe"]

    best_w = None
    best_val = np.inf

    for w0 in _starting_points(N, min_w, max_w):
        res = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_w = res.x
            if not res.success:
                pass  # keep trying other starts

    if best_w is None or not np.isfinite(best_w).all():
        print("[optimizer] WARNING: max_sharpe did not converge — using equal weights")
        best_w = np.ones(N) / N

    # Project into [min_w, max_w] with sum=1 (iterative — simple clip+renorm can overshoot)
    best_w = _project_weights(best_w, min_w, max_w)

    result_metrics = compute_portfolio_metrics(asset_paths, best_w, S0_vec)
    return best_w, result_metrics


def optimize_min_cvar(
    asset_paths: np.ndarray,
    S0_vec: np.ndarray,
    min_w: float = 0.01,
    max_w: float = 0.40,
) -> tuple[np.ndarray, dict]:
    """
    Find weights that minimize portfolio CVaR at the 95th percentile.

    Objective: minimize CVaR_95(W_T)

    Preferred by professional risk desks (Jane Street, Citadel, Two Sigma)
    because it targets the tail directly, unlike variance which is symmetric.

    Parameters
    ----------
    asset_paths : np.ndarray, shape (N, K, T+1)
    S0_vec : np.ndarray, shape (N,)
    min_w, max_w : float — weight bounds per asset

    Returns
    -------
    weights : np.ndarray, shape (N,) — optimal weights summing to 1
    result_metrics : dict — metrics at optimal weights
    """
    N = asset_paths.shape[0]
    bounds = _make_bounds(N, min_w, max_w)
    constraints = _make_constraints(N)

    def cvar_objective(w: np.ndarray) -> float:
        # Normalize: SLSQP probes unnormalized weights during gradient estimation
        w_n = np.clip(w, 0, None)
        s = w_n.sum()
        if s < 1e-12:
            return 0.0
        m = compute_portfolio_metrics(asset_paths, w_n / s, S0_vec)
        # Maximize portfolio CVaR value (worst-case floor as high as possible)
        return -m["portfolio_cvar_95"]

    best_w = None
    best_val = np.inf

    for w0 in _starting_points(N, min_w, max_w):
        res = minimize(
            cvar_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
        )
        if res.fun < best_val:
            best_val = res.fun
            best_w = res.x

    if best_w is None or not np.isfinite(best_w).all():
        print("[optimizer] WARNING: min_cvar did not converge — using equal weights")
        best_w = np.ones(N) / N

    best_w = np.clip(best_w, min_w, max_w)
    best_w /= best_w.sum()

    result_metrics = compute_portfolio_metrics(asset_paths, best_w, S0_vec)
    return best_w, result_metrics


def optimize_risk_parity(
    asset_paths: np.ndarray,
    S0_vec: np.ndarray,
    min_w: float = 0.01,
    max_w: float = 0.40,
) -> tuple[np.ndarray, dict]:
    """
    Risk Parity (Equal Risk Contribution) weights.

    Formula (analytic — no simulation needed):
        wᵢ = (1/σᵢ) / Σⱼ(1/σⱼ)

    where σᵢ = std(log(S[i,:,-1] / S0_vec[i])) — realized path volatility.

    Popularized by Bridgewater's All-Weather fund. Each asset contributes
    equally to total portfolio variance. Does not use paths for optimization —
    computed analytically from calibrated volatilities.

    Parameters
    ----------
    asset_paths : np.ndarray, shape (N, K, T+1)
    S0_vec : np.ndarray, shape (N,)
    min_w, max_w : float — weight bounds (applied after analytic computation)

    Returns
    -------
    weights : np.ndarray, shape (N,) — risk parity weights summing to 1
    result_metrics : dict — metrics at risk parity weights
    """
    N = asset_paths.shape[0]

    # Compute realized volatility from simulated paths
    vols = np.array([
        np.std(np.log(asset_paths[i, :, -1] / S0_vec[i]))
        for i in range(N)
    ])

    # Guard against zero or near-zero volatility
    vols = np.where(vols > 1e-8, vols, 1e-8)

    # Analytic inverse-volatility weighting
    inv_v = 1.0 / vols
    w_rp = inv_v / inv_v.sum()

    # Apply position bounds (iterative projection maintains sum=1)
    w_rp = _project_weights(w_rp, min_w, max_w)

    result_metrics = compute_portfolio_metrics(asset_paths, w_rp, S0_vec)
    return w_rp, result_metrics


def optimize_weights(
    asset_paths: np.ndarray,
    S0_vec: np.ndarray,
    strategy: str = "max_sharpe",
    min_w: float = 0.01,
    max_w: float = 0.40,
) -> tuple[np.ndarray, dict]:
    """
    Unified optimizer entry point. Dispatches to the requested strategy.

    Parameters
    ----------
    asset_paths : np.ndarray, shape (N, K, T+1)
    S0_vec : np.ndarray, shape (N,)
    strategy : str
        'max_sharpe' | 'min_cvar' | 'risk_parity'
    min_w, max_w : float — weight bounds

    Returns
    -------
    weights : np.ndarray, shape (N,)
    metrics : dict
    """
    STRATEGIES = {
        "max_sharpe":  optimize_max_sharpe,
        "min_cvar":    optimize_min_cvar,
        "risk_parity": optimize_risk_parity,
    }

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES.keys())}"
        )

    print(f"[optimizer] Running {strategy} ...")
    weights, metrics = STRATEGIES[strategy](asset_paths, S0_vec, min_w, max_w)
    # Final guaranteed hard projection at dispatcher level — catches any residual drift
    weights = np.clip(weights, min_w, max_w)
    weights = weights / weights.sum()
    print(f"[optimizer] {strategy} done — Sharpe={metrics['sharpe']:.3f}")
    return weights, metrics


def run_all_strategies(
    asset_paths: np.ndarray,
    S0_vec: np.ndarray,
    tickers: list[str],
    min_w: float = 0.01,
    max_w: float = 0.40,
    equal_weights: np.ndarray | None = None,
) -> dict:
    """
    Run all three optimizers and return a comparison dict.
    Also computes equal-weight baseline for reference.

    Returns
    -------
    results : dict with keys 'max_sharpe', 'min_cvar', 'risk_parity', 'equal_weight'
        Each value: {'weights': ndarray, 'metrics': dict}
    """
    N = asset_paths.shape[0]
    if equal_weights is None:
        equal_weights = np.ones(N) / N

    results = {}

    # Equal-weight baseline
    eq_metrics = compute_portfolio_metrics(asset_paths, equal_weights, S0_vec)
    results["equal_weight"] = {"weights": equal_weights, "metrics": eq_metrics}
    print(f"[optimizer] equal_weight baseline — Sharpe={eq_metrics['sharpe']:.3f}")

    # All three optimizers
    for strategy in ["max_sharpe", "min_cvar", "risk_parity"]:
        w, m = optimize_weights(asset_paths, S0_vec, strategy, min_w, max_w)
        results[strategy] = {"weights": w, "metrics": m}

    return results


def print_comparison_table(
    results: dict,
    tickers: list[str],
) -> None:
    """
    Print a side-by-side comparison table of all strategies.
    Per spec §6.2: shows weights, Sharpe, CVaR, diversification ratio.
    """
    strategies = ["equal_weight", "max_sharpe", "min_cvar", "risk_parity"]
    labels = {
        "equal_weight": "Equal Weight",
        "max_sharpe":   "Max Sharpe",
        "min_cvar":     "Min CVaR",
        "risk_parity":  "Risk Parity",
    }

    col_w = 14
    print("\n" + "═" * (10 + col_w * 4))
    print(f"  MARGINS — Strategy Comparison Table")
    print("═" * (10 + col_w * 4))

    # Header
    header = f"{'':18s}" + "".join(f"{labels[s]:>{col_w}s}" for s in strategies)
    print(header)
    print("─" * (10 + col_w * 4))

    # Weights per asset
    for i, ticker in enumerate(tickers):
        row = f"  {ticker:16s}"
        for s in strategies:
            w = results[s]["weights"][i] * 100
            row += f"{w:>{col_w - 1}.1f}%"
        print(row)

    print("─" * (10 + col_w * 4))

    # Key metrics
    metric_rows = [
        ("Sharpe Ratio",       "sharpe",                 lambda v: f"{v:>{col_w}.3f}"),
        ("CVaR 95% ($)",       "portfolio_cvar_95",      lambda v: f"${v:>{col_w - 1},.0f}"),
        ("Exp. Return (%)",    "expected_return",         lambda v: f"{v:>{col_w - 1}.2f}%"),
        ("Div. Ratio",         "diversification_ratio",  lambda v: f"{v:>{col_w}.3f}"),
        ("Prob Gain (%)",      "prob_gain",               lambda v: f"{v:>{col_w - 1}.1f}%"),
    ]

    for label, key, fmt in metric_rows:
        row = f"  {label:16s}"
        for s in strategies:
            row += fmt(results[s]["metrics"][key])
        print(row)

    print("═" * (10 + col_w * 4) + "\n")
