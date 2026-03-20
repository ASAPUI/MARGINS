"""
simulator.py — MARGINS Portfolio Mode
Vectorized correlated multi-asset Monte Carlo simulation engine.

Key design decisions:
- Correlation injected via Cholesky at EVERY time step (not post-hoc on terminal prices)
  This is the scientifically correct approach per spec §2.3 and Burgess (2022)
- Single np.einsum() call for the Cholesky transform — no asset-level loops in hot path
- GBM (Geometric Brownian Motion) as default per-asset model
  Future: model.simulate() dispatch for OU / Heston / Merton Jump Diffusion
- Vectorized over paths; only time steps are looped (unavoidable for path dependency)
- Target: 5 assets × 5,000 paths × 30 steps < 3 seconds on standard CPU
"""

import numpy as np
import time


class GBMModel:
    """
    Geometric Brownian Motion model parameters for a single asset.
    Calibrated from historical log-returns.

    dS = μ·S·dt + σ·S·dW

    Parameters
    ----------
    mu : float
        Annualized drift (log-return mean × 252)
    sigma : float
        Annualized volatility (log-return std × √252)
    name : str
        Asset identifier (ticker)
    """

    def __init__(self, mu: float, sigma: float, name: str = ""):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.name = name

    def __repr__(self):
        return f"GBMModel({self.name}: μ={self.mu:.4f}, σ={self.sigma:.4f})"


def calibrate_models(
    log_returns_df,
    tickers: list[str],
) -> list[GBMModel]:
    """
    Calibrate GBM parameters for each asset from log-return history.

    Parameters
    ----------
    log_returns_df : pd.DataFrame
        Daily log-returns, shape (T, N). From universe.fetch_universe().
    tickers : list[str]
        Asset names in column order.

    Returns
    -------
    models : list[GBMModel]
        One GBMModel per asset, calibrated on the provided log-returns.
    """
    models = []
    print("[simulator] Calibrating GBM models:")
    for ticker in tickers:
        rets = log_returns_df[ticker].values
        mu = float(np.mean(rets) * 252)          # annualized drift
        sigma = float(np.std(rets) * np.sqrt(252))  # annualized vol
        model = GBMModel(mu=mu, sigma=sigma, name=ticker)
        models.append(model)
        print(f"  {ticker:12s}: μ={mu*100:+.2f}%/yr  σ={sigma*100:.2f}%/yr")
    return models


def simulate_portfolio(
    models: list[GBMModel],
    S0_vec: np.ndarray,
    L: np.ndarray,
    n_steps: int,
    n_paths: int,
    dt: float = 1 / 252,
    seed: int = 42,
) -> np.ndarray:
    """
    Vectorized correlated multi-asset Monte Carlo simulation.

    Algorithm (per spec §3.2):
    1. Generate independent N(0,1) shocks: ε, shape (N, K, T)
    2. Apply Cholesky correlation: Z = einsum('ij,jkt->ikt', L, ε)
    3. Simulate GBM paths: S[i,k,t] = S[i,k,t-1] * exp((μᵢ - σᵢ²/2)·dt + σᵢ·√dt·Z[i,k,t])

    Parameters
    ----------
    models : list[GBMModel]
        Calibrated model for each asset. len(models) == N.
    S0_vec : np.ndarray, shape (N,)
        Current prices for each asset (starting point for simulation).
    L : np.ndarray, shape (N, N)
        Lower-triangular Cholesky factor from correlation.compute_cholesky().
    n_steps : int
        Number of trading day steps to simulate (horizon T).
    n_paths : int
        Number of Monte Carlo paths K.
    dt : float
        Time step size in years. Default 1/252 (one trading day).
    seed : int or None
        RNG seed for reproducibility. Set None for random draws.

    Returns
    -------
    paths : np.ndarray, shape (N, K, T+1)
        Simulated price paths. paths[i, k, 0] = S0_vec[i] for all k.
        paths[i, k, t] is the price of asset i on path k at time step t.
    """
    t_start = time.time()

    N = len(models)
    K = n_paths
    T = n_steps

    if seed is not None:
        np.random.seed(seed)

    # --- Step 1: Independent standard normal shocks ---
    # Shape: (N, K, T) — N assets, K paths, T time steps
    eps = np.random.standard_normal((N, K, T))

    # --- Step 2: Apply Cholesky to inject correlation ---
    # Z[i,k,t] = sum_j L[i,j] * eps[j,k,t]
    # einsum 'ij,jkt->ikt': matrix multiply L over asset dimension
    # Result shape: (N, K, T) — same as eps but now correlated across assets
    Z = np.einsum("ij,jkt->ikt", L, eps)

    # --- Step 3: Initialize paths array ---
    # Shape: (N, K, T+1) — T+1 includes t=0 (initial prices)
    paths = np.zeros((N, K, T + 1))
    for i in range(N):
        paths[i, :, 0] = S0_vec[i]  # all K paths start at current price

    # --- Step 4: Simulate GBM paths ---
    # Vectorized over paths K; loop only over time T (path-dependent)
    sqrt_dt = np.sqrt(dt)
    for t in range(1, T + 1):
        for i, model in enumerate(models):
            mu = model.mu
            sigma = model.sigma
            drift = (mu - 0.5 * sigma ** 2) * dt          # deterministic drift
            diffusion = sigma * sqrt_dt * Z[i, :, t - 1]  # stochastic term, shape (K,)
            paths[i, :, t] = paths[i, :, t - 1] * np.exp(drift + diffusion)

    elapsed = time.time() - t_start
    print(
        f"[simulator] {N} assets × {K:,} paths × {T} steps "
        f"completed in {elapsed:.3f}s"
    )

    # Performance budget check (spec §7.3 Check 4)
    if N >= 5 and K >= 5000 and T >= 30 and elapsed > 3.0:
        print(
            f"[simulator] WARNING: Exceeded 3s performance budget ({elapsed:.2f}s). "
            f"Consider reducing paths or using joblib parallelization."
        )

    return paths


def compute_portfolio_values(
    paths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Compute portfolio value paths from asset paths and weights.

    W[k, t] = Σᵢ wᵢ · S[i, k, t]

    Parameters
    ----------
    paths : np.ndarray, shape (N, K, T+1)
        Asset price paths from simulate_portfolio().
    weights : np.ndarray, shape (N,)
        Portfolio weights. Must sum to 1.

    Returns
    -------
    W : np.ndarray, shape (K, T+1)
        Portfolio value path for each of the K Monte Carlo scenarios.
    """
    # einsum 'i,ikt->kt': weighted sum over assets for each path and time step
    W = np.einsum("i,ikt->kt", weights, paths)
    return W
