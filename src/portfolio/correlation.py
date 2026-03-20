"""
correlation.py — MARGINS Portfolio Mode
Ledoit-Wolf covariance shrinkage + Cholesky decomposition.

Key design decisions:
- Always works in annualized log-return space (daily returns × 252)
  so μᵢ and σᵢ are in the same unit space as Σ (avoids scaling errors)
- Ledoit-Wolf shrinkage is MANDATORY — raw sample covariance is prohibited
  (see spec §8: PROHIBITED: Raw sample covariance matrix without shrinkage)
- Ridge fallback (1e-6 × I) handles rare numerical edge cases after shrinkage
- Returns both L (Cholesky factor) and cov_lw (full matrix) for diagnostics
"""

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf


def compute_cholesky(
    log_returns_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Ledoit-Wolf shrinkage and return the Cholesky factor L.

    Parameters
    ----------
    log_returns_df : pd.DataFrame
        Daily log-return DataFrame, shape (T, N).
        Produced by universe.fetch_universe().

    Returns
    -------
    L : np.ndarray, shape (N, N)
        Lower-triangular Cholesky factor such that L @ L.T == Sigma_LW
        to numerical precision.
    cov_lw : np.ndarray, shape (N, N)
        The Ledoit-Wolf shrunk covariance matrix (annualized).

    Raises
    ------
    np.linalg.LinAlgError
        Only if both standard Cholesky and ridge fallback fail (extremely rare).
    """
    # --- 1. Annualize daily log-returns ---
    # Multiply by 252 to put in the same unit space as μᵢ and σᵢ
    rets = log_returns_df.values * 252  # shape (T, N)

    # --- 2. Ledoit-Wolf shrinkage ---
    # Returns (shrunk_covariance, shrinkage_coefficient)
    # ledoit_wolf() expects shape (n_samples, n_features) = (T, N) ✓
    cov_lw, alpha = ledoit_wolf(rets)

    print(
        f"[correlation] Ledoit-Wolf shrinkage intensity α = {alpha:.4f} "
        f"({'high — few obs relative to assets' if alpha > 0.3 else 'moderate'})"
    )

    # --- 3. Cholesky decomposition ---
    try:
        L = np.linalg.cholesky(cov_lw)
    except np.linalg.LinAlgError:
        # Rare edge case: apply small ridge regularization
        ridge = 1e-6 * np.eye(cov_lw.shape[0])
        print("[correlation] WARNING: Cholesky failed — applying ridge 1e-6. Matrix may be near-singular.")
        L = np.linalg.cholesky(cov_lw + ridge)

    # --- 4. Verify: L @ L.T should equal cov_lw to numerical precision ---
    reconstruction_error = np.max(np.abs(L @ L.T - cov_lw))
    assert reconstruction_error < 1e-10, (
        f"Cholesky reconstruction error too large: {reconstruction_error:.2e}"
    )

    print(f"[correlation] Cholesky OK | max reconstruction error: {reconstruction_error:.2e}")

    return L, cov_lw


def get_correlation_matrix(cov_lw: np.ndarray) -> np.ndarray:
    """
    Convert the Ledoit-Wolf covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov_lw : np.ndarray, shape (N, N)
        Annualized shrunk covariance matrix.

    Returns
    -------
    corr : np.ndarray, shape (N, N)
        Correlation matrix with diagonal = 1.0
    """
    std = np.sqrt(np.diag(cov_lw))
    corr = cov_lw / np.outer(std, std)
    # Clip to [-1, 1] to handle floating-point drift
    return np.clip(corr, -1.0, 1.0)


def summarize_correlation(
    cov_lw: np.ndarray,
    tickers: list[str],
) -> None:
    """Print the correlation matrix and per-asset annualized volatilities."""
    corr = get_correlation_matrix(cov_lw)
    vols = np.sqrt(np.diag(cov_lw))

    print("\n── Ledoit-Wolf Correlation Matrix ───────────────────")
    header = f"{'':12s}" + "".join(f"{t:>10s}" for t in tickers)
    print(header)
    for i, ti in enumerate(tickers):
        row = f"{ti:12s}" + "".join(f"{corr[i, j]:>10.3f}" for j in range(len(tickers)))
        print(row)

    print("\n── Annualized Volatilities (from Ledoit-Wolf Σ) ────")
    for i, t in enumerate(tickers):
        print(f"  {t:12s}: {vols[i] * 100:.2f}%")
    print("─────────────────────────────────────────────────────\n")
