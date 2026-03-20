"""
tests/test_metrics.py — MARGINS Phase 2
Unit tests for metrics.py per spec §7.1:

Mandatory tests:
  ✓ Portfolio VaR < sum of individual VaRs for negatively correlated assets
  ✓ diversification_ratio > 1 for anti-correlated assets
  ✓ All metric keys present and correctly typed
  ✓ Sharpe ratio sign correct (positive drift → positive Sharpe)
  ✓ prob_gain ∈ [0, 100]
  ✓ CVaR ≤ VaR (CVaR is always more conservative)
  ✓ Scenario bear < base < bull (monotone percentiles)
  ✓ Spec §7.3 Check 2: portfolio CVaR > min(individual CVaRs) for ρ=-0.5
"""

import sys
import numpy as np

sys.path.insert(0, "/home/claude/margins/src")

from portfolio.simulator import GBMModel, simulate_portfolio
from portfolio.metrics import (
    compute_portfolio_metrics,
    compute_scenario_table,
    compute_per_asset_metrics,
    print_metrics_report,
)

np.random.seed(42)

TICKERS = ["GC=F", "SPY"]
VOLS = np.array([0.15, 0.18])
DRIFTS = np.array([0.05, 0.10])
S0 = np.array([2050.0, 480.0])
N_PATHS = 20_000
N_STEPS = 30

print("=" * 60)
print("MARGINS Phase 2 — metrics.py Unit Tests")
print("=" * 60)


def make_paths(corr: float, seed: int = 42) -> np.ndarray:
    """Generate 2-asset correlated paths with given correlation."""
    COV = np.array([
        [VOLS[0] ** 2, corr * VOLS[0] * VOLS[1]],
        [corr * VOLS[0] * VOLS[1], VOLS[1] ** 2],
    ])
    L = np.linalg.cholesky(COV)
    models = [GBMModel(mu=DRIFTS[i], sigma=VOLS[i], name=TICKERS[i]) for i in range(2)]
    return simulate_portfolio(models, S0, L, N_STEPS, N_PATHS, seed=seed)


# ── TEST 1: All metric keys present, correct types ─────────────
print("\n[TEST 1] All metric keys present and correctly typed")
paths_neg = make_paths(corr=-0.5)
weights = np.array([0.5, 0.5])
m = compute_portfolio_metrics(paths_neg, weights, S0)

REQUIRED_KEYS = [
    "portfolio_var_95", "portfolio_cvar_95", "expected_return",
    "sharpe", "diversification_ratio", "avg_max_drawdown",
    "prob_gain", "correlation_benefit", "portfolio_vol",
    "expected_value", "initial_value", "n_paths", "n_steps",
]
for key in REQUIRED_KEYS:
    assert key in m, f"FAIL: missing key '{key}'"
    assert isinstance(m[key], (int, float)), f"FAIL: key '{key}' has type {type(m[key])}"

print(f"✓ TEST 1 PASSED — all {len(REQUIRED_KEYS)} keys present, correct types\n")


# ── TEST 2: CVaR ≤ VaR (tail mean ≤ threshold) ────────────────
print("[TEST 2] CVaR ≤ VaR (tail mean is more conservative)")
assert m["portfolio_cvar_95"] <= m["portfolio_var_95"], (
    f"FAIL: CVaR {m['portfolio_cvar_95']:.2f} > VaR {m['portfolio_var_95']:.2f}"
)
print(f"  VaR 95%  = ${m['portfolio_var_95']:,.2f}")
print(f"  CVaR 95% = ${m['portfolio_cvar_95']:,.2f}")
print(f"✓ TEST 2 PASSED — CVaR ≤ VaR ✓\n")


# ── TEST 3: Prob of gain in [0, 100] ──────────────────────────
print("[TEST 3] prob_gain ∈ [0, 100]")
assert 0 <= m["prob_gain"] <= 100, f"FAIL: prob_gain = {m['prob_gain']}"
print(f"  prob_gain = {m['prob_gain']:.1f}%")
print(f"✓ TEST 3 PASSED\n")


# ── TEST 4: Sharpe sign correct ────────────────────────────────
print("[TEST 4] Sharpe ratio — positive drift → positive Sharpe")
# High drift assets should produce positive Sharpe
assert m["sharpe"] > 0, f"FAIL: Sharpe = {m['sharpe']:.3f} (expected > 0 with positive drift)"
print(f"  Sharpe = {m['sharpe']:.3f}")
print(f"✓ TEST 4 PASSED\n")


# ── TEST 5: diversification_ratio > 1 for anti-correlated assets
print("[TEST 5] diversification_ratio > 1.0 for ρ = -0.5")
assert m["diversification_ratio"] > 1.0, (
    f"FAIL: diversification_ratio = {m['diversification_ratio']:.4f} (expected > 1)"
)
print(f"  Diversification ratio = {m['diversification_ratio']:.4f}")
print(f"✓ TEST 5 PASSED\n")


# ── TEST 6: Spec §7.1 — Portfolio VaR < sum of individual VaRs ─
print("[TEST 6] Portfolio VaR < sum of individual VaRs (neg correlation)")
ind_var_0 = float(np.percentile(paths_neg[0, :, -1], 5))
ind_var_1 = float(np.percentile(paths_neg[1, :, -1], 5))
sum_ind_vars = weights[0] * ind_var_0 + weights[1] * ind_var_1
port_var = m["portfolio_var_95"]
print(f"  Weighted sum of individual VaRs: ${sum_ind_vars:,.2f}")
print(f"  Portfolio VaR:                   ${port_var:,.2f}")
assert port_var > sum_ind_vars, (
    f"FAIL: Portfolio VaR ${port_var:,.2f} not > weighted sum ${sum_ind_vars:,.2f}\n"
    f"  (Spec §7.1: portfolio VaR < sum — meaning portfolio value is HIGHER in bad scenarios)"
)
print(f"✓ TEST 6 PASSED — diversification raises portfolio floor ✓\n")


# ── TEST 7: Spec §7.3 Check 2 — Portfolio CVaR > min individual CVaRs ──
print("[TEST 7] Spec §7.3 Check 2 — portfolio CVaR > min(individual CVaRs)")
ind_cvar_0 = float(np.mean(paths_neg[0, :, -1][paths_neg[0, :, -1] <= ind_var_0]))
ind_cvar_1 = float(np.mean(paths_neg[1, :, -1][paths_neg[1, :, -1] <= ind_var_1]))
port_cvar = m["portfolio_cvar_95"]
print(f"  CVaR GC=F        : ${ind_cvar_0:,.2f}")
print(f"  CVaR SPY         : ${ind_cvar_1:,.2f}")
print(f"  CVaR Portfolio   : ${port_cvar:,.2f}")
assert port_cvar > min(ind_cvar_0, ind_cvar_1), (
    f"FAIL: portfolio CVaR ${port_cvar:,.2f} not > min individual ${min(ind_cvar_0, ind_cvar_1):,.2f}"
)
print(f"✓ TEST 7 PASSED — diversification benefit confirmed ✓\n")


# ── TEST 8: Scenario table — bear < base < bull ────────────────
print("[TEST 8] Scenario table — bear < base < bull (monotone)")
scenarios = compute_scenario_table(paths_neg, weights)
assert scenarios["bear"] < scenarios["base"] < scenarios["bull"], (
    f"FAIL: scenarios not monotone: {scenarios}"
)
print(f"  Bear: ${scenarios['bear']:,.2f} | Base: ${scenarios['base']:,.2f} | Bull: ${scenarios['bull']:,.2f}")
print(f"✓ TEST 8 PASSED\n")


# ── TEST 9: Correlation effect — neg corr beats pos corr ──────
print("[TEST 9] Negative correlation → better Sharpe & diversification than positive")
paths_pos = make_paths(corr=+0.7, seed=42)
m_pos = compute_portfolio_metrics(paths_pos, weights, S0)
m_neg = compute_portfolio_metrics(paths_neg, weights, S0)

print(f"  Diversification ratio ρ=+0.7 : {m_pos['diversification_ratio']:.4f}")
print(f"  Diversification ratio ρ=-0.5 : {m_neg['diversification_ratio']:.4f}")
assert m_neg["diversification_ratio"] > m_pos["diversification_ratio"], (
    "FAIL: neg correlation should give higher diversification ratio than pos"
)
print(f"✓ TEST 9 PASSED — neg correlation produces more diversification ✓\n")


# ── TEST 10: Full report printout ─────────────────────────────
print("[TEST 10] Full metrics report — visual inspection")
per_asset = compute_per_asset_metrics(paths_neg, TICKERS, S0)
print_metrics_report(m, weights, TICKERS, scenarios, per_asset)
print("✓ TEST 10 PASSED — report rendered without errors\n")


# ── Summary ───────────────────────────────────────────────────
print("=" * 60)
print("✅  ALL PHASE 2 ACCEPTANCE CRITERIA PASSED")
print("=" * 60)
print("  TEST 1:  All metric keys present & typed        ✓")
print("  TEST 2:  CVaR ≤ VaR                            ✓")
print("  TEST 3:  prob_gain ∈ [0, 100]                  ✓")
print("  TEST 4:  Sharpe sign correct                    ✓")
print("  TEST 5:  diversification_ratio > 1 (ρ=-0.5)   ✓")
print("  TEST 6:  Portfolio VaR < sum individual VaRs   ✓")
print("  TEST 7:  Spec §7.3 Check 2 — CVaR benefit      ✓")
print("  TEST 8:  Scenario table monotone               ✓")
print("  TEST 9:  Neg corr > pos corr diversification   ✓")
print("  TEST 10: Report renders cleanly                ✓")
print("\nReady for Phase 3 — optimizer.py")
