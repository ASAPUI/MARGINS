"""
cli_portfolio.py — MARGINS Portfolio Mode — Phase 4
CLI commands: `portfolio` and `optimize`

Usage examples (per spec §6.1 and §6.2):

  python cli_portfolio.py portfolio \\
      --assets GC=F SPY TLT USO BTC-USD \\
      --weights 0.40 0.30 0.20 0.05 0.05 \\
      --days 30 --paths 5000 --period 2y --calib-window 126 \\
      --output portfolio_report.json

  python cli_portfolio.py optimize \\
      --assets GC=F SPY TLT \\
      --strategy all \\
      --days 30 --paths 5000 --max-weight 0.40 \\
      --output optimal_weights.json

Drop this file next to your existing cli.py and import the subcommands,
or run it standalone as shown above.
"""

import sys
import json
import argparse
import numpy as np

sys.path.insert(0, "src")

from portfolio.universe import fetch_universe, get_current_prices, summarize_universe
from portfolio.correlation import compute_cholesky, summarize_correlation
from portfolio.simulator import calibrate_models, simulate_portfolio
from portfolio.metrics import (
    compute_portfolio_metrics,
    compute_scenario_table,
    compute_per_asset_metrics,
    print_metrics_report,
)
from portfolio.optimizer import (
    optimize_weights,
    run_all_strategies,
    print_comparison_table,
)


# ── Shared helpers ────────────────────────────────────────────

def _parse_weights(weight_args, n_assets, tickers):
    """Parse --weights arg: validate they sum to 1, match n_assets."""
    if weight_args is None:
        # Equal weight default
        return np.ones(n_assets) / n_assets

    weights = np.array(weight_args, dtype=float)
    if len(weights) != n_assets:
        raise argparse.ArgumentTypeError(
            f"--weights has {len(weights)} values but --assets has {n_assets} tickers. "
            f"Must match exactly."
        )
    total = weights.sum()
    if abs(total - 1.0) > 0.01:
        raise argparse.ArgumentTypeError(
            f"--weights sum to {total:.4f} but must sum to 1.0. "
            f"Got: {dict(zip(tickers, weights))}"
        )
    # Auto-normalize if within tolerance
    return weights / weights.sum()


def _save_output(data: dict, output_path: str) -> None:
    """Save results dict to JSON or CSV depending on extension."""
    if output_path.endswith(".json"):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=float)
        print(f"\n[cli] Results saved to {output_path}")
    elif output_path.endswith(".csv"):
        import csv
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["key", "value"])
            for k, v in data.items():
                writer.writerow([k, v])
        print(f"\n[cli] Results saved to {output_path}")
    else:
        print(f"\n[cli] WARNING: Unknown output format '{output_path}' — saving as JSON")
        with open(output_path + ".json", "w") as f:
            json.dump(data, f, indent=2, default=float)


# ── Command: portfolio ────────────────────────────────────────

def cmd_portfolio(args) -> dict:
    """
    Run a multi-asset portfolio simulation with given assets and weights.
    Per spec §6.1.

    Output includes: portfolio VaR/CVaR, Sharpe ratio, diversification ratio,
    correlation matrix, per-asset expected return, and scenario table.
    """
    print(f"\n{'═'*58}")
    print(f"  MARGINS — Portfolio Simulation")
    print(f"{'═'*58}")
    print(f"  Assets     : {args.assets}")
    print(f"  Horizon    : {args.days} trading days")
    print(f"  Paths      : {args.paths:,}")
    print(f"  Period     : {args.period}")
    print(f"  Calib win  : {args.calib_window} days")
    print(f"{'═'*58}\n")

    # Step 1: Fetch universe
    combined, log_ret = fetch_universe(
        tickers=args.assets,
        period=args.period,
        calib_window=args.calib_window,
    )
    tickers = list(combined.columns)
    N = len(tickers)
    summarize_universe(combined, log_ret)

    # Step 2: Correlation matrix + Cholesky
    L, cov_lw = compute_cholesky(log_ret)
    summarize_correlation(cov_lw, tickers)

    # Step 3: Calibrate models
    models = calibrate_models(log_ret, tickers)
    S0_vec = get_current_prices(combined)

    # Step 4: Parse weights
    weights = _parse_weights(args.weights, N, tickers)
    print(f"\n[cli] Portfolio weights:")
    for t, w in zip(tickers, weights):
        print(f"  {t:12s}: {w*100:.1f}%")

    # Step 5: Simulate
    paths = simulate_portfolio(
        models=models,
        S0_vec=S0_vec,
        L=L,
        n_steps=args.days,
        n_paths=args.paths,
        seed=42,
    )

    # Step 6: Compute metrics
    metrics = compute_portfolio_metrics(paths, weights, S0_vec, horizon_days=args.days)
    scenarios = compute_scenario_table(paths, weights)
    per_asset = compute_per_asset_metrics(paths, tickers, S0_vec)

    # Step 7: Print report
    print_metrics_report(metrics, weights, tickers, scenarios, per_asset)

    # Step 8: Build output dict
    from portfolio.correlation import get_correlation_matrix
    corr = get_correlation_matrix(cov_lw)
    output = {
        "command":          "portfolio",
        "assets":           tickers,
        "weights":          weights.tolist(),
        "days":             args.days,
        "paths":            args.paths,
        "metrics":          metrics,
        "scenarios":        scenarios,
        "per_asset":        per_asset,
        "correlation_matrix": {
            tickers[i]: {tickers[j]: float(corr[i, j]) for j in range(N)}
            for i in range(N)
        },
    }

    # Step 9: Save output
    if args.output:
        _save_output(output, args.output)

    return output


# ── Command: optimize ─────────────────────────────────────────

def cmd_optimize(args) -> dict:
    """
    Run weight optimization across one or all three strategies.
    Per spec §6.2. Prints side-by-side comparison table.
    """
    print(f"\n{'═'*58}")
    print(f"  MARGINS — Portfolio Optimizer")
    print(f"{'═'*58}")
    print(f"  Assets     : {args.assets}")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Horizon    : {args.days} trading days")
    print(f"  Paths      : {args.paths:,}")
    print(f"  Max weight : {args.max_weight:.0%}")
    print(f"{'═'*58}\n")

    # Steps 1–4: Same universe + calibration pipeline as portfolio
    combined, log_ret = fetch_universe(
        tickers=args.assets,
        period=args.period,
        calib_window=args.calib_window,
    )
    tickers = list(combined.columns)
    N = len(tickers)
    summarize_universe(combined, log_ret)

    L, cov_lw = compute_cholesky(log_ret)
    models = calibrate_models(log_ret, tickers)
    S0_vec = get_current_prices(combined)

    # Simulate paths (shared across all strategies)
    paths = simulate_portfolio(
        models=models,
        S0_vec=S0_vec,
        L=L,
        n_steps=args.days,
        n_paths=args.paths,
        seed=42,
    )

    output = {
        "command":   "optimize",
        "assets":    tickers,
        "days":      args.days,
        "paths":     args.paths,
        "results":   {},
    }

    if args.strategy == "all":
        # Run all 3 + equal-weight baseline → comparison table
        results = run_all_strategies(
            paths, S0_vec, tickers,
            min_w=0.01,
            max_w=args.max_weight,
        )
        print_comparison_table(results, tickers)

        for strategy, data in results.items():
            output["results"][strategy] = {
                "weights": data["weights"].tolist(),
                "metrics": data["metrics"],
            }

    else:
        # Single strategy
        weights, metrics = optimize_weights(
            paths, S0_vec,
            strategy=args.strategy,
            min_w=0.01,
            max_w=args.max_weight,
        )
        print(f"\n  Optimal weights ({args.strategy}):")
        for t, w in zip(tickers, weights):
            print(f"    {t:12s}: {w*100:.1f}%")

        from portfolio.metrics import print_metrics_report
        scenarios = compute_scenario_table(paths, weights)
        per_asset = compute_per_asset_metrics(paths, tickers, S0_vec)
        print_metrics_report(metrics, weights, tickers, scenarios, per_asset)

        output["results"][args.strategy] = {
            "weights": weights.tolist(),
            "metrics": metrics,
        }

    if args.output:
        _save_output(output, args.output)

    return output


# ── Argument parser ───────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with portfolio and optimize subcommands."""
    parser = argparse.ArgumentParser(
        prog="cli_portfolio",
        description="MARGINS Portfolio Mode — Multi-asset Monte Carlo simulation & optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli_portfolio.py portfolio \\
      --assets GC=F SPY TLT USO BTC-USD \\
      --weights 0.40 0.30 0.20 0.05 0.05 \\
      --days 30 --paths 5000

  python cli_portfolio.py optimize \\
      --assets GC=F SPY TLT \\
      --strategy all \\
      --days 30 --paths 5000 --max-weight 0.40
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── Shared arguments (used in both subcommands) ───────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--assets", nargs="+", required=True, metavar="TICKER",
        help="Yahoo Finance ticker symbols (space-separated). E.g. GC=F SPY TLT",
    )
    shared.add_argument(
        "--days", "-d", type=int, default=30, metavar="N",
        help="Forecast horizon in trading days (default: 30)",
    )
    shared.add_argument(
        "--paths", "-p", type=int, default=5000, metavar="K",
        help="Number of Monte Carlo paths (default: 5000)",
    )
    shared.add_argument(
        "--period", default="2y", choices=["6mo", "1y", "2y", "5y"],
        help="Historical data window (default: 2y)",
    )
    shared.add_argument(
        "--calib-window", type=int, default=126, metavar="DAYS",
        help="Calibration window in trading days (default: 126 = 6 months)",
    )
    shared.add_argument(
        "--output", "-o", default=None, metavar="FILE",
        help="Save results to .json or .csv file",
    )

    # ── portfolio subcommand ──────────────────────────────────
    p_portfolio = subparsers.add_parser(
        "portfolio",
        parents=[shared],
        help="Simulate a multi-asset portfolio with given weights",
        description="Run correlated Monte Carlo simulation and compute portfolio risk metrics.",
    )
    p_portfolio.add_argument(
        "--weights", nargs="+", type=float, default=None, metavar="W",
        help="Asset weights (space-separated floats, must sum to 1). Default: equal weight",
    )
    p_portfolio.set_defaults(func=cmd_portfolio)

    # ── optimize subcommand ───────────────────────────────────
    p_optimize = subparsers.add_parser(
        "optimize",
        parents=[shared],
        help="Find optimal weights using max_sharpe / min_cvar / risk_parity",
        description="Optimize portfolio weights across one or all three strategies.",
    )
    p_optimize.add_argument(
        "--strategy",
        default="all",
        choices=["max_sharpe", "min_cvar", "risk_parity", "all"],
        help="Optimization strategy (default: all — runs all 3 + equal-weight baseline)",
    )
    p_optimize.add_argument(
        "--max-weight", type=float, default=0.40, metavar="W",
        help="Maximum allocation per asset, e.g. 0.40 = 40%% (default: 0.40)",
    )
    p_optimize.set_defaults(func=cmd_optimize)

    return parser


# ── Entry point ───────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
