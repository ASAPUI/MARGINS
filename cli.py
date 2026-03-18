"""
Monte Carlo Gold Price Predictor — Command Line Interface
coding and created by me of course.

Full-featured CLI for running simulations, backtests,
and risk analysis from the terminal.

Usage:
    python cli.py simulate --model ou --days 30 --paths 5000
    python cli.py backtest --models gbm ou merton --period 2y
    python cli.py risk     --model heston --days 90
    python cli.py price
    python cli.py (command) --(model or the duration) 
Author: Essabri Ali Rayan
Version: 1.2
"""

import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List

# ── ANSI Colors ────────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GOLD   = "\033[38;5;220m"
    GOLD2  = "\033[38;5;178m"
    GREEN  = "\033[38;5;82m"
    RED    = "\033[38;5;196m"
    BLUE   = "\033[38;5;75m"
    CYAN   = "\033[38;5;51m"
    MUTED  = "\033[38;5;245m"
    WHITE  = "\033[38;5;255m"

def _no_color():
    """Disable colors if not in a terminal."""
    for attr in vars(C):
        if not attr.startswith("_"):
            setattr(C, attr, "")

if not sys.stdout.isatty():
    _no_color()


# ── Banner ─────────────────────────────────────────────────────────────────────
BANNER = f"""
{C.GOLD}╔══════════════════════════════════════════════════════╗
║      Monte Carlo Gold Price Predictor  v1.0          ║
║      Essabri Ali Rayan                               ║
╚══════════════════════════════════════════════════════╝{C.RESET}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────
def print_section(title: str):
    width = 56
    bar   = "─" * width
    print(f"\n{C.GOLD}{bar}{C.RESET}")
    print(f"{C.BOLD}{C.GOLD2}  {title}{C.RESET}")
    print(f"{C.GOLD}{bar}{C.RESET}")


def print_metric(label: str, value: str, change: Optional[str] = None):
    pad   = 28
    color = C.GREEN if change and change.startswith("+") else \
            C.RED   if change and change.startswith("-") else C.GOLD2
    line  = f"  {C.MUTED}{label:<{pad}}{C.RESET}{C.WHITE}{value}{C.RESET}"
    if change:
        line += f"  {color}{change}{C.RESET}"
    print(line)


def print_table(df: pd.DataFrame, title: Optional[str] = None):
    if title:
        print(f"\n{C.BOLD}{title}{C.RESET}")
    col_widths = {col: max(len(str(col)), df[col].astype(str).str.len().max())
                  for col in df.columns}
    header = "  ".join(f"{C.BOLD}{C.GOLD2}{str(col):<{col_widths[col]}}{C.RESET}"
                        for col in df.columns)
    print(f"  {header}")
    print(f"  {C.MUTED}{'  '.join('-' * w for w in col_widths.values())}{C.RESET}")
    for _, row in df.iterrows():
        line = "  ".join(f"{C.WHITE}{str(row[col]):<{col_widths[col]}}{C.RESET}"
                          for col in df.columns)
        print(f"  {line}")


def progress_bar(current: int, total: int, width: int = 40, label: str = ""):
    filled = int(width * current / total)
    bar    = f"{C.GOLD}{'█' * filled}{C.MUTED}{'░' * (width - filled)}{C.RESET}"
    pct    = f"{C.BOLD}{current/total*100:5.1f}%{C.RESET}"
    print(f"\r  {bar} {pct}  {C.MUTED}{label}{C.RESET}", end="", flush=True)


# ── Data Layer ─────────────────────────────────────────────────────────────────
def fetch_prices(period: str = "2y", symbol: str = "GC=F") -> pd.Series:
    """Fetch gold prices from Yahoo Finance with synthetic fallback."""
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError("Empty data")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        prices = df["Close"].dropna()
        print(f"  {C.GREEN}✓{C.RESET} Fetched {len(prices)} days from Yahoo Finance ({symbol})")
        return prices
    except Exception as e:
        print(f"  {C.MUTED}⚠ Yahoo Finance unavailable ({e}), using synthetic data{C.RESET}")
        dates = pd.date_range(end=datetime.today(), periods=504, freq="B")
        rng   = np.random.default_rng(0)
        log_r = rng.normal(0.0002, 0.012, len(dates))
        return pd.Series(1800 * np.exp(np.cumsum(log_r)), index=dates, name="price")


def get_current_price(symbol: str = "GC=F") -> float:
    try:
        import yfinance as yf
        data = yf.Ticker(symbol).history(period="5d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        pass
    return 2350.0


# ── Model Factory ──────────────────────────────────────────────────────────────
MODEL_MAP = {
    "gbm":     ("src.models.gbm",            "GeometricBrownianMotion"),
    "ou":      ("src.models.mean_reversion",  "OrnsteinUhlenbeckModel"),
    "merton":  ("src.models.jump_diffusion",  "MertonJumpModel"),
    "heston":  ("src.models.heston",          "HestonModel"),
    "regime":  ("src.models.regime_switching","RegimeSwitchingModel"),
}

MODEL_LABELS = {
    "gbm":    "GBM",
    "ou":     "Mean Reversion (OU)",
    "merton": "Jump Diffusion",
    "heston": "Heston",
    "regime": "Regime Switching",
}


class FallbackGBM:
    """Minimal GBM used when src/ not available."""
    name = "GBM"

    def __init__(self):
        self.mu = 0.05
        self.sigma = 0.15

    def calibrate(self, data):
        if len(data) > 10:
            returns = np.log(data[1:] / data[:-1]) if data[0] > 10 else data
            self.mu    = float(np.mean(returns) * 252)
            self.sigma = float(np.std(returns)  * np.sqrt(252))

    def simulate(self, S0, n_steps, n_paths=1000, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        dt  = 1 / 252
        Z   = np.random.standard_normal((n_paths, n_steps - 1))
        out = np.zeros((n_paths, n_steps))
        out[:, 0] = S0
        for t in range(1, n_steps):
            drift = (self.mu - 0.5 * self.sigma ** 2) * dt
            diff  = self.sigma * np.sqrt(dt) * Z[:, t - 1]
            out[:, t] = out[:, t - 1] * np.exp(drift + diff)
        return out


def load_model(key: str):
    """Import model class from src/ or fall back gracefully."""
    if key not in MODEL_MAP:
        print(f"  {C.RED}✗ Unknown model '{key}'. Choose: {', '.join(MODEL_MAP)}{C.RESET}")
        sys.exit(1)
    module_path, class_name = MODEL_MAP[key]
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except ImportError:
        print(f"  {C.MUTED}⚠ src/ not found, using fallback GBM for '{key}'{C.RESET}")
        return FallbackGBM


def build_and_calibrate(key: str, prices: pd.Series):
    """Instantiate + calibrate a model."""
    ModelCls = load_model(key)
    model    = ModelCls()

    log_returns = np.log(prices / prices.shift(1)).dropna().values

    if hasattr(model, "calibrate"):
        if key in ("ou", "heston", "regime"):
            model.calibrate(prices.values)
        else:
            model.calibrate(log_returns)

    return model


# ── Risk Calculations ─────────────────────────────────────────────────────────
def compute_risk(paths: np.ndarray, S0: float) -> dict:
    final = paths[:, -1]
    log_r = np.log(final / S0)

    var_95  = np.percentile(final, 5)
    var_99  = np.percentile(final, 1)
    cvar_95 = np.mean(final[final <= var_95]) if np.any(final <= var_95) else var_95

    dd_list = []
    for path in paths:
        peak = np.maximum.accumulate(path)
        dd   = np.min((path - peak) / peak)
        dd_list.append(dd)

    return {
        "mean":         float(np.mean(final)),
        "median":       float(np.median(final)),
        "std":          float(np.std(final)),
        "p5":           float(np.percentile(final, 5)),
        "p25":          float(np.percentile(final, 25)),
        "p75":          float(np.percentile(final, 75)),
        "p95":          float(np.percentile(final, 95)),
        "var_95":       float(var_95),
        "var_99":       float(var_99),
        "cvar_95":      float(cvar_95),
        "prob_up":      float(np.mean(final > S0)),
        "prob_loss_5":  float(np.mean(log_r < -0.05)),
        "prob_loss_10": float(np.mean(log_r < -0.10)),
        "prob_gain_10": float(np.mean(log_r >  0.10)),
        "avg_max_dd":   float(np.mean(dd_list)),
        "vol":          float(np.std(log_r) * 100),
        "skew":         float(pd.Series(log_r).skew()),
        "kurt":         float(pd.Series(log_r).kurt()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_price(args):
    """Fetch and display current gold price."""
    print_section("Current Gold Price")
    try:
        import yfinance as yf
        data = yf.Ticker("GC=F").history(period="5d", interval="1d")
        price   = float(data["Close"].iloc[-1])
        prev    = float(data["Close"].iloc[-2])
        change  = price - prev
        pct     = change / prev * 100
        vol_1d  = float(data["Close"].pct_change().std() * 100)

        print_metric("Symbol",        "GC=F  (COMEX Gold Futures)")
        print_metric("Current Price", f"${price:,.2f}",
                     f"{'+' if change >= 0 else ''}{change:.2f} ({pct:+.2f}%)")
        print_metric("Previous Close",f"${prev:,.2f}")
        print_metric("5-Day Volatility", f"{vol_1d:.2f}%")
        print_metric("Last Updated",  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print(f"  {C.RED}Error fetching price: {e}{C.RESET}")


def cmd_simulate(args):
    """Run Monte Carlo simulation and display results."""
    print_section(f"Simulation  —  {MODEL_LABELS.get(args.model, args.model)}")

    # ── Fetch data ─────────────────────────────────────────────────────────────
    prices = fetch_prices(args.period)
    S0     = get_current_price() if not args.price else args.price

    print_metric("Initial Price",    f"${S0:,.2f}")
    print_metric("Model",            MODEL_LABELS.get(args.model, args.model))
    print_metric("Horizon",          f"{args.days} trading days")
    print_metric("Paths",            f"{args.paths:,}")
    print_metric("Training Period",  args.period)
    print()

    # ── Calibrate ──────────────────────────────────────────────────────────────
    print(f"  {C.MUTED}Calibrating model...{C.RESET}")
    model = build_and_calibrate(args.model, prices)

    # ── Simulate ───────────────────────────────────────────────────────────────
    print(f"  {C.MUTED}Running {args.paths:,} simulations...{C.RESET}")
    seed  = args.seed if args.seed >= 0 else None
    paths = model.simulate(
        S0=S0, n_steps=args.days + 1,
        n_paths=args.paths, random_seed=seed,
    )

    # ── Results ────────────────────────────────────────────────────────────────
    r = compute_risk(paths, S0)
    final = paths[:, -1]

    print_section("Results")
    print_metric("Expected Price",   f"${r['mean']:,.2f}", f"{(r['mean']/S0-1)*100:+.1f}%")
    print_metric("Median Price",     f"${r['median']:,.2f}", f"{(r['median']/S0-1)*100:+.1f}%")
    print_metric("Std Deviation",    f"${r['std']:,.2f}")
    print()
    print_metric("90% CI Lower",     f"${r['p5']:,.2f}")
    print_metric("90% CI Upper",     f"${r['p95']:,.2f}")
    print()
    print_metric("Prob. of Gain",    f"{r['prob_up']*100:.1f}%")
    print_metric("Prob. Gain > 10%", f"{r['prob_gain_10']*100:.1f}%")
    print_metric("Prob. Loss > 10%", f"{r['prob_loss_10']*100:.1f}%")
    print()

    # Scenario table
    scenarios = pd.DataFrame({
        "Scenario":    ["Bear (5%)", "Mild Bear (25%)", "Base (50%)", "Mild Bull (75%)", "Bull (95%)"],
        "Price ($)":   [f"{r['p5']:,.0f}", f"{r['p25']:,.0f}", f"{r['median']:,.0f}",
                        f"{r['p75']:,.0f}", f"{r['p95']:,.0f}"],
        "Change":      [f"{(r['p5']/S0-1)*100:+.1f}%",  f"{(r['p25']/S0-1)*100:+.1f}%",
                        f"{(r['median']/S0-1)*100:+.1f}%", f"{(r['p75']/S0-1)*100:+.1f}%",
                        f"{(r['p95']/S0-1)*100:+.1f}%"],
    })
    print_table(scenarios, "Scenario Analysis")

    # ── Export ─────────────────────────────────────────────────────────────────
    if args.output:
        _export_results(args.output, {
            "model": args.model, "S0": S0, "days": args.days,
            "paths": args.paths, "results": r,
            "timestamp": datetime.now().isoformat(),
        })
        print(f"\n  {C.GREEN}✓ Results saved to {args.output}{C.RESET}")


def cmd_risk(args):
    """Deep risk analysis for a single model."""
    print_section(f"Risk Analysis  —  {MODEL_LABELS.get(args.model, args.model)}")

    prices = fetch_prices(args.period)
    S0     = get_current_price() if not args.price else args.price

    print(f"  {C.MUTED}Calibrating & simulating...{C.RESET}")
    model = build_and_calibrate(args.model, prices)
    seed  = args.seed if args.seed >= 0 else None
    paths = model.simulate(S0=S0, n_steps=args.days + 1,
                           n_paths=args.paths, random_seed=seed)

    r = compute_risk(paths, S0)

    print_section("Value at Risk")
    print_metric("VaR 95%",  f"${r['var_95']:,.2f}", f"{(r['var_95']/S0-1)*100:+.1f}%")
    print_metric("VaR 99%",  f"${r['var_99']:,.2f}", f"{(r['var_99']/S0-1)*100:+.1f}%")
    print_metric("CVaR 95%", f"${r['cvar_95']:,.2f}", f"{(r['cvar_95']/S0-1)*100:+.1f}%")

    print_section("Distribution Statistics")
    print_metric("Realised Volatility", f"{r['vol']:.2f}%")
    print_metric("Skewness",            f"{r['skew']:.4f}")
    print_metric("Kurtosis (excess)",   f"{r['kurt']:.4f}")
    print_metric("Avg Max Drawdown",    f"{r['avg_max_dd']*100:.2f}%")

    print_section("Tail Risk Probabilities")
    print_metric("P(Loss > 5%)",  f"{r['prob_loss_5']*100:.2f}%")
    print_metric("P(Loss > 10%)", f"{r['prob_loss_10']*100:.2f}%")
    print_metric("P(Gain > 10%)", f"{r['prob_gain_10']*100:.2f}%")

    if args.target:
        final = paths[:, -1]
        prob_reach = np.mean(final >= args.target) * 100
        print_section(f"Target Analysis  —  ${args.target:,}")
        print_metric("Probability of reaching target", f"{prob_reach:.2f}%")
        above = final[final >= args.target]
        if len(above):
            print_metric("Average price when above target", f"${np.mean(above):,.2f}")

    if args.output:
        _export_results(args.output, {
            "model": args.model, "S0": S0, "days": args.days,
            "risk_metrics": r, "timestamp": datetime.now().isoformat(),
        })
        print(f"\n  {C.GREEN}✓ Risk report saved to {args.output}{C.RESET}")


def cmd_backtest(args):
    """Walk-forward backtest across one or more models."""
    print_section("Walk-Forward Backtest")

    prices = fetch_prices(args.period)
    models_to_test = args.models if args.models else ["gbm", "ou"]

    print_metric("Models",        ", ".join(MODEL_LABELS.get(m, m) for m in models_to_test))
    print_metric("Price History", f"{len(prices)} days")
    print_metric("Train Window",  f"{args.train_window} days")
    print_metric("Test Window",   f"{args.test_window} days")
    print()

    all_results = {}

    for model_key in models_to_test:
        label = MODEL_LABELS.get(model_key, model_key)
        print(f"\n  {C.GOLD}▸ Backtesting {label}...{C.RESET}")

        n         = len(prices)
        start     = args.train_window
        step      = args.test_window
        windows   = []

        while start + args.test_window <= n:
            windows.append((
                slice(start - args.train_window, start),
                slice(start, min(start + args.test_window, n)),
            ))
            start += step

        window_metrics = []

        for i, (train_sl, test_sl) in enumerate(windows):
            progress_bar(i + 1, len(windows), label=f"window {i+1}/{len(windows)}")

            train_prices = prices.iloc[train_sl]
            test_prices  = prices.iloc[test_sl]

            try:
                model = build_and_calibrate(model_key, train_prices)
                S0_w  = float(train_prices.iloc[-1])
                n_d   = len(test_prices)
                paths = model.simulate(S0=S0_w, n_steps=n_d + 1,
                                       n_paths=args.paths, random_seed=42)
                preds   = np.mean(paths[:, 1:], axis=0)
                actuals = test_prices.values

                errors  = preds - actuals
                mask    = actuals != 0
                mape    = np.mean(np.abs(errors[mask] / actuals[mask])) * 100

                if len(preds) > 1:
                    dir_acc = np.mean((np.diff(preds) > 0) == (np.diff(actuals) > 0)) * 100
                else:
                    dir_acc = 50.0

                window_metrics.append({
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "mae":  float(np.mean(np.abs(errors))),
                    "mape": float(mape),
                    "dir_accuracy": float(dir_acc),
                })
            except Exception as e:
                pass

        print()  # newline after progress bar

        if window_metrics:
            agg = {k: np.mean([m[k] for m in window_metrics]) for k in window_metrics[0]}
            all_results[label] = agg

    # ── Results Table ──────────────────────────────────────────────────────────
    print_section("Backtest Results")
    if all_results:
        rows = []
        for model_label, metrics in all_results.items():
            rows.append({
                "Model":    model_label,
                "RMSE":     f"${metrics['rmse']:,.1f}",
                "MAE":      f"${metrics['mae']:,.1f}",
                "MAPE":     f"{metrics['mape']:.2f}%",
                "Dir Acc":  f"{metrics['dir_accuracy']:.1f}%",
            })
        df = pd.DataFrame(rows)
        print_table(df)

        # Best model
        best = min(all_results.items(), key=lambda x: x[1]["mape"])
        print(f"\n  {C.GREEN}★ Best model by MAPE: {C.BOLD}{best[0]}{C.RESET}  "
              f"{C.MUTED}({best[1]['mape']:.2f}% avg error){C.RESET}")

    if args.output and all_results:
        _export_results(args.output, {
            "backtest_results": all_results,
            "config": {
                "models": models_to_test,
                "train_window": args.train_window,
                "test_window": args.test_window,
                "period": args.period,
            },
            "timestamp": datetime.now().isoformat(),
        })
        print(f"\n  {C.GREEN}✓ Backtest results saved to {args.output}{C.RESET}")


def cmd_compare(args):
    """Compare all models on same data and print leaderboard."""
    print_section("Model Comparison")

    prices = fetch_prices(args.period)
    S0     = get_current_price()
    seed   = args.seed if args.seed >= 0 else None

    results = {}

    for key, label in MODEL_LABELS.items():
        try:
            print(f"  {C.MUTED}Simulating {label}...{C.RESET}", end="\r")
            model = build_and_calibrate(key, prices)
            paths = model.simulate(S0=S0, n_steps=args.days + 1,
                                   n_paths=args.paths, random_seed=seed)
            r = compute_risk(paths, S0)
            results[label] = r
        except Exception as e:
            print(f"  {C.RED}✗ {label}: {e}{C.RESET}")

    print_section("Leaderboard")

    rows = []
    for label, r in results.items():
        rows.append({
            "Model":       label,
            "Expected $":  f"{r['mean']:,.0f}",
            "Change":      f"{(r['mean']/S0-1)*100:+.1f}%",
            "Vol %":       f"{r['vol']:.1f}",
            "VaR 95%":     f"{(r['var_95']/S0-1)*100:+.1f}%",
            "P(Gain)":     f"{r['prob_up']*100:.0f}%",
        })
    df = pd.DataFrame(rows)
    print_table(df)

    if args.output:
        _export_results(args.output, {k: v for k, v in results.items()})
        print(f"\n  {C.GREEN}✓ Comparison saved to {args.output}{C.RESET}")


# ── Export Helpers ─────────────────────────────────────────────────────────────
def _export_results(path: str, data: dict):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    elif ext == ".csv":
        flat = {k: [v] if not isinstance(v, dict) else [str(v)]
                for k, v in data.items()}
        pd.DataFrame(flat).to_csv(path, index=False)
    else:
        # Default: JSON
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gold-mc",
        description="Monte Carlo Gold Price Predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py price
  python cli.py simulate --model ou --days 30 --paths 5000
  python cli.py simulate --model heston --days 90 --output results.json
  python cli.py risk --model merton --days 60 --target 2600
  python cli.py backtest --models gbm ou merton --period 2y
  python cli.py compare --days 30 --output comparison.csv

Models: gbm | ou | merton | heston | regime
        """,
    )
import click
from src.macro import MacroBridge, ParameterAdjuster

# Add to main CLI group
@click.option('--macro/--no-macro', default=None, 
              help='Enable/disable WorldMonitor macro intelligence')
@click.option('--macro-brief', is_flag=True,
              help='Fetch and display AI world brief')
@click.pass_context
def cli(ctx, macro, macro_brief):
    """Gold Option Monte Carlo Simulator with optional Macro Intelligence."""
    ctx.ensure_object(dict)
    
    # Determine macro mode
    ctx.obj['macro_enabled'] = macro if macro is not None else \
        os.getenv('WORLDMONITOR_ENABLED', 'true').lower() == 'true'
    ctx.obj['macro_brief'] = macro_brief
    
    if ctx.obj['macro_enabled']:
        ctx.obj['macro_bridge'] = MacroBridge()


# Modify simulate command
@cli.command()
@click.pass_context
@click.option('--model', '-m', default='gbm', 
              type=click.Choice(['gbm', 'ou', 'merton', 'heston', 'regime']))
@click.option('--days', '-d', default=30, help='Trading days to simulate')
@click.option('--paths', '-p', default=5000, help='Number of simulation paths')
@click.option('--output', '-o', type=click.Path(), help='Output file (JSON/CSV)')
def simulate(ctx, model, days, paths, output):
    """Run Monte Carlo simulation with optional macro adjustments."""
    
    # Fetch gold data and calibrate model (existing logic)
    fetcher = GoldDataFetcher()
    prices = fetcher.fetch_gold_prices()
    
    # Create base model
    model_obj = create_model(model, historical_data=prices['close'].values)
    
    # Macro integration
    macro_signals = None
    adjusted_params = None
    
    if ctx.obj.get('macro_enabled'):
        bridge = ctx.obj['macro_bridge']
        signals = bridge.get_signals_sync()
        
        if ctx.obj.get('macro_brief'):
            brief = bridge.get_brief()
            if brief:
                click.echo(f"\n🌍 World Brief:\n{'-'*40}\n{brief}\n{'-'*40}\n")
        
        # Apply adjustments based on model type
        adjuster = ParameterAdjuster(signals)
        current_price = prices['close'].iloc[-1]
        
        if model == 'gbm':
            adjusted_params = adjuster.adjust_gbm(model_obj.mu, model_obj.sigma)
        elif model == 'ou':
            adjusted_params = adjuster.adjust_ou(
                model_obj.mu, model_obj.sigma, 
                model_obj.theta, model_obj.kappa
            )
        elif model == 'merton':
            adjusted_params = adjuster.adjust_merton(
                model_obj.mu, model_obj.sigma,
                model_obj.lambda_jump, model_obj.mu_j, model_obj.sigma_j
            )
        elif model == 'heston':
            adjusted_params = adjuster.adjust_heston(
                model_obj.mu, model_obj.v0,
                model_obj.theta_v, model_obj.kappa_v,
                model_obj.xi, model_obj.rho
            )
        elif model == 'regime':
            adjusted_params = adjuster.adjust_regime(
                model_obj.mu[0], model_obj.sigma[0],  # calm
                model_obj.mu[1], model_obj.sigma[1],  # crisis
            )
        
        macro_signals = signals.to_dict()
        
        # Display adjustments
        if not signals.is_fallback:
            click.echo(f"📊 Macro Signal: {signals.risk_tier.value.upper()} risk tier")
            click.echo(f"   CII Top-5 Avg: {signals.cii_top5_avg:.1f}")
            click.echo(f"   Active Hotspots: {signals.active_hotspot_count}")
    
    # Run simulation with adjusted params
    if hasattr(model_obj, 'simulate_with_macro') and adjusted_params:
        results = model_obj.simulate_with_macro(
            S0=current_price,
            n_steps=days,
            n_paths=paths,
            macro_params=adjusted_params
        )
    else:
        results = model_obj.simulate(
            S0=current_price,
            n_steps=days,
            n_paths=paths
        )
    
    # Output results
    output_data = {
        'model': model,
        'days': days,
        'paths': paths,
        'current_price': float(current_price),
        'final_prices': results[:, -1].tolist(),
        'statistics': {
            'mean': float(np.mean(results[:, -1])),
            'std': float(np.std(results[:, -1])),
            'var_95': float(np.percentile(results[:, -1], 5)),
            'var_5': float(np.percentile(results[:, -1], 95)),
        }
    }
    
    if macro_signals:
        output_data['macro_signals'] = macro_signals
    if adjusted_params:
        output_data['parameter_adjustments'] = adjusted_params.to_dict()
    
    if output:
        with open(output, 'w') as f:
            json.dump(output_data, f, indent=2)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(json.dumps(output_data, indent=2))
    # ── Global options ─────────────────────────────────────────────────────────
    parser.add_argument("--no-color", action="store_true",
                        help="Disable colored output")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── price ──────────────────────────────────────────────────────────────────
    subparsers.add_parser("price", help="Show current gold price")

    # ── simulate ───────────────────────────────────────────────────────────────
    sim = subparsers.add_parser("simulate", help="Run Monte Carlo simulation")
    sim.add_argument("--model",  "-m", default="ou",   choices=list(MODEL_MAP),
                     help="Stochastic model (default: ou)")
    sim.add_argument("--days",   "-d", type=int, default=30,
                     help="Forecast horizon in trading days (default: 30)")
    sim.add_argument("--paths",  "-p", type=int, default=5000,
                     help="Number of simulation paths (default: 5000)")
    sim.add_argument("--period", default="2y",
                     help="Training data period (default: 2y)")
    sim.add_argument("--price",  type=float, default=None,
                     help="Override current gold price")
    sim.add_argument("--seed",   type=int, default=42,
                     help="Random seed (-1 for random, default: 42)")
    sim.add_argument("--output", "-o", default=None,
                     help="Save results to file (JSON or CSV)")

    # ── risk ───────────────────────────────────────────────────────────────────
    risk = subparsers.add_parser("risk", help="Deep risk analysis")
    risk.add_argument("--model",  "-m", default="ou", choices=list(MODEL_MAP))
    risk.add_argument("--days",   "-d", type=int, default=30)
    risk.add_argument("--paths",  "-p", type=int, default=5000)
    risk.add_argument("--period", default="2y")
    risk.add_argument("--price",  type=float, default=None)
    risk.add_argument("--target", type=float, default=None,
                     help="Calculate probability of reaching this price target")
    risk.add_argument("--seed",   type=int, default=42)
    risk.add_argument("--output", "-o", default=None)

    # ── backtest ───────────────────────────────────────────────────────────────
    bt = subparsers.add_parser("backtest", help="Walk-forward backtest")
    bt.add_argument("--models",   "-m", nargs="+", choices=list(MODEL_MAP),
                    default=["gbm", "ou"],
                    help="Models to test (default: gbm ou)")
    bt.add_argument("--period",   default="5y",
                    help="Historical data period (default: 5y)")
    bt.add_argument("--train-window", type=int, default=252,
                    help="Training window in days (default: 252)")
    bt.add_argument("--test-window",  type=int, default=30,
                    help="Test window in days (default: 30)")
    bt.add_argument("--paths",    "-p", type=int, default=1000)
    bt.add_argument("--output",   "-o", default=None)

    # ── compare ────────────────────────────────────────────────────────────────
    cmp = subparsers.add_parser("compare", help="Compare all models side-by-side")
    cmp.add_argument("--days",   "-d", type=int, default=30)
    cmp.add_argument("--paths",  "-p", type=int, default=3000)
    cmp.add_argument("--period", default="2y")
    cmp.add_argument("--seed",   type=int, default=42)
    cmp.add_argument("--output", "-o", default=None)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.no_color:
        _no_color()

    print(BANNER)

    COMMANDS = {
        "price":    cmd_price,
        "simulate": cmd_simulate,
        "risk":     cmd_risk,
        "backtest": cmd_backtest,
        "compare":  cmd_compare,
    }

    if not args.command:
        parser.print_help()
        sys.exit(0)

    fn = COMMANDS.get(args.command)
    if fn:
        try:
            fn(args)
            print(f"\n  {C.MUTED}Done.{C.RESET}\n")
        except KeyboardInterrupt:
            print(f"\n  {C.MUTED}Interrupted.{C.RESET}\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n  {C.RED}Error: {e}{C.RESET}\n")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
