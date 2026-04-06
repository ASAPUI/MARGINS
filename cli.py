"""
Monte Carlo Gold Price Predictor — Command Line Interface

Full-featured CLI for running simulations, backtests,
and risk analysis from the terminal.

Author: Essabri Ali Rayan
Version: 1.3
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
    for attr in vars(C):
        if not attr.startswith("_"):
            setattr(C, attr, "")

if not sys.stdout.isatty():
    _no_color()

BANNER = f"""
{C.GOLD}╔══════════════════════════════════════════════════════╗
║        Monte Carlo Gold Price Predictor v1.3         ║
║                  Essabri Ali Rayan                   ║
╚══════════════════════════════════════════════════════╝{C.RESET}
"""

# ── Helpers ────────────────────────────────────────────────────────────────────
def print_section(title: str):
    width = 56
    bar = "─" * width
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

# ── Macro intelligence (optional) ─────────────────────────────────────────────
try:
    from src.macro.bridge  import MacroBridge
    from src.macro.adjuster import ParameterAdjuster
    HAS_MACRO = True
except ImportError:
    HAS_MACRO      = False
    MacroBridge    = None
    ParameterAdjuster = None

# ── Data Layer ─────────────────────────────────────────────────────────────────
def fetch_prices(period: str = "2y", symbol: str = "GC=F") -> pd.Series:
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError("Empty data")
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        else:
            df.index = df.index.tz_localize(None)
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
    "gbm":    ("src.models.gbm",              "GeometricBrownianMotion"),
    "ou":     ("src.models.mean_reversion",   "OrnsteinUhlenbeckModel"),
    "merton": ("src.models.jump_diffusion",   "MertonJumpModel"),
    "heston": ("src.models.heston",           "HestonModel"),
    "regime": ("src.models.regime_switching", "RegimeSwitchingModel"),
}

MODEL_LABELS = {
    "gbm":    "GBM",
    "ou":     "Mean Reversion (OU)",
    "merton": "Jump Diffusion",
    "heston": "Heston",
    "regime": "Regime Switching",
}

class FallbackGBM:
    name = "GBM"
    def __init__(self):
        self.mu    = 0.05
        self.sigma = 0.15

    def calibrate(self, data):
        if len(data) > 10:
            returns    = np.log(data[1:] / data[:-1]) if data[0] > 10 else data
            self.mu    = float(np.mean(returns) * 252)
            self.sigma = float(np.std(returns)  * np.sqrt(252))

    def simulate(self, S0, n_steps, n_paths=1000, random_seed=None):
        if random_seed is not None:        # FIX: seed=0 was falsy before
            np.random.seed(random_seed)
        dt  = 1 / 252
        Z   = np.random.standard_normal((n_paths, n_steps - 1))
        out = np.zeros((n_paths, n_steps))
        out[:, 0] = S0
        for t in range(1, n_steps):
            drift       = (self.mu - 0.5 * self.sigma ** 2) * dt
            diff        = self.sigma * np.sqrt(dt) * Z[:, t - 1]
            out[:, t]   = out[:, t - 1] * np.exp(drift + diff)
        return out

def load_model(key: str):
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
    ModelCls    = load_model(key)
    model       = ModelCls()
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    if hasattr(model, "calibrate"):
        if key in ("ou", "heston", "regime"):
            model.calibrate(prices.values)
        else:
            model.calibrate(log_returns)
    return model

# ── Risk Calculations ──────────────────────────────────────────────────────────
def compute_risk(paths: np.ndarray, S0: float) -> dict:
    final  = paths[:, -1]
    log_r  = np.log(final / S0)
    var_95 = np.percentile(final, 5)
    var_99 = np.percentile(final, 1)
    mask95 = final <= var_95
    cvar_95 = float(np.mean(final[mask95])) if mask95.any() else float(var_95)
    dd_list = []
    for path in paths:
        peak = np.maximum.accumulate(path)
        dd   = np.min((path - peak) / peak)
        dd_list.append(dd)
    return {
        "mean":        float(np.mean(final)),
        "median":      float(np.median(final)),
        "std":         float(np.std(final)),
        "p5":          float(np.percentile(final, 5)),
        "p25":         float(np.percentile(final, 25)),
        "p75":         float(np.percentile(final, 75)),
        "p95":         float(np.percentile(final, 95)),
        "var_95":      float(var_95),
        "var_99":      float(var_99),
        "cvar_95":     float(cvar_95),
        "prob_up":     float(np.mean(final > S0)),
        "prob_loss_5": float(np.mean(log_r < -0.05)),
        "prob_loss_10":float(np.mean(log_r < -0.10)),
        "prob_gain_10":float(np.mean(log_r > 0.10)),
        "avg_max_dd":  float(np.mean(dd_list)),
        "vol":         float(np.std(log_r) * 100),
        "skew":        float(pd.Series(log_r).skew()),
        "kurt":        float(pd.Series(log_r).kurt()),
    }

# ── Macro helpers ──────────────────────────────────────────────────────────────
def _apply_macro(args, model_key: str, model_obj, prices: pd.Series):
    if not getattr(args, "macro", False) or not HAS_MACRO:
        return None, None
    print(f"\n  {C.CYAN}↯ Fetching WorldMonitor macro signals...{C.RESET}")
    bridge  = MacroBridge()
    signals = bridge.get_signals_sync()
    if getattr(args, "macro_brief", False):
        brief = bridge.get_brief_sync()
        if brief:
            print(f"\n  {C.GOLD}🌍 World Brief:{C.RESET}")
            print(f"  {'─'*50}")
            for line in brief.split(". "):
                if line.strip():
                    print(f"  {C.MUTED}{line.strip()}.{C.RESET}")
            print(f"  {'─'*50}\n")
    if not signals.is_fallback:
        print(f"  {C.GREEN}✓{C.RESET} Macro: {C.BOLD}{signals.risk_tier.value.upper()}{C.RESET} "
              f"risk | CII avg {signals.cii_top5_avg:.1f} | "
              f"hotspots {signals.active_hotspot_count}")
    adjuster = ParameterAdjuster(signals)
    def _p(attr, default=0.05):
        params = getattr(model_obj, "params", None)
        if params and hasattr(params, attr):
            return getattr(params, attr)
        return getattr(model_obj, attr, default)
    try:
        if model_key == "gbm":
            adj = adjuster.adjust_gbm(_p("mu"), _p("sigma"))
        elif model_key == "ou":
            adj = adjuster.adjust_ou(_p("mu"), _p("sigma"),
                                     _p("theta", 2000.0), _p("kappa", 0.5))
        elif model_key == "merton":
            adj = adjuster.adjust_merton(
                _p("mu"), _p("sigma"),
                _p("lambda_jump", 2.0), _p("mu_jump", -0.05), _p("sigma_jump", 0.10))
        elif model_key == "heston":
            adj = adjuster.adjust_heston(
                _p("mu"), _p("v0", 0.04), _p("theta_v", 0.04),
                _p("kappa_v", 1.5), _p("xi", 0.3), _p("rho", -0.7))
        elif model_key == "regime":
            mu_list    = _p("mu",    [0.05, -0.05])
            sigma_list = _p("sigma", [0.12,  0.25])
            adj = adjuster.adjust_regime(mu_list[0], sigma_list[0],
                                         mu_list[1], sigma_list[1])
        else:
            adj = None
    except Exception as e:
        print(f"  {C.MUTED}⚠ Macro adjustment failed ({e}), running without macro{C.RESET}")
        adj = None
    return adj, signals.to_dict() if not signals.is_fallback else None

# ══════════════════════════════════════════════════════════════════════════════
#  COMMANDS
# ══════════════════════════════════════════════════════════════════════════════
def cmd_price(args):
    print_section("Current Gold Price")
    try:
        import yfinance as yf
        data  = yf.Ticker("GC=F").history(period="5d", interval="1d")
        price = float(data["Close"].iloc[-1])
        prev  = float(data["Close"].iloc[-2])
        change= price - prev
        pct   = change / prev * 100
        vol_1d= float(data["Close"].pct_change().std() * 100)
        print_metric("Symbol",         "GC=F (COMEX Gold Futures)")
        print_metric("Current Price",  f"${price:,.2f}",
                     f"{'+' if change >= 0 else ''}{change:.2f} ({pct:+.2f}%)")
        print_metric("Per Gram",       f"${price/31.1035:,.2f}")
        print_metric("Per Kilogram",   f"${price*32.1507:,.0f}")
        print_metric("Previous Close", f"${prev:,.2f}")
        print_metric("5-Day Volatility",f"{vol_1d:.2f}%")
        print_metric("Last Updated",   datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print(f"  {C.RED}Error fetching price: {e}{C.RESET}")

def cmd_simulate(args):
    print_section(f"Simulation — {MODEL_LABELS.get(args.model, args.model)}")
    prices = fetch_prices(args.period)
    S0     = get_current_price() if not args.price else args.price
    print_metric("Initial Price",   f"${S0:,.2f}")
    print_metric("Model",           MODEL_LABELS.get(args.model, args.model))
    print_metric("Horizon",         f"{args.days} trading days")
    print_metric("Paths",           f"{args.paths:,}")
    print_metric("Training Period", args.period)
    print()
    print(f"  {C.MUTED}Calibrating model...{C.RESET}")
    model = build_and_calibrate(args.model, prices)
    adj, macro_dict = _apply_macro(args, args.model, model, prices)
    print(f"  {C.MUTED}Running {args.paths:,} simulations...{C.RESET}")
    seed  = args.seed if args.seed != 0 else None
    paths = model.simulate(S0=S0, n_steps=args.days + 1,
                           n_paths=args.paths, random_seed=seed)
    r = compute_risk(paths, S0)
    final = paths[:, -1]
    print_section("Results")
    print_metric("Expected Price",  f"${r['mean']:,.2f}",   f"{(r['mean']/S0-1)*100:+.1f}%")
    print_metric("Median Price",    f"${r['median']:,.2f}", f"{(r['median']/S0-1)*100:+.1f}%")
    print_metric("Std Deviation",   f"${r['std']:,.2f}")
    print()
    print_metric("90% CI Lower",    f"${r['p5']:,.2f}")
    print_metric("90% CI Upper",    f"${r['p95']:,.2f}")
    print()
    print_metric("Prob. of Gain",   f"{r['prob_up']*100:.1f}%")
    print_metric("Prob. Gain > 10%",f"{r['prob_gain_10']*100:.1f}%")
    print_metric("Prob. Loss > 10%",f"{r['prob_loss_10']*100:.1f}%")
    scenarios = pd.DataFrame({
        "Scenario":  ["Bear (5%)", "Mild Bear (25%)", "Base (50%)",
                      "Mild Bull (75%)", "Bull (95%)"],
        "Price ($)": [f"{r['p5']:,.0f}", f"{r['p25']:,.0f}", f"{r['median']:,.0f}",
                      f"{r['p75']:,.0f}", f"{r['p95']:,.0f}"],
        "Change":    [f"{(r['p5']/S0-1)*100:+.1f}%",   f"{(r['p25']/S0-1)*100:+.1f}%",
                      f"{(r['median']/S0-1)*100:+.1f}%",f"{(r['p75']/S0-1)*100:+.1f}%",
                      f"{(r['p95']/S0-1)*100:+.1f}%"],
    })
    print_table(scenarios, "Scenario Analysis")
    if args.output:
        out = {"model": args.model, "S0": S0, "days": args.days,
               "paths": args.paths, "results": r,
               "timestamp": datetime.now().isoformat()}
        if macro_dict:
            out["macro_signals"] = macro_dict
        _export_results(args.output, out)
        print(f"\n  {C.GREEN}✓ Results saved to {args.output}{C.RESET}")

def cmd_risk(args):
    print_section(f"Risk Analysis — {MODEL_LABELS.get(args.model, args.model)}")
    prices = fetch_prices(args.period)
    S0     = get_current_price() if not args.price else args.price
    print(f"  {C.MUTED}Calibrating & simulating...{C.RESET}")
    model = build_and_calibrate(args.model, prices)
    adj, _ = _apply_macro(args, args.model, model, prices)
    seed   = args.seed if args.seed != 0 else None
    paths  = model.simulate(S0=S0, n_steps=args.days + 1,
                            n_paths=args.paths, random_seed=seed)
    r = compute_risk(paths, S0)
    print_section("Value at Risk")
    print_metric("VaR 95%",  f"${r['var_95']:,.2f}", f"{(r['var_95']/S0-1)*100:+.1f}%")
    print_metric("VaR 99%",  f"${r['var_99']:,.2f}", f"{(r['var_99']/S0-1)*100:+.1f}%")
    print_metric("CVaR 95%", f"${r['cvar_95']:,.2f}",f"{(r['cvar_95']/S0-1)*100:+.1f}%")
    print_section("Distribution Statistics")
    print_metric("Realised Volatility",f"{r['vol']:.2f}%")
    print_metric("Skewness",           f"{r['skew']:.4f}")
    print_metric("Kurtosis (excess)",  f"{r['kurt']:.4f}")
    print_metric("Avg Max Drawdown",   f"{r['avg_max_dd']*100:.2f}%")
    print_section("Tail Risk Probabilities")
    print_metric("P(Loss > 5%)",  f"{r['prob_loss_5']*100:.2f}%")
    print_metric("P(Loss > 10%)", f"{r['prob_loss_10']*100:.2f}%")
    print_metric("P(Gain > 10%)", f"{r['prob_gain_10']*100:.2f}%")
    if args.target:
        final      = paths[:, -1]
        prob_reach = np.mean(final >= args.target) * 100
        print_section(f"Target Analysis — ${args.target:,}")
        print_metric("Probability of reaching target", f"{prob_reach:.2f}%")
        above = final[final >= args.target]
        if len(above):
            print_metric("Average price when above target", f"${np.mean(above):,.2f}")
    if args.output:
        _export_results(args.output, {
            "model": args.model, "S0": S0, "days": args.days,
            "risk_metrics": r, "timestamp": datetime.now().isoformat()})
        print(f"\n  {C.GREEN}✓ Risk report saved to {args.output}{C.RESET}")

# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST  — rebuilt to reduce MAE/RMSE
# ══════════════════════════════════════════════════════════════════════════════
def _backtest_window(model_key, train_prices, test_prices, n_paths, calib_window, window_idx):
    """
    Run one walk-forward window and return error metrics.
    """
    # FIX 1: calibrate on a recent window only
    recent = train_prices.iloc[-calib_window:]
    model = build_and_calibrate(model_key, recent)

    S0_w = float(train_prices.iloc[-1])
    n_d = len(test_prices)
    if n_d < 2:
        return None

    paths_w = model.simulate(
        S0=S0_w,
        n_steps=n_d + 1,
        n_paths=n_paths,
        random_seed=window_idx,  # Changed from 42 to window_idx for independent randomness
    )

    # Changed from np.mean to np.median as point estimate
    preds = np.median(paths_w[:, 1:], axis=0)  # shape (n_d,)
    actuals = test_prices.values  # shape (n_d,)

    errors = preds - actuals
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # FIX 3: MAPE computed on log returns (not levels)
    log_pred = np.log(preds[1:] / preds[:-1]) if len(preds) > 1 else np.array([])
    log_actual = np.log(actuals[1:] / actuals[:-1]) if len(actuals) > 1 else np.array([])
    if len(log_actual) > 0:
        mape = float(np.mean(np.abs(log_pred - log_actual)) * 100)
    else:
        mape = 0.0

    # New directional accuracy: based on final price vs initial over the full test window
    final_pred_up = np.median(paths_w[:, -1]) > S0_w
    final_actual_up = actuals[-1] > actuals[0]
    dir_acc = 100.0 if (final_pred_up == final_actual_up) else 0.0

    # FIX 4: also compute terminal-only MAE for reference
    terminal_mae = float(abs(preds[-1] - actuals[-1]))

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "dir_accuracy": dir_acc,
        "terminal_mae": terminal_mae,
    }

def cmd_backtest(args):
    print_section("Walk-Forward Backtest")
    prices = fetch_prices(args.period)

    models_to_test = args.models if args.models else ["gbm", "ou"]

    # FIX: shorter calibration window = more recent parameters
    calib_window = min(args.calib_window, args.train_window)

    print_metric("Models",       ", ".join(MODEL_LABELS.get(m, m) for m in models_to_test))
    print_metric("Price History",f"{len(prices)} days")
    print_metric("Train Window", f"{args.train_window} days")
    print_metric("Calib Window", f"{calib_window} days  ← most-recent slice for calibration")
    print_metric("Test Window",  f"{args.test_window} days")
    print_metric("Paths",        f"{args.paths}")
    print()

    all_results = {}

    for model_key in models_to_test:
        label = MODEL_LABELS.get(model_key, model_key)
        print(f"\n  {C.GOLD}▸ Backtesting {label}...{C.RESET}")

        n      = len(prices)
        start  = args.train_window
        step   = args.test_window

        windows = []
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
                       m = _backtest_window(
            model_key, train_prices, test_prices,
            args.paths, calib_window, i,
        )
                if m:
                    window_metrics.append(m)
            except Exception:
                pass

        print()   # end progress bar line

        if window_metrics:
            agg = {k: float(np.mean([m[k] for m in window_metrics]))
                   for k in window_metrics[0]}
            all_results[label] = agg

    print_section("Backtest Results")
    if all_results:
        rows = []
        for model_label, metrics in all_results.items():
            rows.append({
                "Model":        model_label,
                "RMSE":         f"${metrics['rmse']:,.1f}",
                "MAE":          f"${metrics['mae']:,.1f}",
                "MAPE (ret%)":  f"{metrics['mape']:.3f}%",
                "Dir Acc":      f"{metrics['dir_accuracy']:.1f}%",
                "Terminal MAE": f"${metrics['terminal_mae']:,.1f}",
            })
        print_table(pd.DataFrame(rows))

        best = min(all_results.items(), key=lambda x: x[1]["mape"])
        print(f"\n  {C.GREEN}★ Best model by return-MAPE: "
              f"{C.BOLD}{best[0]}{C.RESET}  "
              f"{C.MUTED}({best[1]['mape']:.3f}% avg daily return error){C.RESET}")

        print(f"\n  {C.MUTED}Note: MAE/RMSE in $ measure prediction vs actual price level.")
        print(f"  MAPE (ret%) measures daily log-return error — the more meaningful")
        print(f"  metric for stochastic models that predict distributions, not points.{C.RESET}")

    if args.output and all_results:
        _export_results(args.output, {
            "backtest_results": all_results,
            "config": {
                "models":        models_to_test,
                "train_window":  args.train_window,
                "calib_window":  calib_window,
                "test_window":   args.test_window,
                "period":        args.period,
            },
            "timestamp": datetime.now().isoformat(),
        })
        print(f"\n  {C.GREEN}✓ Backtest results saved to {args.output}{C.RESET}")


def cmd_compare(args):
    print_section("Model Comparison")
    prices = fetch_prices(args.period)
    S0     = get_current_price()
    seed   = args.seed if args.seed != 0 else None
    results = {}
    for key, label in MODEL_LABELS.items():
        try:
            print(f"  {C.MUTED}Simulating {label}...{C.RESET}", end="\r")
            model   = build_and_calibrate(key, prices)
            paths   = model.simulate(S0=S0, n_steps=args.days + 1,
                                     n_paths=args.paths, random_seed=seed)
            results[label] = compute_risk(paths, S0)
        except Exception as e:
            print(f"  {C.RED}✗ {label}: {e}{C.RESET}")
    print_section("Leaderboard")
    rows = []
    for label, r in results.items():
        rows.append({
            "Model":     label,
            "Expected $":f"{r['mean']:,.0f}",
            "Change":    f"{(r['mean']/S0-1)*100:+.1f}%",
            "Vol %":     f"{r['vol']:.1f}",
            "VaR 95%":   f"{(r['var_95']/S0-1)*100:+.1f}%",
            "P(Gain)":   f"{r['prob_up']*100:.0f}%",
        })
    print_table(pd.DataFrame(rows))
    if args.output:
        _export_results(args.output, {k: v for k, v in results.items()})
        print(f"\n  {C.GREEN}✓ Comparison saved to {args.output}{C.RESET}")

# ── Export ─────────────────────────────────────────────────────────────────────
def _export_results(path: str, data: dict):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        flat = {k: [v] if not isinstance(v, dict) else [str(v)]
                for k, v in data.items()}
        pd.DataFrame(flat).to_csv(path, index=False)
    else:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

# ── Argument Parser ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gold-mc",
        description="Monte Carlo Gold Price Predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py price
  python cli.py simulate --model ou --days 30 --paths 5000
  python cli.py risk     --model merton --days 60 --target 2600
  python cli.py backtest --models gbm ou merton heston regime --period 2y
  python cli.py backtest --models gbm ou --calib-window 63   # 3-month recalib
  python cli.py compare  --days 30 --output comparison.csv
""",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    subs = parser.add_subparsers(dest="command", metavar="COMMAND")

    # price
    subs.add_parser("price", help="Show current gold price")

    # simulate
    sim = subs.add_parser("simulate", help="Run Monte Carlo simulation")
    sim.add_argument("--model",  "-m", default="ou", choices=list(MODEL_MAP))
    sim.add_argument("--days",   "-d", type=int, default=30)
    sim.add_argument("--paths",  "-p", type=int, default=5000)
    sim.add_argument("--period",       default="2y")
    sim.add_argument("--price",        type=float, default=None)
    sim.add_argument("--seed",         type=int,   default=42)
    sim.add_argument("--output", "-o", default=None)
    sim.add_argument("--macro",        action="store_true")
    sim.add_argument("--macro-brief",  action="store_true")

    # risk
    risk = subs.add_parser("risk", help="Deep risk analysis")
    risk.add_argument("--model",  "-m", default="ou", choices=list(MODEL_MAP))
    risk.add_argument("--days",   "-d", type=int, default=30)
    risk.add_argument("--paths",  "-p", type=int, default=5000)
    risk.add_argument("--period",       default="2y")
    risk.add_argument("--price",        type=float, default=None)
    risk.add_argument("--target",       type=float, default=None)
    risk.add_argument("--seed",         type=int,   default=42)
    risk.add_argument("--output", "-o", default=None)
    risk.add_argument("--macro",        action="store_true")

    # backtest  ← new --calib-window flag added here
    bt = subs.add_parser("backtest", help="Walk-forward backtest")
    bt.add_argument("--models",       "-m",  nargs="+", choices=list(MODEL_MAP),
                    default=["gbm", "ou"])
    bt.add_argument("--period",              default="5y")
    bt.add_argument("--train-window",        type=int, default=252)
    bt.add_argument("--calib-window",        type=int, default=126,
                    help="Days of recent history used for calibration (default 126 = 6 months). "
                         "Shorter = more responsive to recent market. Must be <= train-window.")
    bt.add_argument("--test-window",         type=int, default=30)
    bt.add_argument("--paths",        "-p",  type=int, default=1000)
    bt.add_argument("--output",       "-o",  default=None)

    # compare
    cmp = subs.add_parser("compare", help="Compare all models side-by-side")
    cmp.add_argument("--days",   "-d", type=int, default=30)
    cmp.add_argument("--paths",  "-p", type=int, default=3000)
    cmp.add_argument("--period",       default="2y")
    cmp.add_argument("--seed",         type=int, default=42)
    cmp.add_argument("--output", "-o", default=None)

    return parser

# ── Entry Point ────────────────────────────────────────────────────────────────
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