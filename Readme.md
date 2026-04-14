# 🥇 Monte Carlo Gold Price Predictor

> A production-grade stochastic simulation engine for gold price forecasting, risk analysis, and walk-forward portfolio backtesting.

**Author:** Essabri Ali Rayan  
**Version:** 2.0.0  
**License:** BSL

---

## 📖 Overview

This project uses **Monte Carlo simulation** to predict gold price movements and quantify risk. It combines 6 advanced stochastic models with real market data, professional risk analytics, an interactive Streamlit dashboard, a full command-line interface, and an institutional-grade walk-forward portfolio backtesting engine (Phase 6).

The system mirrors tools used by professional quantitative traders and hedge funds — calibrating models from historical data, simulating thousands of possible price futures, extracting probability distributions and risk metrics from the results, and validating performance with rigorous out-of-sample testing.

> ⚠️ **Price display note:** All prices are per troy ounce. 1 troy ounce = 31.1035 g, so: `displayed price ÷ 31.1035 ≈ price per gram`

---

## ✨ Features

- **6 Stochastic Models** — GBM, Ornstein-Uhlenbeck, Merton Jump Diffusion, Heston, Regime Switching, LSTM
- **Live Market Data** — Auto-fetches gold prices via Yahoo Finance (`GC=F`)
- **Risk Analytics** — VaR, CVaR, confidence intervals, drawdown analysis
- **Interactive Dashboard** — 4-tab Streamlit web app with Plotly charts
- **Command Line Interface** — Full CLI with 5 commands and JSON/CSV export
- **Phase 6 Walk-Forward Backtest** — 48-fold OOS engine, no lookahead bias, vol-parity ensemble, Diebold-Mariano significance testing
- **Model Comparison** — Leaderboard with statistical significance tests
- **Caching System** — Parquet/HDF5/CSV/Pickle storage with TTL expiration

---

## 🗂️ Project Structure

```
monte-carlo-gold/
│
├── src/
│   ├── data/
│   │   ├── __init__.py          ← Package exports
│   │   ├── fetcher.py           ← Yahoo Finance + FRED data fetching
│   │   ├── cleaner.py           ← Missing values, outlier detection
│   │   ├── features.py          ← Feature engineering (RSI, vol, regimes)
│   │   └── storage.py           ← Multi-backend caching system
│   │
│   ├── models/
│   │   ├── __init__.py          ← Model registry + create_model() factory
│   │   ├── gbm.py               ← Geometric Brownian Motion
│   │   ├── mean_reversion.py    ← Ornstein-Uhlenbeck process
│   │   ├── jump_diffusion.py    ← Merton jump diffusion
│   │   ├── heston.py            ← Heston stochastic volatility
│   │   ├── regime_switching.py  ← Markov regime switching
│   │   └── lstm.py              ← LSTM neural network (v2.0)
│   │
│   └── backtesting/
│       ├── __init__.py          ← Package exports
│       ├── backtester.py        ← Walk-forward testing engine (Phase 6)
│       └── comparison.py        ← Model ranking + statistical tests
│
├── phase6_backtest.py           ← Phase 6 standalone runner
├── app.py                       ← Streamlit web dashboard
├── cli.py                       ← Command-line interface
├── requirements.txt             ← Python dependencies
└── Readme.md                    ← This file
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ASAPUI/MARGINS.git
cd MARGINS
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### 3. Run Phase 6 Walk-Forward Backtest

```bash
python phase6_backtest.py
```

Or via the CLI:

```bash
python cli.py backtest --models gbm ou merton heston regime --period 5y
```

---

## 📦 Requirements

```
Python >= 3.10
yfinance >= 0.2.0
pandas >= 1.5.0
numpy >= 1.24.0
scipy >= 1.10.0
fredapi >= 0.5.0
pyarrow >= 10.0.0     # Parquet support
tables >= 3.8.0       # HDF5 support
torch >= 2.0.0
httpx >= 0.27
cachetools >= 5.3
pydantic >= 2.0
streamlit
plotly
scikit-learn
pytest
```

```bash
pip install -r requirements.txt
```

---

## 🧠 The 6 Stochastic Models

### 1. Geometric Brownian Motion (GBM)

```
dS = μ·S·dt + σ·S·dW
```

The foundational model. Assumes log-normally distributed returns with constant drift and volatility. Used as a **baseline benchmark** for Diebold-Mariano significance testing.

### 2. Ornstein-Uhlenbeck — Mean Reversion

```
dS = κ·(θ − S)·dt + σ·dW
```

Captures gold's tendency to revert to long-term equilibrium. `κ` controls reversion speed, `θ` is the long-term fair value. **Best for long-term forecasting.** Note: AR(1) calibration on daily log-returns can be unstable in high-volatility regimes — apply on weekly/monthly scales for best results.

### 3. Merton Jump Diffusion

```
dS = μ·S·dt + σ·S·dW + J·S·dN
```

Extends GBM with a Poisson jump process to model sudden price shocks from geopolitical crises or central bank interventions. **Best for risk analysis and stress testing.**

### 4. Heston Stochastic Volatility

```
dS = μ·S·dt + √v·S·dW₁
dv = κ(θ−v)·dt + ξ·√v·dW₂
```

Models volatility itself as a stochastic process, capturing volatility clustering and the leverage effect. **Best for options pricing and volatility surface modelling.**

### 5. Regime Switching (Markov Chain) ⭐ Phase 6 Winner

```
dS = μ(sₜ)·S·dt + σ(sₜ)·S·dW
```

Alternates between distinct market states (calm / crisis) using a Markov chain. Each regime has its own drift and volatility. **Best overall Sharpe (1.15) in Phase 6 walk-forward testing** — gold exhibits well-documented regime structure driven by macro and geopolitical cycles.

### 6. LSTM Neural Network *(v2.0)*

Deep learning baseline. Trained on rolling windows of historical log-returns and rolling volatility features. **Best for capturing non-linear dependencies** that classical stochastic models miss.

---

## 🖥️ Dashboard — 4 Tabs

| Tab | Content |
|---|---|
| 📈 Simulation | Fan chart, confidence bands, scenario table, return distribution |
| ⚠️ Risk Analysis | VaR/CVaR metrics, drawdown distribution, target price probability |
| 📊 Market Data | Historical chart, return statistics, rolling volatility |
| ℹ️ Model Guide | Formula reference, pros/cons per model |

---

## 💻 CLI Reference

```
python cli.py <command> [options]

Commands:
  price      Show current live gold price
  simulate   Run Monte Carlo simulation
  risk       Deep risk analysis (VaR, CVaR, drawdowns)
  backtest   Walk-forward backtest across models
  compare    Side-by-side model comparison

Common Options:
  --model   -m    Model: gbm | ou | merton | heston | regime | lstm
  --days    -d    Forecast horizon in trading days (default: 30)
  --paths   -p    Number of simulation paths (default: 5000)
  --period        Training data: 6mo | 1y | 2y | 5y (default: 2y)
  --output  -o    Save results to file (.json or .csv)
  --seed          Random seed for reproducibility (default: 42)
```

### Examples

```bash
# Quick 7-day forecast with GBM
python cli.py simulate --model gbm --days 7

# 1-year Heston simulation, save results
python cli.py simulate --model heston --days 252 --paths 10000 --output forecast.json

# Risk analysis with price target
python cli.py risk --model ou --days 90 --target 2800

# Phase 6: Full walk-forward backtest across all models
python cli.py backtest --models gbm ou merton heston regime --period 5y

# Disable colors (for logging/piping)
python cli.py --no-color simulate --model ou --days 30
```

---

## 📊 Risk Metrics Explained

| Metric | Description |
|---|---|
| **VaR 95%** | Maximum loss expected 95% of the time |
| **CVaR 95%** | Average loss in the worst 5% of scenarios |
| **Max Drawdown** | Worst peak-to-trough price drop across all paths |
| **90% CI** | Range containing 90% of all simulated final prices |
| **Prob. of Gain** | % of paths where final price > current price |
| **Realised Vol** | Annualised standard deviation of simulated returns |

---

## 🔬 Phase 6 — Walk-Forward Portfolio Backtest

Phase 6 implements the **industry-standard walk-forward validation framework** — the same methodology used by institutional quant desks to validate time-series models with zero lookahead bias.

### Methodology

```
├── Training Window (252 days)  →  Calibrate all model parameters via MLE
├── Test Window    (21 days)    →  Evaluate out-of-sample median path forecast
└── Step forward by 21 days    →  Repeat for 48 folds across 5 years
```

- **Signal generation:** Go long when model median forecast > S0, otherwise flat.
- **Ensemble:** Vol-parity weights (inverse rolling volatility) rebalanced each fold.
- **Significance:** Diebold-Mariano test (Newey-West adjusted) vs GBM baseline.

### Phase 6 Results — Full Leaderboard (48 Folds)

| Rank | Model | RMSE | MAPE% | DirAcc% | Sharpe | Calmar | MDD% | Return% |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **Regime** | 47.88 | 3.10 | 50.5% | **1.154** | 4.16 | -16.2% | **+10.03%** |
| 🥈 2 | GBM | 47.08 | 3.07 | 48.6% | 0.585 | 1.34 | -16.3% | +2.88% |
| 🥉 3 | Heston | 47.31 | 3.08 | 50.4% | -0.309 | -0.68 | -16.2% | -3.28% |
| 4 | Merton | 49.66 | 3.24 | 50.3% | 0.389 | 1.11 | **-12.0%** | +1.43% |
| 5 | OU | 78.04 | 5.12 | 49.7% | 0.013 | 0.03 | -25.0% | -3.29% |

**Vol-Parity Ensemble:** Return +6.71% · Sharpe 0.899 · MDD -14.69%

### Diebold-Mariano Significance vs GBM Baseline

| Model | DM Statistic | Direction | Significant? |
|---|---|---|---|
| OU | -6.995 | Worse than GBM | ★ Yes (p<0.05) |
| Merton | -1.280 | Worse than GBM | No |
| Heston | -0.754 | Worse than GBM | No |
| Regime | -0.837 | Worse than GBM | No |

### Key Findings

- **Regime Switching wins overall** (Sharpe 1.15, Return +10%). Gold's macro-driven regime structure — crisis flight-to-safety vs calm carry — gives the Regime model a structural edge.
- **OU is statistically significantly worse than GBM** (DM = -7.0, ★). Daily-frequency AR(1) calibration flips sign in high-volatility windows. OU works better applied on weekly price levels, not daily log-returns.
- **Directional accuracy is ~50% across all models** — consistent with weak-form EMH at the 21-day horizon. Alpha comes from risk sizing (vol-parity ensemble), not raw forecast accuracy.
- **Ensemble outperforms all individual models** on risk-adjusted basis (Sharpe 0.899, MDD -14.7%) — model error diversification is the key driver.
- **Merton has the lowest MDD (-12%)** despite average RMSE, because jump detection causes it to go flat during stressed periods.

### Phase 6 Output Files

| File | Description |
|---|---|
| `phase6_dashboard.html` | Interactive chart dashboard (equity curves, radar, DM test, drawdown) |
| `phase6_leaderboard.csv` | Final rankings with all metrics |
| `phase6_fold_metrics.csv` | Per-fold RMSE/MAE/MAPE/DirAcc for all 240 OOS windows |
| `phase6_equity_curves.csv` | NAV series for 5 models + vol-parity ensemble |

### Forecast Accuracy Targets

| Metric | Formula | Target |
|---|---|---|
| RMSE | √mean((pred − actual)²) | Lower is better |
| MAE | mean(\|pred − actual\|) | Lower is better |
| MAPE | mean(\|error/actual\|) × 100 | < 5% is good |
| Directional Accuracy | % correct up/down calls | > 55% is excellent |
| Sharpe Ratio | Ann. return / Ann. vol | > 1.0 is strong |
| Calmar Ratio | Ann. return / \|Max Drawdown\| | > 1.0 is good |

---

## 🛠️ Usage as a Python Package

```python
from src.models import create_model
from src.data.fetcher import GoldDataFetcher
from src.data.cleaner import DataCleaner

# Fetch and clean data
fetcher = GoldDataFetcher()
raw     = fetcher.fetch_gold_prices('GC=F', period='2y')
cleaner = DataCleaner()
prices  = cleaner.clean_price_data(raw)

# Create and calibrate model
model = create_model('regime', historical_data=prices['close'].values)

# Simulate 10,000 paths for 30 days
paths = model.simulate(S0=2350.0, n_steps=30, n_paths=10000)

# Results
import numpy as np
final_prices = paths[:, -1]
print(f"Expected price: ${np.mean(final_prices):,.2f}")
print(f"95% CI: [${np.percentile(final_prices,5):,.2f}, ${np.percentile(final_prices,95):,.2f}]")
print(f"Prob. of gain:  {np.mean(final_prices > 2350)*100:.1f}%")
```

### Running the Phase 6 Engine Directly

```python
from phase6_backtest import generate_gold_prices, run_walk_forward, build_leaderboard

prices = generate_gold_prices(n_days=1260, seed=42)
df_metrics, eq_df, port_stats = run_walk_forward(
    prices.values, n_paths=1000, train_size=252, test_size=21, seed=42
)
leaderboard = build_leaderboard(df_metrics, port_stats)
print(leaderboard)
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run a specific test file
pytest tests/test_models.py -v
```

---

## 🗺️ Roadmap

- [✅] Add LSTM/neural network model for comparison
- [✅] **Phase 6: Walk-Forward Portfolio Backtest with vol-parity ensemble**
- [✅] **Diebold-Mariano statistical significance ranking**
- [✅] Portfolio mode — simulate multiple assets simultaneously (gold, silver, crypto)
- [ ] Integrate FRED API for live macro features (real rates, DXY, inflation breakevens)
- [ ] Add implied volatility surface from gold options data
- [ ] Deploy dashboard to Streamlit Cloud
- [ ] Add email/Telegram alerts for price target breaches
- [ ] LSTM Phase 6 integration — include in walk-forward leaderboard

---

## 📄 License

BSL License — not for commercial use.

---

## 👤 Author

**Essabri Ali Rayan**  
GitHub: [@ASAPUI](https://github.com/ASAPUI)  
Project: [MARGINS](https://github.com/ASAPUI/MARGINS)  
Contact: [alirayanessabri@gmail.com](mailto:alirayanessabri@gmail.com) · +212 069 402 5836

---

> *"I don't know exactly what will happen, but if I simulate thousands of possible futures and average them, I'll get a very good estimate."* — The Monte Carlo philosophy 🎲
