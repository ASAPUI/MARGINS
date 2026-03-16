
# 🥇 Monte Carlo Gold Price Predictor

> A production-grade stochastic simulation engine for gold price forecasting and risk analysis.

**Author:** Essabri Ali Rayan  
**Version:** 1.2.0  
**License:** MIT

---

## 📖 Overview

This project uses **Monte Carlo simulation** to predict gold price movements and quantify risk. It combines 5 advanced stochastic models with real market data, professional risk analytics, an interactive Streamlit dashboard, and a full command-line interface.

The system mirrors tools used by professional quantitative traders and hedge funds — calibrating models from historical data, simulating thousands of possible price futures, and extracting probability distributions and risk metrics from the results.

---

## ✨ Features

- **5 Stochastic Models** — GBM, Ornstein-Uhlenbeck, Merton Jump Diffusion, Heston, Regime Switching
- **Live Market Data** — Auto-fetches gold prices via Yahoo Finance (GC=F)
- **Risk Analytics** — VaR, CVaR, confidence intervals, drawdown analysis
- **Interactive Dashboard** — 4-tab Streamlit web app with Plotly charts
- **Command Line Interface** — Full CLI with 5 commands and JSON/CSV export
- **Walk-Forward Backtesting** — No-lookahead-bias validation on historical data
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
│   │   └── regime_switching.py  ← Markov regime switching
│   │
│   └── backtesting/
│       ├── __init__.py          ← Package exports
│       ├── backtester.py        ← Walk-forward testing engine
│       └── comparison.py        ← Model ranking + statistical tests
│
├── app.py                       ← Streamlit web dashboard
├── cli.py                       ← Command-line interface
├── example.py                   ← Quick-start usage examples
├── requirements.txt             ← Python dependencies
├── pyproject.toml               ← Package configuration
└── README.md                    ← This file
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ASAPUI/monte-carlo-options.git
cd monte-carlo-options
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### 3. Or Use the CLI

```bash
# Get current gold price
python cli.py price

# Run a simulation (30 days, OU model, 5000 paths)
python cli.py simulate --model ou --days 30 --paths 5000

# Deep risk analysis
python cli.py risk --model merton --days 60 --target 2600

# Backtest multiple models
python cli.py backtest --models gbm ou merton --period 2y

# Compare all models side by side
python cli.py compare --days 30 --output results.json
```

---

## 📦 Requirements

```
Python >= 3.10
numpy
pandas
scipy
numba
yfinance
fredapi          # optional — for macro data
streamlit
plotly
scikit-learn
pytest
```

Install everything at once:

```bash
pip install -r requirements.txt
```

---

## 🧠 The 5 Stochastic Models

### 1. Geometric Brownian Motion (GBM)
```
dS = μ·S·dt + σ·S·dW
```
The foundational model. Assumes log-normally distributed returns with constant drift and volatility. Used as a **baseline benchmark**.

### 2. Ornstein-Uhlenbeck — Mean Reversion ⭐ Most important for gold
```
dS = κ·(θ − S)·dt + σ·dW
```
Captures gold's documented tendency to revert to long-term equilibrium. `κ` controls the speed of reversion, `θ` is the long-term fair value. **Best for long-term forecasting.**

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
Models volatility itself as a stochastic process, capturing volatility clustering and the leverage effect. **Best for options pricing and volatility forecasting.**

### 5. Regime Switching (Markov Chain)
```
dS = μ(sₜ)·S·dt + σ(sₜ)·S·dW
```
Alternates between distinct market states (calm / crisis) using a Markov chain. Each regime has its own drift and volatility parameters. **Best for scenario planning.**

---

## 🖥️ Dashboard — 4 Tabs

| Tab | Content |
|-----|---------|
| 📈 Simulation | Fan chart, confidence bands, scenario table, return distribution |
| ⚠️ Risk Analysis | VaR/CVaR metrics, drawdown distribution, target price probability |
| 📊 Market Data | Historical chart, return statistics, rolling volatility |
| ℹ️ Model Guide | Formula reference, pros/cons for each model |

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
  --model   -m    Model: gbm | ou | merton | heston | regime
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

# Full backtest — all models — 5 years of data
python cli.py backtest --models gbm ou merton heston regime --period 5y

# Disable colors (for logging/piping)
python cli.py --no-color simulate --model ou --days 30
```

---

## 📊 Risk Metrics Explained

| Metric | Description |
|--------|-------------|
| **VaR 95%** | Maximum loss expected 95% of the time |
| **CVaR 95%** | Average loss in the worst 5% of scenarios |
| **Max Drawdown** | Worst peak-to-trough price drop across all paths |
| **90% CI** | Range containing 90% of all simulated final prices |
| **Prob. of Gain** | % of paths where final price > current price |
| **Realised Vol** | Annualised standard deviation of simulated returns |

---

## 🔬 Backtesting Methodology

The backtester uses **walk-forward analysis** — the industry standard for validating time series models without lookahead bias:

```
├── Training Window (252 days default)  →  Calibrate model parameters
├── Test Window    (30 days default)    →  Evaluate out-of-sample predictions
└── Step forward by test window size    →  Repeat across full history
```

**Accuracy metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| RMSE | √mean((pred − actual)²) | Lower is better |
| MAE | mean(\|pred − actual\|) | Lower is better |
| MAPE | mean(\|error/actual\|) × 100 | < 5% is good |
| Directional Accuracy | % correct up/down calls | > 55% is good |

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
model = create_model('ou', historical_data=prices['close'].values)

# Simulate 10,000 paths for 30 days
paths = model.simulate(S0=2350.0, n_steps=30, n_paths=10000)

# Results
import numpy as np
final_prices = paths[:, -1]
print(f"Expected price: ${np.mean(final_prices):,.2f}")
print(f"95% CI: [${np.percentile(final_prices,5):,.2f}, ${np.percentile(final_prices,95):,.2f}]")
print(f"Prob. of gain:  {np.mean(final_prices > 2350)*100:.1f}%")
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
- [ ] Integrate FRED API for live macro features
- [ ] Portfolio mode — simulate multiple assets simultaneously
- [ ] Add implied volatility surface from gold options data
- [ ] Deploy dashboard to Streamlit Cloud
- [ ] Add email/Telegram alerts for price target breaches

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Essabri Ali Rayan**  
GitHub: [@ASAPUI](https://github.com/ASAPUI)  
Project: [monte-carlo-options](https://github.com/ASAPUI/monte-carlo-options)
customise service: +2120694025836 or alirayanessabri@gmail.com
---
## Models

The system supports six models:

| Model | Key | Description |
|-------|-----|-------------|
| Geometric Brownian Motion | `gbm` | Classic Black-Scholes model |
| Ornstein-Uhlenbeck | `ou` | Mean reversion model |
| Merton Jump Diffusion | `merton` | Jump diffusion model |
| Heston | `heston` | Stochastic volatility |
| Regime Switching | `regime` | Markov regime switching |
| **LSTM Neural Network** | `lstm` | **Deep learning baseline (NEW in v2.0)** |

### Using the LSTM Model

```bash
# Run simulation with LSTM
python cli.py simulate --model lstm --days 30

# Compare all models including LSTM
python cli.py compare --days 30

# Backtest specific models
python cli.py backtest --models gbm ou lstm --period 2y
> *"I don't know exactly what will happen, but if I simulate thousands of possible futures and average them, I'll get a very good estimate."* — The Monte Carlo philosophy 🎲
=======
# 🥇 Monte Carlo Gold Price Predictor

> A production-grade stochastic simulation engine for gold price forecasting and risk analysis.

**Author:** Essabri Ali Rayan  
**Version:** 1.0.0  
**License:** MIT

---

## 📖 Overview

This project uses **Monte Carlo simulation** to predict gold price movements and quantify risk. It combines 5 advanced stochastic models with real market data, professional risk analytics, an interactive Streamlit dashboard, and a full command-line interface.

The system mirrors tools used by professional quantitative traders and hedge funds — calibrating models from historical data, simulating thousands of possible price futures, and extracting probability distributions and risk metrics from the results.

---

## ✨ Features

- **5 Stochastic Models** — GBM, Ornstein-Uhlenbeck, Merton Jump Diffusion, Heston, Regime Switching
- **Live Market Data** — Auto-fetches gold prices via Yahoo Finance (GC=F)
- **Risk Analytics** — VaR, CVaR, confidence intervals, drawdown analysis
- **Interactive Dashboard** — 4-tab Streamlit web app with Plotly charts
- **Command Line Interface** — Full CLI with 5 commands and JSON/CSV export
- **Walk-Forward Backtesting** — No-lookahead-bias validation on historical data
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
│   │   └── regime_switching.py  ← Markov regime switching
│   │
│   └── backtesting/
│       ├── __init__.py          ← Package exports
│       ├── backtester.py        ← Walk-forward testing engine
│       └── comparison.py        ← Model ranking + statistical tests
│
├── app.py                       ← Streamlit web dashboard
├── cli.py                       ← Command-line interface
├── example.py                   ← Quick-start usage examples
├── requirements.txt             ← Python dependencies
├── pyproject.toml               ← Package configuration
└── README.md                    ← This file
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ASAPUI/monte-carlo-options.git
cd monte-carlo-options
pip install -r requirements.txt
```

### 2. Launch the Dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### 3. Or Use the CLI

```bash
# Get current gold price
python cli.py price

# Run a simulation (30 days, OU model, 5000 paths)
python cli.py simulate --model ou --days 30 --paths 5000

# Deep risk analysis
python cli.py risk --model merton --days 60 --target 2600

# Backtest multiple models
python cli.py backtest --models gbm ou merton --period 2y

# Compare all models side by side
python cli.py compare --days 30 --output results.json
```

---

## 📦 Requirements

```
Python >= 3.10
numpy
pandas
scipy
numba
yfinance
fredapi          # optional — for macro data
streamlit
plotly
scikit-learn
pytest
torch
```

Install everything at once:

```bash
pip install -r requirements.txt
```

---

## 🧠 The 5 Stochastic Models

### 1. Geometric Brownian Motion (GBM)
```
dS = μ·S·dt + σ·S·dW
```
The foundational model. Assumes log-normally distributed returns with constant drift and volatility. Used as a **baseline benchmark**.

### 2. Ornstein-Uhlenbeck — Mean Reversion ⭐ Most important for gold
```
dS = κ·(θ − S)·dt + σ·dW
```
Captures gold's documented tendency to revert to long-term equilibrium. `κ` controls the speed of reversion, `θ` is the long-term fair value. **Best for long-term forecasting.**

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
Models volatility itself as a stochastic process, capturing volatility clustering and the leverage effect. **Best for options pricing and volatility forecasting.**

### 5. Regime Switching (Markov Chain)
```
dS = μ(sₜ)·S·dt + σ(sₜ)·S·dW
```
Alternates between distinct market states (calm / crisis) using a Markov chain. Each regime has its own drift and volatility parameters. **Best for scenario planning.**

---

## 🖥️ Dashboard — 4 Tabs

| Tab | Content |
|-----|---------|
| 📈 Simulation | Fan chart, confidence bands, scenario table, return distribution |
| ⚠️ Risk Analysis | VaR/CVaR metrics, drawdown distribution, target price probability |
| 📊 Market Data | Historical chart, return statistics, rolling volatility |
| ℹ️ Model Guide | Formula reference, pros/cons for each model |

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
  --model   -m    Model: gbm | ou | merton | heston | regime
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

# Full backtest — all models — 5 years of data
python cli.py backtest --models gbm ou merton heston regime --period 5y

# Disable colors (for logging/piping)
python cli.py --no-color simulate --model ou --days 30
```

---

## 📊 Risk Metrics Explained

| Metric | Description |
|--------|-------------|
| **VaR 95%** | Maximum loss expected 95% of the time |
| **CVaR 95%** | Average loss in the worst 5% of scenarios |
| **Max Drawdown** | Worst peak-to-trough price drop across all paths |
| **90% CI** | Range containing 90% of all simulated final prices |
| **Prob. of Gain** | % of paths where final price > current price |
| **Realised Vol** | Annualised standard deviation of simulated returns |

---

## 🔬 Backtesting Methodology

The backtester uses **walk-forward analysis** — the industry standard for validating time series models without lookahead bias:

```
├── Training Window (252 days default)  →  Calibrate model parameters
├── Test Window    (30 days default)    →  Evaluate out-of-sample predictions
└── Step forward by test window size    →  Repeat across full history
```

**Accuracy metrics:**

| Metric | Formula | Target |
|--------|---------|--------|
| RMSE | √mean((pred − actual)²) | Lower is better |
| MAE | mean(\|pred − actual\|) | Lower is better |
| MAPE | mean(\|error/actual\|) × 100 | < 5% is good |
| Directional Accuracy | % correct up/down calls | > 55% is good |

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
model = create_model('ou', historical_data=prices['close'].values)

# Simulate 10,000 paths for 30 days
paths = model.simulate(S0=2350.0, n_steps=30, n_paths=10000)

# Results
import numpy as np
final_prices = paths[:, -1]
print(f"Expected price: ${np.mean(final_prices):,.2f}")
print(f"95% CI: [${np.percentile(final_prices,5):,.2f}, ${np.percentile(final_prices,95):,.2f}]")
print(f"Prob. of gain:  {np.mean(final_prices > 2350)*100:.1f}%")
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

- [✅ ] Add LSTM/neural network model for comparison
- [ ] Integrate FRED API for live macro features
- [ ] Portfolio mode — simulate multiple assets simultaneously
- [ ] Add implied volatility surface from gold options data
- [ ] Deploy dashboard to Streamlit Cloud

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

**Essabri Ali Rayan**  
GitHub: [@ASAPUI](https://github.com/ASAPUI)  
Project: [monte-carlo-options](https://github.com/ASAPUI/monte-carlo-options)
any customize service you can call:0694025836
---

> *"I don't know exactly what will happen, but if I simulate thousands of possible futures and average them, I'll get a very good estimate."* — The Monte Carlo philosophy 🎲
