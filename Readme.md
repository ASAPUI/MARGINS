# 🥇 MARGINS

> A production-grade stochastic simulation engine for gold price forecasting, risk analysis, and multi-asset portfolio optimization.

**Author:** Essabri Ali Rayan
**Version:** 1.4.2
**License:** BSL

---

## 📖 Overview

MARGINS uses **Monte Carlo simulation** to predict gold price movements, quantify risk, and optimize multi-asset portfolios. It combines 5 advanced stochastic models with real market data, professional risk analytics, an interactive Streamlit dashboard, and a full command-line interface.

The system mirrors tools used by professional quantitative traders and hedge funds — calibrating models from historical data, simulating thousands of possible price futures, and extracting probability distributions and risk metrics from the results.

**New in v2.0 — Portfolio Mode:** Simulate up to 20 correlated assets simultaneously using Cholesky-decomposed covariance matrices and Ledoit-Wolf shrinkage. Optimize allocations across three industry-standard strategies: Max Sharpe, Min CVaR, and Risk Parity.

---

## ⚠️ Price Warning

Prices are displayed **per troy ounce**:

---

## ✨ Features

### Single-Asset Engine (v1.x)
- **5 Stochastic Models** — GBM, Ornstein-Uhlenbeck, Merton Jump Diffusion, Heston, Regime Switching
- **Live Market Data** — Auto-fetches gold prices via Yahoo Finance (GC=F)
- **Risk Analytics** — VaR, CVaR, confidence intervals, drawdown analysis
- **Interactive Dashboard** — 4-tab Streamlit web app with Plotly charts
- **Command Line Interface** — Full CLI with 5 commands and JSON/CSV export
- **Walk-Forward Backtesting** — No-lookahead-bias validation on historical data
- **Model Comparison** — Leaderboard with statistical significance tests
- **Caching System** — Parquet/HDF5/CSV/Pickle storage with TTL expiration

### Portfolio Mode (v1.4.2) — New
- **Multi-Asset Simulation** — 2–20 correlated assets under a single Monte Carlo framework
- **Cholesky Correlation** — Correlation injected at every time step (not post-hoc) — the method used by Bloomberg PORT and MSCI RiskMetrics
- **Ledoit-Wolf Shrinkage** — Mandatory covariance regularization for positive-definite matrices
- **Portfolio Risk Metrics** — Portfolio VaR, CVaR, Sharpe ratio, diversification ratio, max drawdown, correlation benefit
- **3 Optimization Strategies** — Max Sharpe, Min CVaR, Risk Parity
- **Portfolio Dashboard Tab** — Fan chart, correlation heatmap, per-asset paths, optimizer comparison
- **2 New CLI Commands** — portfolio and optimize

---

## ⚡ Quick Start

### 1. Clone & Install
git clone https://github.com/ASAPUI/MARGINS.git
cd MARGINS
pip install -r requirements.txt
### 2. Launch the Dashboard
#for the app:
streamlit run app.py "or just double-click on run.vps"
#only dashbord:
streamlit run app_portfolio_tab.py
Open your browser at http://localhost:8501

### 3. Single-Asset CLI
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


### 4. Portfolio CLI (New in v1.4.2)
python cli_portfolio.py portfolio \\
      --assets GC=F SPY TLT USO BTC-USD \\
      --weights 0.40 0.30 0.20 0.05 0.05 \\
      --days 30 --paths 5000 --period 2y --calib-window 126 \\
      --output portfolio_report.json

> **Windows users:** PowerShell does not support \ line continuation. Use single-line commands or replace \ with a backtick `  ``.

---

## 📦 Requirements
yfinance>=0.2.0
pandas>=1.5.0
numpy>=1.24.0
scipy>=1.10.0
fredapi>=0.5.0
pyarrow>=10.0.0  # For Parquet support
tables>=3.8.0    # For HDF5 support
torch>=2.0.0
httpx>=0.27
cachetools>=5.3
pydantic>=2.0
---

## 🧠 The 5 Stochastic Models

### 1. Geometric Brownian Motion (GBM)

The foundational model. Assumes log-normally distributed returns with constant drift and volatility. Used as a **baseline benchmark**.

### 2. Ornstein-Uhlenbeck — Mean Reversion ⭐ Most important for gold

Captures gold's documented tendency to revert to long-term equilibrium. κ controls the speed of reversion, θ is the long-term fair value. **Best for long-term forecasting.**

### 3. Merton Jump Diffusion

Extends GBM with a Poisson jump process to model sudden price shocks from geopolitical crises or central bank interventions. **Best for risk analysis and stress testing.**

### 4. Heston Stochastic Volatility

Models volatility itself as a stochastic process, capturing volatility clustering and the leverage effect. **Best for options pricing and volatility forecasting.**

### 5. Regime Switching (Markov Chain)

Alternates between distinct market states (calm / crisis) using a Markov chain. Each regime has its own drift and volatility parameters. **Best for scenario planning.**

---

## 📐 Portfolio Mode — Scientific Foundation

Portfolio Mode is built on three published academic pillars:

### Markowitz Mean-Variance (1952)
Portfolio variance is not a weighted average of individual variances:

Correlation must be injected at every simulation step — simulating assets independently and combining them at the end is **mathematically incorrect**.

### Ledoit-Wolf Shrinkage (2004)
The raw sample covariance matrix is noisy and frequently ill-conditioned. Shrinkage produces a well-conditioned, positive-definite matrix:

Using np.cov() directly without shrinkage is **prohibited** in this implementation.

### Cholesky Decomposition for Correlated Simulation
Correlation is applied at every time step via:

This is the approach used by Bloomberg PORT, MSCI RiskMetrics, and all major sell-side risk desks.

---

## 🏗️ Portfolio Mode — Architecture
src/
  portfolio/
    __init__.py      ← exports: simulate_portfolio, optimize_weights
    universe.py      ← asset definitions, price fetching, date alignment
    correlation.py   ← Ledoit-Wolf shrinkage, Cholesky decomposition
    simulator.py     ← correlated Monte Carlo engine (vectorized)
    optimizer.py     ← max Sharpe, min CVaR, risk parity
    metrics.py       ← portfolio VaR, CVaR, Sharpe, diversification


---


---

## 🖥️ Dashboard — 5 Tabs

| Tab | Content |
|-----|---------|
| 📈 Simulation | Fan chart, confidence bands, scenario table, return distribution |
| ⚠️ Risk Analysis | VaR/CVaR metrics, drawdown distribution, target price probability |
| 📊 Market Data | Historical chart, return statistics, rolling volatility |
| ℹ️ Model Guide | Formula reference, pros/cons for each model |
| 💼 Portfolio Mode | Multi-asset simulation, correlation heatmap, optimizer comparison (v2.0) |

---

## 📊 Risk Metrics

### Single-Asset
| Metric | Description |
|--------|-------------|
| VaR 95% | Maximum loss expected 95% of the time |
| CVaR 95% | Average loss in the worst 5% of scenarios |
| Max Drawdown | Worst peak-to-trough price drop across all paths |
| 90% CI | Range containing 90% of all simulated final prices |
| Prob. of Gain | % of paths where final price > current price |
| Realised Vol | Annualised standard deviation of simulated returns |

### Portfolio (v2.0)
| Metric | Description |
|--------|-------------|
| Portfolio VaR 95% | 5th percentile terminal portfolio value |
| Portfolio CVaR 95% | Mean of worst 5% of terminal portfolio values |
| Sharpe Ratio | Annualized risk-adjusted return E[Rp] / σ(Rp) · √252 |
| Diversification Ratio | Weighted vol / portfolio vol — > 1 means diversification benefit |
| Correlation Benefit | Risk reduction from inter-asset correlation |
| Avg Max Drawdown | Average worst peak-to-trough across all paths |

---

## ⚖️ Portfolio Optimization Strategies

### Max Sharpe Ratio

Maximizes risk-adjusted return. Tends to concentrate in 1–3 assets — the 40% position cap is essential.

### Min CVaR (Tail Risk Minimization)

Preferred by professional risk desks. Directly targets the tail of the loss distribution.

### Risk Parity (Analytic)

Each asset contributes equally to total portfolio variance. Popularized by Bridgewater's All-Weather fund. Computed analytically — always converges.

---

## 🔬 Backtesting Methodology

The backtester uses **walk-forward analysis** — the industry standard for validating time series models without lookahead bias:

| Metric | Formula | Target |
|--------|---------|--------|
| RMSE | √mean((pred − actual)²) | Lower is better |
| MAE | mean(\|pred − actual\|) | Lower is better |
| MAPE | mean(\|error/actual\|) × 100 | < 5% is good |
| Directional Accuracy | % correct up/down calls | > 55% is good |

---

---

## 🗺️ Roadmap

- [x] Add LSTM/neural network model for comparison
- [x] Integrate FRED API for live macro features
- [x] Portfolio mode — simulate multiple assets simultaneously
- [x] Cholesky-correlated Monte Carlo engine
- [x] Portfolio dashboard tab (Streamlit)
- [x] Max Sharpe / Min CVaR / Risk Parity optimization
- [ ] Walk-forward portfolio backtest (Phase 6)
- [ ] Black-Litterman views — Bayesian analyst forecast integration (Phase 7)
- [ ] Implied volatility surface from gold options data
- [ ] Deploy dashboard to Streamlit Cloud
- [ ] Email / Telegram alerts for price target breaches

---

## 📋 Changelog

### v1.4.2
- **Portfolio Mode** — full multi-asset Monte Carlo framework (Phases 1–5)
- Added src/portfolio/ — universe, correlation, simulator, metrics, optimizer modules
- Added cli_portfolio.py with portfolio and optimize commands
- Added Portfolio tab to Streamlit dashboard (app_portfolio_tab.py)
- Ledoit-Wolf covariance shrinkage with Cholesky decomposition
- Three optimization strategies: Max Sharpe, Min CVaR, Risk Parity
- Portfolio risk metrics: VaR, CVaR, Sharpe, diversification ratio, drawdown, correlation benefit

### v1.3.2
- Bug fixes and stability improvements
- Walk-forward backtest MAPE fix (log-return error instead of dollar MAE)

### v1.0.0
- Initial release — 5 stochastic models, CLI, Streamlit dashboard

---

## 📄 License

BSL — Free for non-commercial use.
This is an AI-assisted project. Results are for educational and research purposes only. Always verify outputs before making financial decisions.

---

## 👤 Author

**Essabri Ali Rayan**
GitHub: [@ASAPUI](https://github.com/ASAPUI)
Project: [MARGINS](https://github.com/ASAPUI/MARGINS)
Contact: +212 0694025836 | alirayanessabri@gmail.com
linkdin:https://www.linkedin.com/in/essabri-ali-rayan-9a64b13b2/

---

> *"I don't know exactly what will happen, but if I simulate thousands of possible futures and average them, I'll get a very good estimate."* — The Monte Carlo philosophy 🎲
>warning: we don't take any responsability of any lose or decrasse of the value of your asset after the wrong usage of our product and we don't take responsabilities of any bad financial action or desicion taken by any of the user it's just a tool that help to anlyze the gold market thank you for using our product . 
