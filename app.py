"""
Monte Carlo Gold Price Predictor — Streamlit App

A production-grade interactive dashboard for gold price simulation
using multiple stochastic models.

Author: Essabri Ali Rayan
Version: 1.4.4 

"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # noqa: F401 – kept for callers
from datetime import datetime, timedelta  # noqa: F401
from datetime import datetime as dt_datetime
from scipy.stats import skew as sp_skew
from app_portfolio_tab import render_portfolio_tab
import logging
import time  # noqa: F401

# ── Optional autorefresh ────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False

# ── Optional macro bridge ───────────────────────────────────────────────────
try:
    from src.macro import MacroBridge, ParameterAdjuster
    HAS_MACRO = True
except ImportError:
    HAS_MACRO = False
    MacroBridge = None  # type: ignore
    ParameterAdjuster = None  # type: ignore

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Gold Predictor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS —  MARGINS ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Playfair+Display:wght@500&family=DM+Sans:wght@400&display=swap');

/* ── Design tokens ── */
:root {
    --bg-base:       #0B0C14;
    --bg-panel:      #10121E;
    --border:        #1F2240;
    --border-hover:  #2E3160;
    --gold:          #C9A84C;
    --gold-light:    #E8C97A;
    --gold-hover:    rgba(201,168,76,0.10);
    --text-primary:  #ECEDF5;
    --text-secondary:#9395B0;
    --text-muted:    #5A5C78;
    --up:            #22C55E;
    --down:          #EF4444;
}

/* ── Base ── */
html, body, .stApp, [class*="css"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stApp { background-color: var(--bg-base) !important; }
[data-testid="stAppViewContainer"] { background-color: var(--bg-base) !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography hierarchy ── */
/* h1 — page title: Playfair Display */
h1 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}
/* h2/h3 — section headers: IBM Plex Mono uppercase */
h2, h3, h4 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 6px !important;
    margin-bottom: 12px !important;
}
/* Description text: DM Sans */
p, .stMarkdown p {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-secondary) !important;
    font-size: 13px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    font-family: 'IBM Plex Mono', monospace !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 12px 0 !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    padding: 16px !important;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 400 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: var(--text-muted) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 500 !important;
    color: var(--gold-light) !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.16em !important;
    color: var(--text-secondary) !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--gold) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: var(--gold-hover) !important;
}

/* ── Buttons — primary ── */
.stButton > button {
    background: linear-gradient(135deg, #C9A84C, #E8C97A) !important;
    color: #0B0C14 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.55rem 1.5rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--border-hover) !important;
    color: var(--text-primary) !important;
}

/* ── Inputs / selects / sliders ── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
div[data-baseweb="select"] > div {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--gold) !important;
}
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    color: var(--text-primary) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--gold) !important;
    box-shadow: none !important;
}

/* Widget labels */
label[data-testid="stWidgetLabel"],
.stSlider label,
.stSelectbox label,
.stCheckbox label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--text-muted) !important;
}

/* Slider track */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--gold) !important;
    border-color: var(--gold) !important;
}
.stSlider [data-baseweb="slider"] div[class*="Track"] {
    background: var(--border) !important;
}

/* ── Expanders ── */
.stExpander {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}
.stExpander summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--text-secondary) !important;
}

/* ── DataFrames / tables ── */
[data-testid="stDataFrame"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}
[data-testid="stDataFrame"] th {
    background: var(--bg-panel) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: var(--text-muted) !important;
    border-bottom: 1px solid var(--border) !important;
}
[data-testid="stDataFrame"] td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-color: var(--border) !important;
}

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    border: 1px solid var(--border) !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; }

/* ── Custom utility classes ── */
/* Gold-box: used in target analysis */
.gold-box {
    background: rgba(201,168,76,0.06) !important;
    border: 1px solid rgba(201,168,76,0.25) !important;
    border-radius: 0 !important;
    padding: 16px 20px !important;
    margin: 12px 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.gold-box b { color: var(--text-secondary) !important; font-size: 9px !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; }

/* Stat pills */
.stat-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
.stat-pill {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 2px 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: var(--gold-light);
    letter-spacing: 0.05em;
}

/* Section header accent line */
.margins-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
    position: relative;
}
.margins-section::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 32px;
    height: 1px;
    background: var(--gold);
}

/* Sidebar brand */
.margins-brand {
    font-family: 'Playfair Display', serif !important;
    font-size: 22px !important;
    font-weight: 500 !important;
    color: var(--gold) !important;
    letter-spacing: 0.04em !important;
    line-height: 1.2 !important;
}

/* Spinner / progress */
[data-testid="stSpinner"] { color: var(--gold) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }
</style>
""", unsafe_allow_html=True)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

# ── Colour / layout constants (must come BEFORE plotting helpers) ────────────
# FIX 5: moved these up so helper functions can reference them at definition time.
DARK_LAYOUT = dict(
    paper_bgcolor="#0B0C14",
    plot_bgcolor="#0B0C14",
    font=dict(color="#9395B0", family="IBM Plex Mono"),
    xaxis=dict(gridcolor="#1F2240", showgrid=True, zeroline=False,
               tickfont=dict(color="#9395B0", family="IBM Plex Mono", size=10)),
    yaxis=dict(gridcolor="#1F2240", showgrid=True, zeroline=False,
               tickfont=dict(color="#9395B0", family="IBM Plex Mono", size=10)),
    margin=dict(l=40, r=20, t=40, b=40),
)

GOLD  = "#C9A84C"
GOLD2 = "#E8C97A"
RED   = "#EF4444"
GREEN = "#22C55E"
BLUE  = "#5599EE"

# ── Model Imports (graceful fallbacks) ──────────────────────────────────────
MODELS_AVAILABLE = {}

try:
    from src.models.gbm import GeometricBrownianMotion, GBMParameters  # noqa: F401
    MODELS_AVAILABLE["GBM"] = GeometricBrownianMotion
except ImportError:
    pass

# FIX 1: was orphaned inside the previous try/except; now has its own try block.
try:
    from src.models.mean_reversion import OrnsteinUhlenbeckModel, OUParameters  # noqa: F401
    MODELS_AVAILABLE["Mean Reversion (OU)"] = OrnsteinUhlenbeckModel
except ImportError:
    pass

try:
    from src.models.jump_diffusion import MertonJumpModel, MertonParameters  # noqa: F401
    MODELS_AVAILABLE["Jump Diffusion"] = MertonJumpModel
except ImportError:
    pass

try:
    from src.models.heston import HestonModel, HestonParameters, create_heston_model  # FIX 3

    # FIX 2: all methods were unindented (at module level). Re-indented under class.
    class HestonBates:
        """Thin wrapper that exposes Heston+Jumps (Bates) as a drop-in model."""

        def __init__(self):
            self._model = create_heston_model(enable_jumps=True)  # FIX 3
            self.params = self._model.params

        def calibrate(self, prices):
            self._model.calibrate(prices)
            self.params = self._model.params

        def simulate(self, S0, n_steps, n_paths=1000, random_seed=None, antithetic=True):
            return self._model.simulate(
                S0=S0, n_steps=n_steps, n_paths=n_paths,
                random_seed=random_seed, antithetic=antithetic,
            )

    MODELS_AVAILABLE["Heston"] = HestonModel
    MODELS_AVAILABLE["Heston + Jumps (Bates)"] = HestonBates
except ImportError:
    pass

try:
    from src.models.regime_switching import RegimeSwitchingModel
    MODELS_AVAILABLE["Regime Switching"] = RegimeSwitchingModel
except ImportError:
    pass


# ── Demo / Fallback Model ───────────────────────────────────────────────────
class DemoGBM:
    """Minimal GBM fallback when src/ is not installed."""

    name = "GBM (demo)"

    def __init__(self, mu=0.05, sigma=0.15):
        self.mu = mu
        self.sigma = sigma

    def calibrate(self, returns):
        self.mu    = float(np.mean(returns) * 252)
        self.sigma = float(np.std(returns) * np.sqrt(252))

    def simulate(self, S0, n_steps, n_paths=1000, random_seed=None, **kwargs):
        if random_seed:
            np.random.seed(random_seed)
        dt = 1 / 252
        Z  = np.random.standard_normal((n_paths, n_steps - 1))
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0
        for t in range(1, n_steps):
            drift       = (self.mu - 0.5 * self.sigma ** 2) * dt
            diff        = self.sigma * np.sqrt(dt) * Z[:, t - 1]
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diff)
        return paths


if not MODELS_AVAILABLE:
    MODELS_AVAILABLE = {
        "GBM":              DemoGBM,
        "Mean Reversion":   DemoGBM,
        "Jump Diffusion":   DemoGBM,
        "Heston":           DemoGBM,
        "Regime Switching": DemoGBM,
    }


# ── Data Helpers ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_gold_data(period="2y"):
    try:
        import yfinance as yf
        df = yf.Ticker("GC=F").history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError("empty")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[["Close"]].rename(columns={"Close": "price"}).dropna()
    except Exception:
        dates = pd.date_range(end=datetime.today(), periods=504, freq="B")
        rng   = np.random.default_rng(0)
        log_r = rng.normal(0.0002, 0.012, len(dates))
        price = 1800 * np.exp(np.cumsum(log_r))
        return pd.DataFrame({"price": price}, index=dates)


@st.cache_data(ttl=60)
def get_current_price():
    try:
        import yfinance as yf
        data = yf.Ticker("GC=F").history(period="5d", interval="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1]), float(data["Close"].pct_change().iloc[-1] * 100)
    except Exception:
        pass
    return 2350.0, 0.0


def run_simulation(model_name, S0, n_steps, n_paths, mu, sigma, prices, seed=None, macro_signals=None):
    """Instantiate, optionally calibrate, and simulate. Returns path array."""
    ModelCls = MODELS_AVAILABLE[model_name]
    model    = ModelCls()

    # Calibrate from recent price history
    try:
        log_returns = np.log(prices["price"] / prices["price"].shift(1)).dropna().values
        if hasattr(model, "calibrate"):
            if model_name in ("Mean Reversion (OU)", "Heston", "Heston + Jumps (Bates)", "Regime Switching"):
                model.calibrate(prices["price"].values)
            else:
                model.calibrate(log_returns)
    except Exception:
        pass

    # Apply macro parameter adjustments
    if HAS_MACRO and macro_signals is not None and not macro_signals.is_fallback:
        try:
            adjuster = ParameterAdjuster(macro_signals)

            if model_name == "GBM" and hasattr(model, "mu") and hasattr(model, "sigma"):
                p = adjuster.adjust_gbm(model.mu, model.sigma)
                model.mu    = p.mu_adjusted
                model.sigma = p.sigma_adjusted

            elif model_name == "Mean Reversion (OU)" and hasattr(model, "mu"):
                theta = getattr(model, "theta", model.mu)
                kappa = getattr(model, "kappa", 1.0)
                p = adjuster.adjust_ou(model.mu, model.sigma, theta, kappa)
                model.mu    = p.mu_adjusted
                model.sigma = p.sigma_adjusted
                if hasattr(model, "theta") and p.theta_adjusted is not None:
                    model.theta = p.theta_adjusted

            elif model_name == "Jump Diffusion" and hasattr(model, "mu"):
                lambda_j = getattr(model, "lambda_jump", getattr(model, "lambda_", 0.1))
                mu_j     = getattr(model, "mu_j", 0.0)
                sigma_j  = getattr(model, "sigma_j", 0.02)
                p = adjuster.adjust_merton(model.mu, model.sigma, lambda_j, mu_j, sigma_j)
                model.mu    = p.mu_adjusted
                model.sigma = p.sigma_adjusted
                for attr, val in [
                    ("lambda_jump", p.lambda_adjusted),
                    ("lambda_",     p.lambda_adjusted),
                    ("mu_j",        p.mu_j_adjusted),
                    ("sigma_j",     p.sigma_j_adjusted),
                ]:
                    if hasattr(model, attr):
                        setattr(model, attr, val)

            elif model_name in ("Heston", "Heston + Jumps (Bates)") and hasattr(model, "params"):
                p_obj = model.params
                try:
                    p = adjuster.adjust_heston(
                        p_obj.mu, p_obj.v0, p_obj.theta, p_obj.kappa, p_obj.xi, p_obj.rho
                    )
                    model.params.mu    = p.mu_adjusted
                    model.params.v0    = p.sigma_adjusted ** 2
                    model.params.theta = p.theta_v_adjusted
                    model.params.xi    = p.xi_adjusted
                except Exception:
                    pass

            elif model_name == "Regime Switching":
                mu_calm      = getattr(model, "mu_calm",     getattr(model, "mu", 0.05))
                sigma_calm   = getattr(model, "sigma_calm",  getattr(model, "sigma", 0.12))
                mu_crisis    = getattr(model, "mu_crisis",   -0.10)
                sigma_crisis = getattr(model, "sigma_crisis", 0.30)
                p = adjuster.adjust_regime(mu_calm, sigma_calm, mu_crisis, sigma_crisis)
                for attr, val in [
                    ("p_crisis",         p.p_crisis),
                    ("p_calm_to_crisis", p.p_calm_to_crisis),
                ]:
                    if hasattr(model, attr):
                        setattr(model, attr, val)

        except Exception as e:
            logging.warning("Macro adjustment skipped: %s", e)

    # Manual overrides always win
    if mu is not None and hasattr(model, "mu"):
        model.mu = mu
    if sigma is not None and hasattr(model, "sigma"):
        model.sigma = sigma

    if model_name in ("Heston", "Heston + Jumps (Bates)"):
        paths = model.simulate(S0=S0, n_steps=n_steps, n_paths=n_paths, random_seed=seed, antithetic=True)
    else:
        paths = model.simulate(S0=S0, n_steps=n_steps, n_paths=n_paths, random_seed=seed)

    return paths  # FIX 4: was missing — callers received None


# ── Plotting Helpers ─────────────────────────────────────────────────────────
# (DARK_LAYOUT and colour constants now defined above — FIX 5)

def fan_chart(paths, dates_future, S0, current_price_series):
    """Fan chart with confidence bands."""
    p5, p25, p50, p75, p95 = (
        np.percentile(paths, q, axis=0) for q in [5, 25, 50, 75, 95]
    )
    n_show = min(200, paths.shape[0])
    idx    = np.random.choice(paths.shape[0], n_show, replace=False)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=current_price_series.index, y=current_price_series.values,
        name="Historical", line=dict(color=GOLD, width=2),
    ))

    for i in idx:
        fig.add_trace(go.Scatter(
            x=dates_future, y=paths[i],
            mode="lines", line=dict(color="rgba(201,168,76,0.04)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=list(dates_future) + list(dates_future[::-1]),
        y=list(p95) + list(p5[::-1]),
        fill="toself", fillcolor="rgba(201,168,76,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="90% CI", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=list(dates_future) + list(dates_future[::-1]),
        y=list(p75) + list(p25[::-1]),
        fill="toself", fillcolor="rgba(201,168,76,0.14)",
        line=dict(color="rgba(0,0,0,0)"), name="50% CI", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates_future, y=p50,
        name="Median", line=dict(color=GOLD2, width=2.5, dash="dot"),
    ))

    fig.update_layout(
        title="Simulated Gold Price Paths",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        **DARK_LAYOUT,
    )
    return fig


def distribution_chart(final_prices, S0):
    """Histogram of final prices."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=final_prices, nbinsx=80,
        marker_color=GOLD, opacity=0.75, name="Final Price",
    ))
    fig.add_vline(x=float(np.mean(final_prices)),
                  line=dict(color=GOLD2, dash="dash", width=2),
                  annotation_text="Mean", annotation_position="top right")
    fig.add_vline(x=S0,
                  line=dict(color="#AAAACC", dash="dot", width=1.5),
                  annotation_text="Current", annotation_position="top left")
    fig.update_layout(
        title="Distribution of Final Prices",
        xaxis_title="Price (USD)", yaxis_title="Count",
        **DARK_LAYOUT,
    )
    return fig


def risk_gauge(var_95, cvar_95, S0):
    """Risk metrics as horizontal bar."""
    loss_pct  = (S0 - var_95)  / S0 * 100   # FIX 10: added space before *
    closs_pct = (S0 - cvar_95) / S0 * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[loss_pct, closs_pct],
        y=["VaR 95%", "CVaR 95%"],
        orientation="h",
        marker_color=[RED, "#FF7777"],
        text=[f"{loss_pct:.1f}%", f"{closs_pct:.1f}%"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Downside Risk (% of Current Price)",
        xaxis_title="Potential Loss (%)",
        **DARK_LAYOUT,
    )
    return fig


def historical_chart(prices):
    """Full historical price chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices["price"],
        fill="tozeroy", fillcolor="rgba(201,168,76,0.06)",
        line=dict(color=GOLD, width=1.8), name="Gold Price",
    ))
    fig.update_layout(
        title="Historical Gold Price (GC=F)",
        yaxis_title="Price (USD)",
        **DARK_LAYOUT,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TRADE SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_trade_signal(paths, S0, n_steps, macro_signals=None):
    """
    Derive a structured trade recommendation from simulation output.

    Returns a dict with: signal, confidence, entry, stop_loss, take_profit,
    risk_reward, reasons, warnings, and supporting statistics.
    """
    final = paths[:, -1]
    S0    = float(S0)

    mean_final   = float(np.mean(final))
    median_final = float(np.median(final))  # noqa: F841
    p5  = float(np.percentile(final, 5))
    p10 = float(np.percentile(final, 10))   # noqa: F841
    p25 = float(np.percentile(final, 25))
    p75 = float(np.percentile(final, 75))
    p90 = float(np.percentile(final, 90))
    p95 = float(np.percentile(final, 95))

    prob_gain   = float(np.mean(final > S0))
    prob_gain5  = float(np.mean(final > S0 * 1.05))
    prob_loss5  = float(np.mean(final < S0 * 0.95))
    prob_loss10 = float(np.mean(final < S0 * 0.90))  # noqa: F841

    expected_return = (mean_final - S0) / S0

    peak      = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - peak) / peak
    avg_max_dd = float(np.mean(np.min(drawdowns, axis=1)))

    ret_skew = float(sp_skew(final))

    # ── Macro overlay ────────────────────────────────────────────────────────
    macro_risk  = "neutral"
    macro_boost = 0.0       # noqa: F841 – kept for potential future weighting
    macro_warn  = []

    if macro_signals is not None and not macro_signals.is_fallback:
        cii_avg = macro_signals.cii_top5_avg
        cii_max = macro_signals.cii_max
        n_crit  = macro_signals.critical_anomaly_count
        n_high  = macro_signals.high_anomaly_count

        if cii_avg >= 70 or n_crit >= 2:
            macro_risk  = "high"
            macro_boost = 0.05
            macro_warn.append(f"CII Top-5 avg {cii_avg:.0f}/100 — elevated geopolitical stress")
        elif cii_avg >= 55 or n_high >= 3:
            macro_risk  = "elevated"
            macro_boost = 0.02
            macro_warn.append(f"CII Top-5 avg {cii_avg:.0f}/100 — moderate geopolitical risk")
        else:
            macro_risk  = "calm"
            macro_boost = -0.01

        if cii_max >= 80:
            macro_warn.append(f"Single-country crisis risk: CII max {cii_max:.0f}/100")
        if n_crit > 0:
            macro_warn.append(f"{n_crit} critical anomaly event(s) detected")

    # ── Signal scoring ───────────────────────────────────────────────────────
    reasons  = []
    warnings = []
    score    = 0

    if expected_return > 0.05:
        score += 3
        reasons.append(f"Strong expected return: +{expected_return*100:.1f}%")
    elif expected_return > 0.02:
        score += 2
        reasons.append(f"Positive expected return: +{expected_return*100:.1f}%")
    elif expected_return > 0:
        score += 1
        reasons.append(f"Marginal positive expected return: +{expected_return*100:.2f}%")
    elif expected_return < -0.02:
        score -= 2
        reasons.append(f"Negative expected return: {expected_return*100:.1f}%")
    else:
        score -= 1
        reasons.append(f"Slightly negative expected return: {expected_return*100:.2f}%")

    if prob_gain > 0.65:
        score += 2
        reasons.append(f"High probability of gain: {prob_gain*100:.0f}%")
    elif prob_gain > 0.55:
        score += 1
        reasons.append(f"Moderate probability of gain: {prob_gain*100:.0f}%")
    elif prob_gain < 0.40:
        score -= 2
        reasons.append(f"Low probability of gain: {prob_gain*100:.0f}%")

    if ret_skew > 0.3:
        score += 1
        reasons.append(f"Positive return skew ({ret_skew:.2f}) — upside tail larger than downside")
    elif ret_skew < -0.3:
        score -= 1
        warnings.append(f"Negative return skew ({ret_skew:.2f}) — downside tail risk elevated")

    if macro_risk == "high":
        score += 1
        reasons.append("Macro: high geopolitical stress → safe-haven gold demand")
    elif macro_risk == "elevated":
        reasons.append("Macro: elevated geopolitical risk — supportive for gold")
    elif macro_risk == "calm":
        score -= 1
        reasons.append("Macro: calm environment — gold may face headwinds vs risk assets")

    upside   = p75 - S0
    downside = S0 - p25
    rr_ratio = upside / downside if downside > 0 else 0
    if rr_ratio >= 2.0:
        score += 1
        reasons.append(f"Favourable risk/reward: {rr_ratio:.1f}x")
    elif rr_ratio < 1.0:
        score -= 1
        warnings.append(f"Poor risk/reward ratio: {rr_ratio:.1f}x")

    if avg_max_dd < -0.10:
        warnings.append(f"High average drawdown: {avg_max_dd*100:.1f}% across paths")

    if score >= 4:
        signal, confidence = "BUY",  "HIGH"
    elif score >= 2:
        signal, confidence = "BUY",  "MEDIUM"
    elif score >= 1:
        signal, confidence = "BUY",  "LOW"
    elif score <= -3:
        signal, confidence = "SELL", "HIGH"
    elif score <= -1:
        signal, confidence = "SELL", "LOW"
    else:
        signal, confidence = "HOLD", "MEDIUM"

    entry       = S0
    stop_loss   = round(p5,  2)
    take_profit = round(p90, 2)
    rr_final    = (take_profit - entry) / (entry - stop_loss) if entry > stop_loss else 0

    warnings += macro_warn

    return {
        "signal":           signal,
        "confidence":       confidence,
        "score":            score,
        "entry":            round(entry, 2),
        "stop_loss":        stop_loss,
        "take_profit":      take_profit,
        "risk_reward":      round(rr_final, 2),
        "expected_return":  round(expected_return * 100, 2),
        "prob_gain":        round(prob_gain * 100, 1),
        "prob_loss5":       round(prob_loss5 * 100, 1),
        "prob_gain5":       round(prob_gain5 * 100, 1),
        "p25": round(p25, 2), "p75": round(p75, 2),
        "p5":  round(p5,  2), "p95": round(p95, 2),
        "mean_final":       round(mean_final, 2),
        "avg_max_dd":       round(avg_max_dd * 100, 2),
        "macro_risk":       macro_risk,
        "reasons":          reasons,
        "warnings":         warnings,
        "horizon_days":     n_steps,
    }


# ── Macro tab renderer ───────────────────────────────────────────────────────
def render_macro_tab():
    """Render the Macro Intelligence dashboard tab."""
    # FIX 8: guard against missing macro module
    if not HAS_MACRO:
        st.info(
            "The `src.macro` module is not installed. "
            "Macro Intelligence is unavailable in this environment."
        )
        return

    st.markdown("<p class='margins-section'>Macro Intelligence — WorldMonitor</p>", unsafe_allow_html=True)

    if "macro_bridge" not in st.session_state:
        st.session_state.macro_bridge  = MacroBridge()
        st.session_state.macro_signals = None

    bridge = st.session_state.macro_bridge

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 Refresh Signals", type="primary"):
            with st.spinner("Fetching macro data..."):
                signals_result = bridge.get_signals_sync()
                st.session_state.macro_signals = signals_result
                if signals_result.brief_text:
                    st.session_state.macro_brief = signals_result.brief_text
                else:
                    st.session_state.macro_brief = bridge.get_brief_sync()
            st.success("Signals updated")

    with col2:
        st.caption(f"Status: {'🟢 Live' if bridge.is_healthy() else '🟡 Fallback'}")

    with col3:
        enable_macro = st.toggle("Enable Macro Adjustments", value=True)
        st.session_state.macro_enabled = enable_macro

    signals = st.session_state.macro_signals

    if signals is None:
        st.info("Click 'Refresh Signals' to load WorldMonitor data")
        return

    risk_color = {
        "stable":   "#22C55E",
        "elevated": "#E8C97A",
        "high":     "#EF4444",
        "critical": "#EF4444",
        "extreme":  "#EF4444",
    }.get(signals.risk_tier.value, "#9395B0")

    st.markdown(f"""
    <div style='padding:12px 16px;background:#10121E;
                border:1px solid #1F2240;border-left:2px solid {risk_color};
                margin-bottom:12px;'>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:9px;
                    text-transform:uppercase;letter-spacing:0.14em;color:#5A5C78;'>
            Geopolitical Risk Level
        </div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:18px;
                    font-weight:500;color:{risk_color};margin-top:4px;'>
            {signals.risk_tier.value.upper()}
            {'  ·  FALLBACK MODE' if signals.is_fallback else ''}
        </div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:9px;
                    color:#5A5C78;margin-top:4px;'>
            Last updated: {signals.fetched_at.strftime('%Y-%m-%d %H:%M UTC')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CII Top-5 Avg",       f"{signals.cii_top5_avg:.1f}",
              delta=f"{signals.cii_top5_avg - 50:.1f} vs neutral")
    m2.metric("Max Single CII",       f"{signals.cii_max:.1f}")
    m3.metric("High Anomalies",       signals.high_anomaly_count)
    m4.metric("Critical Anomalies",   signals.critical_anomaly_count,
              delta=f"{signals.critical_anomaly_count}" if signals.critical_anomaly_count > 0 else None,
              delta_color="inverse")

    st.markdown("<p class='margins-section'>Country Instability Index — Top 10</p>", unsafe_allow_html=True)
    if signals.cii_scores:
        df = pd.DataFrame([
            {
                "Country":   k,
                "CII Score": v,
                "Risk":      "🔴" if v > 70 else "🟠" if v > 50 else "🟡" if v > 30 else "🟢",
            }
            for k, v in sorted(signals.cii_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

    if signals.anomaly_zscores:
        st.markdown("<p class='margins-section'>Live Anomaly Feed</p>", unsafe_allow_html=True)
        for anomaly in signals.anomaly_zscores[:5]:
            emoji = "🔴" if anomaly.z_score >= 3.0 else "🟠" if anomaly.z_score >= 2.0 else "🟡"
            with st.expander(f"{emoji} {anomaly.region} | {anomaly.event_type} (z={anomaly.z_score:.2f})"):
                st.write(f"**Region:** {anomaly.region}")
                st.write(f"**Type:** {anomaly.event_type}")
                st.write(f"**Z-Score:** {anomaly.z_score:.2f}")
                st.write(f"**Detected:** {anomaly.timestamp}")
                if anomaly.metadata:
                    st.json(anomaly.metadata)

    st.markdown("<p class='margins-section'>AI World Brief</p>", unsafe_allow_html=True)
    brief_text = None
    if signals.brief_text:
        brief_text = signals.brief_text
    elif st.session_state.get("macro_brief"):
        brief_text = st.session_state.macro_brief

    if brief_text:
        st.markdown(f"""
        <div style='background:#10121E;padding:16px 20px;
                    border:1px solid #1F2240;border-left:2px solid #C9A84C;
                    font-family:"DM Sans",sans-serif;font-size:13px;
                    line-height:1.7;color:#ECEDF5;'>
            {brief_text}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:10px;"
            "color:#5A5C78;padding:8px 0;'>"
            "Click Refresh Signals to generate world brief.</div>",
            unsafe_allow_html=True,
        )

    if "last_params" in st.session_state:
        st.markdown("<p class='margins-section'>Applied Parameter Adjustments</p>", unsafe_allow_html=True)
        st.json(st.session_state.last_params.to_dict())


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div class='margins-brand'>MARGINS</div>"
        "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;"
        "text-transform:uppercase;letter-spacing:0.16em;color:#5A5C78;"
        "margin-top:4px;margin-bottom:12px;'>Monte Carlo Gold Predictor</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    REFRESH_MS = 5 * 60 * 1000

    if HAS_AUTOREFRESH:
        refresh_count = st_autorefresh(interval=REFRESH_MS, key="gold_autorefresh")

        prev_count = st.session_state.get("_last_refresh_count", -1)
        if refresh_count != prev_count and refresh_count > 0:
            st.session_state["_last_refresh_count"] = refresh_count
            get_current_price.clear()

            if "macro_bridge" in st.session_state:
                try:
                    bridge_ref = st.session_state.macro_bridge
                    bridge_ref._cache.clear()
                    bridge_ref._brief_cache.clear()
                    new_signals = bridge_ref.get_signals_sync()
                    st.session_state.macro_signals = new_signals
                    if new_signals.brief_text:
                        st.session_state.macro_brief = new_signals.brief_text
                except Exception:
                    pass

            if "paths" in st.session_state:
                try:
                    new_price, _ = get_current_price()
                    new_paths = run_simulation(
                        st.session_state.get("_last_model", list(MODELS_AVAILABLE.keys())[0]),
                        new_price,
                        st.session_state["n_steps"] + 1,
                        st.session_state.get("_last_n_paths", 1000),
                        st.session_state.get("_last_mu"),
                        st.session_state.get("_last_sigma"),
                        fetch_gold_data(st.session_state.get("_last_period", "2y")),
                        seed=st.session_state.get("_last_seed"),
                        macro_signals=(
                            st.session_state.get("macro_signals")
                            if st.session_state.get("macro_enabled", True)
                            else None
                        ),
                    )
                    st.session_state["paths"] = new_paths
                    st.session_state["S0"]    = new_price
                except Exception:
                    pass

    current_price, price_change = get_current_price()
    now_str = dt_datetime.now().strftime("%H:%M:%S")

    delta_color = "#22C55E" if price_change >= 0 else "#EF4444"
    st.markdown(
        f"<div style='margin-bottom:8px;'>"
        f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;"
        f"text-transform:uppercase;letter-spacing:0.14em;color:#5A5C78;"
        f"margin-bottom:4px;'>Current Gold Price</div>"
        f"<span style='font-family:\"IBM Plex Mono\",monospace;font-size:20px;"
        f"font-weight:500;color:#E8C97A;'>${current_price:,.2f}</span>&nbsp;"
        f"<span style='color:{delta_color};font-size:11px;font-family:\"IBM Plex Mono\",monospace;'>"
        f"{'▲' if price_change >= 0 else '▼'} {abs(price_change):.2f}%</span>"
        f"</div>"
        f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:10px;"
        f"color:#5A5C78;margin-bottom:4px;'>"
        f"${current_price/31.1035:,.2f}/g &nbsp;·&nbsp; ${current_price*32.1507:,.0f}/kg"
        f"</div>",
        unsafe_allow_html=True,
    )

    if HAS_AUTOREFRESH:
        st.markdown(
            f"<div style='margin-top:6px;padding:6px 10px;background:var(--bg-base);"
            f"border:1px solid var(--border);'>"
            f"<span style='color:#22C55E;font-family:\"IBM Plex Mono\",monospace;"
            f"font-size:9px;text-transform:uppercase;letter-spacing:0.12em;'>"
            f"● LIVE · AUTO-REFRESH 5 MIN</span><br>"
            f"<span style='color:#5A5C78;font-family:\"IBM Plex Mono\",monospace;"
            f"font-size:9px;'>Last: {now_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='margin-top:4px;font-family:\"IBM Plex Mono\",monospace;"
            f"font-size:9px;color:#5A5C78;'>Last: {now_str}</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Refresh all", key="manual_refresh_btn",
                     help="Refresh price, macro signals and trade signal"):
            get_current_price.clear()
            if "macro_bridge" in st.session_state:
                try:
                    st.session_state.macro_bridge._cache.clear()
                    st.session_state.macro_bridge._brief_cache.clear()
                    new_sig = st.session_state.macro_bridge.get_signals_sync()
                    st.session_state.macro_signals = new_sig
                    if new_sig.brief_text:
                        st.session_state.macro_brief = new_sig.brief_text
                except Exception:
                    pass
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p class='margins-section'>Parameters</p>",
        unsafe_allow_html=True,
    )

    model_name = st.selectbox(
        "Stochastic Model",
        list(MODELS_AVAILABLE.keys()),
        index=min(1, len(MODELS_AVAILABLE) - 1),
    )

    horizon_label = st.selectbox(
        "Prediction Horizon",
        ["7 Days", "30 Days", "90 Days", "1 Year"],
        index=1,
    )
    horizon_map = {"7 Days": 7, "30 Days": 30, "90 Days": 90, "1 Year": 252}
    n_steps     = horizon_map[horizon_label]

    n_paths = st.select_slider(
        "Number of Simulations",
        options=[500, 1_000, 2_000, 5_000, 10_000],
        value=2_000,
    )

    data_period = st.selectbox("Training Data", ["6mo", "1y", "2y", "5y"], index=2)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        "<p class='margins-section'>Manual Overrides</p>",
        unsafe_allow_html=True,
    )
    use_manual = st.checkbox("Override calibrated params")
    if use_manual:
        manual_mu    = st.slider("Drift (μ) annualised",     -0.3, 0.5,  0.05, 0.01)
        manual_sigma = st.slider("Volatility (σ) annualised", 0.05, 0.6, 0.15, 0.01)
    else:
        manual_mu, manual_sigma = None, None

    seed_raw = st.number_input("Random Seed (0 = random)", 0, 9999, 42)
    seed = int(seed_raw) if seed_raw > 0 else None   # FIX 9: clean conversion

    run_btn = st.button("▶  Run Simulation", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<h1 style='font-family:\"Playfair Display\",serif;font-size:2rem;"
    "font-weight:500;color:#ECEDF5;margin-bottom:4px;'>"
    "Monte Carlo Gold Price Predictor</h1>"
    "<p style='font-family:\"DM Sans\",sans-serif;font-size:13px;"
    "color:#9395B0;margin-top:0;margin-bottom:20px;'>"
    "Stochastic simulation engine for gold price forecasting &amp; risk analysis</p>",
    unsafe_allow_html=True,
)


# Load data
prices = fetch_gold_data(data_period)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Simulation",
    "Risk Analysis",
    "Market Data",
    "Model Guide",
    "Macro Intelligence",
    "Trade Signal",
    "Avg Error",
    "Portfolio",
    "Investment Committee",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if run_btn or "paths" not in st.session_state:
        with st.spinner("Running simulations…"):
            try:
                paths = run_simulation(
                    model_name, current_price, n_steps + 1,
                    n_paths, manual_mu, manual_sigma, prices, seed,
                    macro_signals=(
                        st.session_state.get("macro_signals")
                        if st.session_state.get("macro_enabled", True) else None
                    ),
                )
                st.session_state["paths"]       = paths
                st.session_state["model_name"]  = model_name
                st.session_state["n_steps"]     = n_steps
                st.session_state["S0"]          = current_price
                st.session_state["_last_model"]  = model_name
                st.session_state["_last_n_paths"] = n_paths
                st.session_state["_last_period"]  = data_period
                st.session_state["_last_seed"]    = seed
                st.session_state["_last_mu"]      = manual_mu
                st.session_state["_last_sigma"]   = manual_sigma
            except Exception as e:
                st.error(f"Simulation error: {e}")
                st.stop()
    else:
        paths         = st.session_state["paths"]
        model_name    = st.session_state["model_name"]
        n_steps       = st.session_state["n_steps"]
        current_price = st.session_state["S0"]

    final_prices = paths[:, -1]
    S0           = current_price

    mean_price   = np.mean(final_prices)
    median_price = np.median(final_prices)
    p5_price     = np.percentile(final_prices, 5)
    p95_price    = np.percentile(final_prices, 95)
    prob_up      = np.mean(final_prices > S0) * 100
    exp_return   = (mean_price / S0 - 1) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected Price",  f"${mean_price:,.0f}",   f"{exp_return:+.1f}%")
    c2.metric("Median Price",    f"${median_price:,.0f}")
    c3.metric("5th Percentile",  f"${p5_price:,.0f}")
    c4.metric("95th Percentile", f"${p95_price:,.0f}")
    c5.metric("Prob. of Gain",   f"{prob_up:.1f}%")

    st.markdown("")

    last_date    = prices.index[-1]
    dates_future = pd.date_range(start=last_date, periods=n_steps + 1, freq="B")
    recent_prices = prices["price"].iloc[-120:]

    st.plotly_chart(fan_chart(paths, dates_future, S0, recent_prices), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(distribution_chart(final_prices, S0), use_container_width=True)
    with col_b:
        log_returns = np.log(final_prices / S0)
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=log_returns * 100, nbinsx=60,
            marker_color=BLUE, opacity=0.75, name="Log Return %",
        ))
        fig_ret.add_vline(x=0, line=dict(color="#AAAACC", dash="dot"))
        fig_ret.update_layout(
            title="Distribution of Log Returns (%)",
            xaxis_title="Log Return (%)", yaxis_title="Count",
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    st.markdown("<p class='margins-section'>Scenario Summary</p>", unsafe_allow_html=True)
    scenarios = pd.DataFrame({
        "Scenario":    ["Bear (5th pct)", "Mild Bear (25th)", "Base (Median)", "Mild Bull (75th)", "Bull (95th)"],
        "Price":       [f"${np.percentile(final_prices,  5):,.0f}",
                        f"${np.percentile(final_prices, 25):,.0f}",
                        f"${median_price:,.0f}",
                        f"${np.percentile(final_prices, 75):,.0f}",
                        f"${p95_price:,.0f}"],
        "Change":      [f"{(np.percentile(final_prices,  5)/S0-1)*100:+.1f}%",
                        f"{(np.percentile(final_prices, 25)/S0-1)*100:+.1f}%",
                        f"{(median_price/S0-1)*100:+.1f}%",
                        f"{(np.percentile(final_prices, 75)/S0-1)*100:+.1f}%",
                        f"{(p95_price/S0-1)*100:+.1f}%"],
        "Probability": ["5%", "25%", "50%", "25%", "5%"],
    })
    st.dataframe(scenarios, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if "paths" not in st.session_state:
        st.info("Run a simulation first (Tab 1).")
    else:
        paths        = st.session_state["paths"]
        S0           = st.session_state["S0"]
        final_prices = paths[:, -1]

        var_90  = np.percentile(final_prices, 10)
        var_95  = np.percentile(final_prices, 5)
        var_99  = np.percentile(final_prices, 1)
        cvar_95 = np.mean(final_prices[final_prices <= var_95])
        cvar_99 = np.mean(final_prices[final_prices <= var_99])

        max_dd_per_path = []
        for path in paths:
            peak_p   = np.maximum.accumulate(path)
            drawdown = (path - peak_p) / peak_p
            max_dd_per_path.append(float(np.min(drawdown)))
        avg_max_dd  = np.mean(max_dd_per_path) * 100
        vol_realized = np.std(np.log(final_prices / S0)) * 100

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("VaR 95%",          f"${var_95:,.0f}",  f"{(var_95/S0-1)*100:+.1f}%")
        r2.metric("CVaR 95%",         f"${cvar_95:,.0f}", f"{(cvar_95/S0-1)*100:+.1f}%")
        r3.metric("Avg Max Drawdown", f"{avg_max_dd:.1f}%")
        r4.metric("Realised Vol",     f"{vol_realized:.1f}%")

        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(risk_gauge(var_95, cvar_95, S0), use_container_width=True)

        with col2:
            confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
            var_values = [np.percentile(final_prices, (1 - cl) * 100) for cl in confidence_levels]
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(
                x=[f"{int(cl*100)}%" for cl in confidence_levels],
                y=var_values,
                marker_color=[f"rgba(224,85,85,{0.4 + i*0.12})" for i in range(5)],
                text=[f"${v:,.0f}" for v in var_values],
                textposition="outside",
            ))
            fig_var.add_hline(y=S0, line=dict(color=GOLD, dash="dash"),
                              annotation_text="Current Price")
            fig_var.update_layout(
                title="Value at Risk by Confidence Level",
                yaxis_title="Price (USD)",
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_var, use_container_width=True)

        # FIX 6: VRP block now correctly placed inside `with tab2:` / `else:` block
        if st.session_state.get("model_name") in ("Heston", "Heston + Jumps (Bates)"):
            st.markdown("<p class='margins-section'>Variance Risk Premium Signal</p>", unsafe_allow_html=True)
            try:
                from src.models.heston import HestonModel as _HM
                h_model = _HM()
                h_model.calibrate(prices["price"].values)
                implied_vol = np.sqrt(h_model.params.v0) * 1.15
                vrp = h_model.variance_risk_premium(implied_vol=implied_vol)

                v1, v2, v3 = st.columns(3)
                v1.metric("Implied Vol",  f"{np.sqrt(vrp['implied_var']):.1%}")
                v2.metric("Realised Vol", f"{np.sqrt(vrp['realised_var']):.1%}")
                sig_icon = {"short_vol": "🔴", "long_vol": "🟢", "neutral": "🟡"}
                v3.metric("VRP Signal",
                          f"{sig_icon.get(vrp['trading_signal'], '')} {vrp['trading_signal'].upper()}")
                st.caption(
                    f"VRP z-score: {vrp['vrp_z_score']:.2f} "
                    f"— positive = options overpriced vs history"
                )
            except Exception:
                pass

        # ── Target Analysis ───────────────────────────────────────────────────
        st.markdown("<p class='margins-section'>Probability of Reaching a Target Price</p>", unsafe_allow_html=True)
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            target = st.number_input("Target Price ($)", value=int(S0 * 1.10), step=50)
        with col_t2:
            prob_reach   = np.mean(final_prices >= target) * 100
            above_target = final_prices[final_prices >= target]
            avg_if_reach = float(np.mean(above_target)) if len(above_target) > 0 else 0
            st.markdown(f"""
            <div class='gold-box'>
                <b>Probability of reaching ${target:,}</b><br>
                <span style='font-size:2rem;color:{GOLD2};font-weight:700'>{prob_reach:.1f}%</span><br>
                <span style='color:#8888AA;font-size:0.85rem;'>
                    Avg price when above target: ${avg_if_reach:,.0f}
                </span>
            </div>
            """, unsafe_allow_html=True)

        # ── Drawdown Distribution ─────────────────────────────────────────────
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Histogram(
            x=[d * 100 for d in max_dd_per_path], nbinsx=60,
            marker_color=RED, opacity=0.75, name="Max Drawdown %",
        ))
        fig_dd.update_layout(
            title="Distribution of Maximum Drawdowns per Path",
            xaxis_title="Max Drawdown (%)", yaxis_title="Count",
            **DARK_LAYOUT,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Risk Summary Table ────────────────────────────────────────────────
        st.markdown("<p class='margins-section'>Full Risk Summary</p>", unsafe_allow_html=True)
        risk_table = pd.DataFrame({
            "Metric": [
                "VaR 80%", "VaR 90%", "VaR 95%", "VaR 99%",
                "CVaR 95%", "CVaR 99%",
                "Prob. of Loss > 5%", "Prob. of Loss > 10%", "Prob. of Loss > 20%",
                "Prob. of Gain > 5%", "Prob. of Gain > 10%", "Prob. of Gain > 20%",
            ],
            "Value": [
                f"${np.percentile(final_prices, 20):,.0f}",
                f"${var_90:,.0f}",
                f"${var_95:,.0f}",
                f"${var_99:,.0f}",
                f"${cvar_95:,.0f}",
                f"${cvar_99:,.0f}",
                f"{np.mean(final_prices < S0*0.95)*100:.1f}%",
                f"{np.mean(final_prices < S0*0.90)*100:.1f}%",
                f"{np.mean(final_prices < S0*0.80)*100:.1f}%",
                f"{np.mean(final_prices > S0*1.05)*100:.1f}%",
                f"{np.mean(final_prices > S0*1.10)*100:.1f}%",
                f"{np.mean(final_prices > S0*1.20)*100:.1f}%",
            ],
        })
        st.dataframe(risk_table, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<p class='margins-section'>Historical Gold Price</p>", unsafe_allow_html=True)
    st.plotly_chart(historical_chart(prices), use_container_width=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("<p class='margins-section'>Return Statistics</p>", unsafe_allow_html=True)
        log_ret = np.log(prices["price"] / prices["price"].shift(1)).dropna()
        stats_df = pd.DataFrame({
            "Metric": [
                "Annualised Return", "Annualised Volatility",
                "Sharpe Ratio (rf=0)", "Skewness", "Kurtosis",
                "Max Daily Gain", "Max Daily Loss",
            ],
            "Value": [
                f"{log_ret.mean()*252*100:.2f}%",
                f"{log_ret.std()*np.sqrt(252)*100:.2f}%",
                f"{(log_ret.mean()*252)/(log_ret.std()*np.sqrt(252)):.3f}",
                f"{log_ret.skew():.3f}",
                f"{log_ret.kurt():.3f}",
                f"{log_ret.max()*100:.2f}%",
                f"{log_ret.min()*100:.2f}%",
            ],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col_m2:
        st.markdown("<p class='margins-section'>Rolling Volatility</p>", unsafe_allow_html=True)
        roll_vol = log_ret.rolling(30).std() * np.sqrt(252) * 100
        fig_vol  = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol.values,
            fill="tozeroy", fillcolor="rgba(85,153,238,0.10)",
            line=dict(color=BLUE, width=1.5), name="30d Vol",
        ))
        fig_vol.update_layout(
            yaxis_title="Volatility (%)",
            **{k: v for k, v in DARK_LAYOUT.items() if k != "margin"},
            margin=dict(l=40, r=20, t=10, b=40),
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("<p class='margins-section'>Historical Return Distribution</p>", unsafe_allow_html=True)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=log_ret.values * 100, nbinsx=80,
        marker_color=GOLD, opacity=0.75,
    ))
    fig_hist.update_layout(
        xaxis_title="Daily Log Return (%)", yaxis_title="Count",
        **DARK_LAYOUT,
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<p class='margins-section'>Model Reference Guide</p>", unsafe_allow_html=True)

    # FIX 7: Heston entry was missing name/formula/desc/best_for — added them.
    model_info = {
        "GBM": {
            "name":     "Geometric Brownian Motion",
            "formula":  "dS = μ·S·dt + σ·S·dW",
            "desc":     "The foundational stochastic model. Assumes log-normally distributed returns with constant drift and volatility. Best used as a baseline benchmark.",
            "pros":     ["Simple & fast", "Analytically tractable", "Black-Scholes validated"],
            "cons":     ["No mean reversion", "Constant volatility", "No jumps"],
            "best_for": "Baseline comparison",
        },
        "Mean Reversion (OU)": {
            "name":     "Ornstein-Uhlenbeck",
            "formula":  "dS = κ·(θ − S)·dt + σ·dW",
            "desc":     "Gold's most appropriate single-factor model. Captures the documented tendency of gold to revert to long-term equilibrium.",
            "pros":     ["Captures mean reversion", "Reflects gold's behaviour", "Analytically solvable"],
            "cons":     ["Can produce negative prices if σ is large", "Constant reversion speed"],
            "best_for": "Long-term forecasting",
        },
        "Jump Diffusion": {
            "name":     "Merton Jump Diffusion",
            "formula":  "dS = μ·S·dt + σ·S·dW + J·S·dN",
            "desc":     "Extends GBM with a Poisson jump process to model sudden price shocks from geopolitical crises, central bank interventions, or market panics.",
            "pros":     ["Captures tail risk", "Models crisis events", "Realistic skewness"],
            "cons":     ["More parameters to calibrate", "Jump timing is random"],
            "best_for": "Risk analysis & stress testing",
        },
        "Heston": {
            "name":     "Heston Stochastic Volatility",
            "formula":  "dS = μ·S·dt + √v·S·dW₁  |  dv = κ(θ−v)dt + ξ√v·dW₂",
            "desc":     "Models volatility as a mean-reverting stochastic process. Captures volatility clustering, the leverage effect, and the implied-vol smile — all missing from GBM.",
            "pros":     ["MLE-calibrated (v2.0)", "QE discretisation — no negative variance",
                         "Antithetic variates — ~40% accuracy boost", "Captures leverage effect"],
            "cons":     ["Feller condition must hold (2κθ > ξ²)", "More parameters than GBM"],
            "best_for": "Options pricing & vol-surface modelling",
        },
        "Heston + Jumps (Bates)": {
            "name":     "Bates Model (Heston + Jumps)",
            "formula":  "dS = μ·S·dt + √v·S·dW₁ + J·S·dN  |  dv = κ(θ−v)dt + ξ√v·dW₂",
            "desc":     "Heston with a compound-Poisson jump overlay. Captures crisis spikes (COVID, 2008, geopolitical) that pure diffusion cannot model.",
            "pros":     ["Best for crisis periods", "Captures fat tails", "Realistic skewness"],
            "cons":     ["More parameters", "Slower calibration"],
            "best_for": "Risk analysis during high-stress periods",
        },
        "Regime Switching": {
            "name":     "Markov Regime Switching",
            "formula":  "dS = μ(sₜ)·S·dt + σ(sₜ)·S·dW",
            "desc":     "Uses a Markov chain to alternate between distinct market states (calm / crisis). Each regime has its own drift and volatility parameters.",
            "pros":     ["Captures market structural changes", "Excellent for scenario analysis", "Interpretable regimes"],
            "cons":     ["Regime parameters need careful calibration", "Transition matrix estimation requires long history"],
            "best_for": "Scenario planning & long-term forecasting",
        },
    }

    for key, info in model_info.items():
        if key not in MODELS_AVAILABLE:
            continue
        with st.expander(f"**{info['name']}**  —  {key}", expanded=(key == "Mean Reversion (OU)")):
            col_l, col_r = st.columns([3, 2])
            with col_l:
                st.markdown(f"**Formula:** `{info['formula']}`")
                st.markdown(info["desc"])
                st.markdown(f"**Best for:** {info['best_for']}")
            with col_r:
                st.markdown("**Pros**")
                for p in info["pros"]:
                    st.markdown(f"✅ {p}")
                st.markdown("**Cons**")
                for c in info["cons"]:
                    st.markdown(f"⚠️ {c}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MACRO INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    render_macro_tab()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — TRADE SIGNAL
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("<p class='margins-section'>Trade Signal</p>", unsafe_allow_html=True)

    if "paths" not in st.session_state:
        st.info("Run a simulation first in the Simulation tab to generate a trade signal.")
    else:
        paths_ts   = st.session_state["paths"]
        S0_ts      = st.session_state["S0"]
        n_steps_ts = st.session_state["n_steps"]
        macro_sig  = st.session_state.get("macro_signals") if st.session_state.get("macro_enabled", True) else None

        sig = compute_trade_signal(paths_ts, S0_ts, n_steps_ts, macro_sig)

        sig_colors = {
            "BUY":  ("#0B1A0E", "#22C55E"),
            "SELL": ("#1A0B0B", "#EF4444"),
            "HOLD": ("#131208", "#E8C97A"),
        }
        sig_icons  = {"BUY": "▲", "SELL": "▼", "HOLD": "◆"}
        bg, fg = sig_colors.get(sig["signal"], ("#10121E", "#ECEDF5"))
        icon   = sig_icons.get(sig["signal"], "")

        conf_color = {"HIGH": "#22C55E", "MEDIUM": "#E8C97A", "LOW": "#EF4444"}
        cc = conf_color.get(sig["confidence"], "#9395B0")

        st.markdown(f"""
        <div style='background:{bg};border:1px solid {fg};border-left:3px solid {fg};
                    border-radius:0;padding:1.2rem 1.5rem;margin-bottom:1rem;'>
            <div style='display:flex;align-items:center;gap:1.5rem;'>
                <div style='font-size:2.8rem;color:{fg};font-weight:500;
                            font-family:"Playfair Display",serif;line-height:1;'>
                    {icon} {sig["signal"]}
                </div>
                <div>
                    <div style='color:{cc};font-family:"IBM Plex Mono",monospace;
                                font-size:9px;font-weight:400;letter-spacing:0.16em;
                                text-transform:uppercase;'>
                        {sig["confidence"]} CONFIDENCE
                    </div>
                    <div style='color:#5A5C78;font-family:"IBM Plex Mono",monospace;
                                font-size:10px;margin-top:6px;letter-spacing:0.08em;'>
                        {sig["horizon_days"]}-DAY HORIZON &nbsp;·&nbsp;
                        SCORE: {sig["score"]:+d} &nbsp;·&nbsp;
                        MACRO: {sig["macro_risk"].upper()}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<p class='margins-section'>Entry Levels</p>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry (Current)", f"${sig['entry']:,.2f}")
        c2.metric("Stop-Loss",       f"${sig['stop_loss']:,.2f}",
                  delta=f"{(sig['stop_loss']-sig['entry'])/sig['entry']*100:.1f}%",
                  delta_color="inverse")
        c3.metric("Take-Profit",     f"${sig['take_profit']:,.2f}",
                  delta=f"+{(sig['take_profit']-sig['entry'])/sig['entry']*100:.1f}%")
        c4.metric("Risk / Reward",   f"{sig['risk_reward']:.1f}x",
                  delta="good" if sig["risk_reward"] >= 2 else "low",
                  delta_color="normal" if sig["risk_reward"] >= 2 else "inverse")

        st.markdown("<p class='margins-section'>Price Reference</p>", unsafe_allow_html=True)
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Per Troy Oz",  f"${sig['entry']:,.2f}")
        pc2.metric("Per Gram",     f"${sig['entry']/31.1035:,.2f}")
        pc3.metric("Per Kilogram", f"${sig['entry']*32.1507:,.0f}")

        st.markdown("<p class='margins-section'>Probability Breakdown</p>", unsafe_allow_html=True)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("P(Any Gain)",  f"{sig['prob_gain']:.1f}%")
        p2.metric("P(Gain > 5%)", f"{sig['prob_gain5']:.1f}%")
        p3.metric("P(Loss > 5%)", f"{sig['prob_loss5']:.1f}%")
        p4.metric("Avg Drawdown", f"{sig['avg_max_dd']:.1f}%")

        st.markdown("<p class='margins-section'>Simulated Price Range at Horizon</p>", unsafe_allow_html=True)
        fig_range = go.Figure()
        fig_range.add_trace(go.Scatter(
            x=[sig["p5"], sig["p95"]], y=[1, 1], mode="lines",
            line=dict(color="rgba(200,200,200,0.15)", width=40),
            name="90% range", showlegend=True,
            hovertemplate="90%% range: $%{x:,.0f}<extra></extra>",
        ))
        fig_range.add_trace(go.Scatter(
            x=[sig["p25"], sig["p75"]], y=[1, 1], mode="lines",
            line=dict(color="rgba(201,168,76,0.4)", width=30),
            name="IQR (25-75%)", showlegend=True,
            hovertemplate="IQR: $%{x:,.0f}<extra></extra>",
        ))
        fig_range.add_trace(go.Scatter(
            x=[sig["mean_final"]], y=[1], mode="markers+text",
            marker=dict(color=GOLD, size=14, symbol="diamond"),
            text=[f"  Mean<br>  ${sig['mean_final']:,.0f}"],
            textposition="middle right",
            textfont=dict(color=GOLD, size=11),
            name="Mean", showlegend=False,
        ))
        fig_range.add_vline(x=S0_ts,
            line=dict(color="#5599EE", dash="dash", width=1.5),
            annotation_text=f"Entry ${S0_ts:,.0f}",
            annotation_font=dict(color="#5599EE", size=11))
        fig_range.add_vline(x=sig["stop_loss"],
            line=dict(color=RED, dash="dot", width=1.5),
            annotation_text=f"SL ${sig['stop_loss']:,.0f}",
            annotation_font=dict(color=RED, size=11))
        fig_range.add_vline(x=sig["take_profit"],
            line=dict(color=GREEN, dash="dot", width=1.5),
            annotation_text=f"TP ${sig['take_profit']:,.0f}",
            annotation_font=dict(color=GREEN, size=11))

        range_layout = {k: v for k, v in DARK_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")}
        range_layout.update(dict(
            height=180,
            yaxis=dict(visible=False, range=[0, 2]),
            xaxis=dict(title="Gold Price (USD / troy oz)", tickformat="$,.0f",
                       gridcolor="#1F2240", showgrid=True, zeroline=False,
                       tickfont=dict(color="#9395B0", family="IBM Plex Mono", size=10)),
            legend=dict(orientation="h", y=1.3, x=0,
                        font=dict(family="IBM Plex Mono", size=10, color="#9395B0")),
            margin=dict(l=40, r=40, t=50, b=40),
        ))
        fig_range.update_layout(**range_layout)
        st.plotly_chart(fig_range, use_container_width=True)

        col_r, col_w = st.columns(2)
        with col_r:
            st.markdown("<p class='margins-section'>Signal Reasons</p>", unsafe_allow_html=True)
            for r in sig["reasons"]:
                st.markdown(
                    f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:11px;"
                    f"color:#9395B0;padding:3px 0;border-bottom:1px solid #1F2240;'>"
                    f"<span style='color:#22C55E;'>+</span> {r}</div>",
                    unsafe_allow_html=True,
                )
        with col_w:
            st.markdown("<p class='margins-section'>Risk Warnings</p>", unsafe_allow_html=True)
            if sig["warnings"]:
                for w in sig["warnings"]:
                    st.markdown(
                        f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:11px;"
                        f"color:#9395B0;padding:3px 0;border-bottom:1px solid #1F2240;'>"
                        f"<span style='color:#EF4444;'>!</span> {w}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<div style='font-family:\"IBM Plex Mono\",monospace;font-size:11px;"
                    "color:#5A5C78;'>No significant risk warnings.</div>",
                    unsafe_allow_html=True,
                )

        st.markdown(
            "<div style='margin-top:1.5rem;padding:12px 16px;background:#10121E;"
            "border:1px solid #1F2240;border-left:2px solid #5A5C78;'>"
            "<span style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;"
            "text-transform:uppercase;letter-spacing:0.12em;color:#5A5C78;'>"
            "DISCLAIMER</span>"
            "<p style='font-family:\"DM Sans\",sans-serif;font-size:11px;color:#5A5C78;"
            "margin:6px 0 0 0;line-height:1.6;'>"
            "This signal is generated from a Monte Carlo stochastic simulation and is for "
            "<strong style=\"color:#9395B0;\">educational and research purposes only</strong>. "
            "It does not constitute financial advice. Past simulation performance does not "
            "guarantee future results. Always consult a qualified financial advisor before "
            "trading. Gold markets carry significant risk of loss.</p></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — AVERAGE ERROR (BACKTEST)
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("<p class='margins-section'>Model Average Error — Backtest</p>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-family:\"DM Sans\",sans-serif;font-size:12px;color:#9395B0;'>"
        "Walk-forward backtest: trains on older data, predicts on the most recent period, "
        "compares to actual prices.</p>",
        unsafe_allow_html=True,
    )

    if "paths" not in st.session_state:
        st.info("Run a simulation first (Tab 1) to enable error analysis.")
    else:
        # ── Controls ──────────────────────────────────────────────────────────
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            test_window = st.selectbox(
                "Test window (days to predict)",
                [21, 63, 126, 252],
                index=1,
                help="Shorter = more recent, harder test. Longer = more data points but covers trending periods."
            )
        with col_ctrl2:
            n_paths_bt = st.select_slider(
                "Paths per model",
                options=[200, 500, 1000, 2000],
                value=1000,
                help="More paths = more accurate median estimate but slower."
            )

        run_bt = st.button("▶ Run Backtest", use_container_width=True)

        if run_bt or "bt_results" not in st.session_state:
            with st.spinner("Running walk-forward backtest for all models…"):
                test_prices = prices["price"].iloc[-test_window:].values
                train       = prices["price"].iloc[:-test_window].values
                S0_bt       = float(train[-1])
                results     = []

                for m_name, ModelCls in MODELS_AVAILABLE.items():
                    try:
                        model = ModelCls()

                        # ── Calibrate ─────────────────────────────────────────
                        if hasattr(model, "calibrate"):
                            # Heston v2 MLE needs at least 100 prices
                            if m_name in ("Heston", "Heston + Jumps (Bates)"):
                                if hasattr(model, "_model"):
                                    # HestonBates wrapper
                                    model._model.calibrate(train)
                                    model.params = model._model.params
                                else:
                                    model.calibrate(train)
                            elif m_name in ("Mean Reversion (OU)", "Regime Switching"):
                                model.calibrate(train)
                            else:
                                log_r = np.log(train[1:] / train[:-1])
                                model.calibrate(log_r)

                        # ── Simulate ──────────────────────────────────────────
                        if m_name in ("Heston", "Heston + Jumps (Bates)"):
                            sim = model.simulate(
                                S0=S0_bt, n_steps=test_window,
                                n_paths=n_paths_bt, random_seed=42, antithetic=True
                            )
                        else:
                            sim = model.simulate(
                                S0=S0_bt, n_steps=test_window,
                                n_paths=n_paths_bt, random_seed=42
                            )

                        # ── Use MEAN not median — less biased for trending markets ──
                        predicted = np.mean(sim, axis=0)
                        actual    = test_prices
                        n         = min(len(predicted), len(actual))
                        predicted, actual = predicted[:n], actual[:n]

                        mae     = np.mean(np.abs(predicted - actual))
                        rmse    = np.sqrt(np.mean((predicted - actual) ** 2))
                        mape    = np.mean(np.abs((predicted - actual) / actual)) * 100
                        dir_acc = np.mean(
                            np.sign(np.diff(actual)) == np.sign(np.diff(predicted))
                        ) * 100

                        # Feller ratio for Heston models
                        feller = "—"
                        if m_name in ("Heston", "Heston + Jumps (Bates)"):
                            try:
                                p = model.params if hasattr(model, "params") else model._model.params
                                feller = f"{p.feller_ratio():.2f}"
                            except Exception:
                                pass

                        results.append({
                            "Model":             m_name,
                            "MAE ($)":           round(mae,     2),
                            "RMSE ($)":          round(rmse,    2),
                            "MAPE (%)":          round(mape,    2),
                            "Dir. Accuracy (%)": round(dir_acc, 1),
                            "Feller":            feller,
                        })
                    except Exception as e:
                        results.append({
                            "Model":             m_name,
                            "MAE ($)":           None,
                            "RMSE ($)":          None,
                            "MAPE (%)":          None,
                            "Dir. Accuracy (%)": None,
                            "Feller":            f"ERROR: {e}",
                        })

                st.session_state["bt_results"]     = results
                st.session_state["bt_test_window"] = test_window

        results = st.session_state.get("bt_results", [])

        if results:
            df_err = pd.DataFrame(results)
            df_err = df_err.dropna(subset=["MAE ($)"]).sort_values("MAE ($)")
            best   = df_err.iloc[0]["Model"]

            st.success(f"🏆 Best model by MAE: **{best}**  "
                       f"(test window: {st.session_state.get('bt_test_window', '?')} days)")
            st.dataframe(df_err, use_container_width=True, hide_index=True)

            # MAE vs RMSE chart
            fig_err = go.Figure()
            for metric, color in [("MAE ($)", GOLD), ("RMSE ($)", BLUE)]:
                fig_err.add_trace(go.Bar(
                    name=metric,
                    x=df_err["Model"],
                    y=df_err[metric],
                    marker_color=color,
                    opacity=0.8,
                ))
            fig_err.update_layout(
                title="MAE vs RMSE by Model",
                barmode="group",
                yaxis_title="Error ($)",
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_err, use_container_width=True)

            # MAPE chart
            fig_mape = go.Figure()
            fig_mape.add_trace(go.Bar(
                x=df_err["Model"],
                y=df_err["MAPE (%)"],
                marker_color=RED,
                opacity=0.8,
                text=[f"{v:.2f}%" for v in df_err["MAPE (%)"]],
                textposition="outside",
            ))
            fig_mape.update_layout(
                title="Mean Absolute Percentage Error (MAPE) by Model",
                yaxis_title="MAPE (%)",
                **DARK_LAYOUT,
            )
            st.plotly_chart(fig_mape, use_container_width=True)
        else:
            st.error("Could not compute errors. Make sure models loaded correctly.")
#====================================================================================
#tab 8
#====================================================================================
with tab8:
    render_portfolio_tab()
    st.markdown(
        "<p class='margins-section' style='margin-top:2rem;'>How to Run</p>",
        unsafe_allow_html=True,
    )
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — INVESTMENT COMMITTEE SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
with tab9:
    st.markdown("<p class='margins-section'>MARGINS V1.3.0 — Investment Committee Scorecard</p>", unsafe_allow_html=True)

    # ── Scorecard CSS (scoped, no overflow, fully responsive) ─────────────────
    st.markdown("""
    <style>
    /* Prevent Streamlit from clipping the scorecard content */
    .block-container {
        max-width: 100% !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 0.5rem !important;
    }
    [data-testid="column"] {
        min-width: 170px !important;
        flex: 1 1 170px !important;
    }

    /* Summary metric cards */
    .sc-card {
        background: #10121E;
        border: 1px solid #1F2240;
        border-radius: 6px;
        padding: 16px 14px;
        text-align: center;
        box-sizing: border-box;
        width: 100%;
    }
    .sc-card-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #5A5C78;
        margin-bottom: 8px;
    }
    .sc-card-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 38px;
        font-weight: 500;
        line-height: 1;
    }
    .sc-card-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        color: #9395B0;
        margin-top: 6px;
        line-height: 1.4;
    }

    /* Model fit cards */
    .sc-model {
        background: #10121E;
        border: 1px solid #1F2240;
        border-radius: 6px;
        padding: 12px 10px;
        box-sizing: border-box;
        width: 100%;
    }
    .sc-model-name {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #9395B0;
        margin-bottom: 4px;
    }
    .sc-model-desc {
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
        color: #5A5C78;
        margin-bottom: 8px;
    }

    /* Dimension bar rows — the key fix for right-side cutoff */
    .sc-dim-row {
        display: flex;
        align-items: center;
        width: 100%;
        margin-bottom: 10px;
        box-sizing: border-box;
    }
    .sc-dim-name {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #ECEDF5;
        min-width: 160px;
        max-width: 160px;
        padding-right: 12px;
        flex-shrink: 0;
    }
    .sc-dim-track {
        flex: 1;
        min-width: 0;
        background: #1F2240;
        border-radius: 2px;
        height: 6px;
        position: relative;
        margin-right: 10px;
    }
    .sc-dim-fill {
        height: 6px;
        border-radius: 2px;
        position: absolute;
        left: 0; top: 0;
    }
    .sc-dim-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #E8C97A;
        min-width: 26px;
        text-align: right;
        flex-shrink: 0;
    }

    /* Badge pills */
    .sc-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 3px 8px;
        border-radius: 2px;
    }

    /* Overall verdict box */
    .sc-verdict {
        background: rgba(201,168,76,0.06);
        border: 1px solid rgba(201,168,76,0.20);
        border-radius: 6px;
        padding: 24px 28px;
        display: flex;
        align-items: center;
        gap: 28px;
        box-sizing: border-box;
        width: 100%;
    }
    .sc-verdict-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 52px;
        font-weight: 500;
        color: #C9A84C;
        line-height: 1;
        flex-shrink: 0;
    }
    .sc-verdict-denom {
        font-size: 24px;
        color: #5A5C78;
    }
    .sc-verdict-text {
        font-family: 'DM Sans', sans-serif;
        font-size: 13px;
        color: #9395B0;
        line-height: 1.7;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── SECTION 1: Summary metric cards ──────────────────────────────────────
    st.markdown("<p class='margins-section'>Summary Dimensions</p>", unsafe_allow_html=True)

    summary_cols = st.columns([1, 1, 1, 1])
    summary_data = [
        ("Model sophistication", "7", "#E8C97A", "Good breadth, weak calibration"),
        ("Risk framework",       "5", "#EF4444", "Standard metrics, gaps in tail risk"),
        ("Practical utility",    "6", "#E8C97A", "Retail/research grade; not HF-ready"),
        ("Code quality signals", "7", "#E8C97A", "Clean structure, solid CLI+tests"),
    ]
    for col, (label, score, color, sub) in zip(summary_cols, summary_data):
        with col:
            st.markdown(f"""
            <div class="sc-card">
                <div class="sc-card-label">{label}</div>
                <div class="sc-card-score" style="color:{color};">{score}<span style="font-size:18px;color:#5A5C78;">/10</span></div>
                <div class="sc-card-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 2: Model-by-model fit ─────────────────────────────────────────
    st.markdown("<p class='margins-section'>Model-by-Model Fit for Gold</p>", unsafe_allow_html=True)

    model_cols = st.columns([1, 1, 1, 1, 1])
    model_data = [
        ("GBM",           "Baseline only",        "#C9A84C", "#2A1F00", "Weak fit"),
        ("OU / mean rev.","Contested for gold",   "#C9A84C", "#2A1F00", "Conditional"),
        ("Merton JD",     "Crisis / shock risk",  "#22C55E", "#0A2010", "Good fit"),
        ("Heston SV",     "Vol clustering",        "#22C55E", "#0A2010", "Good fit"),
        ("Regime SW",     "Crisis vs calm",        "#5599EE", "#0A1A30", "Promising"),
    ]
    for col, (name, desc, badge_color, badge_bg, badge_label) in zip(model_cols, model_data):
        with col:
            st.markdown(f"""
            <div class="sc-model">
                <div class="sc-model-name">{name}</div>
                <div class="sc-model-desc">{desc}</div>
                <span class="sc-badge" style="color:{badge_color};background:{badge_bg};">{badge_label}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 3: Detailed dimension scores ──────────────────────────────────
    st.markdown("<p class='margins-section'>Detailed Dimension Scores</p>", unsafe_allow_html=True)

    dimension_data = [
        ("Model breadth",        9, "#5599EE"),
        ("Calibration depth",    4, "#5599EE"),
        ("VaR / CVaR quality",   6, "#22C55E"),
        ("Tail risk coverage",   3, "#22C55E"),
        ("Backtest rigor",       6, "#C9A84C"),
        ("Macro factor coverage",2, "#C9A84C"),
        ("Commercial readiness", 4, "#EF4444"),
        ("Documentation clarity",8, "#EF4444"),
    ]

    dim_html = ""
    for name, score, color in dimension_data:
        pct = score * 10
        dim_html += f"""
        <div class="sc-dim-row">
            <div class="sc-dim-name">{name}</div>
            <div class="sc-dim-track">
                <div class="sc-dim-fill" style="width:{pct}%;background:{color};"></div>
            </div>
            <div class="sc-dim-score">{score}</div>
        </div>"""

    st.markdown(dim_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION 4: Overall verdict ────────────────────────────────────────────
    st.markdown("""
    <div class="sc-verdict">
        <div class="sc-verdict-score">6.2<span class="sc-verdict-denom">/10</span></div>
        <div class="sc-verdict-text">
            <strong style="color:#ECEDF5;font-family:'IBM Plex Mono',monospace;font-size:11px;
            text-transform:uppercase;letter-spacing:0.12em;">Strong academic portfolio, incomplete practitioner toolkit.</strong><br>
            Suitable for quantitative research, retail analytics, and educational use.
            Not production-ready for institutional risk desks without macro integration,
            correlation structure, and calibration improvements.
            <br><br>
            <span class="sc-badge" style="color:#C9A84C;background:#2A1F00;">
                ⚠ Not hedge-fund ready in current state
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)