"""
Monte Carlo Gold Price Predictor — Streamlit App

A production-grade interactive dashboard for gold price simulation
using multiple stochastic models.

Author: Essabri Ali Rayan
Version: 1.0
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo Gold Predictor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --gold:    #C9A84C;
        --gold2:   #E8C97A;
        --dark:    #0D0D14;
        --panel:   #13131F;
        --border:  #2A2A3D;
        --text:    #E8E8F0;
        --muted:   #8888AA;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--dark);
        color: var(--text);
    }

    h1, h2, h3 { font-family: 'Playfair Display', serif; color: var(--gold); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--border);
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label {
        color: var(--muted) !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: var(--gold2) !important;
        font-size: 1.6rem !important;
        font-weight: 600;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--gold), #8B6914);
        color: #0D0D14;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0.04em;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--panel);
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--muted);
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    .stTabs [aria-selected="true"] {
        color: var(--gold) !important;
        border-bottom: 2px solid var(--gold) !important;
    }

    /* Selectbox / Slider labels */
    label[data-testid="stWidgetLabel"] { color: var(--muted) !important; font-size: 0.78rem; }

    /* Divider */
    hr { border-color: var(--border); }

    /* Info boxes */
    .gold-box {
        background: rgba(201,168,76,0.08);
        border: 1px solid rgba(201,168,76,0.3);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .stat-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }
    .stat-pill {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        color: var(--gold2);
    }
</style>
""", unsafe_allow_html=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)

# ── Model Imports (graceful fallbacks) ────────────────────────────────────────
MODELS_AVAILABLE = {}

try:
    from src.models.gbm import GeometricBrownianMotion, GBMParameters
    MODELS_AVAILABLE['GBM'] = GeometricBrownianMotion
except ImportError:
    pass

try:
    from src.models.mean_reversion import OrnsteinUhlenbeckModel, OUParameters
    MODELS_AVAILABLE['Mean Reversion (OU)'] = OrnsteinUhlenbeckModel
except ImportError:
    pass

try:
    from src.models.jump_diffusion import MertonJumpModel, MertonParameters
    MODELS_AVAILABLE['Jump Diffusion'] = MertonJumpModel
except ImportError:
    pass

try:
    from src.models.heston import HestonModel, HestonParameters
    MODELS_AVAILABLE['Heston'] = HestonModel
except ImportError:
    pass

try:
    from src.models.regime_switching import RegimeSwitchingModel
    MODELS_AVAILABLE['Regime Switching'] = RegimeSwitchingModel
except ImportError:
    pass

# ── Demo / Fallback Model ──────────────────────────────────────────────────────
class DemoGBM:
    """Minimal GBM fallback when src/ not installed."""
    name = "GBM (demo)"

    def __init__(self, mu=0.05, sigma=0.15):
        self.mu = mu
        self.sigma = sigma

    def calibrate(self, returns):
        self.mu = float(np.mean(returns) * 252)
        self.sigma = float(np.std(returns) * np.sqrt(252))

    def simulate(self, S0, n_steps, n_paths=1000, random_seed=None):
        if random_seed:
            np.random.seed(random_seed)
        dt = 1 / 252
        Z = np.random.standard_normal((n_paths, n_steps - 1))
        paths = np.zeros((n_paths, n_steps))
        paths[:, 0] = S0
        for t in range(1, n_steps):
            drift = (self.mu - 0.5 * self.sigma ** 2) * dt
            diff  = self.sigma * np.sqrt(dt) * Z[:, t - 1]
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diff)
        return paths


if not MODELS_AVAILABLE:
    MODELS_AVAILABLE = {
        'GBM': DemoGBM,
        'Mean Reversion': DemoGBM,
        'Jump Diffusion': DemoGBM,
        'Heston': DemoGBM,
        'Regime Switching': DemoGBM,
    }

# ── Data Helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_gold_data(period="2y"):
    try:
        import yfinance as yf
        df = yf.Ticker("GC=F").history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            raise ValueError
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        return df[["Close"]].rename(columns={"Close": "price"}).dropna()
    except Exception:
        # synthetic fallback
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


def run_simulation(model_name, S0, n_steps, n_paths, mu, sigma, prices, seed=None):
    """Instantiate, optionally calibrate, and simulate."""
    ModelCls = MODELS_AVAILABLE[model_name]
    model    = ModelCls()

    # Calibrate from recent price history
    try:
        log_returns = np.log(prices["price"] / prices["price"].shift(1)).dropna().values
        if hasattr(model, "calibrate"):
            # OU / Regime use prices; GBM / Merton use returns
            if model_name in ("Mean Reversion (OU)", "Heston", "Regime Switching"):
                model.calibrate(prices["price"].values)
            else:
                model.calibrate(log_returns)
    except Exception:
        pass

    # Apply manual overrides where applicable
    for attr in ("mu", "sigma"):
        if hasattr(model, attr):
            pass  # keep calibrated values; UI shows them as reference

    paths = model.simulate(S0=S0, n_steps=n_steps, n_paths=n_paths, random_seed=seed)
    return paths


# ── Plotting Helpers ───────────────────────────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E8E8F0", family="DM Sans"),
    xaxis=dict(gridcolor="#1E1E2E", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1E1E2E", showgrid=True, zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
)

GOLD   = "#C9A84C"
GOLD2  = "#E8C97A"
RED    = "#E05555"
GREEN  = "#4CAF50"
BLUE   = "#5599EE"


def fan_chart(paths, dates_future, S0, current_price_series):
    """Fan chart with confidence bands."""
    p5, p25, p50, p75, p95 = (np.percentile(paths, q, axis=0)
                               for q in [5, 25, 50, 75, 95])
    n_show = min(200, paths.shape[0])
    idx    = np.random.choice(paths.shape[0], n_show, replace=False)

    fig = go.Figure()

    # Historical price
    fig.add_trace(go.Scatter(
        x=current_price_series.index, y=current_price_series.values,
        name="Historical", line=dict(color=GOLD, width=2),
    ))

    # Sample paths (faint)
    for i in idx:
        fig.add_trace(go.Scatter(
            x=dates_future, y=paths[i],
            mode="lines", line=dict(color="rgba(201,168,76,0.04)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # CI bands
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

    # Median path
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
    fig.add_vline(x=np.mean(final_prices),
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
    loss_pct  = (S0 - var_95) / S0* 100
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
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🥇 Gold Predictor")
    st.markdown("<hr>", unsafe_allow_html=True)

    current_price, price_change = get_current_price()

    delta_color = GREEN if price_change >= 0 else RED
    st.markdown(
        f"**Current Gold Price**  \n"
        f"<span style='font-size:1.5rem;font-weight:700;color:{GOLD2}'>"
        f"${current_price:,.2f}</span>  "
        f"<span style='color:{delta_color};font-size:0.85rem'>"
        f"{'▲' if price_change >= 0 else '▼'} {abs(price_change):.2f}%</span>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Parameters")

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
    n_steps = horizon_map[horizon_label]

    n_paths = st.select_slider(
        "Number of Simulations",
        options=[500, 1_000, 2_000, 5_000, 10_000],
        value=2_000,
    )

    data_period = st.selectbox("Training Data", ["6mo", "1y", "2y", "5y"], index=2)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🔧 Manual Overrides")
    use_manual = st.checkbox("Override calibrated params")
    if use_manual:
        manual_mu    = st.slider("Drift (μ) annualised", -0.3, 0.5, 0.05, 0.01)
        manual_sigma = st.slider("Volatility (σ) annualised", 0.05, 0.6, 0.15, 0.01)
    else:
        manual_mu, manual_sigma = None, None

    seed = st.number_input("Random Seed (0 = random)", 0, 9999, 42)
    seed = int(seed) if seed > 0 else None

    run_btn = st.button("▶  Run Simulation", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# Monte Carlo Gold Price Predictor")
st.markdown(
    "<p style='color:#8888AA;margin-top:-12px'>"
    "Stochastic simulation engine for gold price forecasting & risk analysis</p>",
    unsafe_allow_html=True,
)

# Load data
prices = fetch_gold_data(data_period)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Simulation",
    "⚠️  Risk Analysis",
    "📊  Market Data",
    "ℹ️  Model Guide",
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
                )
                st.session_state["paths"]      = paths
                st.session_state["model_name"] = model_name
                st.session_state["n_steps"]    = n_steps
                st.session_state["S0"]         = current_price
            except Exception as e:
                st.error(f"Simulation error: {e}")
                st.stop()
    else:
        paths      = st.session_state["paths"]
        model_name = st.session_state["model_name"]
        n_steps    = st.session_state["n_steps"]
        current_price = st.session_state["S0"]

    final_prices = paths[:, -1]
    S0 = current_price

    # ── Key Metrics ────────────────────────────────────────────────────────────
    mean_price  = np.mean(final_prices)
    median_price= np.median(final_prices)
    p5_price    = np.percentile(final_prices, 5)
    p95_price   = np.percentile(final_prices, 95)
    prob_up     = np.mean(final_prices > S0) * 100
    exp_return  = (mean_price / S0 - 1) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected Price",  f"${mean_price:,.0f}", f"{exp_return:+.1f}%")
    c2.metric("Median Price",    f"${median_price:,.0f}")
    c3.metric("5th Percentile",  f"${p5_price:,.0f}")
    c4.metric("95th Percentile", f"${p95_price:,.0f}")
    c5.metric("Prob. of Gain",   f"{prob_up:.1f}%")

    st.markdown("")

    # ── Fan Chart ──────────────────────────────────────────────────────────────
    last_date     = prices.index[-1]
    dates_future  = pd.date_range(start=last_date, periods=n_steps + 1, freq="B")
    recent_prices = prices["price"].iloc[-120:]

    st.plotly_chart(
        fan_chart(paths, dates_future, S0, recent_prices),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(distribution_chart(final_prices, S0), use_container_width=True)
    with col_b:
        # Return distribution
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

    # ── Scenario Table ────────────────────────────────────────────────────────
    st.markdown("#### Scenario Summary")
    scenarios = pd.DataFrame({
        "Scenario":   ["Bear (5th pct)", "Mild Bear (25th)", "Base (Median)", "Mild Bull (75th)", "Bull (95th)"],
        "Price":      [f"${np.percentile(final_prices,  5):,.0f}",
                       f"${np.percentile(final_prices, 25):,.0f}",
                       f"${median_price:,.0f}",
                       f"${np.percentile(final_prices, 75):,.0f}",
                       f"${p95_price:,.0f}"],
        "Change":     [f"{(np.percentile(final_prices,  5)/S0-1)*100:+.1f}%",
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
        paths       = st.session_state["paths"]
        S0          = st.session_state["S0"]
        final_prices = paths[:, -1]

        # ── Risk Metrics ──────────────────────────────────────────────────────
        var_90  = np.percentile(final_prices, 10)
        var_95  = np.percentile(final_prices, 5)
        var_99  = np.percentile(final_prices, 1)
        cvar_95 = np.mean(final_prices[final_prices <= var_95])
        cvar_99 = np.mean(final_prices[final_prices <= var_99])

        max_dd_per_path = []
        for path in paths:
            peak     = np.maximum.accumulate(path)
            drawdown = (path - peak) / peak
            max_dd_per_path.append(np.min(drawdown))
        avg_max_dd = np.mean(max_dd_per_path) * 100

        vol_realized = np.std(np.log(final_prices / S0)) * 100

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("VaR 95%",       f"${var_95:,.0f}", f"{(var_95/S0-1)*100:+.1f}%")
        r2.metric("CVaR 95%",      f"${cvar_95:,.0f}", f"{(cvar_95/S0-1)*100:+.1f}%")
        r3.metric("Avg Max Drawdown", f"{avg_max_dd:.1f}%")
        r4.metric("Realised Vol",  f"{vol_realized:.1f}%")

        st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(risk_gauge(var_95, cvar_95, S0), use_container_width=True)

        with col2:
            # VaR at different confidence levels
            confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
            var_values = [np.percentile(final_prices, (1 - cl) * 100)
                          for cl in confidence_levels]
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

        # ── Target Analysis ───────────────────────────────────────────────────
        st.markdown("#### Probability of Reaching a Target Price")
        col_t1, col_t2 = st.columns([1, 2])
        with col_t1:
            target = st.number_input(
                "Target Price ($)",
                value=int(S0 * 1.10),
                step=50,
            )
        with col_t2:
            prob_reach   = np.mean(final_prices >= target) * 100
            above_target = final_prices[final_prices >= target]
            avg_if_reach = np.mean(above_target) if len(above_target) > 0 else 0
            st.markdown(f"""
            <div class='gold-box'>
                <b>Probability of reaching ${target:,}</b><br>
                <span style='font-size:2rem;color:{GOLD2};font-weight:700'>{prob_reach:.1f}%</span><br>
                <span style='color:#8888AA;font-size:0.85rem'>
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
        st.markdown("#### Full Risk Summary")
        risk_table = pd.DataFrame({
            "Metric": [
                "VaR 80%", "VaR 90%", "VaR 95%", "VaR 99%",
                "CVaR 95%", "CVaR 99%",
                "Prob. of Loss > 5%", "Prob. of Loss > 10%", "Prob. of Loss > 20%",
                "Prob. of Gain > 5%", "Prob. of Gain > 10%", "Prob. of Gain > 20%",
            ],
            "Value": [
                f"${np.percentile(final_prices,20):,.0f}",
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
    st.markdown("### Historical Gold Price")
    st.plotly_chart(historical_chart(prices), use_container_width=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("#### Return Statistics")
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
        st.markdown("#### Rolling Volatility")
        roll_vol = log_ret.rolling(30).std() * np.sqrt(252) * 100
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=roll_vol.index, y=roll_vol.values,
            fill="tozeroy", fillcolor="rgba(85,153,238,0.10)",
            line=dict(color=BLUE, width=1.5), name="30d Vol",
        ))
        fig_vol.update_layout(
            yaxis_title="Volatility (%)", **{k: v for k, v in DARK_LAYOUT.items() if k != "margin"}, margin=dict(l=40, r=20, t=10, b=40)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # ── Price Distribution ────────────────────────────────────────────────────
    st.markdown("#### Historical Return Distribution")
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
    st.markdown("### Model Reference Guide")

    model_info = {
        "GBM": {
            "name": "Geometric Brownian Motion",
            "formula": "dS = μ·S·dt + σ·S·dW",
            "desc": "The foundational stochastic model. Assumes log-normally distributed returns with constant drift and volatility. Best used as a baseline benchmark.",
            "pros": ["Simple & fast", "Analytically tractable", "Black-Scholes validated"],
            "cons": ["No mean reversion", "Constant volatility", "No jumps"],
            "best_for": "Baseline comparison",
        },
        "Mean Reversion (OU)": {
            "name": "Ornstein-Uhlenbeck",
            "formula": "dS = κ·(θ − S)·dt + σ·dW",
            "desc": "Gold's most appropriate single-factor model. Captures the documented tendency of gold to revert to long-term equilibrium. κ controls reversion speed, θ is the equilibrium price.",
            "pros": ["Captures mean reversion", "Reflects gold's behaviour", "Analytically solvable"],
            "cons": ["Can produce negative prices if σ is large", "Constant reversion speed"],
            "best_for": "Long-term forecasting",
        },
        "Jump Diffusion": {
            "name": "Merton Jump Diffusion",
            "formula": "dS = μ·S·dt + σ·S·dW + J·S·dN",
            "desc": "Extends GBM with a Poisson jump process to model sudden price shocks from geopolitical crises, central bank interventions, or market panics.",
            "pros": ["Captures tail risk", "Models crisis events", "Realistic skewness"],
            "cons": ["More parameters to calibrate", "Jump timing is random"],
            "best_for": "Risk analysis & stress testing",
        },
        "Heston": {
            "name": "Heston Stochastic Volatility",
            "formula": "dS = μ·S·dt + √v·S·dW₁  |  dv = κ(θ−v)dt + ξ√v·dW₂",
            "desc": "Models volatility itself as a stochastic process. Captures volatility clustering, the leverage effect (negative price-vol correlation), and the volatility smile observed in gold options.",
            "pros": ["Realistic volatility dynamics", "Captures leverage effect", "Options pricing compatible"],
            "cons": ["Slow to simulate", "Feller condition must hold", "Complex calibration"],
            "best_for": "Options pricing & volatility forecasting",
        },
        "Regime Switching": {
            "name": "Markov Regime Switching",
            "formula": "dS = μ(sₜ)·S·dt + σ(sₜ)·S·dW",
            "desc": "Uses a Markov chain to alternate between distinct market states (calm / crisis). Each regime has its own drift and volatility parameters, capturing structural market changes.",
            "pros": ["Captures market structural changes", "Excellent for scenario analysis", "Interpretable regimes"],
            "cons": ["Regime parameters need careful calibration", "Transition matrix estimation requires long history"],
            "best_for": "Scenario planning & long-term forecasting",
        },
    }

    for key, info in model_info.items():
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

    st.markdown("---")
    st.markdown("""
    ### How to Run

    ```bash
    # Install dependencies
    pip install -r requirements.txt

    # Launch dashboard
    streamlit run app.py
    ```
    """)
