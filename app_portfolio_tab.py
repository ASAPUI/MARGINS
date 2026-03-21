"""
MARGINS — Portfolio Mode Dashboard Tab
    streamlit run app_portfolio_tab.py
    author : essabri ali rayan
    version :1.4.4
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Lazy imports of portfolio modules (graceful degradation if not found) ──────
try:
    from src.portfolio.universe import fetch_universe
    from src.portfolio.correlation import compute_cholesky
    from src.portfolio.simulator import simulate_portfolio
    from src.portfolio.metrics import compute_portfolio_metrics
    from src.portfolio.optimizer import optimize_weights
    PORTFOLIO_MODULES_AVAILABLE = True
except ImportError:
    PORTFOLIO_MODULES_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# THEME & STYLE
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG       = "#0B0C14"
PANEL_BG      = "#10121E"
BORDER        = "#1F2240"
ACCENT_GOLD   = "#C9A84C"
ACCENT_BLUE   = "#5599EE"
ACCENT_RED    = "#EF4444"
ACCENT_GREEN  = "#22C55E"
TEXT_PRIMARY  = "#ECEDF5"
TEXT_MUTED    = "#5A5C78"
GRID_COLOR    = "#1F2240"

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#0B0C14",
        plot_bgcolor="#0B0C14",
        font=dict(color="#9395B0", family="IBM Plex Mono"),
        xaxis=dict(
            gridcolor="#1F2240", zerolinecolor="#1F2240", linecolor="#1F2240",
            tickfont=dict(color="#9395B0", family="IBM Plex Mono", size=10),
        ),
        yaxis=dict(
            gridcolor="#1F2240", zerolinecolor="#1F2240", linecolor="#1F2240",
            tickfont=dict(color="#9395B0", family="IBM Plex Mono", size=10),
        ),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)

ASSET_COLORS = [
    ACCENT_GOLD, ACCENT_BLUE, "#a78bfa", "#f97316",
    "#06b6d4", "#ec4899", "#84cc16", "#f59e0b",
    "#8b5cf6", "#14b8a6", "#fb923c", "#e879f9",
]

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Playfair+Display:wght@500&family=DM+Sans:wght@400&display=swap');

/* ── Base ── */
[data-testid="stAppViewContainer"] { background: #0B0C14; }
[data-testid="stSidebar"] {
    background: #10121E !important;
    border-right: 1px solid #1F2240 !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography ── */
h1, h2, h3, h4, .margins-title {
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    color: #9395B0 !important;
    border-bottom: 1px solid #1F2240 !important;
    padding-bottom: 6px !important;
}
p, .stMarkdown p {
    font-family: 'DM Sans', sans-serif !important;
    color: #9395B0 !important;
    font-size: 13px !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #10121E !important;
    border: 1px solid #1F2240 !important;
    border-radius: 0 !important;
    padding: 16px !important;
}
[data-testid="metric-container"] label {
    color: #5A5C78 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    font-weight: 400 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 500 !important;
    color: #E8C97A !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Section headers ── */
.section-header {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #9395B0 !important;
    border-bottom: 1px solid #1F2240 !important;
    padding-bottom: 6px !important;
    margin-bottom: 16px !important;
    position: relative !important;
}
.section-header::after {
    content: '' !important;
    position: absolute !important;
    bottom: -1px !important;
    left: 0 !important;
    width: 28px !important;
    height: 1px !important;
    background: #C9A84C !important;
}

/* ── Panel container ── */
.panel {
    background: #10121E !important;
    border: 1px solid #1F2240 !important;
    border-radius: 0 !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}

/* ── Weight badge ── */
.weight-badge {
    display: inline-block !important;
    background: #10121E !important;
    border: 1px solid #1F2240 !important;
    border-radius: 2px !important;
    padding: 2px 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #E8C97A !important;
    letter-spacing: 0.05em !important;
}

/* ── Strategy tags ── */
.strategy-tag {
    display: inline-block !important;
    padding: 2px 10px !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
.tag-sharpe  { background: rgba(201,168,76,0.10);  color: #C9A84C; border: 1px solid rgba(201,168,76,0.25); }
.tag-cvar    { background: rgba(239,68,68,0.10);   color: #EF4444; border: 1px solid rgba(239,68,68,0.25); }
.tag-parity  { background: rgba(85,153,238,0.10);  color: #5599EE; border: 1px solid rgba(85,153,238,0.25); }

/* ── Buttons ── */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: linear-gradient(135deg, #C9A84C, #E8C97A) !important;
    color: #0B0C14 !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0.55rem 1.5rem !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Slider labels ── */
.stSlider label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    color: #5A5C78 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border: 1px solid #1F2240 !important;
    border-radius: 0 !important;
}
[data-testid="stDataFrame"] th {
    background: #10121E !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: #5A5C78 !important;
    border-bottom: 1px solid #1F2240 !important;
}
[data-testid="stDataFrame"] td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    border-color: #1F2240 !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    border: 1px solid #1F2240 !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    gap: 0 !important;
    border-bottom: 1px solid #1F2240 !important;
    background: transparent !important;
}
[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 400 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #9395B0 !important;
    padding: 8px 20px !important;
    background: transparent !important;
    border: none !important;
}
[aria-selected="true"] {
    color: #ECEDF5 !important;
    border-bottom: 2px solid #C9A84C !important;
    background: transparent !important;
}

/* ── Inputs / selects ── */
div[data-baseweb="select"] > div {
    background: #10121E !important;
    border: 1px solid #1F2240 !important;
    border-radius: 0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: #ECEDF5 !important;
}
div[data-baseweb="select"] > div:focus-within { border-color: #C9A84C !important; }
.stNumberInput input, .stTextInput input {
    background: #10121E !important;
    border: 1px solid #1F2240 !important;
    border-radius: 0 !important;
    color: #ECEDF5 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0B0C14; }
::-webkit-scrollbar-thumb { background: #1F2240; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #2E3160; }
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
# MOCK FALLBACKS (when src/portfolio/ modules are not installed)
# ══════════════════════════════════════════════════════════════════════════════

def _mock_run(tickers, weights, n_paths, n_days, calib_window, period):
    """Generate plausible mock data for UI development / demo purposes."""
    np.random.seed(42)
    N, K, T = len(tickers), n_paths, n_days

    # Simulate correlated paths
    mu    = np.array([0.08, 0.12, 0.04, 0.15, 0.25][:N])
    sigma = np.array([0.15, 0.18, 0.07, 0.22, 0.60][:N])
    rho   = 0.35
    cov   = np.full((N, N), rho) * np.outer(sigma, sigma)
    np.fill_diagonal(cov, sigma**2)
    L     = np.linalg.cholesky(cov / 252)

    S0    = np.array([1950, 480, 95, 65000, 35000][:N], dtype=float)
    paths = np.zeros((N, K, T + 1))
    paths[:, :, 0] = S0[:, None]
    eps = np.random.standard_normal((N, K, T))
    Z   = np.einsum("ij,jkt->ikt", L, eps)
    for t in range(1, T + 1):
        for i in range(N):
            drift = (mu[i] - 0.5 * sigma[i]**2) / 252
            paths[i, :, t] = paths[i, :, t-1] * np.exp(drift + Z[i, :, t-1])

    # Portfolio value
    w   = np.array(weights)
    W0  = np.dot(w, S0)
    W   = np.einsum("i,ikt->kt", w * S0 / W0, paths) * W0

    W_T = W[:, -1]
    ret = (W_T / W[:, 0]) - 1
    var_95  = float(np.percentile(W_T, 5))
    cvar_95 = float(np.mean(W_T[W_T <= var_95]))
    peak    = np.maximum.accumulate(W, axis=1)
    dd      = np.min((W - peak) / peak, axis=1)

    ind_vols = [float(np.std(np.log(paths[i, :, -1] / S0[i]))) for i in range(N)]
    wtd_vol  = float(np.dot(w, ind_vols))
    port_vol = float(np.std(np.log(W_T / W[:, 0])))

    metrics = {
        "portfolio_var_95":      var_95,
        "portfolio_cvar_95":     cvar_95,
        "expected_return":       float(np.mean(ret) * 100),
        "sharpe":                float(np.mean(ret) / np.std(ret) * np.sqrt(252)) if np.std(ret) > 0 else 0,
        "diversification_ratio": wtd_vol / port_vol if port_vol > 0 else 1.0,
        "avg_max_drawdown":      float(np.mean(dd) * 100),
        "prob_gain":             float(np.mean(ret > 0) * 100),
    }

    # Correlation matrix (mock)
    log_ret_df = pd.DataFrame(
        np.random.multivariate_normal(mu / 252, cov / 252, 252),
        columns=tickers
    )

    # Mock optimized weights
    opt_weights = {
        "max_sharpe":  np.clip(np.random.dirichlet(np.ones(N) * 2), 0.01, 0.40),
        "min_cvar":    np.clip(np.random.dirichlet(np.ones(N) * 3), 0.01, 0.40),
        "risk_parity": np.array([1 / s for s in sigma]) / sum(1 / s for s in sigma),
    }
    for k in opt_weights:
        opt_weights[k] = opt_weights[k] / opt_weights[k].sum()

    return paths, W, metrics, log_ret_df, opt_weights, S0


# ══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_fan_chart(W, n_days, title="Portfolio Value Distribution"):
    """Fan chart: median path + percentile bands."""
    T = W.shape[1]
    x = list(range(T))

    p5  = np.percentile(W, 5,  axis=0)
    p25 = np.percentile(W, 25, axis=0)
    p50 = np.percentile(W, 50, axis=0)
    p75 = np.percentile(W, 75, axis=0)
    p95 = np.percentile(W, 95, axis=0)

    fig = go.Figure()

    # 5–95 band
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=list(p95) + list(p5[::-1]),
        fill="toself",
        fillcolor="rgba(212,168,83,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5–95%", showlegend=True,
        hoverinfo="skip",
    ))
    # 25–75 band
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=list(p75) + list(p25[::-1]),
        fill="toself",
        fillcolor="rgba(212,168,83,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25–75%", showlegend=True,
        hoverinfo="skip",
    ))
    # Median
    fig.add_trace(go.Scatter(
        x=x, y=p50,
        line=dict(color=ACCENT_GOLD, width=2),
        name="Median",
    ))
    # p5 / p95 borders
    fig.add_trace(go.Scatter(
        x=x, y=p95, line=dict(color=ACCENT_GOLD, width=0.8, dash="dot"),
        name="95th pct", opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p5, line=dict(color=ACCENT_RED, width=0.8, dash="dot"),
        name="5th pct", opacity=0.7,
    ))

    _layout = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **_layout,
        title=dict(text=title, font=dict(size=12, color=TEXT_MUTED), x=0),
        xaxis=dict(title="Trading Days", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=BORDER),
        yaxis=dict(title="Portfolio Value ($)", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=BORDER),
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        height=360,
    )
    return fig


def build_correlation_heatmap(log_ret_df):
    """Annotated correlation heatmap."""
    corr = log_ret_df.corr()
    tickers = list(corr.columns)
    z = corr.values

    colorscale = [
        [0.0,  "#EF4444"],
        [0.5,  "#10121E"],
        [1.0,  "#C9A84C"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=tickers,
        y=tickers,
        colorscale=colorscale,
        zmid=0,
        zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=11, family="IBM Plex Mono"),
        showscale=True,
        colorbar=dict(
            thickness=12,
            tickfont=dict(size=9, color=TEXT_MUTED),
            outlinewidth=0,
        ),
    ))
    _layout = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **_layout,
        title=dict(text="Return Correlation Matrix", font=dict(size=12, color=TEXT_MUTED), x=0),
        height=320,
        xaxis=dict(side="bottom", tickfont=dict(size=10), gridcolor=GRID_COLOR),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10), gridcolor=GRID_COLOR),
    )
    return fig


def build_weight_donut(weights, tickers, title="Current Weights"):
    """Donut chart for portfolio weights."""
    colors = ASSET_COLORS[:len(tickers)]
    fig = go.Figure(go.Pie(
        labels=tickers,
        values=[round(w * 100, 1) for w in weights],
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=DARK_BG, width=2)),
        textfont=dict(family="IBM Plex Mono", size=10),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text=title, font=dict(size=12, color=TEXT_MUTED), x=0),
        showlegend=True,
        legend=dict(
            font=dict(size=10, family="IBM Plex Mono"),
            bgcolor="rgba(0,0,0,0)",
            orientation="v",
            x=1.0,
        ),
        height=280,
        annotations=[dict(
            text=f"{len(tickers)}<br><span style='font-size:10px'>assets</span>",
            x=0.5, y=0.5, font=dict(size=18, family="IBM Plex Mono", color=TEXT_PRIMARY),
            showarrow=False,
        )],
    )
    return fig


def build_optimizer_comparison(opt_weights, tickers, paths, S0, strategy_metrics):
    """Grouped bar chart comparing weight allocations across 3 strategies."""
    strategies = ["max_sharpe", "min_cvar", "risk_parity"]
    labels     = ["Max Sharpe", "Min CVaR", "Risk Parity"]
    colors_bar = [ACCENT_GOLD, ACCENT_RED, ACCENT_BLUE]

    fig = go.Figure()
    for i, (strat, label, color) in enumerate(zip(strategies, labels, colors_bar)):
        w = opt_weights.get(strat, np.ones(len(tickers)) / len(tickers))
        fig.add_trace(go.Bar(
            name=label,
            x=tickers,
            y=[round(v * 100, 1) for v in w],
            marker_color=color,
            marker_line=dict(color=DARK_BG, width=1),
            opacity=0.85,
            text=[f"{v*100:.0f}%" for v in w],
            textposition="outside",
            textfont=dict(size=9, family="IBM Plex Mono", color=TEXT_MUTED),
        ))

    _layout = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **_layout,
        title=dict(text="Optimal Weight Allocation by Strategy", font=dict(size=12, color=TEXT_MUTED), x=0),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
        yaxis=dict(gridcolor=GRID_COLOR, ticksuffix="%", title="Weight (%)"),
        legend=dict(
            orientation="h", x=0, y=1.08,
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        height=320,
    )
    return fig


def build_per_asset_paths(paths, tickers, n_display=50):
    """Small multiples — one path fan per asset."""
    N = paths.shape[0]
    cols = min(N, 3)
    rows = (N + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=tickers,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, ticker in enumerate(tickers):
        r, c = divmod(idx, cols)
        asset_paths = paths[idx]  # (K, T+1)

        p5  = np.percentile(asset_paths, 5,  axis=0)
        p50 = np.percentile(asset_paths, 50, axis=0)
        p95 = np.percentile(asset_paths, 95, axis=0)
        x   = list(range(asset_paths.shape[1]))
        color = ASSET_COLORS[idx % len(ASSET_COLORS)]

        # Sample paths
        for k in range(min(n_display, asset_paths.shape[0])):
            fig.add_trace(go.Scatter(
                x=x, y=asset_paths[k],
                line=dict(color=color, width=0.3),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            ), row=r+1, col=c+1)

        # Median
        fig.add_trace(go.Scatter(
            x=x, y=p50,
            line=dict(color=color, width=2),
            name=ticker,
            showlegend=False,
        ), row=r+1, col=c+1)

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"],
        title=dict(text="Per-Asset Simulated Paths", font=dict(size=12, color=TEXT_MUTED), x=0),
        height=220 * rows,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 10
        ann.font.color = TEXT_MUTED
        ann.font.family = "IBM Plex Mono"

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def render_portfolio_tab():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not PORTFOLIO_MODULES_AVAILABLE:
        st.warning(
            "⚠  `src/portfolio/` modules not found — running in **demo mode** with synthetic data. "
            "Ensure Phases 1–4 are installed to use live market data.",
            icon="⚠️",
        )

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([6, 1])
    with col_title:
        st.markdown(
            "<div style='font-family:\"Playfair Display\",serif;font-size:1.6rem;"
            "font-weight:500;color:#ECEDF5;margin-bottom:2px;'>"
            "MARGINS <span style='color:#C9A84C;font-size:1rem;'>∷</span>"
            " Portfolio Mode</div>"
            "<p style='font-family:\"DM Sans\",sans-serif;font-size:13px;"
            "color:#9395B0;margin-top:2px;'>"
            "Multi-Asset Monte Carlo Simulator — Phase 5 Dashboard</p>",
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            "<div style='text-align:right;padding-top:14px;'>"
            "<span style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;"
            "text-transform:uppercase;letter-spacing:0.14em;color:#C9A84C;'>"
            "v2.0</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color:#1F2240;margin:4px 0 20px;'>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # SIDEBAR — Controls
    # ═══════════════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown(
            "<p class='section-header'>Assets</p>",
            unsafe_allow_html=True,
        )

        PRESET_UNIVERSES = {
            "Gold + Equities + Bonds": ["GC=F", "SPY", "TLT"],
            "Gold + Equities + Crypto": ["GC=F", "SPY", "BTC-USD"],
            "All-Weather (5 assets)":  ["GC=F", "SPY", "TLT", "USO", "BTC-USD"],
            "Custom": [],
        }

        preset = st.selectbox(
            "Universe preset",
            list(PRESET_UNIVERSES.keys()),
            index=0,
        )

        if preset == "Custom":
            raw = st.text_input(
                "Tickers (space-separated)",
                value="GC=F SPY TLT",
                help="Yahoo Finance tickers, e.g. GC=F SPY TLT BTC-USD",
            )
            tickers = [t.strip().upper() for t in raw.split() if t.strip()]
        else:
            tickers = PRESET_UNIVERSES[preset]

        tickers = tickers[:8]  # cap at 8

        if len(tickers) < 2:
            st.error("Select at least 2 assets.")
            return

        st.markdown("<hr style='border-color:#1F2240;'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Weights</p>", unsafe_allow_html=True)

        weight_mode = st.radio(
            "Weight input",
            ["Equal weight", "Manual sliders"],
            horizontal=True,
            label_visibility="collapsed",
        )

        if weight_mode == "Equal weight":
            raw_weights = [1.0 / len(tickers)] * len(tickers)
            st.markdown(
                f"<div style='color:#9395B0;font-family:\"IBM Plex Mono\",monospace;font-size:11px;"
                f"margin-top:4px;'>Each asset: "
                f"<span style='color:#C9A84C;'>{100/len(tickers):.1f}%</span></div>",
                unsafe_allow_html=True,
            )
        else:
            raw_weights = []
            for i, ticker in enumerate(tickers):
                default = min(40, max(1, int(100 / len(tickers))))
                w = st.slider(
                    f"{ticker}",
                    min_value=1, max_value=40,
                    value=default,
                    step=1,
                    format="%d%%",
                    key=f"w_{ticker}_{i}",
                )
                raw_weights.append(float(w))

            total = sum(raw_weights)
            if abs(total - 100) > 0.5:
                st.markdown(
                    f"<div style='color:#EF4444;font-family:\"IBM Plex Mono\",monospace;"
                    f"font-size:11px;margin:4px 0;'>Σ = {total:.0f}% "
                    f"— will auto-normalize to 100%</div>",
                    unsafe_allow_html=True,
                )

        weights = np.array(raw_weights, dtype=float)
        weights = weights / weights.sum()

        st.markdown("<hr style='border-color:#1F2240;'>", unsafe_allow_html=True)
        st.markdown("<p class='section-header'>Simulation</p>", unsafe_allow_html=True)

        n_paths = st.select_slider(
            "Monte Carlo paths",
            options=[500, 1000, 2000, 5000, 10000],
            value=2000,
        )
        n_days = st.slider(
            "Forecast horizon (days)",
            min_value=5, max_value=252,
            value=30, step=5,
        )
        calib_window = st.select_slider(
            "Calibration window (days)",
            options=[63, 126, 252],
            value=126,
        )
        period = st.selectbox(
            "Historical data period",
            ["6mo", "1y", "2y", "5y"],
            index=2,
        )

        st.markdown("<hr style='border-color:#1F2240;'>", unsafe_allow_html=True)

        run_btn = st.button(
            "▶  RUN SIMULATION",
            use_container_width=True,
            type="primary",
        )
        optimize_btn = st.button(
            "◆  OPTIMIZE WEIGHTS",
            use_container_width=True,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SESSION STATE
    # ═══════════════════════════════════════════════════════════════════════════
    if "portfolio_results" not in st.session_state:
        st.session_state.portfolio_results = None
    if "opt_results" not in st.session_state:
        st.session_state.opt_results = None
    if "last_tickers" not in st.session_state:
        st.session_state.last_tickers = None

    # Clear results when tickers change
    if st.session_state.last_tickers != tickers:
        st.session_state.portfolio_results = None
        st.session_state.opt_results = None
        st.session_state.last_tickers = tickers

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN SIMULATION
    # ═══════════════════════════════════════════════════════════════════════════
    if run_btn:
        with st.spinner("Running correlated Monte Carlo simulation…"):
            try:
                if PORTFOLIO_MODULES_AVAILABLE:
                    combined, log_ret = fetch_universe(
                        tickers, period=period, calib_window=calib_window
                    )
                    L, cov_lw = compute_cholesky(log_ret)

                    # Build per-asset model stubs (use GBM defaults)
                    class _SimpleModel:
                        def __init__(self, mu, sigma):
                            self.mu, self.sigma = mu, sigma

                    ann_ret = log_ret.mean() * 252
                    ann_vol = log_ret.std() * np.sqrt(252)
                    models  = [_SimpleModel(ann_ret[t], ann_vol[t]) for t in tickers]
                    S0_vec  = combined.iloc[-1].values.astype(float)

                    paths = simulate_portfolio(
                        models, S0_vec, L,
                        n_steps=n_days, n_paths=n_paths, dt=1/252,
                    )
                    metrics = compute_portfolio_metrics(paths, weights, S0_vec)

                    W0  = float(np.dot(weights, S0_vec))
                    W   = np.einsum("i,ikt->kt", weights * S0_vec / W0, paths) * W0
                    log_ret_df = log_ret

                else:
                    paths, W, metrics, log_ret_df, _, S0_vec = _mock_run(
                        tickers, weights, n_paths, n_days, calib_window, period
                    )

                st.session_state.portfolio_results = {
                    "paths": paths,
                    "W": W,
                    "metrics": metrics,
                    "log_ret_df": log_ret_df,
                    "tickers": tickers,
                    "weights": weights,
                    "n_days": n_days,
                    "S0_vec": S0_vec if PORTFOLIO_MODULES_AVAILABLE else
                              np.array([1950, 480, 95, 65000, 35000][:len(tickers)], dtype=float),
                }
                st.success("Simulation complete.", icon="✅")

            except Exception as e:
                st.error(f"Simulation failed: {e}")
                st.exception(e)

    # ═══════════════════════════════════════════════════════════════════════════
    # RUN OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    if optimize_btn:
        with st.spinner("Optimizing weights across 3 strategies…"):
            try:
                if PORTFOLIO_MODULES_AVAILABLE and st.session_state.portfolio_results:
                    res   = st.session_state.portfolio_results
                    paths = res["paths"]
                    S0    = res["S0_vec"]
                    def _extract_weights(raw, strategy, n):
                        """Normalize whatever optimize_weights returns into a (n,) float array."""
                        # Case 1: returns a dict keyed by strategy name
                        if isinstance(raw, dict):
                            raw = raw.get(strategy) or raw.get(list(raw.keys())[0])
                        # Case 2: scipy OptimizeResult
                        if hasattr(raw, 'x'):
                            raw = raw.x
                        # Case 3: force to flat float array
                        out = np.asarray(raw, dtype=float).flatten()
                        if len(out) != n:
                            out = np.ones(n) / n
                        out = np.clip(out, 0.01, 0.40)
                        out = out / out.sum()
                        return out

                    def _safe_weights(paths, S0, strategy, n):
                        raw = optimize_weights(paths, S0, strategy=strategy)
                        # optimize_weights returns (weights_array, metrics_dict) — unpack it
                        if isinstance(raw, tuple):
                            raw = raw[0]
                        return _extract_weights(raw, strategy, n)

                    N = paths.shape[0]
                    opt_weights = {
                        s: _safe_weights(paths, S0, s, N)
                        for s in ["max_sharpe", "min_cvar", "risk_parity"]
                    }
                    # Compute metrics per strategy
                    strategy_metrics = {}
                    for s, w in opt_weights.items():
                        w_flat = np.asarray(w, dtype=float).flatten()
                        strategy_metrics[s] = compute_portfolio_metrics(paths, w_flat, S0)

                elif st.session_state.portfolio_results:
                    # Use mock opt from last run
                    _, _, _, _, opt_weights, S0 = _mock_run(
                        tickers, weights, n_paths, n_days, calib_window, period
                    )
                    paths = st.session_state.portfolio_results["paths"]
                    strategy_metrics = {}
                    for s, w in opt_weights.items():
                        _, W_opt, m, _, _, _ = _mock_run(
                            tickers, w.tolist(), n_paths, n_days, calib_window, period
                        )
                        strategy_metrics[s] = m

                else:
                    # Run fresh mock
                    _, _, _, _, opt_weights, S0 = _mock_run(
                        tickers, weights, n_paths, n_days, calib_window, period
                    )
                    strategy_metrics = {}
                    for s, w in opt_weights.items():
                        _, _, m, _, _, _ = _mock_run(
                            tickers, w.tolist(), n_paths, n_days, calib_window, period
                        )
                        strategy_metrics[s] = m

                st.session_state.opt_results = {
                    "opt_weights": opt_weights,
                    "strategy_metrics": strategy_metrics,
                    "tickers": tickers,
                }
                st.success("Optimization complete.", icon="✅")

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                st.exception(e)

    # ═══════════════════════════════════════════════════════════════════════════
    # RESULTS DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════
    res = st.session_state.portfolio_results

    if res is None:
        st.markdown(
            "<div style='text-align:center;padding:80px 0;border:1px solid #1F2240;"
            "background:#10121E;'>"
            "<p style='font-family:\"IBM Plex Mono\",monospace;font-size:11px;"
            "color:#5A5C78;letter-spacing:0.18em;text-transform:uppercase;'>"
            "Select Assets · Configure · Run Simulation</p>"
            "<p style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;"
            "color:#2E3160;margin-top:8px;letter-spacing:0.12em;'>"
            "Cholesky-correlated Monte Carlo · Ledoit-Wolf shrinkage · "
            "Markowitz optimization</p></div>",
            unsafe_allow_html=True,
        )
        return

    metrics    = res["metrics"]
    W          = res["W"]
    paths      = res["paths"]
    log_ret_df = res["log_ret_df"]
    _tickers   = res["tickers"]
    _weights   = res["weights"]
    _n_days    = res["n_days"]
    S0_vec     = res["S0_vec"]

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tabs = st.tabs(["Overview", "Paths", "Correlation", "Per Asset", "Optimizer"])

    # ── Tab 1: Overview ────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("<p class='section-header'>Risk Metrics</p>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        exp_ret = metrics.get("expected_return", 0)
        sharpe  = metrics.get("sharpe", 0)
        var95   = metrics.get("portfolio_var_95", 0)
        cvar95  = metrics.get("portfolio_cvar_95", 0)
        div_r   = metrics.get("diversification_ratio", 1.0)
        mdd     = metrics.get("avg_max_drawdown", 0)
        prob_g  = metrics.get("prob_gain", 50)

        with c1:
            st.metric(
                "Expected Return",
                f"{exp_ret:+.2f}%",
                delta="annualized" if _n_days == 252 else f"over {_n_days}d",
            )
        with c2:
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.3f}",
                delta="annualized",
            )
        with c3:
            st.metric(
                "VaR 95%",
                f"${var95:,.0f}",
                delta="5th percentile terminal",
                delta_color="off",
            )
        with c4:
            st.metric(
                "CVaR 95%",
                f"${cvar95:,.0f}",
                delta="avg of worst 5%",
                delta_color="off",
            )

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Diversification Ratio", f"{div_r:.3f}", delta="> 1 = diversified" if div_r > 1 else "no benefit")
        with c6:
            st.metric("Avg Max Drawdown", f"{mdd:.2f}%", delta_color="inverse")
        with c7:
            st.metric("Prob of Gain", f"{prob_g:.1f}%")
        with c8:
            st.metric("Assets", str(len(_tickers)))

        st.markdown("<br>", unsafe_allow_html=True)

        # Scenario table
        st.markdown("<p class='section-header'>Scenario Analysis</p>", unsafe_allow_html=True)

        W_T = W[:, -1]
        scenarios = {
            "Bear (5th pct)":   np.percentile(W_T, 5),
            "Bear (10th pct)":  np.percentile(W_T, 10),
            "Base (Median)":    np.percentile(W_T, 50),
            "Bull (90th pct)":  np.percentile(W_T, 90),
            "Bull (95th pct)":  np.percentile(W_T, 95),
        }
        W0_val = float(np.mean(W[:, 0]))

        scen_df = pd.DataFrame([
            {
                "Scenario":      name,
                "Portfolio $":   f"${val:,.0f}",
                "Return":        f"{(val/W0_val - 1)*100:+.2f}%",
                "vs. Initial":   "▼ LOSS" if val < W0_val else "▲ GAIN",
            }
            for name, val in scenarios.items()
        ])

        st.dataframe(
            scen_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Scenario":    st.column_config.TextColumn(width="medium"),
                "Portfolio $": st.column_config.TextColumn(width="small"),
                "Return":      st.column_config.TextColumn(width="small"),
                "vs. Initial": st.column_config.TextColumn(width="small"),
            },
        )

        # Donut
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(
            build_weight_donut(_weights, _tickers),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ── Tab 2: Paths / Fan Chart ───────────────────────────────────────────────
    with tabs[1]:
        st.markdown("<p class='section-header'>Portfolio Value Distribution</p>", unsafe_allow_html=True)
        st.plotly_chart(
            build_fan_chart(W, _n_days),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Return distribution histogram
        st.markdown("<p class='section-header'>Terminal Return Distribution</p>", unsafe_allow_html=True)
        W_T  = W[:, -1]
        W0_v = float(np.mean(W[:, 0]))
        rets = (W_T / W0_v - 1) * 100

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=rets,
            nbinsx=80,
            marker_color=ACCENT_GOLD,
            marker_line=dict(color=DARK_BG, width=0.3),
            opacity=0.85,
            name="Return distribution",
        ))
        fig_hist.add_vline(
            x=float(np.percentile(rets, 5)),
            line=dict(color=ACCENT_RED, dash="dash", width=1.5),
            annotation_text="VaR 95%",
            annotation_font=dict(color=ACCENT_RED, size=10, family="IBM Plex Mono"),
        )
        fig_hist.add_vline(
            x=float(np.median(rets)),
            line=dict(color=ACCENT_GOLD, dash="dot", width=1.5),
            annotation_text="Median",
            annotation_font=dict(color=ACCENT_GOLD, size=10, family="IBM Plex Mono"),
        )
        _hist_layout = {k: v for k, v in PLOTLY_TEMPLATE["layout"].items() if k not in ("xaxis", "yaxis")}
        fig_hist.update_layout(
            **_hist_layout,
            xaxis=dict(title="Return (%)", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=BORDER),
            yaxis=dict(title="Frequency", gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, linecolor=BORDER),
            showlegend=False,
            height=280,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 3: Correlation ─────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("<p class='section-header'>Correlation Structure</p>", unsafe_allow_html=True)
        st.plotly_chart(
            build_correlation_heatmap(log_ret_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Correlation table
        corr = log_ret_df.corr().round(4)
        st.markdown("<p class='section-header' style='margin-top:16px;'>Pairwise Coefficients</p>", unsafe_allow_html=True)
        st.dataframe(corr, use_container_width=True)

        st.markdown(
            "<p style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;color:#5A5C78;"
            "margin-top:8px;letter-spacing:0.08em;'>Estimated on annualized log-returns · "
            f"Ledoit-Wolf shrinkage · {calib_window}-day calibration window</p>",
            unsafe_allow_html=True,
        )

    # ── Tab 4: Per-Asset ───────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("<p class='section-header'>Per-Asset Simulated Paths</p>", unsafe_allow_html=True)
        st.plotly_chart(
            build_per_asset_paths(paths, _tickers, n_display=40),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Per-asset summary table
        st.markdown("<p class='section-header' style='margin-top:16px;'>Per-Asset Summary</p>", unsafe_allow_html=True)
        rows = []
        for i, ticker in enumerate(_tickers):
            ap    = paths[i]          # (K, T+1)
            ap_T  = ap[:, -1]
            ap_0  = ap[:, 0]
            r_i   = (ap_T / ap_0) - 1
            rows.append({
                "Asset":          ticker,
                "Weight":         f"{_weights[i]*100:.1f}%",
                "Current Price":  f"${float(S0_vec[i]):,.2f}",
                "Median Return":  f"{float(np.median(r_i)*100):+.2f}%",
                "Volatility":     f"{float(np.std(r_i)*100):.2f}%",
                "VaR 95%":        f"${float(np.percentile(ap_T, 5)):,.2f}",
                "Prob Gain":      f"{float(np.mean(r_i > 0)*100):.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Tab 5: Optimizer ──────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("<p class='section-header'>Weight Optimization</p>", unsafe_allow_html=True)

        opt_res = st.session_state.opt_results

        if opt_res is None:
            st.markdown(
                "<div style='text-align:center;padding:60px 0;border:1px solid #1F2240;"
                "background:#10121E;'>"
                "<p style='font-family:\"IBM Plex Mono\",monospace;font-size:11px;"
                "color:#5A5C78;letter-spacing:0.18em;text-transform:uppercase;'>"
                "Run Simulation First · Then Click ◆ Optimize Weights</p></div>",
                unsafe_allow_html=True,
            )
        else:
            opt_weights      = opt_res["opt_weights"]
            strategy_metrics = opt_res["strategy_metrics"]
            _opt_tickers     = opt_res["tickers"]

            # Comparison bar chart
            st.plotly_chart(
                build_optimizer_comparison(opt_weights, _opt_tickers, paths, S0_vec, strategy_metrics),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Strategy metrics table
            st.markdown("<p class='section-header' style='margin-top:16px;'>Strategy Comparison</p>", unsafe_allow_html=True)

            strat_rows = []
            strat_labels = {
                "max_sharpe":  "Max Sharpe",
                "min_cvar":    "Min CVaR",
                "risk_parity": "Risk Parity",
            }
            for s_key, s_label in strat_labels.items():
                w  = opt_weights.get(s_key, np.ones(len(_opt_tickers)) / len(_opt_tickers))
                sm = strategy_metrics.get(s_key, {})
                row = {"Strategy": s_label}
                for i, t in enumerate(_opt_tickers):
                    row[t] = f"{w[i]*100:.1f}%"
                row["Sharpe"]     = f"{sm.get('sharpe', 0):.3f}"
                row["Exp Return"] = f"{sm.get('expected_return', 0):+.2f}%"
                row["CVaR 95%"]   = f"${sm.get('portfolio_cvar_95', 0):,.0f}"
                row["Div Ratio"]  = f"{sm.get('diversification_ratio', 1):.3f}"
                strat_rows.append(row)

            st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

            # Three donuts side by side
            st.markdown("<p class='section-header' style='margin-top:16px;'>Allocation Breakdown</p>", unsafe_allow_html=True)
            dcols = st.columns(3)
            strategy_titles = {
                "max_sharpe":  "Max Sharpe",
                "min_cvar":    "Min CVaR",
                "risk_parity": "Risk Parity",
            }
            for i, (s_key, s_title) in enumerate(strategy_titles.items()):
                with dcols[i]:
                    w = opt_weights.get(s_key, np.ones(len(_opt_tickers)) / len(_opt_tickers))
                    st.plotly_chart(
                        build_weight_donut(w, _opt_tickers, title=s_title),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

            st.markdown(
                "<p style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;"
                "color:#5A5C78;margin-top:12px;letter-spacing:0.08em;'>"
                "Weight bounds: [1%, 40%] · "
                "Max Sharpe &amp; Min CVaR via SLSQP · Risk Parity analytic (1/σᵢ)</p>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    st.set_page_config(
        page_title="MARGINS · Portfolio Mode",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_portfolio_tab()