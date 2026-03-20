"""
universe.py — MARGINS Portfolio Mode
Fetches and aligns price history for all assets on a common trading-day index.

Key design decisions:
- Uses yf.download() with group_by='ticker' for efficient bulk fetch
- Inner join on date index ensures all assets have identical T observations
- Normalizes timezone to UTC before joining (avoids GC=F / BTC-USD misalignment)
- Returns both full aligned history and the most recent calib_window slice
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def fetch_universe(
    tickers: list[str],
    period: str = "2y",
    calib_window: int = 126,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch and align closing prices for all tickers on a common date index.

    Parameters
    ----------
    tickers : list[str]
        Yahoo Finance ticker symbols, e.g. ['GC=F', 'SPY', 'TLT', 'BTC-USD']
    period : str
        Historical data window: '6mo' | '1y' | '2y' | '5y'
    calib_window : int
        Number of most-recent trading days used for calibration (default 126 = 6 months)

    Returns
    -------
    combined : pd.DataFrame
        Full aligned price history, shape (T_full, N), columns = tickers
    log_ret : pd.DataFrame
        Log-return DataFrame for the most recent calib_window days,
        shape (calib_window - 1, N). Used for covariance estimation.

    Raises
    ------
    ValueError
        If fewer than 2 tickers survive alignment, or calib_window > available history.
    """
    if len(tickers) < 2:
        raise ValueError("Portfolio mode requires at least 2 tickers.")

    # --- 1. Bulk download via yfinance ---
    raw = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=True,   # adjusts for splits & dividends
        progress=False,
        threads=True,
    )

    # Extract Close prices — handle both single and multi-ticker DataFrames
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        # Single ticker fallback (shouldn't happen given len check above)
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    # --- 2. Normalize timezone to UTC then strip tz (make tz-naive) ---
    # GC=F (gold futures) and BTC-USD (crypto) have different tz representations
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC").tz_localize(None)
    else:
        prices.index = prices.index.tz_localize(None)

    # Keep only the tickers we requested (yfinance may reorder columns)
    available = [t for t in tickers if t in prices.columns]
    if len(available) < 2:
        raise ValueError(
            f"Only {len(available)} tickers returned data. "
            f"Check ticker symbols: {tickers}"
        )
    prices = prices[available]

    # --- 3. Inner join — drop any date where ANY asset has missing data ---
    combined = prices.dropna(how="any")

    if len(combined) < calib_window + 10:
        raise ValueError(
            f"Only {len(combined)} aligned trading days available. "
            f"Need at least {calib_window + 10} for reliable calibration. "
            f"Try a longer period (e.g. '5y') or reduce calib_window."
        )

    # --- 4. Slice most recent calib_window days ---
    calib_prices = combined.iloc[-calib_window:]

    # --- 5. Compute log-returns (annualized space used in correlation.py) ---
    log_ret = np.log(calib_prices / calib_prices.shift(1)).dropna()

    print(
        f"[universe] Loaded {len(available)} assets | "
        f"{len(combined)} aligned trading days | "
        f"calib window: {len(log_ret)} return observations"
    )
    print(f"[universe] Date range: {combined.index[0].date()} → {combined.index[-1].date()}")
    print(f"[universe] Assets: {available}")

    return combined, log_ret


def get_current_prices(combined: pd.DataFrame) -> np.ndarray:
    """
    Return the most recent closing price for each asset as a numpy array.
    Shape: (N,)
    """
    return combined.iloc[-1].values.astype(float)


def summarize_universe(combined: pd.DataFrame, log_ret: pd.DataFrame) -> None:
    """Print a quick summary of the universe for diagnostic purposes."""
    print("\n── Universe Summary ─────────────────────────────────")
    print(f"  Assets      : {list(combined.columns)}")
    print(f"  Full history: {len(combined)} trading days")
    print(f"  Calib obs   : {len(log_ret)} log-return observations")
    print(f"  Latest prices:")
    for col in combined.columns:
        print(f"    {col:12s}: ${combined[col].iloc[-1]:>10.4f}")
    print(f"  Annualized volatilities (from log-returns × √252):")
    ann_vol = log_ret.std() * np.sqrt(252)
    for col, vol in ann_vol.items():
        print(f"    {col:12s}: {vol * 100:.2f}%")
    print("─────────────────────────────────────────────────────\n")
