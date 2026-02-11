"""
data_provider.py – OHLCV data fetching & caching.
Fetches candle data from yfinance, converts timestamps to PKT,
and provides a simple in-memory cache to avoid redundant API calls.
"""

import datetime as dt
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    PAIRS, PairConfig, PKT_UTC_OFFSET,
    TIMEFRAME_DAILY, TIMEFRAME_1H, TIMEFRAME_15M,
    FETCH_PERIOD_DAILY, FETCH_PERIOD_1H, FETCH_PERIOD_15M,
)

# ───────────────── Simple in-memory cache ──────────────────────
_cache: Dict[Tuple[str, str], pd.DataFrame] = {}


def _period_for_interval(interval: str) -> str:
    """Return the yfinance fetch period for a given interval."""
    return {
        TIMEFRAME_DAILY: FETCH_PERIOD_DAILY,
        TIMEFRAME_1H:    FETCH_PERIOD_1H,
        TIMEFRAME_15M:   FETCH_PERIOD_15M,
    }.get(interval, "30d")


def fetch_ohlcv(
    pair_key: str,
    interval: str = TIMEFRAME_15M,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a pair and interval.

    Parameters
    ----------
    pair_key : str
        Key into config.PAIRS (e.g. "XAUUSD").
    interval : str
        yfinance interval string ("1d", "1h", "15m").
    force_refresh : bool
        If True, bypass cache.

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex in PKT (UTC+5).
    """
    cache_key = (pair_key, interval)
    if not force_refresh and cache_key in _cache:
        return _cache[cache_key].copy()

    pair_cfg: PairConfig = PAIRS[pair_key]
    period = _period_for_interval(interval)

    ticker = yf.Ticker(pair_cfg.symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(
            f"No data returned for {pair_cfg.display_name} ({pair_cfg.symbol}) "
            f"interval={interval} period={period}"
        )

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # Convert index to PKT
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET)))
    df.index.name = "datetime_pkt"

    # Drop rows with NaN
    df.dropna(inplace=True)

    _cache[cache_key] = df.copy()
    return df


def get_latest_closed_candle(
    pair_key: str, interval: str = TIMEFRAME_15M
) -> pd.Series:
    """Return the most recent fully-closed candle as a Series."""
    df = fetch_ohlcv(pair_key, interval)
    # The last row may still be forming – use second-to-last
    if len(df) < 2:
        return df.iloc[-1]
    return df.iloc[-2]


def get_multi_timeframe_data(pair_key: str, force_refresh: bool = False):
    """
    Convenience: fetch Daily, 1H, and 15m data for a pair.

    Returns
    -------
    dict with keys "daily", "1h", "15m" → DataFrames
    """
    return {
        "daily": fetch_ohlcv(pair_key, TIMEFRAME_DAILY, force_refresh),
        "1h":    fetch_ohlcv(pair_key, TIMEFRAME_1H,    force_refresh),
        "15m":   fetch_ohlcv(pair_key, TIMEFRAME_15M,   force_refresh),
    }


def generate_demo_data(pair_key: str) -> dict:
    """
    Generate synthetic OHLCV data for demo/testing when yfinance is unavailable.
    Produces deterministic data with realistic structure.
    """
    np.random.seed(42)
    pair_cfg = PAIRS[pair_key]
    now = pd.Timestamp.now(tz=dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET)))

    base_prices = {
        "XAUUSD": 2650.0,
        "XAGUSD": 31.5,
        "EURUSD": 1.0850,
    }
    base = base_prices.get(pair_key, 100.0)
    volatility = base * 0.005  # 0.5% daily vol

    result = {}
    for tf_label, interval, periods, freq in [
        ("daily", TIMEFRAME_DAILY, 200, "1D"),
        ("1h",    TIMEFRAME_1H,    500, "1h"),
        ("15m",   TIMEFRAME_15M,   400, "15min"),
    ]:
        dates = pd.date_range(
            end=now, periods=periods, freq=freq,
            tz=dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET)),
        )
        # Random walk close
        returns = np.random.normal(0.0001, 0.003, size=periods)
        close = base * np.cumprod(1 + returns)
        # Derive OHLV from close
        spread = volatility * (0.5 if tf_label == "daily" else 0.2 if tf_label == "1h" else 0.1)
        high = close + np.abs(np.random.normal(0, spread, periods))
        low  = close - np.abs(np.random.normal(0, spread, periods))
        opn  = close + np.random.normal(0, spread * 0.3, periods)
        vol  = np.random.randint(100, 10000, size=periods).astype(float)

        df = pd.DataFrame({
            "Open": opn, "High": high, "Low": low,
            "Close": close, "Volume": vol,
        }, index=dates)
        df.index.name = "datetime_pkt"
        result[tf_label] = df

    return result
