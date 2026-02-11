"""
features.py – Technical indicator computation layer.
All indicators implemented natively with pandas/numpy (no TA-Lib dependency).
Every function operates on a DataFrame with OHLCV columns and returns
the DataFrame with new indicator columns appended in-place.
"""

import numpy as np
import pandas as pd

from config import (
    EMA_FAST, EMA_MID, EMA_SLOW,
    RSI_PERIOD, ADX_PERIOD, ATR_PERIOD,
    BB_PERIOD, BB_STD, BREAKOUT_LOOKBACK,
)


# ═══════════════════════════════════════════════════════════════
#  Moving Averages
# ═══════════════════════════════════════════════════════════════

def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA 20, 50, 200 to the DataFrame."""
    df[f"ema_{EMA_FAST}"]  = df["Close"].ewm(span=EMA_FAST,  adjust=False).mean()
    df[f"ema_{EMA_MID}"]   = df["Close"].ewm(span=EMA_MID,   adjust=False).mean()
    df[f"ema_{EMA_SLOW}"]  = df["Close"].ewm(span=EMA_SLOW,  adjust=False).mean()
    return df


def ema_slope(series: pd.Series, lookback: int = 5) -> pd.Series:
    """Compute slope of a series over `lookback` bars (normalized to %)."""
    return (series - series.shift(lookback)) / series.shift(lookback) * 100


# ═══════════════════════════════════════════════════════════════
#  RSI (Wilder's smoothing)
# ═══════════════════════════════════════════════════════════════

def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """Add RSI column using Wilder's exponential smoothing."""
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)
    return df


# ═══════════════════════════════════════════════════════════════
#  ATR (Average True Range)
# ═══════════════════════════════════════════════════════════════

def add_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
    """Add ATR using Wilder's smoothing."""
    high = df["High"]
    low  = df["Low"]
    prev_close = df["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["atr"] = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return df


# ═══════════════════════════════════════════════════════════════
#  ADX (Average Directional Index)
# ═══════════════════════════════════════════════════════════════

def add_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    """Add ADX, +DI, -DI columns."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Zero out when the other is larger
    cond = plus_dm > minus_dm
    plus_dm  = plus_dm.where(cond, 0)
    minus_dm = minus_dm.where(~cond, 0)

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder-smoothed
    alpha = 1 / period
    atr_smooth   = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_smooth  = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    minus_smooth = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    plus_di  = 100 * plus_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_smooth / atr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    df["plus_di"]  = plus_di
    df["minus_di"] = minus_di
    df["adx"]      = adx
    return df


# ═══════════════════════════════════════════════════════════════
#  Bollinger Bands
# ═══════════════════════════════════════════════════════════════

def add_bollinger(
    df: pd.DataFrame,
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
) -> pd.DataFrame:
    """Add Bollinger Bands (upper, middle, lower) and bandwidth."""
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()

    df["bb_upper"] = mid + num_std * std
    df["bb_mid"]   = mid
    df["bb_lower"] = mid - num_std * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    return df


# ═══════════════════════════════════════════════════════════════
#  Breakout Proximity
# ═══════════════════════════════════════════════════════════════

def add_breakout_proximity(
    df: pd.DataFrame, lookback: int = BREAKOUT_LOOKBACK
) -> pd.DataFrame:
    """
    Compute proximity of current close to 20-bar high/low.
    Returns values in [0, 1] where 1 = at the high, 0 = at the low.
    """
    rolling_high = df["High"].rolling(lookback).max()
    rolling_low  = df["Low"].rolling(lookback).min()
    rng = (rolling_high - rolling_low).replace(0, np.nan)

    df["breakout_proximity"] = (df["Close"] - rolling_low) / rng
    df["rolling_high"] = rolling_high
    df["rolling_low"]  = rolling_low
    return df


# ═══════════════════════════════════════════════════════════════
#  Volatility Percentile
# ═══════════════════════════════════════════════════════════════

def add_volatility_percentile(
    df: pd.DataFrame, lookback: int = 100
) -> pd.DataFrame:
    """ATR percentile rank over `lookback` bars (0–100)."""
    if "atr" not in df.columns:
        add_atr(df)
    df["vol_pctile"] = df["atr"].rolling(lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
        raw=False,
    )
    return df


# ═══════════════════════════════════════════════════════════════
#  Candle Pattern Helpers
# ═══════════════════════════════════════════════════════════════

def is_rejection_candle(df: pd.DataFrame) -> pd.Series:
    """
    Detect rejection / pin-bar candles.
    A bullish rejection has a long lower wick (≥ 60% of range) and small body.
    A bearish rejection has a long upper wick.
    Returns +1 (bullish rejection), -1 (bearish rejection), 0 (neither).
    """
    body   = (df["Close"] - df["Open"]).abs()
    rng    = (df["High"] - df["Low"]).replace(0, np.nan)
    upper_wick = df["High"] - df[["Close", "Open"]].max(axis=1)
    lower_wick = df[["Close", "Open"]].min(axis=1) - df["Low"]

    body_ratio = body / rng
    bull_rej = (lower_wick / rng >= 0.6) & (body_ratio <= 0.3)
    bear_rej = (upper_wick / rng >= 0.6) & (body_ratio <= 0.3)

    result = pd.Series(0, index=df.index)
    result[bull_rej] = 1
    result[bear_rej] = -1
    return result


def is_reversal_candle(df: pd.DataFrame) -> pd.Series:
    """
    Detect engulfing-type reversal candles.
    Returns +1 (bullish engulfing), -1 (bearish engulfing), 0 (neither).
    """
    prev_body = (df["Close"].shift(1) - df["Open"].shift(1))
    curr_body = (df["Close"] - df["Open"])

    bull_engulf = (prev_body < 0) & (curr_body > 0) & (curr_body.abs() > prev_body.abs())
    bear_engulf = (prev_body > 0) & (curr_body < 0) & (curr_body.abs() > prev_body.abs())

    result = pd.Series(0, index=df.index)
    result[bull_engulf] = 1
    result[bear_engulf] = -1
    return result


# ═══════════════════════════════════════════════════════════════
#  Pullback Depth (Fibonacci)
# ═══════════════════════════════════════════════════════════════

def add_pullback_depth(df: pd.DataFrame, swing_lookback: int = 20) -> pd.DataFrame:
    """
    Measure pullback depth as fraction of recent swing.
    Values near 0.382–0.618 indicate ideal pullback zone.
    """
    swing_high = df["High"].rolling(swing_lookback).max()
    swing_low  = df["Low"].rolling(swing_lookback).min()
    swing_range = (swing_high - swing_low).replace(0, np.nan)

    # How far price has pulled back from the swing high (for longs)
    df["pullback_depth_long"] = (swing_high - df["Close"]) / swing_range
    # How far price has pulled back from the swing low (for shorts)
    df["pullback_depth_short"] = (df["Close"] - swing_low) / swing_range
    return df


# ═══════════════════════════════════════════════════════════════
#  RSI Divergence (simple)
# ═══════════════════════════════════════════════════════════════

def add_rsi_divergence(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Detect simple RSI divergence.
    Bullish divergence: price makes lower low but RSI makes higher low.
    Bearish divergence: price makes higher high but RSI makes lower high.
    Returns +1 (bullish), -1 (bearish), 0 (none).
    """
    if "rsi" not in df.columns:
        add_rsi(df)

    price_min = df["Close"].rolling(lookback).min()
    rsi_min   = df["rsi"].rolling(lookback).min()
    price_max = df["Close"].rolling(lookback).max()
    rsi_max   = df["rsi"].rolling(lookback).max()

    # Current close vs min/max of window
    bull_div = (df["Close"] <= price_min * 1.002) & (df["rsi"] > rsi_min + 3)
    bear_div = (df["Close"] >= price_max * 0.998) & (df["rsi"] < rsi_max - 3)

    divergence = pd.Series(0, index=df.index)
    divergence[bull_div] = 1
    divergence[bear_div] = -1
    df["rsi_divergence"] = divergence
    return df


# ═══════════════════════════════════════════════════════════════
#  Master: compute all features at once
# ═══════════════════════════════════════════════════════════════

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicator computations to a DataFrame."""
    add_emas(df)
    add_rsi(df)
    add_atr(df)
    add_adx(df)
    add_bollinger(df)
    add_breakout_proximity(df)
    add_volatility_percentile(df)
    add_pullback_depth(df)
    add_rsi_divergence(df)
    df["ema_50_slope"] = ema_slope(df[f"ema_{EMA_MID}"])
    df["rejection"]    = is_rejection_candle(df)
    df["reversal"]     = is_reversal_candle(df)
    return df
