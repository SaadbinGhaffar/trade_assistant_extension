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


# ═══════════════════════════════════════════════════════════════
#  Market Structure Detection
# ═══════════════════════════════════════════════════════════════

def detect_market_structure(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    """
    Detect market structure: swing highs/lows and structure breaks.
    
    Returns DataFrame with columns:
    - swing_high: 1 if swing high, 0 otherwise
    - swing_low: 1 if swing low, 0 otherwise
    - structure: 'bullish', 'bearish', or 'neutral'
    - structure_break: 1 if BOS (break of structure), 0 otherwise
    """
    df["swing_high"] = 0
    df["swing_low"] = 0
    
    # Identify swing points
    for i in range(lookback, len(df) - lookback):
        # Swing high: highest high in window
        if df["High"].iloc[i] == df["High"].iloc[i-lookback:i+lookback+1].max():
            df.loc[df.index[i], "swing_high"] = 1
        
        # Swing low: lowest low in window
        if df["Low"].iloc[i] == df["Low"].iloc[i-lookback:i+lookback+1].min():
            df.loc[df.index[i], "swing_low"] = 1
    
    # Determine market structure
    structure = []
    structure_break = []
    
    swing_highs = df[df["swing_high"] == 1]["High"].values
    swing_lows = df[df["swing_low"] == 1]["Low"].values
    
    for i in range(len(df)):
        if i < lookback * 2:
            structure.append("neutral")
            structure_break.append(0)
            continue
        
        # Get recent swings
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
        
        # Higher highs and higher lows = bullish
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            hh = all(recent_highs[j] < recent_highs[j+1] for j in range(len(recent_highs)-1))
            hl = all(recent_lows[j] < recent_lows[j+1] for j in range(len(recent_lows)-1))
            
            # Lower highs and lower lows = bearish
            lh = all(recent_highs[j] > recent_highs[j+1] for j in range(len(recent_highs)-1))
            ll = all(recent_lows[j] > recent_lows[j+1] for j in range(len(recent_lows)-1))
            
            if hh and hl:
                current_structure = "bullish"
            elif lh and ll:
                current_structure = "bearish"
            else:
                current_structure = "neutral"
        else:
            current_structure = "neutral"
        
        # Detect structure break
        prev_structure = structure[-1] if structure else "neutral"
        is_break = 1 if current_structure != prev_structure and current_structure != "neutral" else 0
        
        structure.append(current_structure)
        structure_break.append(is_break)
    
    df["market_structure"] = structure
    df["structure_break"] = structure_break
    
    return df


# ═══════════════════════════════════════════════════════════════
#  VWAP (Volume-Weighted Average Price)
# ═══════════════════════════════════════════════════════════════

def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VWAP and standard deviation bands.
    
    VWAP = Cumulative(Price × Volume) / Cumulative(Volume)
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    
    df["vwap"] = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
    
    # Calculate VWAP standard deviation bands
    df["vwap_diff"] = typical_price - df["vwap"]
    df["vwap_std"] = df["vwap_diff"].rolling(20).std()
    df["vwap_upper"] = df["vwap"] + 2 * df["vwap_std"]
    df["vwap_lower"] = df["vwap"] - 2 * df["vwap_std"]
    
    # Position relative to VWAP (-1 to 1)
    df["vwap_position"] = (df["Close"] - df["vwap"]) / df["vwap_std"].replace(0, np.nan)
    df["vwap_position"] = df["vwap_position"].fillna(0).clip(-3, 3) / 3
    
    return df


# ═══════════════════════════════════════════════════════════════
#  Liquidity Zones (Volume-Weighted Support/Resistance)
# ═══════════════════════════════════════════════════════════════

def detect_liquidity_zones(
    df: pd.DataFrame,
    lookback: int = 50,
    num_zones: int = 3,
    tolerance_pct: float = 0.3,
) -> dict:
    """
    Detect key liquidity zones using volume-weighted swing points.
    
    Returns dict with:
        "support": list of support levels
        "resistance": list of resistance levels
        "strength": dict mapping level to strength score
    """
    if len(df) < lookback:
        return {"support": [], "resistance": [], "strength": {}}
    
    # Identify volume-weighted swing points
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(df) - 5):
        window_high = df["High"].iloc[i-lookback:i+5]
        window_low = df["Low"].iloc[i-lookback:i+5]
        
        if df["High"].iloc[i] == window_high.max():
            volume_weight = df["Volume"].iloc[i] / df["Volume"].iloc[i-lookback:i+5].mean()
            swing_highs.append({
                "price": df["High"].iloc[i],
                "volume": volume_weight,
                "index": i,
            })
        
        if df["Low"].iloc[i] == window_low.min():
            volume_weight = df["Volume"].iloc[i] / df["Volume"].iloc[i-lookback:i+5].mean()
            swing_lows.append({
                "price": df["Low"].iloc[i],
                "volume": volume_weight,
                "index": i,
            })
    
    # Cluster nearby levels
    def cluster_zones(zones, tolerance_pct):
        if not zones:
            return []
        
        zones = sorted(zones, key=lambda x: x["price"])
        clustered = []
        current_cluster = [zones[0]]
        
        for zone in zones[1:]:
            cluster_avg = np.average(
                [z["price"] for z in current_cluster],
                weights=[z["volume"] for z in current_cluster]
            )
            tolerance = cluster_avg * (tolerance_pct / 100)
            
            if abs(zone["price"] - cluster_avg) <= tolerance:
                current_cluster.append(zone)
            else:
                # Finalize cluster
                avg_price = np.average(
                    [z["price"] for z in current_cluster],
                    weights=[z["volume"] for z in current_cluster]
                )
                strength = sum(z["volume"] for z in current_cluster) * len(current_cluster)
                recency = max(z["index"] for z in current_cluster) / len(df)
                
                clustered.append({
                    "price": avg_price,
                    "strength": strength * (0.5 + recency * 0.5),
                    "touches": len(current_cluster),
                })
                current_cluster = [zone]
        
        # Add final cluster
        if current_cluster:
            avg_price = np.average(
                [z["price"] for z in current_cluster],
                weights=[z["volume"] for z in current_cluster]
            )
            strength = sum(z["volume"] for z in current_cluster) * len(current_cluster)
            recency = max(z["index"] for z in current_cluster) / len(df)
            
            clustered.append({
                "price": avg_price,
                "strength": strength * (0.5 + recency * 0.5),
                "touches": len(current_cluster),
            })
        
        return clustered
    
    resistance_zones = cluster_zones(swing_highs, tolerance_pct)
    support_zones = cluster_zones(swing_lows, tolerance_pct)
    
    # Sort by strength and take top N
    resistance_zones = sorted(resistance_zones, key=lambda x: x["strength"], reverse=True)[:num_zones]
    support_zones = sorted(support_zones, key=lambda x: x["strength"], reverse=True)[:num_zones]
    
    # Extract prices
    resistance_prices = sorted([z["price"] for z in resistance_zones], reverse=True)
    support_prices = sorted([z["price"] for z in support_zones])
    
    # Build strength map
    strength = {}
    for z in resistance_zones:
        strength[z["price"]] = z["strength"]
    for z in support_zones:
        strength[z["price"]] = z["strength"]
    
    return {
        "support": support_prices,
        "resistance": resistance_prices,
        "strength": strength,
    }


# ═══════════════════════════════════════════════════════════════
#  Momentum Oscillators
# ═══════════════════════════════════════════════════════════════

def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator (%K and %D)."""
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    
    df["stoch_k"] = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(d_period).mean()
    df["stoch_k"] = df["stoch_k"].fillna(50)
    df["stoch_d"] = df["stoch_d"].fillna(50)
    
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD (Moving Average Convergence Divergence)."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    
    return df


def add_roc(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Add Rate of Change (ROC) momentum indicator."""
    df["roc"] = ((df["Close"] - df["Close"].shift(period)) / df["Close"].shift(period) * 100).fillna(0)
    return df


# ═══════════════════════════════════════════════════════════════
#  Master: compute all features at once
# ═══════════════════════════════════════════════════════════════

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all indicator computations to a DataFrame, including new world-class features."""
    # Original features
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
    df["rejection"] = is_rejection_candle(df)
    df["reversal"] = is_reversal_candle(df)
    
    # New world-class features
    detect_market_structure(df, lookback=10)
    add_vwap(df)
    add_stochastic(df, k_period=14, d_period=3)
    add_macd(df, fast=12, slow=26, signal=9)
    add_roc(df, period=10)
    
    return df
