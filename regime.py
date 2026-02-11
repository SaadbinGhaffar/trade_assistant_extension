"""
regime.py – Market regime detection on the 1H timeframe.
Classifies the market into TREND, RANGE, or VOLATILE regimes
and returns a confidence score for the detected regime.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config import EMA_FAST, EMA_MID, EMA_SLOW
from features import compute_all_features


class Regime(Enum):
    TREND    = "Trend"
    RANGE    = "Range"
    VOLATILE = "Volatile"


@dataclass
class RegimeResult:
    """Result of regime detection on a single bar."""
    regime: Regime
    confidence: float          # 0–1
    adx_value: float
    atr_expansion: float       # current ATR / rolling median ATR
    bb_width: float
    ma_aligned: bool           # EMA 20 > 50 > 200 (bull) or reverse (bear)
    direction: Optional[str]   # "bullish" | "bearish" | None (range)


def detect_regime(df_1h: pd.DataFrame) -> RegimeResult:
    """
    Detect regime from the latest closed 1H candle.

    Rules
    -----
    - **Trend**: ADX > 25, ATR expanding (ratio > 1.1), MA alignment
    - **Range**: ADX < 20, Bollinger bandwidth contracting, oscillatory
    - **Volatile**: high ATR expansion but no directional alignment

    Parameters
    ----------
    df_1h : pd.DataFrame
        1H OHLCV data with at least 100 bars.

    Returns
    -------
    RegimeResult
    """
    # Ensure features are computed
    if "adx" not in df_1h.columns:
        compute_all_features(df_1h)

    # Use second-to-last row (latest fully closed candle)
    idx = -2 if len(df_1h) >= 2 else -1
    row = df_1h.iloc[idx]

    adx_val  = row.get("adx", 0)
    atr_val  = row.get("atr", 0)
    bb_width = row.get("bb_width", 0)

    # ATR expansion: current ATR vs 50-bar median
    atr_median = df_1h["atr"].rolling(50).median().iloc[idx]
    atr_expansion = atr_val / atr_median if atr_median and atr_median > 0 else 1.0

    # MA alignment check
    ema_f = row.get(f"ema_{EMA_FAST}", 0)
    ema_m = row.get(f"ema_{EMA_MID}", 0)
    ema_s = row.get(f"ema_{EMA_SLOW}", 0)

    bull_aligned = ema_f > ema_m > ema_s
    bear_aligned = ema_f < ema_m < ema_s
    ma_aligned = bull_aligned or bear_aligned

    direction = None
    if bull_aligned:
        direction = "bullish"
    elif bear_aligned:
        direction = "bearish"

    # ── Scoring logic ──────────────────────────────────────────
    trend_score = 0.0
    range_score = 0.0

    # ADX contribution
    if adx_val > 25:
        trend_score += 0.4
    elif adx_val > 20:
        trend_score += 0.2
    elif adx_val < 20:
        range_score += 0.4
    if adx_val < 15:
        range_score += 0.2

    # ATR expansion
    if atr_expansion > 1.2:
        trend_score += 0.25
    elif atr_expansion > 1.05:
        trend_score += 0.1
    elif atr_expansion < 0.8:
        range_score += 0.25

    # MA alignment
    if ma_aligned:
        trend_score += 0.25
    else:
        range_score += 0.15

    # Bollinger width (low = consolidation/range)
    bb_width_median = df_1h["bb_width"].rolling(50).median().iloc[idx]
    if bb_width_median and bb_width_median > 0:
        bb_ratio = bb_width / bb_width_median
        if bb_ratio < 0.8:
            range_score += 0.2
        elif bb_ratio > 1.3:
            trend_score += 0.1

    # ── Classify ───────────────────────────────────────────────
    if trend_score >= 0.6 and ma_aligned:
        regime = Regime.TREND
        confidence = min(trend_score, 1.0)
    elif range_score >= 0.5:
        regime = Regime.RANGE
        confidence = min(range_score, 1.0)
        direction = None
    elif atr_expansion > 1.3 and not ma_aligned:
        regime = Regime.VOLATILE
        confidence = 0.5 + (atr_expansion - 1.3) * 0.5
        confidence = min(confidence, 1.0)
    else:
        # Default to whichever score is higher
        if trend_score >= range_score:
            regime = Regime.TREND
            confidence = trend_score
        else:
            regime = Regime.RANGE
            confidence = range_score

    return RegimeResult(
        regime=regime,
        confidence=confidence,
        adx_value=adx_val,
        atr_expansion=round(atr_expansion, 3),
        bb_width=round(bb_width, 3),
        ma_aligned=ma_aligned,
        direction=direction,
    )
