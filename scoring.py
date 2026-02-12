"""
scoring.py – Dual-sided transparent scoring engine.
Computes separate Long and Short scores across 5 modules,
each contributing weighted points summing to 100.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    WEIGHT_DAILY_BIAS, WEIGHT_REGIME_ALIGNMENT,
    WEIGHT_ENTRY_QUALITY, WEIGHT_VOLATILITY_SESSION,
    WEIGHT_RISK_QUALITY, TOTAL_SCORE,
    EMA_FAST, EMA_MID, EMA_SLOW, PairConfig,
)
from regime import Regime, RegimeResult
from session_filter import SessionInfo


# ═══════════════════════════════════════════════════════════════
#  Score Containers
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModuleScore:
    """Score from a single scoring module."""
    long_score: float
    short_score: float
    max_score: float
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class TotalScore:
    """Aggregated scores across all modules."""
    long_total: float = 0.0
    short_total: float = 0.0
    daily_bias: Optional[ModuleScore] = None
    regime_alignment: Optional[ModuleScore] = None
    entry_quality: Optional[ModuleScore] = None
    volatility_session: Optional[ModuleScore] = None
    risk_quality: Optional[ModuleScore] = None

    @property
    def dominance_spread(self) -> float:
        return abs(self.long_total - self.short_total)

    @property
    def dominant_side(self) -> str:
        return "Long" if self.long_total >= self.short_total else "Short"

    @property
    def daily_bias_label(self) -> str:
        if self.daily_bias:
            return "Bullish" if self.daily_bias.long_score > self.daily_bias.short_score else "Bearish"
        return "Neutral"


# ═══════════════════════════════════════════════════════════════
#  Module 1: Daily Bias Score (25 pts)
# ═══════════════════════════════════════════════════════════════

def score_daily_bias(df_daily: pd.DataFrame) -> ModuleScore:
    """
    Score daily directional bias.

    Components (each scored for both long and short):
    - Price vs 200 EMA             (7 pts)
    - 50 EMA slope                 (6 pts)
    - ADX > 20 (trend strength)    (6 pts)
    - Near breakout high/low       (6 pts)
    """
    MAX = WEIGHT_DAILY_BIAS  # 25
    row = df_daily.iloc[-2] if len(df_daily) >= 2 else df_daily.iloc[-1]

    long_pts = 0.0
    short_pts = 0.0
    details = {}

    # 1) Price vs 200 EMA (7 pts)
    close = row["Close"]
    ema200 = row.get(f"ema_{EMA_SLOW}", close)
    pct_from_ema = (close - ema200) / ema200 * 100 if ema200 != 0 else 0

    if close > ema200:
        long_pts += min(7, 3.5 + abs(pct_from_ema) * 0.7)
        short_pts += max(0, 3.5 - abs(pct_from_ema) * 0.7)
    else:
        short_pts += min(7, 3.5 + abs(pct_from_ema) * 0.7)
        long_pts += max(0, 3.5 - abs(pct_from_ema) * 0.7)
    details["price_vs_ema200"] = f"{'Above' if close > ema200 else 'Below'} ({pct_from_ema:+.2f}%)"

    # 2) 50 EMA slope (6 pts)
    slope = row.get("ema_50_slope", 0)
    if slope > 0:
        long_pts += min(6, 3 + abs(slope) * 2)
        short_pts += max(0, 3 - abs(slope) * 2)
    else:
        short_pts += min(6, 3 + abs(slope) * 2)
        long_pts += max(0, 3 - abs(slope) * 2)
    details["ema50_slope"] = f"{slope:+.3f}%"

    # 3) ADX strength (6 pts)
    adx = row.get("adx", 0)
    plus_di = row.get("plus_di", 0)
    minus_di = row.get("minus_di", 0)

    if adx > 20:
        trend_pts = min(6, (adx - 20) * 0.3 + 3)
        if plus_di > minus_di:
            long_pts += trend_pts
            short_pts += max(0, 6 - trend_pts)
        else:
            short_pts += trend_pts
            long_pts += max(0, 6 - trend_pts)
    else:
        # Weak trend → neutral scoring
        long_pts += 3
        short_pts += 3
    details["adx"] = f"{adx:.1f} (+DI={plus_di:.1f}, -DI={minus_di:.1f})"

    # 4) Near breakout (6 pts)
    bp = row.get("breakout_proximity", 0.5)
    if bp > 0.8:
        long_pts += min(6, (bp - 0.5) * 12)
        short_pts += max(0, 6 - (bp - 0.5) * 12)
    elif bp < 0.2:
        short_pts += min(6, (0.5 - bp) * 12)
        long_pts += max(0, 6 - (0.5 - bp) * 12)
    else:
        long_pts += 3
        short_pts += 3
    details["breakout_proximity"] = f"{bp:.3f}"

    # 5) Market Structure (5 pts) - NEW
    structure = row.get("market_structure", "neutral")
    structure_break = row.get("structure_break", 0)
    
    if structure == "bullish":
        long_pts += 5
        short_pts += 1
        if structure_break == 1:
            long_pts += 2  # Bonus for fresh break
    elif structure == "bearish":
        short_pts += 5
        long_pts += 1
        if structure_break == 1:
            short_pts += 2
    else:
        long_pts += 2.5
        short_pts += 2.5
    details["market_structure"] = f"{structure} (break={structure_break})"

    # Clamp to max
    long_pts = min(long_pts, MAX + 5)  # Increased max to 30
    short_pts = min(short_pts, MAX + 5)

    # Add Label
    details["label"] = "Bullish" if long_pts > short_pts else "Bearish"

    return ModuleScore(
        long_score=round(long_pts, 1),
        short_score=round(short_pts, 1),
        max_score=MAX,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════
#  Module 2: Regime Alignment Score (20 pts)
# ═══════════════════════════════════════════════════════════════

def score_regime_alignment(
    regime: RegimeResult, df_1h: pd.DataFrame
) -> ModuleScore:
    """
    Score how well the current setup aligns with the detected regime.

    - Trend regime rewards with-trend trades, penalises counter-trend.
    - Range regime rewards mean-reversion setups equally on both sides.
    - Volatile regime caps scoring (uncertain environment).
    """
    MAX = WEIGHT_REGIME_ALIGNMENT  # 20
    long_pts = 0.0
    short_pts = 0.0
    details = {}

    row = df_1h.iloc[-2] if len(df_1h) >= 2 else df_1h.iloc[-1]

    if regime.regime == Regime.TREND:
        base = MAX * regime.confidence  # up to 20
        if regime.direction == "bullish":
            long_pts = base
            short_pts = base * 0.25  # counter-trend penalty
        elif regime.direction == "bearish":
            short_pts = base
            long_pts = base * 0.25
        else:
            long_pts = short_pts = base * 0.5
        details["regime"] = f"Trend ({regime.direction}, conf={regime.confidence:.2f})"

    elif regime.regime == Regime.RANGE:
        # Range rewards both sides depending on position within range
        rsi = row.get("rsi", 50)
        bb_upper = row.get("bb_upper", 0)
        bb_lower = row.get("bb_lower", 0)
        close = row["Close"]

        base = MAX * regime.confidence * 0.8  # slightly lower base for range
        if bb_upper and bb_lower and (bb_upper - bb_lower) > 0:
            position = (close - bb_lower) / (bb_upper - bb_lower)
        else:
            position = 0.5

        # Near bottom of range → favor long
        if position < 0.3:
            long_pts = base * 0.9
            short_pts = base * 0.3
        elif position > 0.7:
            short_pts = base * 0.9
            long_pts = base * 0.3
        else:
            long_pts = base * 0.5
            short_pts = base * 0.5
        details["regime"] = f"Range (pos={position:.2f}, conf={regime.confidence:.2f})"

    else:  # VOLATILE
        # Cap at 40% of max
        cap = MAX * 0.4
        long_pts = cap * 0.5
        short_pts = cap * 0.5
        details["regime"] = f"Volatile (ATR_exp={regime.atr_expansion})"

    details["adx"] = f"{regime.adx_value:.1f}"
    details["ma_aligned"] = str(regime.ma_aligned)

    return ModuleScore(
        long_score=round(min(long_pts, MAX), 1),
        short_score=round(min(short_pts, MAX), 1),
        max_score=MAX,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════
#  Module 3: Entry Quality Score (30 pts)
# ═══════════════════════════════════════════════════════════════

def score_entry_quality(
    df_15m: pd.DataFrame,
    regime: RegimeResult,
) -> ModuleScore:
    """
    Score entry quality on the 15m chart.

    - Trend regime → Pullback scoring (depth, RSI, rejection candle, volume)
    - Range regime → Mean reversion (BB touch, RSI extreme, divergence, reversal)
    """
    MAX = WEIGHT_ENTRY_QUALITY  # 30
    row = df_15m.iloc[-2] if len(df_15m) >= 2 else df_15m.iloc[-1]

    long_pts = 0.0
    short_pts = 0.0
    details = {}

    if regime.regime == Regime.TREND:
        # ── Trend Pullback Scoring ──
        # a) Pullback depth 38–61%: ideal zone (10 pts)
        pb_long  = row.get("pullback_depth_long",  0)
        pb_short = row.get("pullback_depth_short", 0)

        def pullback_score(depth: float) -> float:
            if 0.35 <= depth <= 0.65:
                # Peak score at 50% retracement
                return 10 * (1 - 2 * abs(depth - 0.5))
            elif 0.2 <= depth < 0.35 or 0.65 < depth <= 0.8:
                return 4.0
            return 1.0

        long_pts += pullback_score(pb_long)
        short_pts += pullback_score(pb_short)
        details["pullback_long"]  = f"{pb_long:.3f}"
        details["pullback_short"] = f"{pb_short:.3f}"

        # b) RSI in sweet zone (7 pts)
        rsi = row.get("rsi", 50)
        if 40 <= rsi <= 50:
            long_pts += 7  # bullish pullback RSI zone
        elif 35 <= rsi < 40 or 50 < rsi <= 55:
            long_pts += 4
        else:
            long_pts += 1

        if 50 <= rsi <= 60:
            short_pts += 7  # bearish pullback RSI zone
        elif 45 <= rsi < 50 or 60 < rsi <= 65:
            short_pts += 4
        else:
            short_pts += 1
        details["rsi"] = f"{rsi:.1f}"

        # c) Rejection candle quality (7 pts)
        rej = row.get("rejection", 0)
        if rej == 1:
            long_pts += 7
            short_pts += 1
        elif rej == -1:
            short_pts += 7
            long_pts += 1
        else:
            long_pts += 2
            short_pts += 2
        details["rejection"] = f"{rej:+.0f}"

        # d) Volume confirmation (6 pts)
        vol = row.get("Volume", 0)
        vol_mean = df_15m["Volume"].rolling(20).mean().iloc[-2] if len(df_15m) >= 22 else df_15m["Volume"].mean()
        if vol_mean and vol_mean > 0:
            vol_ratio = vol / vol_mean
            vol_score = min(6, vol_ratio * 3)
        else:
            vol_score = 3.0
        long_pts += vol_score
        short_pts += vol_score
        details["volume_ratio"] = f"{vol / vol_mean:.2f}" if vol_mean and vol_mean > 0 else "N/A"

    elif regime.regime == Regime.RANGE:
        # ── Mean Reversion Scoring ──
        close    = row["Close"]
        bb_upper = row.get("bb_upper", close)
        bb_lower = row.get("bb_lower", close)
        rsi      = row.get("rsi", 50)
        div      = row.get("rsi_divergence", 0)
        rev      = row.get("reversal", 0)

        # a) Bollinger outer band touch (10 pts)
        bb_range = bb_upper - bb_lower if (bb_upper - bb_lower) > 0 else 1
        upper_dist = (bb_upper - close) / bb_range
        lower_dist = (close - bb_lower) / bb_range

        if lower_dist < 0.05:
            long_pts += 10
            short_pts += 1
        elif lower_dist < 0.15:
            long_pts += 6
            short_pts += 2
        else:
            long_pts += 2

        if upper_dist < 0.05:
            short_pts += 10
            long_pts += 1
        elif upper_dist < 0.15:
            short_pts += 6
            long_pts += 2
        else:
            short_pts += 2
        details["bb_position"] = f"upper_dist={upper_dist:.3f}, lower_dist={lower_dist:.3f}"

        # b) RSI extreme (7 pts)
        if rsi < 30:
            long_pts += 7
            short_pts += 1
        elif rsi < 40:
            long_pts += 4
            short_pts += 2
        elif rsi > 70:
            short_pts += 7
            long_pts += 1
        elif rsi > 60:
            short_pts += 4
            long_pts += 2
        else:
            long_pts += 2
            short_pts += 2
        details["rsi"] = f"{rsi:.1f}"

        # c) Divergence (7 pts)
        if div == 1:
            long_pts += 7
            short_pts += 1
        elif div == -1:
            short_pts += 7
            long_pts += 1
        else:
            long_pts += 2
            short_pts += 2
        details["divergence"] = f"{div:+.0f}"

        # d) Reversal candle (6 pts)
        if rev == 1:
            long_pts += 6
            short_pts += 1
        elif rev == -1:
            short_pts += 6
            long_pts += 1
        else:
            long_pts += 2
            short_pts += 2
        details["reversal_candle"] = f"{rev:+.0f}"

    # ── Universal Entry Enhancements (apply to all regimes) ──
    
    # e) VWAP Position (3 pts) - NEW
    vwap_pos = row.get("vwap_position", 0)
    if vwap_pos > 0.3:  # Above VWAP = bullish
        long_pts += min(3, vwap_pos * 3)
        short_pts += max(0, 3 - vwap_pos * 3)
    elif vwap_pos < -0.3:  # Below VWAP = bearish
        short_pts += min(3, abs(vwap_pos) * 3)
        long_pts += max(0, 3 - abs(vwap_pos) * 3)
    else:
        long_pts += 1.5
        short_pts += 1.5
    details["vwap_position"] = f"{vwap_pos:+.2f}"
    
    # f) Momentum Confirmation (3 pts) - NEW
    stoch_k = row.get("stoch_k", 50)
    macd_hist = row.get("macd_histogram", 0)
    
    # Stochastic oversold/overbought
    if stoch_k < 30:
        long_pts += 1.5
    elif stoch_k > 70:
        short_pts += 1.5
    else:
        long_pts += 0.5
        short_pts += 0.5
    
    # MACD histogram direction
    if macd_hist > 0:
        long_pts += 1.5
    elif macd_hist < 0:
        short_pts += 1.5
    else:
        long_pts += 0.5
        short_pts += 0.5
    
    details["momentum"] = f"Stoch={stoch_k:.1f}, MACD_H={macd_hist:.4f}"

    if regime.regime == Regime.VOLATILE:
        # Volatile regime → reduce both scores
        long_pts = MAX * 0.2
        short_pts = MAX * 0.2
        details["note"] = "Volatile regime – entry quality capped"

    return ModuleScore(
        long_score=round(min(long_pts, MAX), 1),
        short_score=round(min(short_pts, MAX), 1),
        max_score=MAX,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════
#  Module 4: Volatility & Session Score (15 pts)
# ═══════════════════════════════════════════════════════════════

def score_volatility_session(
    df_15m: pd.DataFrame,
    session: SessionInfo,
) -> ModuleScore:
    """
    Score based on session timing, volatility percentile, and spread.

    - Session bonus (5 pts): Overlap +5, London +3, Outside 0
    - Volatility percentile 40–70 is ideal (5 pts)
    - Spread/liquidity estimate (5 pts)
    """
    MAX = WEIGHT_VOLATILITY_SESSION  # 15
    row = df_15m.iloc[-2] if len(df_15m) >= 2 else df_15m.iloc[-1]

    # Common score (session is not directional)
    score = 0.0
    details = {}

    # 1) Session bonus (5 pts max after normalization)
    session_pts = session.score_bonus  # 0, 3, or 5
    score += session_pts
    details["session"] = f"{session.name} (+{session_pts})"

    # 2) Volatility percentile (5 pts) – sweet spot 40–70
    vol_pct = row.get("vol_pctile", 50)
    if 40 <= vol_pct <= 70:
        vol_pts = 5.0
    elif 25 <= vol_pct < 40 or 70 < vol_pct <= 85:
        vol_pts = 3.0
    elif vol_pct > 85:
        vol_pts = 1.0  # too volatile
    else:
        vol_pts = 2.0  # too quiet
    score += vol_pts
    details["vol_percentile"] = f"{vol_pct:.1f} ({vol_pts:.0f}pts)"

    # 3) Spread / liquidity proxy (5 pts)
    # Use high-low range vs ATR as a proxy for spread quality
    atr = row.get("atr", 1)
    hl_range = row["High"] - row["Low"]
    if atr > 0:
        range_ratio = hl_range / atr
        if 0.7 <= range_ratio <= 1.5:
            spread_pts = 5.0
        elif 0.4 <= range_ratio < 0.7:
            spread_pts = 3.0
        else:
            spread_pts = 2.0
    else:
        spread_pts = 2.5
    score += spread_pts
    details["spread_quality"] = f"HL/ATR={hl_range/atr:.2f}" if atr > 0 else "N/A"

    # 4) Time-of-Day Pattern (2 pts) - NEW
    # This would require current hour from session, placeholder for now
    # In production, check if current hour in OPTIMAL_HOURS for the pair
    time_bonus = 0.0  # Placeholder - implement in main.py with actual time
    score += time_bonus
    details["time_pattern"] = "N/A (implement in main)"

    score = min(score, MAX + 2)  # Increased max to 17

    return ModuleScore(
        long_score=round(score, 1),
        short_score=round(score, 1),  # session/vol is non-directional
        max_score=MAX,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════
#  Module 5: Risk Quality Score (10 pts)
# ═══════════════════════════════════════════════════════════════

def score_risk_quality(
    df_15m: pd.DataFrame,
    pair_cfg: PairConfig,
    account_size: float,
) -> ModuleScore:
    """
    Score the risk/reward quality of the current setup.

    - R:R ≥ 1.5 (4 pts)
    - Stop size reasonable / ATR-based (3 pts)
    - Risk ≤ 1% of account (3 pts)
    """
    MAX = WEIGHT_RISK_QUALITY  # 10
    row = df_15m.iloc[-2] if len(df_15m) >= 2 else df_15m.iloc[-1]

    long_pts = 0.0
    short_pts = 0.0
    details = {}

    atr = row.get("atr", 0)
    close = row["Close"]

    # ATR-based stop distance
    stop_dist = atr * pair_cfg.atr_stop_multiplier
    target_dist = stop_dist * pair_cfg.min_rr  # R:R target

    # 1) Reward/Risk ratio feasibility (4 pts)
    # Check if price structure allows the target
    recent_high = df_15m["High"].iloc[-20:].max() if len(df_15m) >= 20 else df_15m["High"].max()
    recent_low  = df_15m["Low"].iloc[-20:].min() if len(df_15m) >= 20 else df_15m["Low"].min()

    long_room  = recent_high - close
    short_room = close - recent_low

    # Long R:R
    if stop_dist > 0:
        long_rr = long_room / stop_dist
        short_rr = short_room / stop_dist
    else:
        long_rr = short_rr = 0

    if long_rr >= 2.0:
        long_pts += 4
    elif long_rr >= 1.5:
        long_pts += 3
    elif long_rr >= 1.0:
        long_pts += 1.5
    details["long_rr"] = f"{long_rr:.2f}"

    if short_rr >= 2.0:
        short_pts += 4
    elif short_rr >= 1.5:
        short_pts += 3
    elif short_rr >= 1.0:
        short_pts += 1.5
    details["short_rr"] = f"{short_rr:.2f}"

    # 2) Stop size reasonable (3 pts)
    # Stop should be 0.5×–2.5× ATR — not too tight, not too wide
    stop_atr_ratio = pair_cfg.atr_stop_multiplier
    if 1.0 <= stop_atr_ratio <= 2.0:
        stop_quality = 3.0
    elif 0.5 <= stop_atr_ratio < 1.0 or 2.0 < stop_atr_ratio <= 3.0:
        stop_quality = 2.0
    else:
        stop_quality = 1.0
    long_pts += stop_quality
    short_pts += stop_quality
    details["stop_dist"] = f"{stop_dist:.5f}"
    details["atr_multiplier"] = f"{stop_atr_ratio:.1f}x"

    # 3) Risk ≤ 1% (3 pts)
    risk_dollar = account_size * (pair_cfg.risk_pct / 100)
    if pair_cfg.risk_pct <= 1.0:
        risk_pts = 3.0
    elif pair_cfg.risk_pct <= 1.5:
        risk_pts = 2.0
    else:
        risk_pts = 1.0
    long_pts += risk_pts
    short_pts += risk_pts
    details["risk_pct"] = f"{pair_cfg.risk_pct}%"
    details["risk_dollar"] = f"${risk_dollar:.2f}"

    return ModuleScore(
        long_score=round(min(long_pts, MAX), 1),
        short_score=round(min(short_pts, MAX), 1),
        max_score=MAX,
        details=details,
    )


# ═══════════════════════════════════════════════════════════════
#  Master Scorer: aggregate all modules
# ═══════════════════════════════════════════════════════════════

def compute_total_score(
    df_daily: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    regime: RegimeResult,
    session: SessionInfo,
    pair_cfg: PairConfig,
    account_size: float,
) -> TotalScore:
    """
    Compute the full dual-sided score across all 5 modules.

    Returns a TotalScore with all sub-scores and totals.
    """
    db   = score_daily_bias(df_daily)
    ra   = score_regime_alignment(regime, df_1h)
    eq   = score_entry_quality(df_15m, regime)
    vs   = score_volatility_session(df_15m, session)
    rq   = score_risk_quality(df_15m, pair_cfg, account_size)

    total = TotalScore(
        long_total=round(db.long_score + ra.long_score + eq.long_score + vs.long_score + rq.long_score, 1),
        short_total=round(db.short_score + ra.short_score + eq.short_score + vs.short_score + rq.short_score, 1),
        daily_bias=db,
        regime_alignment=ra,
        entry_quality=eq,
        volatility_session=vs,
        risk_quality=rq,
    )

    return total
