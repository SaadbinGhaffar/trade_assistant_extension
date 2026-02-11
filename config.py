"""
config.py – Central configuration for the Forex Trading Assistant.
All constants, pair definitions, session windows, risk params, and scoring weights.
"""

from dataclasses import dataclass, field
from typing import Dict

# ─────────────────────────── Account ───────────────────────────
ACCOUNT_SIZE = 100.0          # USD
MAX_DAILY_LOSS_PCT = 2.0      # %
MAX_WEEKLY_LOSS_PCT = 5.0     # %
CONSECUTIVE_LOSS_THRESHOLD = 2  # after N consecutive losses → halve risk

# ─────────────────────────── Timezone ──────────────────────────
PKT_UTC_OFFSET = 5            # Pakistan Standard Time = UTC+5

# ─────────────────────────── Sessions (PKT hours, 24h) ────────
# Primary: 5 PM – 9 PM PKT  (London–NY overlap)
# Secondary: 12 PM – 5 PM PKT (London session only)
SESSION_OVERLAP_START = 0    # 12 AM PKT (Testing: allow all day)
SESSION_OVERLAP_END   = 24    # 12 AM PKT
SESSION_LONDON_START  = 0     # 12 AM PKT
SESSION_LONDON_END    = 24    # 12 AM PKT

SESSION_SCORE_OVERLAP = 5
SESSION_SCORE_LONDON  = 3
SESSION_SCORE_OUTSIDE = 0

# ─────────────────────────── Pairs ─────────────────────────────
@dataclass
class PairConfig:
    """Configuration for a single Forex pair."""
    symbol: str               # yfinance ticker
    display_name: str
    pip_value: float          # value of 1 pip in quote currency
    pip_size: float           # size of 1 pip (e.g. 0.01 for XAU)
    contract_size: float      # standard lot size
    risk_pct: float           # max risk % per trade
    atr_stop_multiplier: float = 1.5
    min_rr: float = 1.5      # minimum reward:risk ratio

PAIRS: Dict[str, PairConfig] = {
    "XAUUSD": PairConfig(
        symbol="GC=F",
        display_name="Gold/USD",
        pip_value=10.0,    # Futures: $10 per point (0.1 move = $1)
        pip_size=0.1,      # Tick size
        contract_size=100, # 1 contract
        risk_pct=1.0,
        atr_stop_multiplier=1.5,
    ),
    "XAGUSD": PairConfig(
        symbol="SI=F",
        display_name="XAG/USD",
        pip_value=0.01,
        pip_size=0.001,
        contract_size=5000,    # 1 lot = 5000 oz
        risk_pct=0.50,
        atr_stop_multiplier=1.5,
    ),
    "EURUSD": PairConfig(
        symbol="EURUSD=X",
        display_name="EUR/USD",
        pip_value=0.0001,
        pip_size=0.0001,
        contract_size=100_000, # 1 standard lot
        risk_pct=1.0,
        atr_stop_multiplier=1.5,
    ),
}

# ─────────────────────────── Scoring Weights ───────────────────
WEIGHT_DAILY_BIAS       = 25
WEIGHT_REGIME_ALIGNMENT = 20
WEIGHT_ENTRY_QUALITY    = 30
WEIGHT_VOLATILITY_SESSION = 15
WEIGHT_RISK_QUALITY     = 10
TOTAL_SCORE             = 100

# ─────────────────────────── Eligibility ───────────────────────
MIN_SCORE_THRESHOLD     = 65
MIN_DOMINANCE_SPREAD    = 15
MAX_TRADES_PER_SESSION  = 2

# ─────────────────────────── Indicator Defaults ────────────────
EMA_FAST   = 20
EMA_MID    = 50
EMA_SLOW   = 200
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD  = 20
BB_STD     = 2.0
BREAKOUT_LOOKBACK = 20  # 20-day high/low breakout

# ─────────────────────────── Timeframes (yfinance intervals) ──
TIMEFRAME_DAILY = "1d"
TIMEFRAME_1H    = "1h"
TIMEFRAME_15M   = "15m"

# Data fetch periods for each timeframe
FETCH_PERIOD_DAILY = "5y"     # 5 years to ensure >200 EMA is valid
FETCH_PERIOD_1H    = "730d"   # ~2 years (yfinance limit for hourly)
FETCH_PERIOD_15M   = "60d"    # ~2 months (yfinance limit for 15m)

# ═══════════════════════════════════════════════════════════════
#  NEW: World-Class Feature Parameters
# ═══════════════════════════════════════════════════════════════

# ─────────────────────────── Pair Correlations ────────────────────
# Correlation matrix for risk management (avoid overexposure)
PAIR_CORRELATIONS = {
    ("XAUUSD", "XAGUSD"): 0.85,  # Gold and Silver highly correlated
    ("EURUSD", "GBPUSD"): 0.75,  # EUR and GBP moderately correlated
    ("EURUSD", "USDCHF"): -0.80, # EUR/USD and USD/CHF inversely correlated
}
CORRELATION_THRESHOLD = 0.7  # Block if correlation > 0.7

# ─────────────────────────── Time-of-Day Patterns ──────────────────
# Optimal trading hours per pair (PKT timezone)
OPTIMAL_HOURS = {
    "XAUUSD": [(18, 21)],  # 6-9 PM PKT (NY open)
    "EURUSD": [(12, 15)],  # 12-3 PM PKT (London open)
    "GBPUSD": [(12, 15)],  # 12-3 PM PKT (London open)
    "XAGUSD": [(18, 21)],  # 6-9 PM PKT (NY open)
}

# ─────────────────────────── Adaptive ATR Multipliers ──────────────
# Volatility-based stop sizing
ATR_MULTIPLIERS = {
    "low": 1.2,      # vol_percentile < 30
    "normal": 1.5,   # vol_percentile 30-70
    "high": 2.0,     # vol_percentile > 70
}

# ─────────────────────────── Liquidity Zones ───────────────────────
LIQUIDITY_ZONE_LOOKBACK = 50
LIQUIDITY_ZONE_COUNT = 3
LIQUIDITY_ZONE_TOLERANCE = 0.3  # % tolerance for clustering

# ─────────────────────────── Position Scaling ──────────────────────
# Score-based position sizing
POSITION_SCALE = {
    (65, 75): 0.5,   # 50% position
    (75, 85): 0.75,  # 75% position
    (85, 100): 1.0,  # 100% position
}

# ─────────────────────────── News Event Filter ─────────────────────
NEWS_BLOCK_MINUTES_BEFORE = 30
NEWS_BLOCK_MINUTES_AFTER = 30
NEWS_IMPACT_LEVELS = ["high"]  # Block only high-impact news
