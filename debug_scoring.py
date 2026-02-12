
import pandas as pd
from data_provider import fetch_ohlcv
from regime import detect_regime
from scoring import compute_total_score
from governance import assess_eligibility
from risk_manager import RiskManager
from session_filter import classify_candle_session, SessionInfo
import config

# Force session hours to be open (just in case config isn't reloaded)
config.SESSION_OVERLAP_START = 0
config.SESSION_OVERLAP_END = 24

print("Fetching data for GC=F...")
try:
    df_daily = fetch_ohlcv("XAUUSD", "1d", force_refresh=True)
    df_1h = fetch_ohlcv("XAUUSD", "1h", force_refresh=True)
    df_15m = fetch_ohlcv("XAUUSD", "15m", force_refresh=True)
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

print(f"Data Loaded: Daily={len(df_daily)}, 1H={len(df_1h)}, 15m={len(df_15m)}")

if len(df_daily) < 200 or len(df_1h) < 50:
    print("Insufficient data for indicators.")
    exit()

# Test on the last 10 candles
print("\nTesting last 10 candles...")
risk_mgr = RiskManager(account_size=10000)

for i in range(len(df_15m) - 10, len(df_15m)):
    current_time = df_15m.index[i]
    
    # Slice data
    daily_slice = df_daily[df_daily.index <= current_time]
    h1_slice = df_1h[df_1h.index <= current_time]
    m15_slice = df_15m.iloc[:i+1] # up to current
    
    # Detect regime
    regime = detect_regime(h1_slice)
    
    # Session
    session = classify_candle_session(current_time)
    
    # Score
    scores = compute_total_score(
        daily_slice, h1_slice, m15_slice, regime, session,
        config.PAIRS["XAUUSD"], risk_mgr.account_size
    )
    
    # Eligibility
    eligibility = assess_eligibility(scores, session, risk_mgr)
    
    print(f"Time: {current_time}")
    print(f"  Regime: {regime.regime.name}")
    print(f"  Session: {session.name} (Tradeable: {session.is_tradeable})")
    print(f"  Long Score: {scores.long_total} | Short Score: {scores.short_total}")
    print(f"  Eligible: {eligibility.is_eligible}")
    if not eligibility.is_eligible:
        reasons = [f"{b.rule}: {b.detail}" for b in eligibility.block_reasons]
        print(f"  Block Reasons: {reasons}")
    print("-" * 50)
