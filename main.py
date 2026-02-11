"""
main.py – Orchestrator for the Forex Trading Assistant.
Loops over pairs, fetches multi-timeframe data, computes features,
detects regime, scores entries, checks eligibility, and renders dashboard.

Usage:
    python main.py              # Live mode (fetches real data via yfinance)
    python main.py --demo       # Demo mode (uses synthetic data)
    python main.py --pair XAUUSD  # Single pair mode
    python main.py --mc          # Run Monte Carlo simulation
"""

import argparse
import sys
import os
import datetime as dt
from typing import Optional

# Fix Windows console encoding for special characters
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ.setdefault('PYTHONUTF8', '1')

import numpy as np
import pandas as pd

from config import (
    PAIRS, ACCOUNT_SIZE, PairConfig, PKT_UTC_OFFSET,
)
from data_provider import get_multi_timeframe_data, generate_demo_data
from features import compute_all_features
from regime import detect_regime
from scoring import compute_total_score
from session_filter import classify_session, SessionInfo, get_pkt_now
from risk_manager import RiskManager
from governance import assess_eligibility
from dashboard import render_dashboard
from performance_tracker import PerformanceTracker


def analyze_pair(
    pair_key: str,
    data: dict,
    session: SessionInfo,
    risk_mgr: RiskManager,
    tracker: PerformanceTracker,
    verbose: bool = True,
) -> dict:
    """
    Run the full analysis pipeline for one pair.

    Parameters
    ----------
    pair_key : str
        Key into config.PAIRS.
    data : dict
        Multi-timeframe data {"daily": df, "1h": df, "15m": df}.
    session : SessionInfo
    risk_mgr : RiskManager
    tracker : PerformanceTracker
    verbose : bool
        If True, print full dashboard.

    Returns
    -------
    dict with analysis results.
    """
    pair_cfg: PairConfig = PAIRS[pair_key]

    # ── Step 1: Compute features on all timeframes ──
    for tf_key in ["daily", "1h", "15m"]:
        compute_all_features(data[tf_key])

    # ── Step 2: Detect regime from 1H ──
    regime = detect_regime(data["1h"])

    # ── Step 3: Compute dual-sided scores ──
    scores = compute_total_score(
        df_daily=data["daily"],
        df_1h=data["1h"],
        df_15m=data["15m"],
        regime=regime,
        session=session,
        pair_cfg=pair_cfg,
        account_size=ACCOUNT_SIZE,
    )

    # ── Step 4: Compute stops & position sizing ──
    df_15m = data["15m"]
    last_row = df_15m.iloc[-2] if len(df_15m) >= 2 else df_15m.iloc[-1]
    entry_price = last_row["Close"]
    atr = last_row.get("atr", 0)

    direction = scores.dominant_side.lower()
    stop_target = risk_mgr.compute_stop_target(
        entry_price=entry_price,
        atr=atr,
        pair_cfg=pair_cfg,
        direction=direction,
        rr_ratio=pair_cfg.min_rr,
    )
    position = risk_mgr.compute_position_size(
        pair_cfg=pair_cfg,
        stop_distance=stop_target.stop_distance,
    )

    # ── Step 5: Assess eligibility ──
    eligibility = assess_eligibility(
        scores=scores,
        session=session,
        risk_mgr=risk_mgr,
    )

    # ── Step 6: Render dashboard ──
    candle_time = str(last_row.name) if hasattr(last_row, "name") else ""

    if verbose:
        render_dashboard(
            pair_key=pair_key,
            pair_cfg=pair_cfg,
            scores=scores,
            regime=regime,
            session=session,
            eligibility=eligibility,
            stop_target=stop_target,
            position=position,
            risk_mgr=risk_mgr,
            candle_time=candle_time,
        )

    return {
        "pair": pair_key,
        "scores": scores,
        "regime": regime,
        "eligibility": eligibility,
        "stop_target": stop_target,
        "position": position,
        "candle_time": candle_time,
    }


def run_demo():
    """
    Demo mode: generate synthetic data and show dashboard for all pairs
    across 3 simulated candle closes.
    """
    print("\n" + "=" * 62)
    print("  [DEMO] DEMO MODE - Synthetic Data (3 Candle Closes)")
    print("=" * 62)

    risk_mgr = RiskManager(account_size=ACCOUNT_SIZE)
    tracker = PerformanceTracker()

    # Simulate 3 candle closes with different session times
    demo_sessions = [
        SessionInfo(name="London",  is_tradeable=True,  score_bonus=3,
                    current_time_pkt=dt.datetime(2026, 2, 11, 14, 0,
                        tzinfo=dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET)))),
        SessionInfo(name="Overlap", is_tradeable=True,  score_bonus=5,
                    current_time_pkt=dt.datetime(2026, 2, 11, 18, 0,
                        tzinfo=dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET)))),
        SessionInfo(name="Outside", is_tradeable=False, score_bonus=0,
                    current_time_pkt=dt.datetime(2026, 2, 11, 22, 0,
                        tzinfo=dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET)))),
    ]

    for candle_idx, session in enumerate(demo_sessions, 1):
        print(f"\n{'-' * 62}")
        print(f"  [#{candle_idx}] CANDLE CLOSE  |  Session: {session.name}  |  "
              f"Time: {session.current_time_pkt.strftime('%I:%M %p PKT')}")
        print(f"{'-' * 62}")

        for pair_key in PAIRS:
            data = generate_demo_data(pair_key)
            analyze_pair(
                pair_key=pair_key,
                data=data,
                session=session,
                risk_mgr=risk_mgr,
                tracker=tracker,
            )

    # Show Monte Carlo simulation
    print("\n" + "=" * 62)
    print("  [MC] MONTE CARLO RISK ANALYSIS")
    print("=" * 62)
    try:
        from advanced import MonteCarloSimulator
        mc = MonteCarloSimulator(
            win_rate=0.55,
            avg_win=1.2,
            avg_loss=-0.8,
            n_trades=50,
            n_simulations=1000,
            initial_equity=ACCOUNT_SIZE,
        )
        mc.print_summary()
    except Exception as e:
        print(f"  Monte Carlo unavailable: {e}")

    # Bayesian regime example
    print("=" * 62)
    print("  [BAYES] BAYESIAN REGIME PROBABILITIES (Example)")
    print("=" * 62)
    try:
        from advanced import BayesianRegimeModel
        model = BayesianRegimeModel()
        probs = model.compute_probabilities(adx=28, atr_expansion=1.25, bb_width_ratio=1.1)
        print(f"  ADX=28, ATR_exp=1.25, BB_ratio=1.1")
        print(f"  P(Trend):    {probs['trend_prob']*100:.1f}%")
        print(f"  P(Range):    {probs['range_prob']*100:.1f}%")
        print(f"  P(Volatile): {probs['volatile_prob']*100:.1f}%")
        print()
    except Exception as e:
        print(f"  Bayesian model unavailable: {e}")

    # Performance summary (will be empty if no trades recorded)
    print(tracker.summary_text())


def run_live(pair_filter: Optional[str] = None):
    """
    Live mode: fetch real data from yfinance, analyze, and display.
    """
    print("\n" + "=" * 62)
    print("  [LIVE] LIVE MODE - Real Market Data")
    print("=" * 62)

    session = classify_session()
    risk_mgr = RiskManager(account_size=ACCOUNT_SIZE)
    tracker = PerformanceTracker()

    print(f"  Current Time:  {session.current_time_pkt.strftime('%Y-%m-%d %I:%M %p PKT')}")
    print(f"  Session:       {session.name} ({'Tradeable' if session.is_tradeable else 'CLOSED'})")
    print()

    pairs_to_analyze = {pair_filter: PAIRS[pair_filter]} if pair_filter and pair_filter in PAIRS else PAIRS

    for pair_key in pairs_to_analyze:
        try:
            print(f"  Fetching data for {PAIRS[pair_key].display_name}...")
            data = get_multi_timeframe_data(pair_key)
            analyze_pair(
                pair_key=pair_key,
                data=data,
                session=session,
                risk_mgr=risk_mgr,
                tracker=tracker,
            )
        except Exception as e:
            print(f"  [!] Error analyzing {pair_key}: {e}")
            print(f"    Try running with --demo flag for offline testing.\n")

    print(tracker.summary_text())


def main():
    parser = argparse.ArgumentParser(
        description="Forex Trading Assistant – Semi-Automated Decision Tool"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run in demo mode with synthetic data"
    )
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Analyze a single pair (e.g., XAUUSD, XAGUSD, EURUSD)"
    )
    parser.add_argument(
        "--mc", action="store_true",
        help="Run Monte Carlo simulation only"
    )

    args = parser.parse_args()

    if args.mc:
        from advanced import MonteCarloSimulator
        mc = MonteCarloSimulator(
            win_rate=0.55,
            avg_win=1.2,
            avg_loss=-0.8,
            n_trades=100,
            n_simulations=2000,
            initial_equity=ACCOUNT_SIZE,
        )
        mc.print_summary()
        return

    if args.demo:
        run_demo()
    else:
        run_live(pair_filter=args.pair)


if __name__ == "__main__":
    main()
