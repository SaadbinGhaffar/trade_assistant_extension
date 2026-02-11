"""
server.py – Local API server for the TradingView Chrome Extension.
Exposes the trading assistant logic via JSON API.
"""
import sys
import os

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import datetime as dt

# Ensure current directory is in path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import PAIRS, ACCOUNT_SIZE, PairConfig, PKT_UTC_OFFSET
from data_provider import get_multi_timeframe_data, generate_demo_data
from session_filter import classify_session, SessionInfo
from risk_manager import RiskManager
from performance_tracker import PerformanceTracker
from main import analyze_pair

app = Flask(__name__)
CORS(app)  # Enable CORS for browser extension access

# Shared instances (recreated per request in CLI, but kept alive here)
risk_mgr = RiskManager(account_size=ACCOUNT_SIZE)
tracker = PerformanceTracker()


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "ok", "service": "Forex Assistant API"})


@app.route('/analyze', methods=['GET'])
def analyze():
    """
    Analyze a pair and return results in JSON.
    Query params:
      pair (required): e.g. "XAUUSD"
      demo (optional): "true" to use synthetic data
    """
    pair_input = request.args.get('pair', '').upper()
    use_demo = request.args.get('demo', 'false').lower() == 'true'

    # Map common TradingView tickers to our keys
    # TradingView might send "XAUUSD", "GOLD", "GC1!", etc.
    # Simple mapping logic:
    pair_key = None
    if "XAU" in pair_input or "GOLD" in pair_input or "GC" in pair_input:
        pair_key = "XAUUSD"
    elif "XAG" in pair_input or "SILVER" in pair_input or "SI" in pair_input:
        pair_key = "XAGUSD"
    elif "EUR" in pair_input:
        pair_key = "EURUSD"
    
    if not pair_key:
        return jsonify({"error": f"Unsupported pair: {pair_input}"}), 400

    try:
        session = classify_session()
        
        if use_demo:
            data = generate_demo_data(pair_key)
        else:
            data = get_multi_timeframe_data(pair_key)

        # Run analysis (same logic as main.py)
        # analyze_pair prints to stdout, but we also want the return dict
        # We suppress stdout to keep server logs clean? Optionally.
        result = analyze_pair(
            pair_key=pair_key,
            data=data,
            session=session,
            risk_mgr=risk_mgr,
            tracker=tracker,
            verbose=False,  # Don't print dashboard to server console
        )

        scores = result["scores"]
        regime = result["regime"]
        eligibility = result["eligibility"]
        stop_target = result["stop_target"]
        position = result["position"]

        # Serialize for JSON response
        response = {
            "pair": pair_key,
            "session": {
                "name": session.name,
                "is_tradeable": session.is_tradeable,
                "score_bonus": session.score_bonus,
                "time": session.current_time_pkt.strftime("%I:%M %p"),
            },
            "scores": {
                "long_total": scores.long_total,
                "short_total": scores.short_total,
                "dominant_side": scores.dominant_side,
                "dominance_spread": round(scores.dominance_spread, 1),
                "breakdown": {
                    "daily_bias": _extract_score(scores.daily_bias),
                    "regime_alignment": _extract_score(scores.regime_alignment),
                    "entry_quality": _extract_score(scores.entry_quality),
                    "volatility_session": _extract_score(scores.volatility_session),
                    "risk_quality": _extract_score(scores.risk_quality),
                }
            },
            "regime": {
                "type": regime.regime.value,
                "confidence": round(regime.confidence, 2),
                "direction": regime.direction,
            },
            "eligibility": {
                "is_eligible": eligibility.is_eligible,
                "direction": eligibility.direction,
                "block_reasons": [b.rule for b in eligibility.block_reasons],
            },
            "trade_setup": {
                "direction": eligibility.direction,
                "entry": stop_target.entry_price,
                "stop": stop_target.stop_price,
                "target": stop_target.target_price,
                "risk_reward": stop_target.rr_ratio,
                "lot_size": position.lot_size,
                "risk_amount": position.risk_dollars,
                "risk_pct": position.risk_pct,
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _extract_score(mod):
    if not mod:
        return None
    return {
        "long": mod.long_score,
        "short": mod.short_score,
        "max": mod.max_score,
        "details": mod.details
    }


if __name__ == '__main__':
    print("Market Assistant API running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
