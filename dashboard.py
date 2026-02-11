"""
dashboard.py – Console dashboard output for the trading assistant.
Renders a transparent, color-coded dashboard for each pair/candle close
showing all score components, position sizing, stop/target, and eligibility.
"""

import sys
import os
from typing import Optional

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
except ImportError:
    # Fallback: no-op color codes
    class _NoColor:
        def __getattr__(self, _): return ""
    Fore = Style = _NoColor()

from config import PairConfig
from scoring import TotalScore
from regime import RegimeResult
from session_filter import SessionInfo
from risk_manager import StopTarget, PositionSize, RiskManager
from governance import EligibilityResult


def _color_score(score: float, max_score: float) -> str:
    """Return color-coded score string."""
    pct = (score / max_score * 100) if max_score > 0 else 0
    if pct >= 65:
        return f"{Fore.GREEN}{score:.1f}{Style.RESET_ALL}"
    elif pct >= 50:
        return f"{Fore.YELLOW}{score:.1f}{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}{score:.1f}{Style.RESET_ALL}"


def _bar(score: float, max_score: float, width: int = 20) -> str:
    """Render a visual score bar."""
    filled = int(score / max_score * width) if max_score > 0 else 0
    filled = min(filled, width)
    return "#" * filled + "." * (width - filled)


def render_dashboard(
    pair_key: str,
    pair_cfg: PairConfig,
    scores: TotalScore,
    regime: RegimeResult,
    session: SessionInfo,
    eligibility: EligibilityResult,
    stop_target: Optional[StopTarget],
    position: Optional[PositionSize],
    risk_mgr: RiskManager,
    candle_time: str = "",
) -> str:
    """
    Render a full dashboard string for one pair at one candle close.

    Returns the formatted string (also prints to stdout).
    """
    sep = "=" * 62
    thin = "-" * 62

    lines = []
    lines.append(f"\n{Fore.CYAN}{sep}")
    lines.append(f"  FOREX TRADING ASSISTANT - CANDLE CLOSE ANALYSIS")
    lines.append(f"{sep}{Style.RESET_ALL}")

    # Header
    lines.append(f"  {Fore.WHITE}PAIR:{Style.RESET_ALL}       {Fore.CYAN}{pair_cfg.display_name}{Style.RESET_ALL}")
    lines.append(f"  {Fore.WHITE}TIMEFRAME:{Style.RESET_ALL}   15m")
    lines.append(f"  {Fore.WHITE}SESSION:{Style.RESET_ALL}     {session.name} (+{session.score_bonus} pts)")
    if candle_time:
        lines.append(f"  {Fore.WHITE}CANDLE:{Style.RESET_ALL}      {candle_time}")
    lines.append(f"  {thin}")

    # ── Main Scores ──
    long_c  = _color_score(scores.long_total, 100)
    short_c = _color_score(scores.short_total, 100)
    lines.append(f"  {Fore.WHITE}LONG SCORE:{Style.RESET_ALL}   {long_c} / 100   {_bar(scores.long_total, 100)}")
    lines.append(f"  {Fore.WHITE}SHORT SCORE:{Style.RESET_ALL}  {short_c} / 100   {_bar(scores.short_total, 100)}")
    lines.append(f"  {Fore.WHITE}Dominance:{Style.RESET_ALL}    {scores.dominance_spread:.1f}  ->  {Fore.YELLOW}{scores.dominant_side}{Style.RESET_ALL}")
    lines.append(f"  {thin}")

    # ── Module Breakdown ──
    lines.append(f"  {Fore.WHITE}{'MODULE':<28} {'LONG':>7} {'SHORT':>7} {'MAX':>5}{Style.RESET_ALL}")
    lines.append(f"  {'-'*50}")

    modules = [
        ("Daily Bias", scores.daily_bias),
        ("Regime Alignment", scores.regime_alignment),
        ("Entry Quality", scores.entry_quality),
        ("Volatility & Session", scores.volatility_session),
        ("Risk Quality", scores.risk_quality),
    ]
    for name, mod in modules:
        if mod:
            l = _color_score(mod.long_score, mod.max_score)
            s = _color_score(mod.short_score, mod.max_score)
            lines.append(f"  {name:<28} {l:>16} {s:>16} /{mod.max_score:.0f}")

    lines.append(f"  {thin}")

    # ── Detailed Labels ──
    db_label = scores.daily_bias_label
    db = scores.daily_bias
    db_score = max(db.long_score, db.short_score) if db else 0
    lines.append(f"  Daily Bias:       {Fore.YELLOW}{db_label}{Style.RESET_ALL} | Score: {db_score:.1f} / 25")

    r_label = regime.regime.value
    ra = scores.regime_alignment
    ra_score = max(ra.long_score, ra.short_score) if ra else 0
    lines.append(f"  Regime:           {Fore.YELLOW}{r_label}{Style.RESET_ALL} | Score: {ra_score:.1f} / 20")

    eq = scores.entry_quality
    eq_score = max(eq.long_score, eq.short_score) if eq else 0
    lines.append(f"  Entry Quality:    {eq_score:.1f} / 30")

    vs = scores.volatility_session
    vs_score = max(vs.long_score, vs.short_score) if vs else 0
    lines.append(f"  Vol & Session:    {vs_score:.1f} / 15")

    rq = scores.risk_quality
    rq_score = max(rq.long_score, rq.short_score) if rq else 0
    lines.append(f"  Risk Quality:     {rq_score:.1f} / 10")

    lines.append(f"  {thin}")

    # ── Position Sizing & Stops ──
    if position and stop_target:
        lines.append(f"  {Fore.WHITE}POSITION SIZING{Style.RESET_ALL}")
        lines.append(f"  Lot Size:    {position.lot_size:.2f}")
        lines.append(f"  Risk:        ${position.risk_dollars:.2f} ({position.risk_pct:.2f}%)")
        if position.adjusted_for_consecutive_loss:
            lines.append(f"  {Fore.RED}[!] Risk reduced 50% (consecutive losses){Style.RESET_ALL}")
        lines.append(f"  Stop Loss:   {stop_target.stop_price:.5f}  (ATR x {stop_target.atr_multiplier})")
        lines.append(f"  Take Profit: {stop_target.target_price:.5f}  (R:{stop_target.rr_ratio:.1f})")
        lines.append(f"  Entry:       {stop_target.entry_price:.5f}")
        lines.append(f"  {thin}")

    # ── Risk State ──
    rs = risk_mgr.state
    lines.append(f"  {Fore.WHITE}RISK STATE{Style.RESET_ALL}")
    lines.append(f"  Daily P&L:          ${rs.daily_pnl:.2f}")
    lines.append(f"  Weekly P&L:         ${rs.weekly_pnl:.2f}")
    lines.append(f"  Trades (session):   {rs.trades_this_session}")
    lines.append(f"  Consec Losses:      {rs.consecutive_losses}")
    if rs.risk_reduction_active:
        lines.append(f"  {Fore.RED}[!] Risk reduction ACTIVE (50%){Style.RESET_ALL}")
    lines.append(f"  {thin}")

    # ── Eligibility ──
    if eligibility.is_eligible:
        status_str = f"{Fore.GREEN}[OK] ELIGIBLE - {eligibility.direction}{Style.RESET_ALL}"
    else:
        reasons = " | ".join(b.rule for b in eligibility.block_reasons)
        status_str = f"{Fore.RED}[X] BLOCKED - {reasons}{Style.RESET_ALL}"

    lines.append(f"  {Fore.WHITE}TRADE STATUS:{Style.RESET_ALL}  {status_str}")

    if not eligibility.is_eligible:
        for br in eligibility.block_reasons:
            lines.append(f"    -> {Fore.RED}{br.rule}:{Style.RESET_ALL} {br.detail}")

    lines.append(f"{Fore.CYAN}{sep}{Style.RESET_ALL}\n")

    output = "\n".join(lines)
    print(output)
    return output
