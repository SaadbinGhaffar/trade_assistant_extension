"""
governance.py – Trade eligibility & blocking logic.
Enforces score thresholds, dominance spread, session limits,
drawdown caps, and news-event blocking.
"""

from dataclasses import dataclass, field
from typing import List

from config import (
    MIN_SCORE_THRESHOLD, MIN_DOMINANCE_SPREAD,
    MAX_TRADES_PER_SESSION,
)
from scoring import TotalScore
from session_filter import SessionInfo
from risk_manager import RiskManager


@dataclass
class BlockReason:
    """A single reason a trade is blocked."""
    rule: str
    detail: str


@dataclass
class EligibilityResult:
    """Full eligibility assessment for a trade."""
    is_eligible: bool
    direction: str             # "Long" | "Short" | "None"
    long_score: float
    short_score: float
    dominance_spread: float
    block_reasons: List[BlockReason] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.is_eligible:
            return f"✅ ELIGIBLE – {self.direction} ({self.long_score:.0f}L / {self.short_score:.0f}S)"
        reasons = "; ".join(r.rule for r in self.block_reasons)
        return f"🚫 BLOCKED – {reasons}"


def assess_eligibility(
    scores: TotalScore,
    session: SessionInfo,
    risk_mgr: RiskManager,
    news_upcoming: bool = False,
) -> EligibilityResult:
    """
    Determine whether a trade is eligible based on all governance rules.

    Rules checked:
    1. Session must be tradeable
    2. Score ≥ 65
    3. Dominance spread ≥ 15
    4. Max 2 trades per session
    5. Daily loss cap not breached
    6. Weekly loss cap not breached
    7. No high-impact news upcoming
    """
    blocks: List[BlockReason] = []

    # Determine dominant side
    if scores.long_total >= scores.short_total:
        direction = "Long"
        dominant_score = scores.long_total
    else:
        direction = "Short"
        dominant_score = scores.short_total

    # ── Rule 1: Session check ──
    if not session.is_tradeable:
        blocks.append(BlockReason(
            rule="SESSION_CLOSED",
            detail=f"Current session: {session.name}. Trading only allowed 12 PM - 9 PM PKT."
        ))

    # ── Rule 2: Minimum score ──
    if dominant_score < MIN_SCORE_THRESHOLD:
        blocks.append(BlockReason(
            rule="SCORE_TOO_LOW",
            detail=f"Dominant score {dominant_score:.1f} < threshold {MIN_SCORE_THRESHOLD}"
        ))

    # ── Rule 3: Dominance spread ──
    spread = scores.dominance_spread
    if spread < MIN_DOMINANCE_SPREAD:
        blocks.append(BlockReason(
            rule="LOW_DOMINANCE",
            detail=f"Spread {spread:.1f} < min {MIN_DOMINANCE_SPREAD}"
        ))

    # ── Rule 4: Session trade limit ──
    if risk_mgr.state.trades_this_session >= MAX_TRADES_PER_SESSION:
        blocks.append(BlockReason(
            rule="SESSION_LIMIT",
            detail=f"Already {risk_mgr.state.trades_this_session} trades this session (max {MAX_TRADES_PER_SESSION})"
        ))

    # ── Rule 5: Daily loss cap ──
    if risk_mgr.check_daily_loss():
        blocks.append(BlockReason(
            rule="DAILY_LOSS_CAP",
            detail=f"Daily P&L: ${risk_mgr.state.daily_pnl:.2f}"
        ))

    # ── Rule 6: Weekly loss cap ──
    if risk_mgr.check_weekly_loss():
        blocks.append(BlockReason(
            rule="WEEKLY_LOSS_CAP",
            detail=f"Weekly P&L: ${risk_mgr.state.weekly_pnl:.2f}"
        ))

    # ── Rule 7: News blocking ──
    if news_upcoming:
        blocks.append(BlockReason(
            rule="NEWS_EVENT",
            detail="High-impact news event upcoming"
        ))

    is_eligible = len(blocks) == 0

    return EligibilityResult(
        is_eligible=is_eligible,
        direction=direction if is_eligible else "None",
        long_score=scores.long_total,
        short_score=scores.short_total,
        dominance_spread=spread,
        block_reasons=blocks,
    )
