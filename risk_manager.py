"""
risk_manager.py - Position sizing, drawdown tracking, and risk controls.
Computes ATR-adjusted stops, dynamic lot sizes, and enforces
daily/weekly loss caps with consecutive-loss risk reduction.
"""

from dataclasses import dataclass
from typing import List, Optional

from config import (
    ACCOUNT_SIZE, MAX_DAILY_LOSS_PCT, MAX_WEEKLY_LOSS_PCT,
    CONSECUTIVE_LOSS_THRESHOLD, PairConfig,
)


@dataclass
class StopTarget:
    """ATR-adjusted stop and target levels."""
    entry_price: float
    stop_price: float
    target_price: float
    stop_distance: float      # in price units
    target_distance: float
    atr_value: float
    atr_multiplier: float
    rr_ratio: float           # reward-to-risk ratio


@dataclass
class PositionSize:
    """Calculated position size."""
    lot_size: float
    risk_dollars: float
    risk_pct: float
    stop_pips: float
    pip_value_per_lot: float
    adjusted_for_consecutive_loss: bool = False


@dataclass
class RiskState:
    """Current risk management state."""
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    consecutive_losses: int = 0
    trades_today: int = 0
    trades_this_session: int = 0
    daily_loss_limit_hit: bool = False
    weekly_loss_limit_hit: bool = False
    risk_reduction_active: bool = False


class RiskManager:
    """
    Manages position sizing, stop/target calculation, and drawdown tracking.
    """

    def __init__(self, account_size: float = ACCOUNT_SIZE):
        self.account_size = account_size
        self.state = RiskState()
        self._trade_log: List[dict] = []

    # ─────────────────────── Stop & Target ─────────────────────

    def compute_stop_target(
        self,
        entry_price: float,
        atr: float,
        pair_cfg: PairConfig,
        direction: str = "long",
        rr_ratio: float = 1.5,
    ) -> StopTarget:
        """
        Compute ATR-adjusted stop and target prices.

        Parameters
        ----------
        entry_price : float
        atr : float
            Current ATR value from the 15m chart.
        pair_cfg : PairConfig
        direction : str
            "long" or "short"
        rr_ratio : float
            Minimum reward:risk ratio (default 1.5).
        """
        stop_dist = atr * pair_cfg.atr_stop_multiplier
        target_dist = stop_dist * rr_ratio

        if direction == "long":
            stop_price   = entry_price - stop_dist
            target_price = entry_price + target_dist
        else:
            stop_price   = entry_price + stop_dist
            target_price = entry_price - target_dist

        return StopTarget(
            entry_price=round(entry_price, 5),
            stop_price=round(stop_price, 5),
            target_price=round(target_price, 5),
            stop_distance=round(stop_dist, 5),
            target_distance=round(target_dist, 5),
            atr_value=round(atr, 5),
            atr_multiplier=pair_cfg.atr_stop_multiplier,
            rr_ratio=rr_ratio,
        )

    # ─────────────────────── Position Sizing ───────────────────

    def compute_position_size(
        self,
        pair_cfg: PairConfig,
        stop_distance: float,
    ) -> PositionSize:
        """
        Dynamic lot sizing: (account × risk%) / (stop_pips × pip_value_per_lot).

        Applies consecutive-loss risk reduction if active.
        """
        risk_pct = pair_cfg.risk_pct

        # Consecutive loss reduction: halve risk after N losses
        adjusted = False
        if self.state.consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD:
            risk_pct *= 0.5
            adjusted = True

        risk_dollars = self.account_size * (risk_pct / 100)

        # Convert stop distance to pips
        stop_pips = stop_distance / pair_cfg.pip_size

        # Pip value per standard lot
        pip_value_per_lot = pair_cfg.pip_size * pair_cfg.contract_size

        # Lot size
        if pip_value_per_lot > 0 and stop_pips > 0:
            lot_size = risk_dollars / (stop_pips * pip_value_per_lot)
        else:
            lot_size = 0.01  # minimum fallback

        # Clamp to reasonable bounds
        lot_size = max(0.01, round(lot_size, 2))

        return PositionSize(
            lot_size=lot_size,
            risk_dollars=round(risk_dollars, 2),
            risk_pct=risk_pct,
            stop_pips=round(stop_pips, 1),
            pip_value_per_lot=pip_value_per_lot,
            adjusted_for_consecutive_loss=adjusted,
        )

    # ─────────────────────── Drawdown Checks ───────────────────

    def check_daily_loss(self) -> bool:
        """Return True if daily loss limit is breached."""
        limit = self.account_size * (MAX_DAILY_LOSS_PCT / 100)
        hit = self.state.daily_pnl <= -limit
        self.state.daily_loss_limit_hit = hit
        return hit

    def check_weekly_loss(self) -> bool:
        """Return True if weekly loss limit is breached."""
        limit = self.account_size * (MAX_WEEKLY_LOSS_PCT / 100)
        hit = self.state.weekly_pnl <= -limit
        self.state.weekly_loss_limit_hit = hit
        return hit

    def is_risk_reduction_active(self) -> bool:
        """Return True if consecutive loss threshold is reached."""
        active = self.state.consecutive_losses >= CONSECUTIVE_LOSS_THRESHOLD
        self.state.risk_reduction_active = active
        return active

    # ─────────────────────── Trade Recording ───────────────────

    def record_trade(self, pnl: float, details: dict = None):
        """Record a completed trade and update running state."""
        self.state.daily_pnl += pnl
        self.state.weekly_pnl += pnl
        self.state.trades_today += 1
        self.state.trades_this_session += 1

        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        self._trade_log.append({
            "pnl": pnl,
            "daily_pnl": self.state.daily_pnl,
            "consecutive_losses": self.state.consecutive_losses,
            **(details or {}),
        })

    def reset_session_count(self):
        """Reset per-session trade count (call at session change)."""
        self.state.trades_this_session = 0

    def reset_daily(self):
        """Reset daily counters (call at start of new day)."""
        self.state.daily_pnl = 0.0
        self.state.trades_today = 0

    def reset_weekly(self):
        """Reset weekly counters (call at start of new week)."""
        self.state.weekly_pnl = 0.0
