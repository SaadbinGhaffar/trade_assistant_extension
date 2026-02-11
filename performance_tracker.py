"""
performance_tracker.py – Rolling trade log, stats, and performance tracking.
Persists trade history to a JSON file for session-over-session analysis.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict

import numpy as np

from config import PKT_UTC_OFFSET


TRADE_LOG_FILE = os.path.join(os.path.dirname(__file__), "trade_log.json")


@dataclass
class TradeRecord:
    """Single trade record."""
    timestamp: str
    pair: str
    direction: str        # "Long" | "Short"
    entry_price: float
    stop_price: float
    target_price: float
    lot_size: float
    risk_pct: float
    pnl: Optional[float] = None       # filled after trade closes
    outcome: Optional[str] = None      # "Win" | "Loss" | "BE"
    long_score: float = 0.0
    short_score: float = 0.0
    session: str = ""
    regime: str = ""
    notes: str = ""


@dataclass
class PerformanceStats:
    """Aggregated performance metrics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    best_trade: float = 0.0
    worst_trade: float = 0.0


class PerformanceTracker:
    """
    Tracks traded signals, outcomes, and computes rolling performance stats.
    """

    def __init__(self, log_file: str = TRADE_LOG_FILE):
        self.log_file = log_file
        self.trades: List[TradeRecord] = []
        self._load()

    # ─────────────────────── Persistence ───────────────────────

    def _load(self):
        """Load trade log from disk."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                self.trades = [TradeRecord(**t) for t in data]
            except (json.JSONDecodeError, TypeError):
                self.trades = []

    def _save(self):
        """Persist trade log to disk."""
        with open(self.log_file, "w") as f:
            json.dump([asdict(t) for t in self.trades], f, indent=2)

    # ─────────────────────── Recording ─────────────────────────

    def add_signal(self, record: TradeRecord):
        """Record a new trade signal (before outcome is known)."""
        self.trades.append(record)
        self._save()

    def close_trade(self, index: int, pnl: float, outcome: str):
        """Mark a trade as closed with its P&L."""
        if 0 <= index < len(self.trades):
            self.trades[index].pnl = pnl
            self.trades[index].outcome = outcome
            self._save()

    def close_last_trade(self, pnl: float):
        """Close the most recent open trade."""
        outcome = "Win" if pnl > 0 else ("Loss" if pnl < 0 else "BE")
        if self.trades:
            self.trades[-1].pnl = pnl
            self.trades[-1].outcome = outcome
            self._save()

    # ─────────────────────── Statistics ─────────────────────────

    def compute_stats(self, last_n: Optional[int] = None) -> PerformanceStats:
        """
        Compute performance statistics over the last N closed trades.
        If last_n is None, uses all closed trades.
        """
        closed = [t for t in self.trades if t.pnl is not None]
        if last_n:
            closed = closed[-last_n:]

        if not closed:
            return PerformanceStats()

        pnls = [t.pnl for t in closed]
        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win  = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        total_win  = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf")
        expectancy = np.mean(pnls) if pnls else 0

        # Consecutive streaks
        max_con_w = max_con_l = cur_w = cur_l = 0
        for p in pnls:
            if p > 0:
                cur_w += 1
                cur_l = 0
            elif p < 0:
                cur_l += 1
                cur_w = 0
            else:
                cur_w = cur_l = 0
            max_con_w = max(max_con_w, cur_w)
            max_con_l = max(max_con_l, cur_l)

        return PerformanceStats(
            total_trades=len(closed),
            wins=len(wins),
            losses=len(losses),
            breakevens=len(pnls) - len(wins) - len(losses),
            win_rate=round(win_rate, 1),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 2),
            expectancy=round(expectancy, 2),
            total_pnl=round(sum(pnls), 2),
            max_consecutive_wins=max_con_w,
            max_consecutive_losses=max_con_l,
            best_trade=round(max(pnls), 2) if pnls else 0,
            worst_trade=round(min(pnls), 2) if pnls else 0,
        )

    # ─────────────────────── Queries ───────────────────────────

    def trades_today(self) -> List[TradeRecord]:
        """Return trades from today (PKT)."""
        now = datetime.now(timezone(timedelta(hours=PKT_UTC_OFFSET)))
        today_str = now.strftime("%Y-%m-%d")
        return [t for t in self.trades if t.timestamp.startswith(today_str)]

    def trades_this_week(self) -> List[TradeRecord]:
        """Return trades from this week (Mon–Sun, PKT)."""
        now = datetime.now(timezone(timedelta(hours=PKT_UTC_OFFSET)))
        week_start = now - timedelta(days=now.weekday())
        week_start_str = week_start.strftime("%Y-%m-%d")
        return [t for t in self.trades if t.timestamp >= week_start_str]

    def summary_text(self, last_n: int = 20) -> str:
        """Return a formatted performance summary string."""
        stats = self.compute_stats(last_n)
        return (
            f"──── Performance (last {last_n} trades) ────\n"
            f"  Trades: {stats.total_trades}  |  "
            f"Win Rate: {stats.win_rate}%  |  "
            f"PF: {stats.profit_factor}\n"
            f"  Avg Win: ${stats.avg_win}  |  "
            f"Avg Loss: ${stats.avg_loss}  |  "
            f"Expectancy: ${stats.expectancy}\n"
            f"  Total P&L: ${stats.total_pnl}  |  "
            f"Best: ${stats.best_trade}  |  "
            f"Worst: ${stats.worst_trade}\n"
            f"  Max Consec Wins: {stats.max_consecutive_wins}  |  "
            f"Max Consec Losses: {stats.max_consecutive_losses}\n"
        )
