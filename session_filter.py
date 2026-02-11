"""
session_filter.py – PKT trading session classification and gating.
Determines which session is active and whether trading is allowed.
"""

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import (
    PKT_UTC_OFFSET,
    SESSION_OVERLAP_START, SESSION_OVERLAP_END,
    SESSION_LONDON_START, SESSION_LONDON_END,
    SESSION_SCORE_OVERLAP, SESSION_SCORE_LONDON, SESSION_SCORE_OUTSIDE,
)


@dataclass
class SessionInfo:
    """Session classification result."""
    name: str                  # "Overlap" | "London" | "Outside"
    is_tradeable: bool
    score_bonus: int           # points to add to session score
    current_time_pkt: dt.datetime


def get_pkt_now() -> dt.datetime:
    """Return current time in PKT (UTC+5)."""
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET)))


def classify_session(timestamp: Optional[dt.datetime] = None) -> SessionInfo:
    """
    Classify the trading session for a given PKT timestamp.

    Parameters
    ----------
    timestamp : datetime, optional
        Time to classify. Defaults to current PKT time.

    Returns
    -------
    SessionInfo
    """
    if timestamp is None:
        timestamp = get_pkt_now()

    # Ensure we have a timezone-aware PKT datetime
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(
            tzinfo=dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET))
        )

    hour = timestamp.hour

    if SESSION_OVERLAP_START <= hour < SESSION_OVERLAP_END:
        return SessionInfo(
            name="Overlap",
            is_tradeable=True,
            score_bonus=SESSION_SCORE_OVERLAP,
            current_time_pkt=timestamp,
        )
    elif SESSION_LONDON_START <= hour < SESSION_LONDON_END:
        return SessionInfo(
            name="London",
            is_tradeable=True,
            score_bonus=SESSION_SCORE_LONDON,
            current_time_pkt=timestamp,
        )
    else:
        return SessionInfo(
            name="Outside",
            is_tradeable=False,
            score_bonus=SESSION_SCORE_OUTSIDE,
            current_time_pkt=timestamp,
        )


def classify_candle_session(candle_time: pd.Timestamp) -> SessionInfo:
    """Classify session for a specific candle's close timestamp."""
    if candle_time.tzinfo is None:
        candle_time = candle_time.tz_localize("UTC").tz_convert(
            dt.timezone(dt.timedelta(hours=PKT_UTC_OFFSET))
        )
    return classify_session(candle_time.to_pydatetime())
