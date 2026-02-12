"""
backtest.py – Historical backtesting engine for the Forex Trading Assistant.

Simulates trades on historical data, tracks performance, and generates metrics.

Usage:
    python backtest.py --pair XAUUSD --days 180
    python backtest.py --pair EURUSD --start 2025-08-01 --end 2026-02-01
    python backtest.py --all-pairs --days 90
"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np

from config import PAIRS, ACCOUNT_SIZE, PKT_UTC_OFFSET
from data_provider import get_multi_timeframe_data
from features import compute_all_features
from regime import detect_regime
from scoring import compute_total_score
from session_filter import classify_candle_session
from risk_manager import RiskManager
from governance import assess_eligibility
from performance_tracker import TradeRecord, PerformanceStats


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    pair: str
    start_date: str
    end_date: str
    initial_capital: float = ACCOUNT_SIZE
    commission_pct: float = 0.0  # Commission per trade (%)
    slippage_pips: float = 0.0   # Slippage in pips


@dataclass
class BacktestTrade:
    """Single backtested trade."""
    entry_time: str
    exit_time: str
    pair: str
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    pnl: float
    pnl_pct: float
    r_multiple: float  # Actual R achieved
    long_score: float
    short_score: float
    regime: str
    session: str
    outcome: str  # "Win", "Loss", "BE"


@dataclass
class BacktestResults:
    """Complete backtest results."""
    config: BacktestConfig
    trades: List[BacktestTrade]
    equity_curve: List[float]
    timestamps: List[str]
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    expectancy: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr: float = 0.0
    
    # Regime breakdown
    regime_stats: Dict[str, Dict] = None
    session_stats: Dict[str, Dict] = None


class Backtester:
    """Historical backtesting engine."""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = [config.initial_capital]
        self.timestamps: List[str] = []
        self.current_equity = config.initial_capital
        
    def run(self) -> BacktestResults:
        """Execute backtest on historical data."""
        print(f"\n{'='*70}")
        print(f"  BACKTESTING: {self.config.pair}")
        print(f"  Period: {self.config.start_date} to {self.config.end_date}")
        print(f"  Initial Capital: ${self.config.initial_capital}")
        print(f"{'='*70}\n")
        
        # Fetch historical data
        print("Fetching historical data...")
        pair_cfg = PAIRS[self.config.pair]
        data = get_multi_timeframe_data(self.config.pair)
        
        if not data or data["daily"].empty:
            print("❌ Failed to fetch data")
            return None
        
        # Filter data by date range
        df_15m = data["15m"]
        df_15m.index = pd.to_datetime(df_15m.index)
        
        # Localize start/end dates to PKT
        tz = timezone(timedelta(hours=PKT_UTC_OFFSET))
        start = pd.to_datetime(self.config.start_date).replace(tzinfo=tz)
        # End date should include the full last day
        end = pd.to_datetime(self.config.end_date).replace(hour=23, minute=59, second=59, tzinfo=tz)
        
        df_15m = df_15m[(df_15m.index >= start) & (df_15m.index <= end)]
        
        if df_15m.empty:
            print("❌ No data in specified date range")
            return None
        
        print(f"✓ Loaded {len(df_15m)} candles")
        print(f"  Date range: {df_15m.index[0]} to {df_15m.index[-1]}")
        
        # Simulate trades
        print("\nSimulating trades...")
        risk_mgr = RiskManager(account_size=self.current_equity)
        
        for i in range(200, len(df_15m)):  # Start after warmup period
            # Get current candle data
            current_time = df_15m.index[i]
            
            # Prepare data slices (up to current candle)
            data_slice = {
                "daily": data["daily"][:current_time],
                "1h": data["1h"][:current_time],
                "15m": df_15m[:i+1],
            }
            
            # Skip if insufficient data
            if len(data_slice["daily"]) < 200 or len(data_slice["1h"]) < 50:
                continue
            
            # Analyze current setup
            try:
                regime = detect_regime(data_slice["1h"])
                # if i % 50 == 0:
                #      print(f"Candle {current_time} | Regime: {regime.regime.name}")
                session = classify_candle_session(current_time)
                
                scores = compute_total_score(
                    df_daily=data_slice["daily"],
                    df_1h=data_slice["1h"],
                    df_15m=data_slice["15m"],
                    regime=regime,
                    session=session,
                    pair_cfg=pair_cfg,
                    account_size=self.current_equity,
                )
                
                eligibility = assess_eligibility(
                    scores=scores,
                    session=session,
                    risk_mgr=risk_mgr,
                    news_upcoming=False,
                )
                
                # Check if trade is eligible
                if not eligibility.is_eligible:
                    # Debug: print why it failed occasionally
                    # Log more frequently for debugging
                    if i % 50 == 0:
                        print(f"Candle {current_time} | Score: L={eligibility.long_score:.1f} S={eligibility.short_score:.1f} | {eligibility.summary}")
                    continue
                
                # Simulate trade
                trade = self._simulate_trade(
                    entry_time=current_time,
                    data_slice=data_slice,
                    df_future=df_15m[i+1:],
                    scores=scores,
                    regime=regime,
                    session=session,
                    pair_cfg=pair_cfg,
                    risk_mgr=risk_mgr,
                )
                
                if trade:
                    self.trades.append(trade)
                    self.current_equity += trade.pnl
                    self.equity_curve.append(self.current_equity)
                    self.timestamps.append(str(trade.exit_time))
                    
                    # Update risk manager
                    risk_mgr.account_size = self.current_equity
                    
                    if len(self.trades) % 10 == 0:
                        print(f"  Trades: {len(self.trades)} | Equity: ${self.current_equity:.2f}")
                        
            except Exception as e:
                # Skip candles with errors
                continue
        
        print(f"\n✓ Simulation complete: {len(self.trades)} trades executed")
        
        # Calculate metrics
        results = self._calculate_metrics()
        return results
    
    def _simulate_trade(
        self,
        entry_time,
        data_slice,
        df_future,
        scores,
        regime,
        session,
        pair_cfg,
        risk_mgr,
    ) -> Optional[BacktestTrade]:
        """Simulate a single trade."""
        
        # Get entry price (next candle open)
        if df_future.empty:
            return None
        
        entry_price = df_future.iloc[0]["Open"]
        direction = scores.dominant_side.lower()
        
        # Calculate stop and target
        last_row = data_slice["15m"].iloc[-1]
        atr = last_row.get("atr", 0)
        vol_percentile = last_row.get("vol_percentile", 50)
        
        stop_target = risk_mgr.compute_stop_target(
            entry_price=entry_price,
            atr=atr,
            pair_cfg=pair_cfg,
            direction=direction,
            rr_ratio=pair_cfg.min_rr,
            vol_percentile=vol_percentile,
        )
        
        # Calculate position size
        position_size = risk_mgr.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_target.stop_price,
            risk_pct=pair_cfg.risk_pct,
            account_size=self.current_equity,
            pair_cfg=pair_cfg,
        )
        
        # Simulate trade execution on future candles
        for j in range(len(df_future)):
            candle = df_future.iloc[j]
            high = candle["High"]
            low = candle["Low"]
            
            # Check if stop or target hit
            if direction == "long":
                if low <= stop_target.stop_price:
                    # Stop hit
                    exit_price = stop_target.stop_price
                    exit_time = df_future.index[j]
                    outcome = "Loss"
                    break
                elif high >= stop_target.target_price:
                    # Target hit
                    exit_price = stop_target.target_price
                    exit_time = df_future.index[j]
                    outcome = "Win"
                    break
            else:  # short
                if high >= stop_target.stop_price:
                    # Stop hit
                    exit_price = stop_target.stop_price
                    exit_time = df_future.index[j]
                    outcome = "Loss"
                    break
                elif low <= stop_target.target_price:
                    # Target hit
                    exit_price = stop_target.target_price
                    exit_time = df_future.index[j]
                    outcome = "Win"
                    break
        else:
            # Trade not closed within available data
            return None
        
        # Calculate P&L
        if direction == "long":
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size
        
        pnl_pct = (pnl / self.current_equity) * 100
        
        # Calculate R-multiple
        risk_amount = abs(entry_price - stop_target.stop_price) * position_size
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0
        
        return BacktestTrade(
            entry_time=str(entry_time),
            exit_time=str(exit_time),
            pair=self.config.pair,
            direction=direction.capitalize(),
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_target.stop_price,
            take_profit=stop_target.target_price,
            position_size=position_size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            r_multiple=r_multiple,
            long_score=scores.long_total,
            short_score=scores.short_total,
            regime=regime.regime.name,
            session=session.name,
            outcome=outcome,
        )
    
    def _calculate_metrics(self) -> BacktestResults:
        """Calculate performance metrics from trades."""
        
        if not self.trades:
            return BacktestResults(
                config=self.config,
                trades=[],
                equity_curve=self.equity_curve,
                timestamps=self.timestamps,
            )
        
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        total_win = sum(wins)
        total_loss = abs(sum(losses))
        profit_factor = (total_win / total_loss) if total_loss > 0 else float("inf")
        
        expectancy = np.mean(pnls)
        
        # Returns
        total_return = self.current_equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100
        
        # Sharpe Ratio (annualized)
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Max Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = equity_array - running_max
        max_dd = np.min(drawdown)
        max_dd_pct = (max_dd / self.config.initial_capital) * 100
        
        # Average R:R
        avg_rr = np.mean([t.r_multiple for t in self.trades])
        
        # Regime breakdown
        regime_stats = self._calculate_regime_stats()
        session_stats = self._calculate_session_stats()
        
        return BacktestResults(
            config=self.config,
            trades=self.trades,
            equity_curve=self.equity_curve,
            timestamps=self.timestamps,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 1),
            profit_factor=round(profit_factor, 2),
            sharpe_ratio=round(sharpe, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 1),
            total_return=round(total_return, 2),
            total_return_pct=round(total_return_pct, 1),
            expectancy=round(expectancy, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            avg_rr=round(avg_rr, 2),
            regime_stats=regime_stats,
            session_stats=session_stats,
        )
    
    def _calculate_regime_stats(self) -> Dict:
        """Calculate performance by regime."""
        regimes = {}
        for trade in self.trades:
            regime = trade.regime
            if regime not in regimes:
                regimes[regime] = {"trades": [], "wins": 0}
            regimes[regime]["trades"].append(trade.pnl)
            if trade.outcome == "Win":
                regimes[regime]["wins"] += 1
        
        stats = {}
        for regime, data in regimes.items():
            total = len(data["trades"])
            wins = data["wins"]
            win_rate = (wins / total * 100) if total > 0 else 0
            avg_pnl = np.mean(data["trades"])
            stats[regime] = {
                "trades": total,
                "win_rate": round(win_rate, 1),
                "avg_pnl": round(avg_pnl, 2),
            }
        return stats
    
    def _calculate_session_stats(self) -> Dict:
        """Calculate performance by session."""
        sessions = {}
        for trade in self.trades:
            session = trade.session
            if session not in sessions:
                sessions[session] = {"trades": [], "wins": 0}
            sessions[session]["trades"].append(trade.pnl)
            if trade.outcome == "Win":
                sessions[session]["wins"] += 1
        
        stats = {}
        for session, data in sessions.items():
            total = len(data["trades"])
            wins = data["wins"]
            win_rate = (wins / total * 100) if total > 0 else 0
            avg_pnl = np.mean(data["trades"])
            stats[session] = {
                "trades": total,
                "win_rate": round(win_rate, 1),
                "avg_pnl": round(avg_pnl, 2),
            }
        return stats


def print_results(results: BacktestResults):
    """Print backtest results to console."""
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS: {results.config.pair}")
    print(f"  Period: {results.config.start_date} to {results.config.end_date}")
    print(f"{'='*70}\n")
    
    print(f"Total Trades:      {results.total_trades}")
    print(f"Winning Trades:    {results.winning_trades}")
    print(f"Losing Trades:     {results.losing_trades}")
    print(f"Win Rate:          {results.win_rate}%")
    print(f"Profit Factor:     {results.profit_factor}")
    print(f"Sharpe Ratio:      {results.sharpe_ratio}")
    print(f"Expectancy:        ${results.expectancy}")
    print(f"Avg Win:           ${results.avg_win}")
    print(f"Avg Loss:          ${results.avg_loss}")
    print(f"Avg R:R:           {results.avg_rr}R")
    print(f"\nTotal Return:      ${results.total_return} ({results.total_return_pct}%)")
    print(f"Max Drawdown:      ${results.max_drawdown} ({results.max_drawdown_pct}%)")
    print(f"Final Equity:      ${results.equity_curve[-1]:.2f}")
    
    if results.regime_stats:
        print(f"\n{'─'*70}")
        print("Performance by Regime:")
        print(f"{'─'*70}")
        for regime, stats in results.regime_stats.items():
            print(f"  {regime:12} | Trades: {stats['trades']:3} | Win Rate: {stats['win_rate']:5.1f}% | Avg P&L: ${stats['avg_pnl']:7.2f}")
    
    if results.session_stats:
        print(f"\n{'─'*70}")
        print("Performance by Session:")
        print(f"{'─'*70}")
        for session, stats in results.session_stats.items():
            print(f"  {session:12} | Trades: {stats['trades']:3} | Win Rate: {stats['win_rate']:5.1f}% | Avg P&L: ${stats['avg_pnl']:7.2f}")
    
    print(f"\n{'='*70}\n")


def save_results(results: BacktestResults, filename: str = "backtest_results.json"):
    """Save backtest results to JSON file."""
    data = {
        "config": asdict(results.config),
        "metrics": {
            "total_trades": results.total_trades,
            "win_rate": results.win_rate,
            "profit_factor": results.profit_factor,
            "sharpe_ratio": results.sharpe_ratio,
            "max_drawdown_pct": results.max_drawdown_pct,
            "total_return_pct": results.total_return_pct,
            "expectancy": results.expectancy,
            "avg_rr": results.avg_rr,
        },
        "trades": [asdict(t) for t in results.trades],
        "equity_curve": results.equity_curve,
        "timestamps": results.timestamps,
        "regime_stats": results.regime_stats,
        "session_stats": results.session_stats,
    }
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Backtest the Forex Trading Assistant")
    parser.add_argument("--pair", type=str, default="XAUUSD", help="Trading pair (e.g., XAUUSD)")
    parser.add_argument("--days", type=int, help="Number of days to backtest (from today)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=ACCOUNT_SIZE, help="Initial capital")
    parser.add_argument("--output", type=str, default="backtest_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Determine date range
    if args.days:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    elif args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        # Default: last 90 days
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    
    # Create config
    config = BacktestConfig(
        pair=args.pair,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
    )
    
    # Run backtest
    backtester = Backtester(config)
    results = backtester.run()
    
    if results:
        # Print results
        print_results(results)
        
        # Save to file
        save_results(results, args.output)
        
        print(f"\n✓ Backtest complete!")
        print(f"  Run 'python backtest_report.py {args.output}' to generate HTML report")
    else:
        print("\n❌ Backtest failed")


if __name__ == "__main__":
    main()
