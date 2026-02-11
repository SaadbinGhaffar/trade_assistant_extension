"""
advanced.py – Optional advanced features.
1. Bayesian regime probability model
2. Monte Carlo simulation for risk & expectancy
3. Equity curve slope filter
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

from regime import Regime


# ═══════════════════════════════════════════════════════════════
#  1. Bayesian Regime Probability
# ═══════════════════════════════════════════════════════════════

class BayesianRegimeModel:
    """
    Simple Bayesian model that estimates regime probabilities
    using conjugate priors (Beta distribution) updated with
    observed ADX/ATR/BB features.

    Prior: P(Trend) = P(Range) = 0.5
    Likelihood: based on how well current features match
    each regime's typical distribution.
    """

    def __init__(self):
        # Beta prior parameters (alpha, beta) for each regime
        self.trend_prior = (2.0, 2.0)   # uniform-ish prior
        self.range_prior = (2.0, 2.0)

    def compute_probabilities(
        self,
        adx: float,
        atr_expansion: float,
        bb_width_ratio: float,
    ) -> dict:
        """
        Compute P(Trend | data) and P(Range | data) using Bayes' theorem.

        Parameters
        ----------
        adx : float
            Current ADX value.
        atr_expansion : float
            ATR / median(ATR) ratio.
        bb_width_ratio : float
            Current BB width / median(BB width).

        Returns
        -------
        dict with keys "trend_prob", "range_prob", "volatile_prob"
        """
        # Likelihood of observing these features under each regime
        # Trend: high ADX, expanding ATR, wider BB
        trend_likelihood = (
            sp_stats.norm.pdf(adx, loc=30, scale=8) *
            sp_stats.norm.pdf(atr_expansion, loc=1.3, scale=0.3) *
            sp_stats.norm.pdf(bb_width_ratio, loc=1.2, scale=0.3)
        )

        # Range: low ADX, contracting ATR, narrow BB
        range_likelihood = (
            sp_stats.norm.pdf(adx, loc=15, scale=5) *
            sp_stats.norm.pdf(atr_expansion, loc=0.8, scale=0.2) *
            sp_stats.norm.pdf(bb_width_ratio, loc=0.7, scale=0.2)
        )

        # Volatile: high ATR but inconsistent direction
        volatile_likelihood = (
            sp_stats.norm.pdf(adx, loc=20, scale=10) *
            sp_stats.norm.pdf(atr_expansion, loc=1.5, scale=0.4) *
            sp_stats.norm.pdf(bb_width_ratio, loc=1.5, scale=0.4)
        )

        # Priors (uniform)
        p_trend = 0.4
        p_range = 0.4
        p_volatile = 0.2

        # Posterior (unnormalized)
        post_trend    = trend_likelihood * p_trend
        post_range    = range_likelihood * p_range
        post_volatile = volatile_likelihood * p_volatile

        # Normalize
        total = post_trend + post_range + post_volatile
        if total == 0:
            total = 1e-10

        return {
            "trend_prob":    round(post_trend / total, 4),
            "range_prob":    round(post_range / total, 4),
            "volatile_prob": round(post_volatile / total, 4),
        }


# ═══════════════════════════════════════════════════════════════
#  2. Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Monte Carlo simulation to estimate expected equity paths,
    drawdown distribution, and risk of ruin.
    """

    def __init__(
        self,
        win_rate: float = 0.55,
        avg_win: float = 1.5,
        avg_loss: float = -1.0,
        n_trades: int = 100,
        n_simulations: int = 1000,
        initial_equity: float = 100.0,
    ):
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.n_trades = n_trades
        self.n_simulations = n_simulations
        self.initial_equity = initial_equity

    def run(self) -> dict:
        """
        Run the Monte Carlo simulation.

        Returns
        -------
        dict with:
            - median_equity: median final equity
            - p5_equity: 5th percentile (worst-case)
            - p95_equity: 95th percentile (best-case)
            - max_drawdown_median: median max drawdown
            - risk_of_ruin: probability of equity going below 50%
            - expectancy: per-trade expectancy
            - equity_paths: array of shape (n_simulations, n_trades+1)
        """
        np.random.seed(None)

        equity_paths = np.zeros((self.n_simulations, self.n_trades + 1))
        equity_paths[:, 0] = self.initial_equity

        max_drawdowns = np.zeros(self.n_simulations)

        for sim in range(self.n_simulations):
            equity = self.initial_equity
            peak = equity

            for t in range(1, self.n_trades + 1):
                if np.random.random() < self.win_rate:
                    pnl = abs(np.random.normal(self.avg_win, self.avg_win * 0.3))
                else:
                    pnl = -abs(np.random.normal(abs(self.avg_loss), abs(self.avg_loss) * 0.3))

                equity += pnl
                equity = max(equity, 0)  # can't go below 0
                equity_paths[sim, t] = equity
                peak = max(peak, equity)

            # Max drawdown for this simulation
            peaks = np.maximum.accumulate(equity_paths[sim])
            dd = (peaks - equity_paths[sim]) / np.where(peaks > 0, peaks, 1)
            max_drawdowns[sim] = dd.max()

        final_equities = equity_paths[:, -1]

        return {
            "median_equity":       round(np.median(final_equities), 2),
            "p5_equity":           round(np.percentile(final_equities, 5), 2),
            "p95_equity":          round(np.percentile(final_equities, 95), 2),
            "mean_equity":         round(np.mean(final_equities), 2),
            "max_drawdown_median": round(np.median(max_drawdowns) * 100, 1),
            "max_drawdown_p95":    round(np.percentile(max_drawdowns, 95) * 100, 1),
            "risk_of_ruin":        round((final_equities < self.initial_equity * 0.5).mean() * 100, 1),
            "expectancy":          round(self.win_rate * self.avg_win + (1 - self.win_rate) * self.avg_loss, 4),
            "equity_paths":        equity_paths,
        }

    def print_summary(self, results: Optional[dict] = None):
        """Print a formatted summary of MC results."""
        if results is None:
            results = self.run()

        print("\n──── Monte Carlo Simulation Results ────")
        print(f"  Simulations:        {self.n_simulations}")
        print(f"  Trades per sim:     {self.n_trades}")
        print(f"  Win Rate:           {self.win_rate*100:.1f}%")
        print(f"  Avg Win / Loss:     ${self.avg_win:.2f} / ${self.avg_loss:.2f}")
        print(f"  Expectancy/trade:   ${results['expectancy']:.4f}")
        print(f"  ── Equity Outcomes ──")
        print(f"  Median Final:       ${results['median_equity']:.2f}")
        print(f"  5th Percentile:     ${results['p5_equity']:.2f}")
        print(f"  95th Percentile:    ${results['p95_equity']:.2f}")
        print(f"  ── Risk Metrics ──")
        print(f"  Median Max DD:      {results['max_drawdown_median']:.1f}%")
        print(f"  95th %ile Max DD:   {results['max_drawdown_p95']:.1f}%")
        print(f"  Risk of Ruin (<50%): {results['risk_of_ruin']:.1f}%")
        print()


# ═══════════════════════════════════════════════════════════════
#  3. Equity Curve Slope Filter
# ═══════════════════════════════════════════════════════════════

class EquityCurveFilter:
    """
    Reduces risk when the equity curve slope turns negative.
    Uses linear regression over the last N equity points.
    """

    def __init__(self, lookback: int = 10, reduction_factor: float = 0.5):
        self.lookback = lookback
        self.reduction_factor = reduction_factor

    def should_reduce_risk(self, equity_series: List[float]) -> Tuple[bool, float]:
        """
        Check if the equity curve slope is negative.

        Parameters
        ----------
        equity_series : list of float
            Recent equity values (oldest first).

        Returns
        -------
        (should_reduce, slope)
        """
        if len(equity_series) < self.lookback:
            return False, 0.0

        recent = np.array(equity_series[-self.lookback:])
        x = np.arange(len(recent))

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, recent)

        # Reduce if slope is negative and statistically significant
        should_reduce = slope < 0 and p_value < 0.3

        return should_reduce, round(slope, 4)

    def adjusted_risk_pct(
        self, base_risk_pct: float, equity_series: List[float]
    ) -> float:
        """Return adjusted risk percentage based on equity curve."""
        should_reduce, slope = self.should_reduce_risk(equity_series)
        if should_reduce:
            return base_risk_pct * self.reduction_factor
        return base_risk_pct
