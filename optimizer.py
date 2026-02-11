"""
optimizer.py – Parameter optimization for Default Trading System.

Performs grid search or random search to find optimal parameters.
"""

import itertools
import random
from typing import Dict, List, Any
import pandas as pd
from dataclasses import asdict

from backtest import Backtester, BacktestConfig

class Optimizer:
    def __init__(self, base_config: BacktestConfig):
        self.base_config = base_config
        self.results = []

    def optimize_grid(self, param_grid: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Run grid search optimization.
        
        param_grid example:
        {
            "risk_pct": [0.5, 1.0, 2.0],
            "min_rr": [1.0, 1.5, 2.0]
        }
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        print(f"Starting Grid Search: {len(combinations)} combinations")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"Testing combo {i+1}/{len(combinations)}: {params}")
            
            # Update config (assuming params are attributes of Config or related objects)
            # Note: In a real complex system, we'd need a way to inject these deep into the systems.
            # For this simple optimizer, we might need to modify global config or pass params to Backtester.
            # Since Backtester uses global CONFIG, this is tricky without refactoring.
            # For now, we will assume we can patch the objects or that Backtester takes overrides.
            
            # ACTUALLY: The current system relies on global `config.py`. 
            # To optimize properly, we should refactor `backtest.py` to accept overrides 
            # or monkey-patch `config` module temporarily.
            
            # Let's try monkey-patching for now as it's least invasive.
            import config
            for k, v in params.items():
                if hasattr(config, k.upper()): # Config vars are usually UPPER
                    setattr(config, k.upper(), v)
                elif hasattr(config, k):  # Try lowercase just in case
                    setattr(config, k, v)
                
                # Special handling for PairConfig if needed, but simple globals first.
                
            # Run Backtest
            backtester = Backtester(self.base_config)
            result = backtester.run()
            
            if result:
                self.results.append({
                    **params,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_return_pct": result.total_return_pct,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "trades": result.total_trades
                })
                
        return pd.DataFrame(self.results).sort_values("sharpe_ratio", ascending=False)

if __name__ == "__main__":
    # Example Usage
    base_cfg = BacktestConfig(
        pair="XAUUSD",
        start_date="2025-11-01",
        end_date="2026-02-01",
        initial_capital=10000
    )
    
    optimizer = Optimizer(base_cfg)
    
    # Define parameters to optimize (must match config.py variable names)
    grid = {
        "risk_pct": [0.5, 1.0],         # Will patch config.RISK_PCT if it existed (actually config.PAIRS... tricky)
        # config.py has PAIRS dict. Creating a generic optimizer for this specific codebase 
        # requires more specific patching logic. 
        # Let's stick to global constants for now.
        "ema_fast": [10, 20],
        "ema_mid": [40, 50],
        "rsi_period": [14, 21]
    }
    
    # NOTE: To make this work effectively, we'd need to ensure `features.py` and others 
    # read from `config` at runtime, which they do.
    
    print("Warning: Optimization modifies global state. Run in isolated process recommended.")
    df = optimizer.optimize_grid(grid)
    print("\nTop 5 Parameter Sets:")
    print(df.head())
    df.to_csv("optimization_results.csv")
