"""
backtest_report.py – Generates HTML performance reports from backtest results.

Usage:
    python backtest_report.py backtest_results.json
"""

import json
import argparse
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def load_results(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def generate_html(data, output_file):
    metrics = data['metrics']
    config = data['config']
    trades = data['trades']
    equity_curve = data['equity_curve']
    timestamps = data['timestamps']
    
    # Create Equity Curve Chart
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, equity_curve, label='Equity', color='#007bfb', linewidth=2)
    plt.title(f"Equity Curve - {config['pair']}")
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Convert plot to base64
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Generate HTML content
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Backtest Report - {config['pair']}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f0f2f5; color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #1a1a1a; }}
            .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #eee; padding-bottom: 20px; margin-bottom: 30px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
            .metric-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #e1e4e8; }}
            .metric-val {{ font-size: 24px; font-weight: bold; color: #007bfb; margin: 10px 0; }}
            .metric-label {{ font-size: 14px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
            .chart-container {{ margin-bottom: 40px; text-align: center; }}
            img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #eee; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background: #f8f9fa; font-weight: 600; }}
            .win {{ color: #28a745; font-weight: bold; }}
            .loss {{ color: #dc3545; font-weight: bold; }}
            .config-section {{ background: #eef6fc; padding: 15px; border-radius: 8px; margin-bottom: 30px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>Backtest Report: {config['pair']}</h1>
                    <p>{config['start_date']} to {config['end_date']} | Initial Capital: ${config['initial_capital']}</p>
                </div>
                <div>
                    <h2>{metrics['total_return_pct']}% Return</h2>
                </div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-val">{metrics['total_trades']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-val">{metrics['win_rate']}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-val">{metrics['profit_factor']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-val">{metrics['sharpe_ratio']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-val">{metrics['max_drawdown_pct']}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg R:R</div>
                    <div class="metric-val">{metrics['avg_rr']}R</div>
                </div>
            </div>

            <div class="chart-container">
                <h3>Equity Curve</h3>
                <img src="data:image/png;base64,{plot_url}" alt="Equity Curve">
            </div>

            <h2>Recent Trades</h2>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Price</th>
                        <th>P&L</th>
                        <th>Score</th>
                        <th>Regime</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # Add last 20 trades
    for trade in trades[-20:]:  # Show last 20
        pnl_class = "win" if trade['pnl'] > 0 else "loss" if trade['pnl'] < 0 else ""
        html += f"""
                    <tr>
                        <td>{trade['entry_time']}</td>
                        <td>{trade['direction']}</td>
                        <td>{trade['entry_price']:.4f}</td>
                        <td class="{pnl_class}">${trade['pnl']:.2f}</td>
                        <td>L:{trade['long_score']} S:{trade['short_score']}</td>
                        <td>{trade['regime']}</td>
                    </tr>
        """
    
    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate HTML report from backtest results")
    parser.add_argument("results_file", help="Path to backtest_results.json")
    parser.add_argument("--output", help="Output HTML file path", default="backtest_report.html")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: File {args.results_file} not found.")
        return
        
    data = load_results(args.results_file)
    generate_html(data, args.output)

if __name__ == "__main__":
    main()
