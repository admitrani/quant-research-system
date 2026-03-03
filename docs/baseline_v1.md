# Baseline Backtest (v1)

## Overview

This document defines the professional baseline strategy used as benchmark for ML evaluation.

The baseline is intentionally simple and non-optimized.

---

## Strategy Definition

- Strategy: EMA Crossover
- Fast EMA: 10
- Slow EMA: 20
- Mode: Long only
- Position sizing: 100% available capital
- Initial capital: 100,000
- Data source: Gold v1 dataset
- Timeframe: 1H

---

## Cost Model

Broker simulation:
- Commission: 0.18% per side
- Slippage: 0.05% per side
- Applied to every trade

---

## Backtesting Engine

- Framework: Backtrader
- Data: storage/gold/btcusdt_1h_v1.parquet
- No lookahead bias
- No parameter tuning
- No walk-forward optimization

---

## Metrics Computation

### Sharpe Ratio:

Computed manually:

Sharpe = (mean_daily_return / std_daily_return) * sqrt(365)

Daily returns derived from TimeReturn analyzer.

### CAGR:

CAGR = (Final Value / Initial Value)^(1/years) - 1

### Calmar Ratio:

Calmar = CAGR / Max Drawdown

### Trade Metrics

Extracted from TradeAnalyzer:
- Total trades
- Net PnL
- Gross profit
- Gross loss
- Profit factor
- Expectancy per trade

---

## Results

- Final value: 145,932
- CAGR: 5.41%
- Sharpe: 0.32
- Max Drawdown: 70.19%
- Calmar: 0.077
- Profit Factor: 1.04
- Trades: 308

---

## Interpretation

The baseline shows:
- Positive long-term return
- Very high drawdown
- Weak risk-adjusted performance
- Marginal profitability after costs
- It is considered a valid but weak benchmark.

---

## Version Freeze

This baseline corresponds to:
- Gold v1 dataset
- Cost model v1
- EMA parameters (10,20)

Any modification requires version increment.
