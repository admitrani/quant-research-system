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
- Execution buffer: 0.30%
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

## EMA Baseline Results

- Final Value: $9,998
- CAGR: -35.9%
- Sharpe: -1.25
- Sortino: -6.73
- Volatility: 1.554
- Max Drawdown: 93.0%
- Calmar: -0.386
- Total Trades: 382
- Win Rate: 22.0%
- Profit Factor: 0.717
- Expectancy per Trade: -$235.61
- Exposure: 18.5%
- Avg Trade Duration: 21.9 hours

Source: `metrics_ema_v1.csv`

---

## Buy & Hold Results

- Final Value: $232,964
- CAGR: +17.8%
- Sharpe: 0.572
- Sortino: 3.99
- Volatility: 2.871
- Max Drawdown: 77.1%
- Calmar: 0.231
- Exposure: 100%

Source: `metrics_buy_hold_v1.csv`

---

## Important Note on Active Period

Both EMA and ML strategies touched the `minimum_capital` threshold ($10,000)
before the end of 2022 and flatlined for the remaining ~3 years of the OOS period.
The metrics above reflect this short active period, not the full 5-year OOS window.

Buy & Hold remained fully invested throughout the entire period. Comparing CAGR
or Sharpe directly between strategies with different effective durations requires
this context.

---

## Conclusion

EMA crossover does not outperform Buy & Hold under realistic costs.
Buy & Hold becomes the primary benchmark for ML comparison.
