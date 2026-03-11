# BTC/USDT ML Trading System

Quantitative trading system for BTC/USDT 1H using Machine Learning with walk-forward validation. Built with production-grade architecture.

## System Overview

The system consists of three independent pipelines:

1. **Data Pipeline** — Ingestion → Silver → Gold (Medallion architecture)
2. **Research Pipeline** — Walk-forward validation → Model comparison → Robustness → Candidate selection
3. **Backtest Pipeline** — ML strategy → EMA baseline → Buy & Hold → Metrics → Equity comparison

### Architecture

Binance API → Raw (Bronze) → Silver (DuckDB) → Gold (Parquet)
↓
Research Pipeline (XGBoost, RF, Logistic)
↓
Candidate Selection (XGBoost, depth=4, threshold=0.6)
↓
Backtest Pipeline (Backtrader)
↓
Metrics & Equity Comparison

## Execution
```bash
# Pipeline 1: Data ingestion and transformation
python -m orchestration.pipeline

# Pipeline 2: Research (walk-forward + model selection)
python -m models.research.research_pipeline

# Pipeline 3: Backtest (ML + baselines + metrics)
python -m backtests.v1.backtest_pipeline
```

Each pipeline supports partial execution with `--stage <stage_name>`.

## v1 Configuration

| Parameter              | Value                    |
|------------------------|--------------------------|
| Asset                  | BTC/USDT                 |
| Timeframe              | 1H                       |
| Historical Range       | 2019-01-01 → 2026-03-03  |
| Label Horizon          | t+3 (3 hours)            |
| Mode                   | Long / Flat              |
| Initial Training       | 2 years                  |
| Test Window            | 6 months                 |
| OOS Windows            | 10                       |
| Selected Model         | XGBoost (depth=4, threshold=0.6) |
| Commission per side    | 0.18%                    |
| Slippage per side      | 0.05%                    |
| Initial Capital        | $100,000                 |
| Risk Fraction          | 1.0 (all-in)            |

Full configuration: `config/v1.yaml`

## v1 Results

| Metric          | ML Strategy | EMA Baseline | Buy & Hold |
|-----------------|-------------|--------------|------------|
| Final Value     | $9,995      | $9,998       | $232,964   |
| CAGR            | -27.2%      | -35.9%       | +17.8%     |
| Sharpe          | -2.32       | -1.25        | 0.57       |
| Max Drawdown    | 90.6%       | 93.0%        | 77.1%      |
| Total Trades    | 665         | 382          | —          |
| Win Rate        | 45.9%       | 22.0%        | —          |
| Profit Factor   | 0.557       | 0.717        | —          |

Both ML and EMA touched minimum capital ($10k) before end of 2022.
Buy & Hold remained active for the full OOS period (2021-2026).

## Features (Gold v1)

1. `return_1h` — 1-hour price return
2. `return_3h` — 3-hour rolling return
3. `return_12h` — 12-hour rolling return
4. `volatility_12h` — 12-hour rolling volatility
5. `ma20_distance` — Distance to 20-period moving average
6. `volume_zscore` — 20-hour volume z-score

## Key Design Decisions

- **Config-driven**: All parameters in `config/v1.yaml`, no hardcoded values
- **Medallion architecture**: Bronze/Silver/Gold with immutable Gold
- **Walk-forward validation**: Expanding window, retraining each period
- **Unified metrics**: Custom Sharpe (daily returns × √365) across all strategies
- **Anti-leakage**: `future_return` never used as feature; LAG-based features only

## Future Extensions (v2)

- Magnitude-filtered label (`future_return_6h > 0.0075`)
- Additional features (`return_24h`, etc.)
- Reduced risk fraction
- Multi-asset expansion

## Tech Stack

Python, Backtrader, XGBoost, DuckDB, Pandas, scikit-learn

## Project Structure

config/              # v1.yaml, config_loader.py
ingestion/           # API client, market data, audit
transformations/     # Silver/Gold SQL models
orchestration/       # Pipeline stages
models/              # Walk-forward, robustness, candidate selection
strategies/          # ML strategy (Backtrader)
backtests/v1/        # ML backtest, EMA/B&H baselines, metrics
storage/             # Bronze/Silver/Gold parquet files
tests/               # 71 tests covering critical paths
docs/                # Specifications, data freeze, risk policy

