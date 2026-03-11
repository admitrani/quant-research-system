# ML Strategy Backtest (v1)

## Overview

This document describes the ML strategy backtest — the final evaluation of the selected
XGBoost model under realistic trading conditions using Backtrader.

The research pipeline (walk-forward, robustness, candidate selection) identified XGBoost
with depth=4 and threshold=0.60 as the best candidate. This backtest validates that
candidate in a full simulation with transaction costs, fractional BTC sizing, and
capital management rules.

---

## Strategy Design

### Signal Generation

The ML strategy uses walk-forward signal generation:

1. For each OOS window, the model is retrained on all available history
2. Predicted probabilities are generated for the test period
3. Probabilities are attached to the Gold dataset as `ml_prob`

Implementation: `backtests/v1/ml/signal_engine.py`

### Entry / Exit Logic

- **Entry:** `ml_prob > entry_threshold` (0.60) and no open position
- **Exit:** `ml_prob < exit_threshold` (0.50) and position open
- **Mode:** Long only (no short positions)
- **Sizing:** `cash * risk_fraction / (price * total_cost_factor)`

The strategy operates bar-by-bar. It does NOT hold for a fixed number of bars —
the holding period depends on how many consecutive bars the probability stays above
the exit threshold.

Implementation: `strategies/ml_strategy_v1.py`

### Capital Management

- Initial capital: $100,000
- Risk fraction: 1.0 (full allocation)
- Minimum capital kill-switch: $10,000
- Max drawdown limit: 1.0 (effectively disabled — minimum_capital is the real guard)
- Fractional BTC sizing with execution buffer
- Min position size threshold to prevent floating point residuals

Implementation: Controlled via `config/v1.yaml`, enforced in strategy `next()`.

---

## Cost Model

| Parameter          | Value  |
|--------------------|--------|
| Commission/side    | 0.18%  |
| Slippage/side      | 0.05%  |
| Execution buffer   | 0.30%  |
| Round-trip cost    | ~0.46% |
| Broker model       | IBKR   |

Costs are applied by Backtrader's broker engine on every order execution.

---

## Results

| Metric              | Value       |
|---------------------|-------------|
| Final Value         | $9,995      |
| CAGR                | -36.9%      |
| Sharpe              | -2.32       |
| Sortino             | -5.41    |
| Volatility          | 0.933       |
| Max Drawdown        | 90.6%       |
| Calmar              | -0.407      |
| Total Trades        | 665         |
| Win Rate            | 45.9%       |
| Profit Factor       | 0.557       |
| Avg Trade Return    | -$135.35    |
| Exposure            | 6.0%        |
| Avg Trade Duration  | 3.96 hours  |

Source: `metrics_ml_v1.csv`, `comparison_v1.csv`

---

## Analysis

### Why the strategy lost money

The ML strategy touched the minimum_capital threshold ($10,000) before end of 2022
and flatlined for the remaining ~3 years. The key drivers:

1. **Transaction costs dominate:** With 665 trades and ~0.46% round-trip cost, the
   strategy paid approximately $30,500 in total costs — a significant fraction of
   the initial capital when combined with losing trades.

2. **Risk fraction = 1.0 with compounding:** Each incorrect signal destroys a large
   fraction of capital. The first few incorrect trades in the 2021 OOS window caused
   rapid capital erosion from which the strategy never recovered.

3. **Low exposure (6%):** The high threshold (0.60) means the model only signals BUY
   on ~6% of bars. When it does, it goes all-in. This creates a high-conviction,
   low-frequency pattern that is extremely sensitive to the quality of each signal.

4. **Bear market 2022:** The OOS period includes the 2022 crypto crash. A long-only
   strategy with aggressive sizing is structurally disadvantaged in a sustained downtrend.

### Research vs Backtest gap

| Metric       | Research (vectorized) | Backtest (Backtrader) |
|-------------|----------------------|----------------------|
| Sharpe       | 0.735                | -2.32                |
| Equity       | 3.12x                | 0.10x                |

The gap is explained by:

- **Transaction costs:** Research metrics exclude costs. The backtest applies full cost model.
- **Compounding:** Research uses simple signal × return. Backtest compounds with real equity.
- **Capital kill-switch:** Once equity hits $10k, the backtest stops trading. Research
  continues evaluating signals through the entire OOS period.

This gap is expected and documented. The research pipeline serves to compare models
(all under the same costless conditions), while the backtest is the source of truth
for strategy viability.

---

## Key Implementation Details

### Fractional BTC sizing

At BTC prices >$30k, a $100k account can only buy fractional BTC. The sizing formula
accounts for commission, slippage, and buffer to prevent MARGIN errors:

```
size = (cash * risk_fraction) / (price * (1 + commission + slippage + buffer))
```

### Floating point residual guard

After closing a position, Backtrader may report a tiny residual size (e.g., 1e-15)
due to floating point arithmetic. The strategy uses `min_position_size` threshold
to distinguish between real positions and residuals:

```
current_position = self.position.size > self.p.min_position_size
```

### Equity tracking

A custom EquityObserver records equity at every bar for the equity curve CSV.
The equity plot aligns ML (hourly) with EMA and Buy & Hold (daily) using
`resample("D").last().ffill()`.

---

## Artifacts

| File                       | Description                    |
|----------------------------|--------------------------------|
| `metrics_ml_v1.csv`        | All computed metrics           |
| `equity_ml_v1.csv`         | Hourly equity curve            |
| `equity_ML_v1.png`         | Equity curve plot              |
| `comparison_v1.csv`        | Side-by-side with EMA and B&H  |
| `equity_comparison_v1.png` | Combined equity comparison     |

---

## Pipeline Integration

The ML backtest is the first stage of the backtest pipeline:

```bash
python -m backtests.v1.backtest_pipeline
```

Stages: `ml → ema → buy_and_hold → comparison`

Can be run individually: `python -m backtests.v1.backtest_pipeline --stage ml`

---

## Metrics Methodology

### Sharpe Ratio (unified)

All three strategies compute Sharpe identically:

```
daily_returns = hourly_returns.resample("D").apply(λx: (1+x).prod() - 1)
Sharpe = (mean_daily / std_daily) × √365
```

### CAGR (unified)

All three strategies use the same formula:

```
CAGR = (final_value / initial_capital) ^ (1/years) - 1
```

### Sortino and Volatility

Computed on hourly returns with annualization factor √8760.

---

## v2 Implications

The v1 results establish the baseline. The key areas for v2 improvement:

- **Magnitude-filtered label:** `future_return_6h > 0.0075` to eliminate the neutral
  zone from training (signals with expected return below transaction costs).
- **Reduced risk fraction:** Lower than 1.0 to prevent catastrophic compounding losses.
- **Additional features:** `return_24h` and others to capture longer-term dynamics.
- **Extended label horizon:** 6h instead of 3h for more signal clarity.

These changes target the two main failure modes: the model signaling on low-magnitude
moves (unprofitable after costs) and the aggressive sizing destroying capital too quickly.
