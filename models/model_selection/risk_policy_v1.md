# Risk Policy & Position Sizing(v1)

## Overview

This module defines the **capital management and risk control rules** of the trading system.

Previous stages focused on:

- ML model training
- Walk-forward validation
- Robustness testing
- Candidate model selection

However, a predictive model alone does not constitute a complete trading system. A real system also requires clear rules for **capital allocation and risk management**.

The goal of this module is to define a **risk policy layer independent from the ML model**, which will later be used by the execution engine.

This separation ensures:

- consistent evaluation across experiments
- reproducibility of results
- protection against catastrophic losses

---

## Risk Policy Configuration

Risk parameters are defined in:


config/v1.yaml


Example configuration:

```yaml
system:
  risk:
    initial_capital: 100000
    position_sizing: fixed_fraction
    risk_fraction: 1.0
    compounding: true
    max_positions: 1
    max_drawdown_limit: 0.40
    minimum_capital: 10000
```

---

## Selected Parameters

### Initial Capital

initial_capital = 100000

The system starts with the same capital used during earlier research experiments. This ensures consistency between:

- research backtests
- model comparison
- robustness testing

The absolute value of capital does not affect performance metrics such as Sharpe ratio but defines the scale of the equity curve.

### Position Sizing

position_sizing = fixed_fraction

risk_fraction = 1.0

The system uses fixed fraction position sizing, allocating 100% of available capital when a trading signal is generated.

This choice was made because:

- Research experiments were conducted using full allocation.
- The strategy does not currently use stop-loss based risk sizing.
- It keeps the system simple for the first version.

More advanced sizing techniques (ATR-based sizing, volatility targeting, Kelly criterion, etc.) may be introduced in later versions.

### Compounding

compounding = true

Position size is based on current equity, allowing the system to grow (or shrink) over time.

This reflects how capital evolves in real trading systems.

### Max Simultaneous Positions

max_positions = 1

The current system trades:

- one asset (BTC)
- one timeframe (1h)

Therefore only one position can exist at a time.

This constraint simplifies the architecture and reflects the current strategy design.

### Max Drawdown Limit

max_drawdown_limit = 0.40

Trading stops if drawdown exceeds 40%.

This acts as a risk kill-switch, protecting the system from:

- model degradation
- regime changes
- unexpected data issues
- pipeline failures

The value was selected based on typical drawdowns observed in systematic strategies with Sharpe ratios around 1.

### Minimum Capital

minimum_capital = 10000

If equity falls below this level, trading stops.

This prevents the system from continuing to operate after severe capital loss.

---

## Stop Trading Conditions

The system stops trading when any of the following occurs:

### Maximum drawdown exceeded

drawdown > 40%

### Capital below minimum threshold

equity < 10000

### Data or execution integrity errors

Examples include:
- corrupted datasets
- NaN values in features
- model unable to generate signals

These safeguards prevent the system from operating with invalid inputs.

---

## Utility Functions

Two helper functions were implemented in:

models/utils.py

### Load Risk Policy

get_risk_policy()

Loads the risk configuration from the YAML configuration file.

### Validate Risk Policy

validate_risk_policy()

Ensures that risk parameters are valid, including:

- positive capital values
- valid risk fraction range
- valid drawdown limits

This prevents invalid configurations from being used during research or execution.

---

## Outcome

After this module, the system now includes:

- ML model selection
- robustness testing
- candidate model selection
- explicit capital management rules

This completes the research layer of the trading system.
