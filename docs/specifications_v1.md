# v1 — System Specification (Frozen Configuration)

## Objective

Validate whether Machine Learning (ML) adds statistically significant value over a simple technical baseline under a controlled, reproducible architecture.

This version (v1) is designed as a structural validation experiment before any multi-asset expansion or model complexity increase.

---

## Universe Definition

- **Asset:** BTCUSDT  
- **Timeframe:** 1H  
- **Data Source:** Binance API  
- **Historical Range:** 2019-01-01 → latest available at freeze date  
- **Execution Target:** Interactive Brokers (IBKR)

This version intentionally restricts the universe to a single asset and timeframe to avoid structural data snooping.

---

## Label Definition

- **Horizon:** t+3 (3 hours forward)
- **Mode:** Long / Flat
- **Definition:** 
    - future_return(t) = Close(t+3) / Close(t) - 1
    - label(t) = 1 if future_return > 0 else 0

Short positions are not included in v1.

---

## Validation Scheme

- **Method:** Expanding Walk-Forward
- **Initial Training Window:** 3 years
- **Test Window:** 1 year
- **Retraining:** At the beginning of each test window
- **Threshold:** 0.5 (fixed)

No parameter optimization is performed after observing results.

---

## Cost Model

- **Broker Model:** Interactive Brokers (IBKR)
- **Commission per side:** 0.18%
- **Slippage per side:** 0.05%
- **Approximate round-trip cost:** 0.46%

Costs are applied consistently to both baseline and ML strategies.

---

## Data Integrity Decisions

- Full raw rebuild from 2019-01-01.
- Historical gaps (<0.1%) accepted as exchange-level discontinuities.
- No synthetic gap filling.
- Last incomplete candle automatically removed at Silver layer.
- Dataset considered frozen as v1 baseline.

---

## Anti-Data-Snooping Policy

The following elements are frozen in v1:

- Asset
- Timeframe
- Historical start
- Label horizon
- Model mode (Long/Flat)
- Validation scheme
- Retraining frequency
- Threshold
- Cost structure

Any modification requires explicit version increment.

---

## Version Declaration

This document formalizes the structural configuration of:

> v1 — Structural ML Validation (Single Asset)

Future iterations:
- v2 → Generalization (multi-asset)
- v3 → New hypothesis (e.g., Long/Short, regression, regime modeling)

---

## Gold Layer Implementation Decision

Gold v1 is implemented entirely in SQL (DuckDB).

Rationale:

- Deterministic and reproducible transformations.
- Clear separation between data engineering and modeling layers.
- Alignment with project architecture.
- Avoidance of notebook-driven data mutation.
- Reduced risk of temporal leakage through explicit window functions.

Python is reserved exclusively for modeling stages.

---

## Feature Selection Rationale (v1)

Gold v1 intentionally uses a minimal structural feature set.

### Hypothesis Being Tested

The objective of v1 is to validate whether basic price and volatility dynamics contain exploitable predictive information at a 3-hour horizon under realistic trading costs.

This version does NOT attempt feature maximization.

---

## Feature Set (v1)

The following features are included:

1. 1H return  
2. 3H rolling return  
3. 12H rolling return  
4. 12H rolling volatility  
5. Distance to MA20  
6. 20H volume z-score  

---

## Rationale Per Feature

### 1H Return
Captures immediate short-term price dynamics.

### 3H Return
Aligned with label horizon (t+3). Ensures structural coherence.

### 12H Return
Provides medium-scale momentum context.

### 12H Volatility
Captures regime dependency of predictability.

### MA20 Distance
Measures structural deviation from local mean.

### Volume Z-Score
Captures abnormal participation levels.

---

## Excluded Features

The following feature groups are intentionally excluded in v1:

- RSI, MACD and derivative oscillators
- Calendar effects
- Intraday session indicators
- Regime clustering features
- Nonlinear technical composites

Each excluded group represents a separate hypothesis and will be evaluated in later versions (v3+).

---

## Versioning Policy

Any addition of new features requires version increment.

---

## Target Storage Policy

Gold v1 stores both:

- `future_return` (continuous target)
- `label` (binary classification target)

`future_return` is computed as:

future_return(t) = Close(t+3) / Close(t) - 1

The binary label is derived from it:

label(t) = 1 if future_return > 0 else 0

Rationale:

- Preserve magnitude information.
- Enable threshold sensitivity analysis.
- Allow future regression modeling without rebuilding Gold.
- Improve diagnostic transparency.

`future_return` is never used as an input feature.

---

## Edge Handling Policy

Gold v1 applies strict edge handling to avoid leakage.

### Initial Rows

Rows lacking sufficient rolling history are removed.

No partial rolling windows are allowed.

### Final Rows

Rows where `future_return` is NULL (insufficient forward horizon) are removed.

### Missing Historical Data

Exchange-level historical gaps (<0.1%) are preserved.

No forward fill, back fill or interpolation is applied.

### General Rule

Gold contains only rows with:

- Fully defined features
- Fully defined target
- No synthetic data

---

## Physical Dataset Versioning

Gold datasets are immutable.

Gold v1 is stored as:

storage/gold/btcusdt_1h_v1.parquet

Versioning Rules:

- The file is never overwritten.
- Structural changes require version increment.
- Each dataset version corresponds to a frozen configuration.
- Gold versions must be traceable to a specific Git tag.

This guarantees full experimental reproducibility.