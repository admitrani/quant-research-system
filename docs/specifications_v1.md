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

> 4.8 v1 — Structural ML Validation (Single Asset)

Future iterations:
- v2 → Generalization (multi-asset)
- v3 → New hypothesis (e.g., Long/Short, regression, regime modeling)

---
