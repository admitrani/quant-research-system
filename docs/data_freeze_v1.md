# Data Freeze — BTCUSDT 1H (v1)

## Overview

This document formalizes the raw dataset freeze used for v1.

The objective of this freeze is to ensure:

- Reproducibility
- Temporal integrity
- Explicit data quality documentation
- Clear version control before model training

---

## Dataset Scope

- **Symbol:** BTCUSDT  
- **Interval:** 1H  
- **Data Source:** Binance API  
- **Range:** 2019-01-01 → latest available at freeze date  
- **Pipeline Mode:** Full rebuild (non-incremental)  

This dataset replaces any previous historical ingestion starting from 2020.

---

## Raw Integrity Report

| Metric                        | Value      |
|------------------------------|------------|
| Total candles                | 62,757     |
| Duplicates                   | 0          |
| Sorted (monotonic timestamps)| True       |
| Gaps detected                | 20         |
| Missing candles              | 60         |
| Missing ratio                | ~0.095%    |
| Max consecutive missing      | 10 candles |
| Mean missing per gap         | 3 candles  |
| Median missing per gap       | 2 candles  |

---

## Gap Investigation

After full rebuild, temporal gaps were detected by the raw audit system.

### Backfill Attempt

- Automated gap backfill executed.
- Binance API returned **0 candles** for missing ranges.
- Revalidation confirmed identical gap structure.

### Interpretation

The missing candles correspond to historical exchange-level discontinuities and are not caused by:

- Pagination errors
- Incremental ingestion errors
- Duplicate overwrites
- Timestamp misalignment

Gaps are distributed across multiple years and do not form large structural holes.

---

## Methodological Decision

The dataset is accepted **as-is** with the documented gaps.

No artificial correction is applied:

- ❌ No forward-fill  
- ❌ No synthetic imputation  
- ❌ No interpolation  
- ❌ No manual insertion  

### Justification

1. Missing ratio < 0.1% of total dataset.
2. Gaps are small and isolated.
3. Imputation would introduce synthetic bias.
4. Realistic modeling requires accepting exchange-level data imperfections.
5. Integrity > artificial continuity.

---

## Implications for Modeling

- Rolling features will operate on available observations.
- No timestamp duplication or disorder exists.
- Walk-forward validation remains valid.
- Label computation (t+3) is unaffected structurally.

This level of discontinuity is considered acceptable for professional quantitative research.

---

## Version Declaration

This dataset is officially frozen as:

> RAW BTCUSDT 1H — v1 Baseline Dataset

Any modification to:
- Historical range
- Source
- Gap treatment
- Structural integrity

Will require explicit version increment and documentation.

---
