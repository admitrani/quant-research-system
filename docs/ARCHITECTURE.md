# System Architecture

## 1. Storage Strategy

Raw layer behaves as a Data Lake (file-based, append-only).

Silver/Gold simulate analytical warehouse layers using DuckDB.

Current mode: Batch-first, OLAP-oriented.

---

## 2. File Format

Storage format: Parquet  
Reason:

- Columnar compression
- Efficient OLAP scans
- ML compatibility
- Backtesting performance

CSV allowed only for debugging.

---

## 3. Partitioning Strategy

Primary partition key:
- year
- month

Reason:
- Time-series workload
- Window-based backtesting
- Controlled historical rewriting

---

## 4. Ingestion Modes

### Incremental

- Uses watermark (max open_time from raw)
- Fetches only new data
- Append-only write

### Gap Backfill

- Detects temporal gaps
- Fetches only missing ranges
- Preserves existing data
- Re-validates after repair

### Range Reprocess

- Rewrites only rows inside specified window
- Does NOT drop full partitions
- Preserves out-of-range data
- Controlled historical recovery

---

## 5. Data Guarantees

The system guarantees:

- No duplication at Silver layer
- Deterministic transformations
- Idempotent incremental runs
- Recovery convergence
- No silent corruption

Raw is treated as:

- Source of truth
- Immutable in incremental mode
- Rewrite-only under explicit reprocess

---

## 6. Validation Strategy

### Raw Validation

- Duplicate detection
- Sorted check
- Gap detection
- Missing candle count

### Silver Validation

- Schema validation
- Null checks
- Price/volume sanity rules
- Order validation
- Runtime anomaly detection

System is designed to fail fast on structural violations.

---

## 7. Orchestration Model

Pipeline execution is stage-based:

- ingestion
- silver
- gold
- model

Supports:

- Partial stage execution
- Controlled backfill
- Controlled reprocess
- Runtime logging

DAG execution order:

ingestion → silver → gold → model

---

## 8. Recovery Model

There are three recovery levels:

1. Retry (transient failures)
2. Gap backfill (partial historical loss)
3. Range reprocess (structural correction)

This layered recovery model prevents:

- Full reload necessity
- Historical corruption
- Cascade failures

---

## 9. Upgrade Path

Future improvements:

- Metadata tracking table (run registry)
- Live ingestion mode
- Cloud object storage
- Distributed compute
- Data contracts
- Observability dashboard