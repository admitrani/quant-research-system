# Quant Data Engineering System

## 🎯 Objective

This project builds a production-grade data engineering system
for financial market data ingestion, transformation, and recovery.

The long-term goal is to support quantitative trading strategies
with a robust, scalable, reproducible, and fault-tolerant pipeline.

This repository focuses on:

- Reliable incremental ingestion from APIs
- Medallion architecture (Bronze / Silver / Gold)
- SQL-based transformations (DuckDB)
- Runtime data quality validation
- Gap detection and automatic backfill
- Controlled historical reprocessing
- DAG-based orchestration
- Idempotent execution guarantees

---

## 🏗 Architecture Overview

The system follows a **Medallion Architecture** with operational safeguards.

### 🥉 Bronze (Raw Data Lake)

- Immutable storage
- Append-only (incremental mode)
- Partitioned by year/month
- Stored exactly as received from source
- Source of truth
- Supports:
  - Incremental loading
  - Gap detection
  - Surgical backfill
  - Range reprocessing

### 🥈 Silver (Clean Layer)

- Structured and typed schema
- Deduplicated
- Ordered and validated
- Runtime data integrity checks
- Analytical-ready dataset
- Exported to Parquet + materialized in DuckDB

### 🥇 Gold (Curated Layer)

- Strategy-ready datasets
- Feature engineering layer
- Aggregated / ML-ready tables
- Built on top of validated Silver data

---

## 🔁 Pipeline Execution Modes

The pipeline supports multiple controlled execution paths:

### 1️⃣ Incremental Mode (Default)

Fetches only new candles using watermark logic.

python -m orchestration.pipeline --stage ingestion

---

### 2️⃣ Gap Backfill Mode

Detects historical gaps and fills only missing candles.

python -m orchestration.pipeline --stage ingestion --backfill

Characteristics:
- No full reload
- No duplication
- Re-validates after repair
- Idempotent

---

### 3️⃣ Range Reprocessing Mode

Rewrites a specific historical window without affecting other data.

python -m orchestration.pipeline
--stage ingestion
--reprocess-start YYYY-MM-DD
--reprocess-end YYYY-MM-DD

Characteristics:
- Rewrites only affected rows
- Does not drop full partitions
- Preserves data outside range
- Fully controlled recovery

---

## 🧪 Data Quality & Validation

Runtime validation includes:

- Schema validation (Silver layer)
- Duplicate detection
- Null checks
- Range validation
- Ordering validation
- Temporal gap detection
- Volume anomaly detection (z-score)
- Post-backfill revalidation

The system is designed to fail fast on structural violations.

---

## 🧠 Design Principles

- Idempotent execution
- Deterministic transformations
- Recovery-first architecture
- Separation of raw vs transformed logic
- No silent failures
- Layered responsibility
- Append-only raw storage

---

## 📊 Execution Flow

API → Incremental Load → Raw Lake → Validation  
→ Silver Transformation → Runtime Checks → Export  
→ Gold (future feature layer)

---

## 🔄 Development Workflow

This repository follows a feature-branch workflow:

- main: production-ready code
- feature/*: isolated development branches

All new features are developed in feature branches
and merged into main.

---

## 🚀 Future Extensions

- ML feature store integration
- Live event-driven ingestion
- Metadata tracking table (run history)
- Cloud storage migration
- Orchestrator migration (Airflow/Prefect)
- Distributed compute support
