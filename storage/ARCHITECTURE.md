Raw layer behaves as Data Lake (append-only, file-based storage).

Silver/Gold simulate analytical warehouse layers.

Future upgrade path: migrate Silver/Gold to SQL engine (DuckDB/Postgres/Cloud warehouse).

System analytical mode: OLAP-first.
Operational layer (future execution) may require OLTP.

Storage format: Columnar (Parquet)
Reason: OLAP-oriented workload (feature engineering, ML training, historical backtesting).

File format standard: Parquet
Reason: Columnar storage, compression efficiency, OLAP performance, ML compatibility.
CSV allowed only for debugging or export.

Partitioning strategy:
- Primary partition key: year/month
- Optional clustering key: symbol
Reason: Time-series workload with window-based backtesting.

Data modeling approach:
- Silver: structured fact/dimension separation.
- Gold: analytical denormalized datasets for ML.
- Primary key for prices: (asset_id, timestamp).
