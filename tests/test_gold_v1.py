import duckdb
from pathlib import Path


def test_gold_v1_integrity():

    # Check file exists
    gold_path = Path("storage/gold/btcusdt_1h_v1.parquet")
    assert gold_path.exists(), "Gold v1 parquet file does not exist."

    con = duckdb.connect("warehouse.duckdb")

    try:
        # Load gold table
        con.execute("DROP TABLE IF EXISTS gold_v1_test")
        con.execute(f"""
            CREATE TABLE gold_v1_test AS
            SELECT * FROM read_parquet('{gold_path}')
        """)

        # Basic schema validation
        columns = con.execute("PRAGMA table_info('gold_v1_test')").fetchall()
        column_names = [col[1] for col in columns]

        required_ohlc = [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume"
        ]

        for col in required_ohlc:
            assert col in column_names, f"Missing column {col} in Gold v1."

        # No NULL values in critical columns
        null_check = con.execute("""
            SELECT
                SUM(CASE WHEN return_1h IS NULL THEN 1 ELSE 0 END) +
                SUM(CASE WHEN return_3h IS NULL THEN 1 ELSE 0 END) +
                SUM(CASE WHEN return_12h IS NULL THEN 1 ELSE 0 END) +
                SUM(CASE WHEN volatility_12h IS NULL THEN 1 ELSE 0 END) +
                SUM(CASE WHEN ma20_distance IS NULL THEN 1 ELSE 0 END) +
                SUM(CASE WHEN volume_zscore IS NULL THEN 1 ELSE 0 END) +
                SUM(CASE WHEN future_return IS NULL THEN 1 ELSE 0 END)
            FROM gold_v1_test
        """).fetchone()[0]

        assert null_check == 0, "Gold v1 contains NULL values."

        # No duplicate timestamps
        duplicates = con.execute("""
            SELECT COUNT(*) - COUNT(DISTINCT open_time)
            FROM gold_v1_test
        """).fetchone()[0]

        assert duplicates == 0, "Gold v1 contains duplicate timestamps."

        # Sorted order validation
        unordered = con.execute("""
            SELECT COUNT(*)
            FROM (
                SELECT open_time,
                       LAG(open_time) OVER (ORDER BY open_time) AS prev_time
                FROM gold_v1_test
            )
            WHERE prev_time IS NOT NULL
            AND open_time <= prev_time
        """).fetchone()[0]

        assert unordered == 0, "Gold v1 is not strictly ordered by time."

        # Basic sanity check: dataset not empty
        row_count = con.execute("SELECT COUNT(*) FROM gold_v1_test").fetchone()[0]

        assert row_count > 1000, "Gold v1 unexpectedly small."

    finally:
        con.close()