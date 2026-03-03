import duckdb
import pandas as pd
from pathlib import Path

def test_gold_v1_integrity():

    # Create in-memory DuckDB
    con = duckdb.connect(":memory:")

    # Create minimal mock silver_prices table
    df = pd.DataFrame({
        "open_time_utc": pd.date_range("2020-01-01", periods=200, freq="h"),
        "open_price": range(200),
        "high_price": range(1, 201),
        "low_price": range(200),
        "close_price": range(1, 201),
        "volume": [1000 + i for i in range(200)]
    })

    con.register("silver_prices", df)

    # Load gold SQL
    sql_path = Path("transformations/models/gold/gold_v1.sql")
    gold_sql = sql_path.read_text()

    # Execute transformation
    con.execute(f"CREATE TABLE gold_v1 AS {gold_sql}")

    # Check table exists
    result = con.execute("SELECT COUNT(*) FROM gold_v1").fetchone()[0]

    assert result > 0, "Gold transformation returned empty result."

    con.close()