import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

def test_gold_v1_integrity():

    np.random.seed(42)

    # Create connection
    con = duckdb.connect(":memory:")

    prices = 40000 + np.cumsum(np.random.normal(0, 100, 200))
    
    df = pd.DataFrame({
        "open_time_utc": pd.date_range("2020-01-01", periods=200, freq="h"),
        "open_price": prices * 0.999,
        "high_price": prices * 1.001,
        "low_price": prices * 0.998,
        "close_price": prices,
        "volume": np.abs(np.random.normal(500, 50, 200))
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
    