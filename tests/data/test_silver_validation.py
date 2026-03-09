import duckdb
import pandas as pd
from transformations.validators.silver_validator import validate_silver_prices


def test_silver_validator_passes_clean_data():

    con = duckdb.connect(":memory:")

    df = pd.DataFrame({
        "open_time": pd.Series([1, 2, 3], dtype="int64"),
        "open_price": pd.Series([100.0, 101.0, 102.0], dtype="float64"),
        "high_price": pd.Series([101.0, 102.0, 103.0], dtype="float64"),
        "low_price": pd.Series([99.0, 100.0, 101.0], dtype="float64"),
        "close_price": pd.Series([100.5, 101.5, 102.5], dtype="float64"),
        "volume": pd.Series([10.0, 12.0, 15.0], dtype="float64"),
        "close_time": pd.Series([2, 3, 4], dtype="int64"),
        "quote_asset_volume": pd.Series([1000.0, 1200.0, 1500.0], dtype="float64"),
        "number_of_trades": pd.Series([5, 6, 7], dtype="int64"),
        "taker_buy_base_volume": pd.Series([5.0, 6.0, 7.0], dtype="float64"),
        "taker_buy_quote_volume": pd.Series([500.0, 600.0, 700.0], dtype="float64"),
        "open_time_utc": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    })

    con.register("df", df)
    con.execute("CREATE TABLE silver_prices AS SELECT * FROM df")

    validate_silver_prices(con)

    con.close()