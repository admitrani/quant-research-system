import pandas as pd
from pathlib import Path
from unittest.mock import patch
from ingestion.reprocess import run_range_reprocess


def test_reprocess_rewrites_range(tmp_path, monkeypatch):

    # Create Raw structure
    base = tmp_path / "storage/raw/source=binance/dataset=klines/symbol=BTCUSDT/interval=1h/year=2021/month=05"
    base.mkdir(parents=True)

    old_df = pd.DataFrame({
        "0": [1620000000000, 1620003600000],  # timestamps dummy
    })

    old_file = base / "old.parquet"
    old_df.to_parquet(old_file, index=False)

    # Mock fetch_klines
    new_df = pd.DataFrame({
        "0": [1620000000000, 1620003600000, 1620007200000],
    })

    monkeypatch.chdir(tmp_path)

    with patch("ingestion.reprocess.fetch_klines", return_value=new_df):

        inserted = run_range_reprocess(
            symbol="BTCUSDT",
            interval="1h",
            start_date="2021-05-03",
            end_date="2021-05-04"
        )

        assert inserted == 3

        parquet_files = list(base.glob("*.parquet"))

        dfs = [pd.read_parquet(f) for f in parquet_files]
        df_after = pd.concat(dfs, ignore_index=True)

        assert df_after["0"].duplicated().sum() == 0

        assert sorted(df_after["0"].tolist()) == sorted(new_df["0"].tolist())
    