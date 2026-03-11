import pandas as pd
from pathlib import Path
from ingestion.market_data import get_last_timestamp


def test_get_last_timestamp_empty(tmp_path, monkeypatch):

    # Crear estructura vacía
    base = tmp_path / "storage/raw/source=binance/dataset=klines/symbol=BTCUSDT/interval=1h"
    base.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)

    result = get_last_timestamp("BTCUSDT", "1h")
    assert result is None


def test_get_last_timestamp_with_data(tmp_path, monkeypatch):

    base = tmp_path / "storage/raw/source=binance/dataset=klines/symbol=BTCUSDT/interval=1h/year=2020/month=01"
    base.mkdir(parents=True)

    df = pd.DataFrame({
        "0": [1000, 2000, 3000]
    })

    file_path = base / "data.parquet"
    df.to_parquet(file_path, index=False)

    monkeypatch.chdir(tmp_path)

    result = get_last_timestamp("BTCUSDT", "1h")
    assert result == 3000
