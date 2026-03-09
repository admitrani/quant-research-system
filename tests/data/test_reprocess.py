import pandas as pd
from pathlib import Path
from unittest.mock import patch
from ingestion.reprocess import run_range_reprocess


def test_reprocess_rewrites_range(tmp_path):


    # Crear estructura raw previa
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

    with patch("ingestion.reprocess.fetch_klines", return_value=new_df):

        # Cambiar cwd temporalmente
        import os
        original_cwd = Path.cwd()
        os.chdir(tmp_path)

        try:
            inserted = run_range_reprocess(
                symbol="BTCUSDT",
                interval="1h",
                start_date="2021-05-03",
                end_date="2021-05-04"
            )

            # Validaciones
            # Debe haber insertado 3 filas
            assert inserted == 3

            # Verificar que solo existe un archivo nuevo
            parquet_files = list(base.glob("*.parquet"))

            # Leer todos los archivos y concatenar
            dfs = [pd.read_parquet(f) for f in parquet_files]
            df_after = pd.concat(dfs, ignore_index=True)

            # No debe haber duplicados
            assert df_after["0"].duplicated().sum() == 0

            # Debe contener exactamente los timestamps nuevos
            assert sorted(df_after["0"].tolist()) == sorted(new_df["0"].tolist())

        finally:
            os.chdir(original_cwd)