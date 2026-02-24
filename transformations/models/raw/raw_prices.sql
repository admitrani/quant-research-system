SELECT *
FROM read_parquet('storage/raw/source=binance/dataset=klines/**/*.parquet')