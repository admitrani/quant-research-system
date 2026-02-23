WITH raw_data AS (
    SELECT *
    FROM read_parquet('storage/raw/source=binance/dataset=klines/symbol=BTCUSDT/interval=1h/**/*.parquet')
),

typed AS (
    SELECT
        CAST("0" AS BIGINT)  AS open_time,
        CAST("1" AS DOUBLE)  AS open_price,
        CAST("2" AS DOUBLE)  AS high_price,
        CAST("3" AS DOUBLE)  AS low_price,
        CAST("4" AS DOUBLE)  AS close_price,
        CAST("5" AS DOUBLE)  AS volume,
        CAST("6" AS BIGINT)  AS close_time,
        CAST("7" AS DOUBLE)  AS quote_asset_volume,
        CAST("8" AS BIGINT)  AS number_of_trades,
        CAST("9" AS DOUBLE)  AS taker_buy_base_volume,
        CAST("10" AS DOUBLE) AS taker_buy_quote_volume
    FROM raw_data
),

with_timestamp AS (
    SELECT *,
           TO_TIMESTAMP(open_time / 1000) AT TIME ZONE 'UTC' AS open_time_utc
    FROM typed
),

deduplicated AS (
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY open_time ORDER BY open_time) AS rn
        FROM with_timestamp
    )
    WHERE rn = 1
)

SELECT
    open_time,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    close_time,
    quote_asset_volume,
    number_of_trades,
    taker_buy_base_volume,
    taker_buy_quote_volume,
    open_time_utc
FROM deduplicated
ORDER BY open_time;