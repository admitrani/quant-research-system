WITH silver AS (
    SELECT *
    FROM read_parquet('storage/silver/.../**/*.parquet')
),

cleaned AS (
    SELECT s.*
    FROM silver s
    WHERE NOT EXISTS (
        SELECT 1
        FROM bad_candles b
        WHERE s.open_time = b.open_time
    )
)

SELECT *
FROM cleaned;