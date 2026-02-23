WITH base AS (
    SELECT *
    FROM silver_prices
),

returns AS (
    SELECT
        open_time,
        close_price,
        LAG(close_price) OVER (ORDER BY open_time) AS prev_close,
        (close_price / LAG(close_price) OVER (ORDER BY open_time) - 1) AS return_pct
    FROM base
),

ma AS (
    SELECT
        *,
        AVG(close_price) OVER (
            ORDER BY open_time
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS ma_20
    FROM returns
)

SELECT *
FROM ma
ORDER BY open_time;