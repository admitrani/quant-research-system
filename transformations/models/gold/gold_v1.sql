WITH base AS (
    SELECT *
    FROM silver_prices
),

-- Basic returns
returns AS (
    SELECT
        *,
        -- 1H return
        (close_price / LAG(close_price) OVER (ORDER BY open_time) - 1) 
            AS return_1h

    FROM base
),

-- Rolling features
rolling_features AS (
    SELECT
        *,
        
        -- 3H rolling return
        (close_price / LAG(close_price, 3) OVER (ORDER BY open_time) - 1)
            AS return_3h,

        -- 12H rolling return
        (close_price / LAG(close_price, 12) OVER (ORDER BY open_time) - 1)
            AS return_12h,

        -- 12H rolling volatility (std of 1H returns)
        STDDEV(return_1h) OVER (
            ORDER BY open_time
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
        ) AS volatility_12h,

        -- MA20
        AVG(close_price) OVER (
            ORDER BY open_time
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS ma20,

        -- Volume mean and std for z-score
        AVG(volume) OVER (
            ORDER BY open_time
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS volume_mean_20,

        STDDEV(volume) OVER (
            ORDER BY open_time
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS volume_std_20

    FROM returns
),

-- Final feature engineering
features AS (
    SELECT
        *,
        
        -- Distance to MA20
        (close_price / ma20 - 1) AS ma20_distance,

        -- Volume z-score
        CASE 
            WHEN volume_std_20 > 0 
            THEN (volume - volume_mean_20) / volume_std_20
            ELSE NULL
        END AS volume_zscore

    FROM rolling_features
),

-- Target (t+3)
target AS (
    SELECT
        *,
        
        (LEAD(close_price, 3) OVER (ORDER BY open_time) / close_price - 1)
            AS future_return

    FROM features
),

-- Binary label
labeled AS (
    SELECT
        *,
        CASE 
            WHEN future_return > 0 THEN 1
            ELSE 0
        END AS label
    FROM target
),

-- Edge handling (NULL removal)
cleaned AS (
    SELECT *
    FROM labeled
    WHERE
        return_1h IS NOT NULL
        AND return_3h IS NOT NULL
        AND return_12h IS NOT NULL
        AND volatility_12h IS NOT NULL
        AND ma20_distance IS NOT NULL
        AND volume_zscore IS NOT NULL
        AND future_return IS NOT NULL
)

SELECT
    open_time,
    open_time_utc,

    open_price,
    high_price,
    low_price,
    close_price,
    volume,

    return_1h,
    return_3h,
    return_12h,
    volatility_12h,
    ma20_distance,
    volume_zscore,

    future_return,
    label

FROM cleaned
ORDER BY open_time;