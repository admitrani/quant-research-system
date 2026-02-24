import logging

logger = logging.getLogger(__name__)


def validate_silver_prices(con):
    logger.info("Running Silver model tests...")

    # Test 1: Check for unique values
    duplicate_count = con.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT open_time)
        FROM silver_prices
    """).fetchone()[0]

    # Test 2: Nulls
    nulls = con.execute("""
        SELECT
            SUM(CASE WHEN open_time IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN close_price IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END)
        FROM silver_prices
    """).fetchone()

    # Test 3: Range
    invalid_prices = con.execute("""
        SELECT COUNT(*)
        FROM silver_prices
        WHERE close_price <= 0;
    """).fetchone()[0]

    # Test 4: Temporal consistency
    unordered_rows = con.execute("""
        SELECT COUNT(*)
        FROM (
            SELECT
                open_time,
                LAG(open_time) OVER (ORDER BY open_time) AS prev_time
            FROM silver_prices
        )
        WHERE prev_time IS NOT NULL
        AND open_time <= prev_time;
    """).fetchone()[0]

    # Logging
    logger.info(f"Duplicate rows: {duplicate_count}")
    logger.info(f"Null open_time: {nulls[0]}")
    logger.info(f"Null close_price: {nulls[1]}")
    logger.info(f"Null volume: {nulls[2]}")
    logger.info(f"Invalid prices (<= 0): {invalid_prices}")
    logger.info(f"Unordered rows: {unordered_rows}")

    # fail fast
    if (
        duplicate_count > 0
        or nulls[0] > 0
        or nulls[1] > 0
        or nulls[2] > 0
        or invalid_prices > 0
        or unordered_rows > 0
    ):
        logger.error("Silver model validation failed.")
        raise Exception("Silver validation error")
    
    logger.info("Silver model validation passed.")

