import logging

logger = logging.getLogger(__name__)


def validate_silver_prices(con):

    # Schema Validation (Data Contract)
    logger.info("Running Silver schema validation...")

    columns = con.execute("PRAGMA table_info('silver_prices')").fetchall()
    
    column_names = [col[1] for col in columns]
    column_types = {col[1]: col[2] for col in columns}

    # Required columns (minimum contract)
    required_columns = [
        "open_time",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "open_time_utc"
    ]

    missing_columns = [col for col in required_columns if col not in column_names]

    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise Exception("Schema validation failed: missing columns")

    # Type validation
    expected_types = {
        "open_time": "BIGINT",
        "open_price": "DOUBLE",
        "high_price": "DOUBLE",
        "low_price": "DOUBLE",
        "close_price": "DOUBLE",
        "volume": "DOUBLE",
        "close_time": "BIGINT",
        "quote_asset_volume": "DOUBLE",
        "number_of_trades": "BIGINT",
        "taker_buy_base_volume": "DOUBLE",
        "taker_buy_quote_volume": "DOUBLE",
        "open_time_utc": "TIMESTAMP"
    }

    for col, expected_type in expected_types.items():
        actual_type = column_types.get(col)

        if actual_type is None or expected_type not in actual_type.upper():
            logger.error(f"Type mismatch for {col}. Expected: {expected_type}, Found: {actual_type}")
            raise Exception("Schema type validation failed: type mismatch")

    logger.info("Schema validation passed.")

    # Data integrity tests
    logger.info("Running Silver data integrity tests...")

    # Test 1: Check for unique values
    duplicate_count = con.execute("""SELECT COUNT(*) - COUNT(DISTINCT open_time) FROM silver_prices""").fetchone()[0]

    # Test 2: Nulls
    nulls = con.execute("""
        SELECT
            SUM(CASE WHEN open_time IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN close_price IS NULL THEN 1 ELSE 0 END),
            SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END)
        FROM silver_prices
    """).fetchone()

    # Test 3: Range checks
    invalid_prices = con.execute("""SELECT COUNT(*) FROM silver_prices WHERE close_price <= 0""").fetchone()[0]
    invalid_volume = con.execute("""SELECT COUNT(*) FROM silver_prices WHERE volume < 0""").fetchone()[0]
    invalid_trades = con.execute("""SELECT COUNT(*) FROM silver_prices WHERE number_of_trades < 0""").fetchone()[0]

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

    logger.info(f"Duplicate rows: {duplicate_count}")
    logger.info(f"Null open_time: {nulls[0]}")
    logger.info(f"Null close_price: {nulls[1]}")
    logger.info(f"Null volume: {nulls[2]}")
    logger.info(f"Invalid prices (<= 0): {invalid_prices}")
    logger.info(f"Invalid volume (< 0): {invalid_volume}")
    logger.info(f"Invalid trades (< 0): {invalid_trades}")
    logger.info(f"Unordered rows: {unordered_rows}")

    # fail fast
    if (
        duplicate_count > 0
        or nulls[0] > 0
        or nulls[1] > 0
        or nulls[2] > 0
        or invalid_prices > 0
        or invalid_volume > 0
        or invalid_trades > 0
        or unordered_rows > 0
    ):
        logger.error("Silver model validation failed.")
        raise Exception("Silver validation error")
    
    logger.info("Silver model validation passed.")

