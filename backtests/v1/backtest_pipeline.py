import logging
import argparse
import time
import pandas as pd
from pathlib import Path

from models.utils import get_risk_policy, validate_risk_policy


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "backtest.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger(__name__)


# Stage runners

def run_ml():
    from backtests.v1.ml.run_backtest import run_backtest
    logger.info("Running ML strategy backtest...")
    run_backtest()  # metrics are logged inside run_backtest
    logger.info("ML backtest completed.")


def run_ema():
    from backtests.v1.baseline.ema_baseline import run_backtest as run_ema_backtest
    logger.info("Running EMA baseline backtest...")
    run_ema_backtest()
    logger.info("EMA backtest completed.")


def run_buy_and_hold():
    from backtests.v1.baseline.ema_baseline import run_buy_and_hold as run_bh_backtest
    logger.info("Running Buy & Hold backtest...")
    run_bh_backtest()
    logger.info("Buy & Hold backtest completed.")


def run_metrics():
    """Load results from all three backtests and produce a combined comparison CSV."""

    logger.info("Building comparison table...")

    ml_path   = Path("backtests/v1/ml/results/metrics_ml_v1.csv")
    ema_path  = Path("backtests/v1/baseline/results/metrics_ema_v1.csv")
    bh_path   = Path("backtests/v1/baseline/results/metrics_buy_hold_v1.csv")

    missing = [p for p in [ml_path, ema_path, bh_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot build comparison — missing results files: {[str(p) for p in missing]}. "
            "Run all three backtest stages first."
        )

    risk = get_risk_policy()
    initial_capital = risk["initial_capital"]

    def _expectancy_pct(avg_trade_return_usd):
        """
        Normalise avg trade P&L by initial capital.
        expectancy_pct = avg_trade_return_usd / initial_capital * 100
        Capital-independent: interpretable across strategies with the same starting equity.
        """
        if avg_trade_return_usd is None:
            return None
        return (float(avg_trade_return_usd) / initial_capital) * 100

    ml_raw = pd.read_csv(ml_path, index_col=0)["value"]
    ml_avg_usd = ml_raw.get("Avg Trade Return")
    ml = {
        "final_value": ml_raw.get("Final Value"),
        "cagr": ml_raw.get("CAGR"),
        "sharpe": ml_raw.get("Sharpe"),
        "sortino": ml_raw.get("Sortino"),
        "volatility": ml_raw.get("Volatility"),
        "max_drawdown_pct": ml_raw.get("Max Drawdown"),
        "calmar_ratio": ml_raw.get("Calmar"),
        "total_trades": ml_raw.get("Trades"),
        "win_rate": ml_raw.get("Win Rate"),
        "profit_factor": ml_raw.get("Profit Factor"),
        "avg_trade_return_usd": ml_avg_usd,
        "expectancy_pct": _expectancy_pct(ml_avg_usd),
        "exposure": ml_raw.get("Exposure"),
        "avg_trade_duration": ml_raw.get("Avg Trade Duration"),
    }

    ema = pd.read_csv(ema_path).iloc[0].to_dict()
    bh  = pd.read_csv(bh_path).iloc[0].to_dict()

    ema_avg_usd = ema.get("expectancy_per_trade")
    ema_norm = {
        "final_value": ema.get("final_value"),
        "cagr": ema.get("cagr"),
        "sharpe": ema.get("sharpe"),
        "sortino": ema.get("sortino"),
        "volatility": ema.get("volatility"),
        "max_drawdown_pct": ema.get("max_drawdown_pct"),
        "calmar_ratio": ema.get("calmar_ratio"),
        "total_trades": ema.get("total_trades"),
        "win_rate": ema.get("win_rate"),
        "profit_factor": ema.get("profit_factor"),
        "avg_trade_return_usd": ema_avg_usd,
        "expectancy_pct": _expectancy_pct(ema_avg_usd),
        "exposure": ema.get("exposure"),
        "avg_trade_duration": ema.get("avg_trade_duration"),
    }

    # Buy & Hold opens exactly one position and never closes it, so
    # per-trade metrics (win_rate, profit_factor, expectancy) are not
    # applicable and are set to None to avoid misleading comparisons.
    bh_norm = {
        "final_value": bh.get("final_value"),
        "cagr": bh.get("cagr"),
        "sharpe": bh.get("sharpe"),
        "sortino": bh.get("sortino"),
        "volatility": bh.get("volatility"),
        "max_drawdown_pct": bh.get("max_drawdown_pct"),
        "calmar_ratio": bh.get("calmar_ratio"),
        "total_trades": None,
        "win_rate": None,
        "profit_factor": None,
        "avg_trade_return_usd": None,
        "expectancy_pct": None,
        "exposure": bh.get("exposure"),
        "avg_trade_duration": None,
    }

    comparison = pd.DataFrame({
        "ML Strategy": ml,
        "EMA Baseline": ema_norm,
        "Buy & Hold": bh_norm,
    })

    output_path = Path("backtests/v1/results")
    output_path.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path / "comparison_v1.csv")

    logger.info("Comparison table saved to backtests/v1/results/comparison_v1.csv")
    logger.info("\n" + comparison.to_string())



def run_equity():
    from backtests.v1.equity_plot import plot_combined_equity
    logger.info("Generating combined equity curve plot...")
    plot_combined_equity()
    logger.info("Equity plot completed.")

# Pipeline

STAGES = [
    ("ml", run_ml),
    ("ema", run_ema),
    ("buy_and_hold", run_buy_and_hold),
    ("metrics", run_metrics),
    ("equity", run_equity),
]


def run_backtest_pipeline(stage=None):

    logger.info("=" * 60)
    logger.info("Starting backtest pipeline")

    pipeline_start = time.time()

    risk = get_risk_policy()
    validate_risk_policy(risk)

    logger.info("Risk policy loaded:")
    for k, v in risk.items():
        logger.info(f"  {k}: {v}")

    stages = list(STAGES)

    if stage and stage != "all":
        stage_names = [name for name, _ in stages]
        if stage not in stage_names:
            logger.error(f"Invalid stage: {stage}. Valid options: {stage_names}")
            raise ValueError(f"Invalid stage: {stage}")
        start_index = stage_names.index(stage)
        stages = stages[start_index:]

    try:
        for stage_name, stage_func in stages:

            logger.info(f"Stage: {stage_name}")
            stage_start = time.time()

            stage_func()

            duration = time.time() - stage_start
            logger.info(f"Stage {stage_name} completed in {duration:.2f}s")

        total_duration = time.time() - pipeline_start
        logger.info(f"Backtest pipeline finished successfully in {total_duration:.2f}s")

    except Exception:
        logger.exception("Backtest pipeline failed.")
        raise

    finally:
        logger.info("=" * 60)


# Entry point

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run v1 backtest pipeline")

    parser.add_argument(
        "--stage",
        type=str,
        choices=["ml", "ema", "buy_and_hold", "metrics", "equity", "all"],
        default="all",
        help=(
            "Stage to start from. Options: ml | ema | buy_and_hold | metrics | equity | all. "
            "Stages run sequentially from the selected stage onwards. "
            "Use 'metrics' alone only after all three backtest stages have run."
        )
    )

    args = parser.parse_args()

    run_backtest_pipeline(stage=args.stage)
