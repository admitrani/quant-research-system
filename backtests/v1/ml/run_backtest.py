import pandas as pd
import numpy as np
import backtrader as bt
import logging
from pathlib import Path

from backtests.v1.ml.signal_engine import generate_ml_probabilities
from backtests.v1.ml.data_feed import MLDataFeed
from strategies.ml_strategy_v1 import MLStrategyV1
from backtests.v1.ml.metrics import extract_backtest_metrics
from backtests.v1.ml.equity_observer import EquityObserver
from backtests.v1.ml.plots import plot_equity_curve
from models.utils import get_risk_policy, get_execution_costs, get_exit_threshold

logger = logging.getLogger(__name__)


def run_backtest():

    # Generate ML probabilities — candidate config is loaded inside signal_engine
    df, candidate = generate_ml_probabilities()

    risk = get_risk_policy()
    costs = get_execution_costs()
    exit_threshold = get_exit_threshold()

    logger.info(f"Candidate model: {candidate['model']} | threshold: {candidate['threshold']} | max_depth: {candidate['max_depth']}")
    logger.info(f"Entry threshold: {candidate['threshold']} | Exit threshold: {exit_threshold}")
    logger.info(f"Costs — commission: {costs['commission']} | slippage: {costs['slippage']} | buffer: {costs['buffer']}")
    logger.info(f"Capital: {risk['initial_capital']} | Min capital: {risk['minimum_capital']} | Max DD limit: {risk['max_drawdown_limit']}")

    cerebro = bt.Cerebro()
    cerebro.addobserver(EquityObserver)

    data = MLDataFeed(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(
        MLStrategyV1,
        entry_threshold=candidate["threshold"],
        exit_threshold=exit_threshold,
        risk_fraction=risk["risk_fraction"],
        commission=costs["commission"],
        slippage=costs["slippage"],
        buffer=costs["buffer"],
        minimum_capital=risk["minimum_capital"],
        max_drawdown_limit=risk["max_drawdown_limit"],
        min_position_size=risk["min_position_size"],
    )

    cerebro.broker.setcash(risk["initial_capital"])
    cerebro.broker.setcommission(commission=costs["commission"])
    cerebro.broker.set_slippage_perc(perc=costs["slippage"])

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    results = cerebro.run()
    strat = results[0]

    equity_curve = strat.observers.equityobserver.lines.equity.array
    equity_series = pd.Series(equity_curve[:len(df)], index=df.index)
    plot_equity_curve(equity_series)
    equity_df = pd.DataFrame({"equity": equity_series})

    metrics = extract_backtest_metrics(cerebro, strat)

    output_path = Path(__file__).parent / "results"
    output_path.mkdir(exist_ok=True)
    metrics.to_frame(name="value").to_csv(output_path / "metrics_ml_v1.csv")
    equity_df.to_csv(output_path / "equity_ml_v1.csv", index=False)

    logger.info(f"ML final value: {metrics['Final Value']:.2f}")
    logger.info(f"ML CAGR: {metrics['CAGR']:.4f}")
    logger.info(f"ML Sharpe: {metrics['Sharpe']:.4f}")
    logger.info(f"ML Max DD: {metrics['Max Drawdown']:.2f}%")
    logger.info(f"ML Trades: {int(metrics['Trades'])} | Win Rate: {metrics['Win Rate']:.4f} | Profit Factor: {metrics['Profit Factor']:.4f}")
    logger.info(f"ML Exposure: {metrics['Exposure']:.4f} | Avg Trade Duration: {metrics['Avg Trade Duration']:.2f} bars")
    logger.info(f"Results saved to: {output_path}")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    run_backtest()