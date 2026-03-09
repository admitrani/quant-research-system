import logging
import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from models.utils import get_backtest_start, get_execution_costs, get_risk_policy, get_annualization_factor

logger = logging.getLogger(__name__)


# Strategy definition

class EMABaseline(bt.Strategy):

    params = (
        ("fast_period", 10),
        ("slow_period", 20),
        ("execution_buffer", None),
        ("minimum_capital", None),
        ("min_position_size", None),
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.fast_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)
        self.bars_in_market = 0
        self.total_bars = 0
        logger.info(
            f"EMABaseline initialised | "
            f"fast_period: {self.p.fast_period} | slow_period: {self.p.slow_period} | "
            f"execution_buffer: {self.p.execution_buffer:.4f} | minimum_capital: {self.p.minimum_capital}"
        )

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            logger.warning(f"ORDER FAILED | Status: {order.getstatusname()} | Size: {order.size}")

    def next(self):
        self.total_bars += 1
        if self.position:
            self.bars_in_market += 1

        if self.broker.getvalue() < self.p.minimum_capital:
            return

        if not self.position:
            if self.crossover > 0:
                size = (self.broker.get_cash() * self.p.execution_buffer) / self.data.close[0]
                if size > self.p.min_position_size:
                    self.buy(size=size)
        else:
            if self.crossover < 0:
                self.close()


class BuyAndHold(bt.Strategy):

    params = (
        ("execution_buffer", None),
    )

    def __init__(self):
        self.bought = False
        self.total_bars = 0
        self.bars_in_market = 0
        logger.info(f"BuyAndHold initialised | execution_buffer: {self.p.execution_buffer:.4f}")

    def next(self):
        self.total_bars += 1
        if not self.bought:
            size = (self.broker.get_cash() * self.p.execution_buffer) / self.data.close[0]
            if size > 1e-8:
                self.buy(size=size)
                self.bought = True
        if self.position:
            self.bars_in_market += 1


# Data loading

def load_data():

    project_root = Path(__file__).resolve().parents[3]
    gold_path = project_root / "storage" / "gold" / "btcusdt_1h_v1.parquet"

    df = pd.read_parquet(gold_path)
    backtest_start = get_backtest_start()

    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"]).dt.tz_localize(None)
    df.set_index("open_time_utc", inplace=True)
    df = df[df.index >= backtest_start]

    df = df[["open_price", "high_price", "low_price", "close_price", "volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]

    logger.info(f"Data loaded | bars: {len(df)} | from: {df.index[0].date()} | to: {df.index[-1].date()}")

    return df


def compute_metrics_from_returns(returns_series, annualization_factor):

    returns_daily = returns_series.resample("D").apply(lambda x: (1 + x).prod() - 1).dropna()
    mean_d = returns_daily.mean()
    std_d = returns_daily.std()
    sharpe = (mean_d / std_d) * np.sqrt(365) if std_d != 0 else None

    returns_arr = returns_series.values
    std_h = returns_arr.std()
    mean_h = returns_arr.mean()
    volatility = std_h * np.sqrt(annualization_factor)

    downside = returns_arr[returns_arr < 0]
    downside_std = downside.std() * np.sqrt(annualization_factor) if len(downside) > 0 else 0
    sortino = mean_h / downside_std if downside_std > 0 else None

    return sharpe, sortino, volatility


# Backtest runner

def run_backtest():

    costs = get_execution_costs()
    risk = get_risk_policy()
    execution_buffer = 1 - (costs["commission"] + costs["slippage"] + costs["buffer"])
    annualization_factor = get_annualization_factor()

    logger.info("Starting EMA baseline backtest")
    logger.info(f"Costs — commission: {costs['commission']} | slippage: {costs['slippage']} | buffer: {costs['buffer']}")
    logger.info(f"Capital: {risk['initial_capital']} | Min capital: {risk['minimum_capital']}")

    cerebro = bt.Cerebro()

    df = load_data()
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(
        EMABaseline,
        execution_buffer=execution_buffer,
        minimum_capital=risk["minimum_capital"],
        min_position_size=risk["min_position_size"],
    )

    cerebro.broker.setcash(risk["initial_capital"])
    cerebro.broker.setcommission(commission=costs["commission"])
    cerebro.broker.set_slippage_perc(costs["slippage"])

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro.run()
    strat = results[0]

    exposure = strat.bars_in_market / strat.total_bars if strat.total_bars > 0 else 0

    returns_dict = strat.analyzers.timereturn.get_analysis()
    equity_df = pd.DataFrame(list(returns_dict.items()), columns=["datetime", "return"])
    equity_df["equity_curve"] = (1 + equity_df["return"]).cumprod() * risk["initial_capital"]

    returns_series = equity_df.set_index("datetime")["return"]
    sharpe, sortino, volatility = compute_metrics_from_returns(returns_series, annualization_factor)

    final_value = strat.broker.getvalue()
    max_dd = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]

    years = (equity_df["datetime"].iloc[-1] - equity_df["datetime"].iloc[0]).days / 365
    cagr = (final_value / risk["initial_capital"]) ** (1 / years) - 1 if years > 0 else None
    calmar = cagr / (max_dd / 100) if max_dd > 0 else None

    trades = strat.analyzers.trades.get_analysis()
    gross_profit = trades["won"]["pnl"]["total"]
    gross_loss = abs(trades["lost"]["pnl"]["total"])
    net_pnl = trades["pnl"]["net"]["total"]
    total_closed = trades["total"]["closed"]
    won = trades["won"]["total"]

    profit_factor = gross_profit / gross_loss if gross_loss != 0 else None
    expectancy = net_pnl / total_closed if total_closed != 0 else None
    win_rate = won / total_closed if total_closed > 0 else None
    avg_trade_duration = trades.get("len", {}).get("average", None)

    results_path = Path(__file__).parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "final_value": final_value,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": volatility,
        "max_drawdown_pct": max_dd,
        "calmar_ratio": calmar,
        "total_trades": total_closed,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "net_pnl": net_pnl,
        "exposure": exposure,
        "avg_trade_duration": avg_trade_duration,
    }

    pd.DataFrame([metrics]).to_csv(results_path / "metrics_ema_v1.csv", index=False)
    equity_df.to_csv(results_path / "equity_ema_v1.csv", index=False)

    logger.info(f"EMA final value: {final_value:.2f}")
    logger.info(f"EMA CAGR: {cagr:.4f} | Sharpe: {sharpe:.4f} | Max DD: {max_dd:.2f}%")
    logger.info(f"EMA Trades: {total_closed} | Win Rate: {win_rate:.4f} | Profit Factor: {profit_factor:.4f}")
    logger.info(f"EMA Exposure: {exposure:.4f} | Avg Trade Duration: {avg_trade_duration} bars")
    logger.info(f"Results saved to: {results_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["datetime"], equity_df["equity_curve"])
    plt.title("v1 EMA Baseline Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / "equity_ema_v1.png")
    plt.close()

    return cerebro, strat, metrics, equity_df


def run_buy_and_hold():

    costs = get_execution_costs()
    risk = get_risk_policy()
    execution_buffer = 1 - (costs["commission"] + costs["slippage"] + costs["buffer"])
    annualization_factor = get_annualization_factor()

    logger.info("Starting Buy & Hold backtest")
    logger.info(f"Costs — commission: {costs['commission']} | slippage: {costs['slippage']} | buffer: {costs['buffer']}")
    logger.info(f"Capital: {risk['initial_capital']}")

    cerebro_bh = bt.Cerebro()

    df = load_data()
    data = bt.feeds.PandasData(dataname=df)
    cerebro_bh.adddata(data)

    cerebro_bh.addstrategy(
        BuyAndHold,
        execution_buffer=execution_buffer,
    )

    cerebro_bh.broker.setcash(risk["initial_capital"])
    cerebro_bh.broker.setcommission(commission=costs["commission"])
    cerebro_bh.broker.set_slippage_perc(costs["slippage"])

    cerebro_bh.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro_bh.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro_bh.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro_bh.addanalyzer(bt.analyzers.Returns, _name="returns")

    results = cerebro_bh.run()
    strat_bh = results[0]

    exposure = strat_bh.bars_in_market / strat_bh.total_bars if strat_bh.total_bars > 0 else 0

    returns_dict = strat_bh.analyzers.timereturn.get_analysis()
    equity_bh = pd.DataFrame(list(returns_dict.items()), columns=["datetime", "return"])
    equity_bh["equity_curve"] = (1 + equity_bh["return"]).cumprod() * risk["initial_capital"]

    returns_series_bh = equity_bh.set_index("datetime")["return"]
    sharpe, sortino, volatility = compute_metrics_from_returns(returns_series_bh, annualization_factor)

    final_value = strat_bh.broker.getvalue()
    max_dd = strat_bh.analyzers.drawdown.get_analysis()["max"]["drawdown"]

    years = (equity_bh["datetime"].iloc[-1] - equity_bh["datetime"].iloc[0]).days / 365
    cagr = (final_value / risk["initial_capital"]) ** (1 / years) - 1 if years > 0 else None
    calmar = cagr / (max_dd / 100) if max_dd > 0 else None

    trades = strat_bh.analyzers.trades.get_analysis()
    total_closed = trades.get("total", {}).get("closed", 0)
    avg_trade_duration = None  # Buy & Hold never closes a position

    results_path = Path(__file__).parent / "results"
    results_path.mkdir(parents=True, exist_ok=True)

    metrics_bh = {
        "final_value": final_value,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": volatility,
        "max_drawdown_pct": max_dd,
        "calmar_ratio": calmar,
        "total_trades": total_closed,
        "win_rate": None,
        "profit_factor": None,
        "expectancy_per_trade": None,
        "net_pnl": final_value - risk["initial_capital"],
        "exposure": exposure,
        "avg_trade_duration": avg_trade_duration,
    }

    pd.DataFrame([metrics_bh]).to_csv(results_path / "metrics_buy_hold_v1.csv", index=False)
    equity_bh.to_csv(results_path / "equity_buy_hold_v1.csv", index=False)

    logger.info(f"Buy & Hold final value: {final_value:.2f}")
    logger.info(f"Buy & Hold CAGR: {cagr:.4f} | Sharpe: {sharpe:.4f} | Max DD: {max_dd:.2f}%")
    logger.info(f"Results saved to: {results_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_bh["datetime"], equity_bh["equity_curve"])
    plt.title("v1 Buy and Hold Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / "equity_buy_hold_v1.png")
    plt.close()

    return cerebro_bh, strat_bh, metrics_bh, equity_bh


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_backtest()
    run_buy_and_hold()