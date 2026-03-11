import logging
import backtrader as bt
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from models.utils import get_backtest_start, get_execution_costs, get_risk_policy, get_annualization_factor
from config.config_loader import get_gold_path

logger = logging.getLogger(__name__)


# Strategies

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

    gold_path = get_gold_path()

    df = pd.read_parquet(gold_path)
    backtest_start = get_backtest_start()

    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"]).dt.tz_localize(None)
    df.set_index("open_time_utc", inplace=True)
    df = df[df.index >= backtest_start]

    df = df[["open_price", "high_price", "low_price", "close_price", "volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]

    logger.info(f"Data loaded | bars: {len(df)} | from: {df.index[0].date()} | to: {df.index[-1].date()}")

    return df


# Shared helpers

def compute_metrics_from_returns(returns_series, annualization_factor):
    """Compute Sharpe (daily), Sortino (hourly), and Volatility (hourly)."""

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


def _setup_cerebro(df, strategy_class, strategy_params, costs, risk):
    """Common Backtrader cerebro setup for all baseline strategies."""

    cerebro = bt.Cerebro()

    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)

    cerebro.addstrategy(strategy_class, **strategy_params)

    cerebro.broker.setcash(risk["initial_capital"])
    cerebro.broker.setcommission(commission=costs["commission"])
    cerebro.broker.set_slippage_perc(costs["slippage"])

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    return cerebro


def _build_equity_df(strat, initial_capital):
    """Build equity DataFrame from TimeReturn analyzer."""

    returns_dict = strat.analyzers.timereturn.get_analysis()
    equity_df = pd.DataFrame(list(returns_dict.items()), columns=["datetime", "return"])
    equity_df["equity_curve"] = (1 + equity_df["return"]).cumprod() * initial_capital

    return equity_df


def _compute_base_metrics(strat, cerebro, equity_df, risk, annualization_factor):
    """Compute metrics common to all strategies: Sharpe, CAGR, Calmar, etc."""

    returns_series = equity_df.set_index("datetime")["return"]
    sharpe, sortino, volatility = compute_metrics_from_returns(returns_series, annualization_factor)

    final_value = cerebro.broker.getvalue()
    max_dd = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]

    years = (equity_df["datetime"].iloc[-1] - equity_df["datetime"].iloc[0]).days / 365.25
    cagr = (final_value / risk["initial_capital"]) ** (1 / years) - 1 if years > 0 else None
    calmar = cagr / (max_dd / 100) if max_dd and max_dd > 0 else None

    exposure = strat.bars_in_market / strat.total_bars if strat.total_bars > 0 else 0

    return {
        "final_value": final_value,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility": volatility,
        "max_drawdown_pct": max_dd,
        "calmar_ratio": calmar,
        "exposure": exposure,
    }


def _save_results(metrics, equity_df, results_path, name, plot_title):
    """Save metrics CSV, equity CSV, and equity plot."""

    results_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(results_path / f"metrics_{name}_v1.csv", index=False)
    equity_df.to_csv(results_path / f"equity_{name}_v1.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["datetime"], equity_df["equity_curve"])
    plt.title(plot_title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_path / f"equity_{name}_v1.png")
    plt.close()


# Backtest runners

def run_backtest():

    costs = get_execution_costs()
    risk = get_risk_policy()
    execution_buffer = 1 - (costs["commission"] + costs["slippage"] + costs["buffer"])
    annualization_factor = get_annualization_factor()

    logger.info("Starting EMA baseline backtest")
    logger.info(f"Costs — commission: {costs['commission']} | slippage: {costs['slippage']} | buffer: {costs['buffer']}")
    logger.info(f"Capital: {risk['initial_capital']} | Min capital: {risk['minimum_capital']}")

    df = load_data()

    cerebro = _setup_cerebro(
        df=df,
        strategy_class=EMABaseline,
        strategy_params={
            "execution_buffer": execution_buffer,
            "minimum_capital": risk["minimum_capital"],
            "min_position_size": risk["min_position_size"],
        },
        costs=costs,
        risk=risk,
    )

    results = cerebro.run()
    strat = results[0]

    equity_df = _build_equity_df(strat, risk["initial_capital"])
    base = _compute_base_metrics(strat, cerebro, equity_df, risk, annualization_factor)

    # EMA-specific trade metrics
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

    metrics = {
        **base,
        "total_trades": total_closed,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "net_pnl": net_pnl,
        "avg_trade_duration": avg_trade_duration,
    }

    results_path = Path(__file__).parent / "results"
    _save_results(metrics, equity_df, results_path, "ema", "v1 EMA Baseline Equity Curve")

    logger.info(f"EMA final value: {base['final_value']:.2f}")
    logger.info(f"EMA CAGR: {base['cagr']:.4f} | Sharpe: {base['sharpe']:.4f} | Max DD: {base['max_drawdown_pct']:.2f}%")
    logger.info(f"EMA Trades: {total_closed} | Win Rate: {win_rate:.4f} | Profit Factor: {profit_factor:.4f}")
    logger.info(f"EMA Exposure: {base['exposure']:.4f} | Avg Trade Duration: {avg_trade_duration} bars")
    logger.info(f"Results saved to: {results_path}")

    return cerebro, strat, metrics, equity_df


def run_buy_and_hold():

    costs = get_execution_costs()
    risk = get_risk_policy()
    execution_buffer = 1 - (costs["commission"] + costs["slippage"] + costs["buffer"])
    annualization_factor = get_annualization_factor()

    logger.info("Starting Buy & Hold backtest")
    logger.info(f"Costs — commission: {costs['commission']} | slippage: {costs['slippage']} | buffer: {costs['buffer']}")
    logger.info(f"Capital: {risk['initial_capital']}")

    df = load_data()

    cerebro = _setup_cerebro(
        df=df,
        strategy_class=BuyAndHold,
        strategy_params={
            "execution_buffer": execution_buffer,
        },
        costs=costs,
        risk=risk,
    )

    results = cerebro.run()
    strat = results[0]

    equity_df = _build_equity_df(strat, risk["initial_capital"])
    base = _compute_base_metrics(strat, cerebro, equity_df, risk, annualization_factor)

    # Buy & Hold opens exactly one position and never closes it, so
    # per-trade metrics (win_rate, profit_factor, expectancy) are not
    # applicable and are set to None to avoid misleading comparisons.
    trades = strat.analyzers.trades.get_analysis()
    total_closed = trades.get("total", {}).get("closed", 0)

    metrics = {
        **base,
        "total_trades": total_closed,
        "win_rate": None,
        "profit_factor": None,
        "expectancy_per_trade": None,
        "net_pnl": base["final_value"] - risk["initial_capital"],
        "avg_trade_duration": None,
    }

    results_path = Path(__file__).parent / "results"
    _save_results(metrics, equity_df, results_path, "buy_hold", "v1 Buy and Hold Equity Curve")

    logger.info(f"Buy & Hold final value: {base['final_value']:.2f}")
    logger.info(f"Buy & Hold CAGR: {base['cagr']:.4f} | Sharpe: {base['sharpe']:.4f} | Max DD: {base['max_drawdown_pct']:.2f}%")
    logger.info(f"Results saved to: {results_path}")

    return cerebro, strat, metrics, equity_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    run_backtest()
    run_buy_and_hold()
    