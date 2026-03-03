import backtrader as bt
import pandas as pd
from pathlib import Path
import csv
import matplotlib.pyplot as plt

# Strategy definition

class EMABaseline(bt.Strategy):

    params = (
        ("fast_period", 10),
        ("slow_period", 20),
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.p.fast_period)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.p.slow_period)
        self.crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)

    def next(self):

        if not self.position:
            if self.crossover > 0:
                size = int(self.broker.get_cash() / self.data.close[0])
                self.buy(size=size)
        else:
            if self.crossover < 0:
                self.sell()

# Data loading

def load_data():

    file_path = Path("storage/gold/btcusdt_1h_v1.parquet")

    df = pd.read_parquet(file_path)

    df["open_time_utc"] = pd.to_datetime(df["open_time_utc"]).dt.tz_localize(None)
    df.set_index("open_time_utc", inplace=True)

    df = df[["open_price", "high_price", "low_price", "close_price", "volume"]]

    df.columns = ["open", "high", "low", "close", "volume"]

    return df

# Backtest runner

def run_backtest():

    cerebro = bt.Cerebro()
    
    df = load_data()
    data = bt.feeds.PandasData(dataname=df)

    cerebro.adddata(data)
    cerebro.addstrategy(EMABaseline)

    # Initial capital
    cerebro.broker.setcash(100000)

    # Commission + slippage
    cerebro.broker.setcommission(commission=0.0018) # 0.18%
    cerebro.broker.set_slippage_perc(0.0005)        # 0.05%

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    results = cerebro.run()
    strat = results[0]

    # Equity (daily returns)
    returns_dict = strat.analyzers.timereturn.get_analysis()
    equity_df = pd.DataFrame(list(returns_dict.items()), columns=["datetime", "return"])
    equity_df["equity_curve"] = (1 + equity_df["return"]).cumprod() * 100000

    final_value = strat.broker.getvalue()
    max_dd = strat.analyzers.drawdown.get_analysis()["max"]["drawdown"]

    #CAGR and Calmar
    start_value = 100000
    end_value = final_value
    years = (equity_df["datetime"].iloc[-1] - equity_df["datetime"].iloc[0]).days / 365
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else None
    calmar = cagr / (max_dd / 100) if max_dd > 0 else None

    # Sharpe
    mean_daily = equity_df["return"].mean()
    std_daily = equity_df["return"].std()
    sharpe = (mean_daily / std_daily) * (365 ** 0.5) if std_daily != 0 else None

    # Trade metrics
    trades = strat.analyzers.trades.get_analysis()
    gross_profit = trades["won"]["pnl"]["total"]
    gross_loss = abs(trades["lost"]["pnl"]["total"])
    net_pnl = trades["pnl"]["net"]["total"]
    total_closed = trades["total"]["closed"]

    profit_factor = gross_profit / gross_loss if gross_loss != 0 else None
    expectancy = net_pnl / total_closed if total_closed != 0 else None

    # Export
    results_path = Path("backtests/baseline/results")
    results_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        "final_value": final_value,
        "sharpe_daily": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": total_closed,
        "net_pnl": net_pnl,
        "profit_factor": profit_factor,
        "expectancy_per_trade": expectancy,
        "cagr": cagr,
        "calmar_ratio": calmar
    }

    pd.DataFrame([metrics]).to_csv(results_path / "metrics_v1.csv", index=False)
    equity_df.to_csv(results_path / "equity_v1.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(equity_df["datetime"], equity_df["equity_curve"])
    plt.title("v1 Baseline Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(results_path / "equity_v1.png")
    plt.close()

    return cerebro, strat, metrics, equity_df

if __name__ == "__main__":
    cerebro, strat, metrics, equity_df = run_backtest()

    print("Final Portfolio Value:", metrics["final_value"])
    print("Sharpe (manual daily):", metrics["sharpe_daily"])
    print("Max Drawdown %:", metrics["max_drawdown_pct"])
    print("Profit Factor:", metrics["profit_factor"])
    print("Expectancy per trade:", metrics["expectancy_per_trade"])
