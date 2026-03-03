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
        self.bars_in_market = 0
        self.total_bars = 0

    def next(self):

        self.total_bars += 1
        if self.position:
            self.bars_in_market += 1

        if not self.position:
            if self.crossover > 0:
                size = int(self.broker.get_cash() / self.data.close[0])
                self.buy(size=size)
        else:
            if self.crossover < 0:
                self.sell()

class BuyAndHold(bt.Strategy):
    
    def __init__(self):
        self.bought = False
        self.total_bars = 0
        self.bars_in_market = 0

    def next(self):

        self.total_bars += 1
        if not self.bought:
            size = int(self.broker.get_cash() / self.data.close[0])
            self.buy(size=size)
            self.bought = True
        
        if self.position:
            self.bars_in_market += 1

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

    exposure = strat.bars_in_market / strat.total_bars if strat.total_bars > 0 else 0

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
        "calmar_ratio": calmar,
        "exposure": exposure
    }

    pd.DataFrame([metrics]).to_csv(results_path / "metrics_ema_v1.csv", index=False)
    equity_df.to_csv(results_path / "equity_ema_v1.csv", index=False)

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

    cerebro_bh = bt.Cerebro()
    
    df = load_data()
    data = bt.feeds.PandasData(dataname=df)

    cerebro_bh.adddata(data)
    cerebro_bh.addstrategy(BuyAndHold)

    # Initial capital
    cerebro_bh.broker.setcash(100000)

    # Commission + slippage
    cerebro_bh.broker.setcommission(commission=0.0018) # 0.18%
    cerebro_bh.broker.set_slippage_perc(0.0005)        # 0.05%

    # Analyzers
    cerebro_bh.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro_bh.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro_bh.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")

    results = cerebro_bh.run()
    strat_bh = results[0]

    exposure = strat_bh.bars_in_market / strat_bh.total_bars if strat_bh.total_bars > 0 else 0

    # Equity (daily returns)
    returns_dict = strat_bh.analyzers.timereturn.get_analysis()
    equity_bh = pd.DataFrame(list(returns_dict.items()), columns=["datetime", "return"])
    equity_bh["equity_curve"] = (1 + equity_bh["return"]).cumprod() * 100000

    final_value = strat_bh.broker.getvalue()
    max_dd = strat_bh.analyzers.drawdown.get_analysis()["max"]["drawdown"]

    #CAGR and Calmar
    start_value = 100000
    end_value = final_value
    years = (equity_bh["datetime"].iloc[-1] - equity_bh["datetime"].iloc[0]).days / 365
    cagr = (end_value / start_value) ** (1 / years) - 1 if years > 0 else None
    calmar = cagr / (max_dd / 100) if max_dd > 0 else None

    # Sharpe
    mean_daily = equity_bh["return"].mean()
    std_daily = equity_bh["return"].std()
    sharpe = (mean_daily / std_daily) * (365 ** 0.5) if std_daily != 0 else None

    # Trade metrics
    trades = strat_bh.analyzers.trades.get_analysis()
    total_closed = trades.get("total", {}).get("closed", 0)

    # Export
    results_path = Path("backtests/baseline/results")
    results_path.mkdir(parents=True, exist_ok=True)

    metrics_bh = {
        "final_value": final_value,
        "sharpe_daily": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": total_closed,
        "cagr": cagr,
        "calmar_ratio": calmar,
        "exposure": exposure
    }

    pd.DataFrame([metrics_bh]).to_csv(results_path / "metrics_buy_hold_v1.csv", index=False)
    equity_bh.to_csv(results_path / "equity_buy_hold_v1.csv", index=False)

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
    cerebro, strat, metrics, equity_df = run_backtest()
    print("EMA Baseline completed.")

    cerebro_bh, strat_bh, metrics_bh, equity_bh = run_buy_and_hold()
    print("Buy and Hold baselinecompleted.")

    print("Final Portfolio Value:", metrics["final_value"])
    print("Sharpe (manual daily):", metrics["sharpe_daily"])
    print("Max Drawdown %:", metrics["max_drawdown_pct"])
    print("Profit Factor:", metrics["profit_factor"])
    print("Expectancy per trade:", metrics["expectancy_per_trade"])

    print("Buy and Hold Final Portfolio Value:", metrics_bh["final_value"])
    print("Buy and Hold Sharpe (manual daily):", metrics_bh["sharpe_daily"])
    print("Buy and Hold Max Drawdown %:", metrics_bh["max_drawdown_pct"])
    print("Buy and Hold Profit Factor:", metrics_bh["profit_factor"])
    print("Buy and Hold Expectancy per trade:", metrics_bh["expectancy_per_trade"])
