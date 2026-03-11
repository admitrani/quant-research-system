import pandas as pd
import numpy as np
from models.utils import get_annualization_factor, get_risk_policy

def safe_get(d, *keys, default=0):
    for k in keys:
        if k not in d:
            return default
        d = d[k]
    return d
    

def compute_volatility(returns, annualization_factor):

    return np.std(returns) * np.sqrt(annualization_factor)


def compute_sortino(returns, annualization_factor):

    downside = returns[returns < 0]

    if len(downside) == 0:
        return None
    
    downside_std = downside.std()
    
    if downside_std == 0:
        return None
    
    return (np.mean(returns) * np.sqrt(annualization_factor)) / downside_std


def compute_calmar(cagr, max_dd):

    if max_dd == 0:
        return None

    return cagr / (abs(max_dd) / 100)


def extract_backtest_metrics(cerebro, strat):

    analyzers = strat.analyzers

    annualization_factor = get_annualization_factor()

    time_returns = analyzers.timereturn.get_analysis()
    returns_series = pd.Series(time_returns)
    returns_daily = returns_series.resample("D").apply(lambda x: (1 + x).prod() - 1).dropna()
    mean_r = returns_daily.mean()
    std_r = returns_daily.std()
    sharpe = (mean_r / std_r) * np.sqrt(365) if std_r != 0 else None

    drawdown = analyzers.drawdown.get_analysis()
    max_dd = safe_get(drawdown, "max", "drawdown")

    final_value = cerebro.broker.getvalue()
    initial_capital = get_risk_policy()["initial_capital"]
    total_days = (returns_series.index[-1] - returns_series.index[0]).days
    years = total_days / 365.25
    if years > 0 and final_value > 0:
        cagr = (final_value / initial_capital) ** (1 / years) - 1
    else:
        cagr = 0.0

    trades = analyzers.trades.get_analysis()
    total_trades = safe_get(trades, "total", "total")

    won = safe_get(trades, "won", "total")
    lost = safe_get(trades, "lost", "total")

    win_rate = won / total_trades if total_trades > 0 else None

    gross_profit = safe_get(trades, "won", "pnl", "total")
    gross_loss = safe_get(trades, "lost", "pnl", "total")
    profit_factor = None
    if gross_loss != 0:
        profit_factor = gross_profit / abs(gross_loss)

    volatility = compute_volatility(returns_series, annualization_factor)
    sortino = compute_sortino(returns_series, annualization_factor)

    calmar = compute_calmar(cagr, max_dd)

    avg_trade_return = None
    if total_trades > 0:
        pnl_net = safe_get(trades, "pnl", "net", "total")
        avg_trade_return = pnl_net / total_trades
    
    exposure = None
    if strat.total_bars > 0:
        exposure = strat.bars_in_market / strat.total_bars

    avg_trade_duration = None
    if strat.trade_durations:
        avg_trade_duration = np.mean(strat.trade_durations)
    
    results = {
        "Final Value": final_value,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Profit Factor": profit_factor,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Volatility": volatility,
        "Trades": total_trades,
        "Win Rate": win_rate,
        "Avg Trade Return": avg_trade_return,
        "Exposure": exposure,
        "Avg Trade Duration": avg_trade_duration,
    }

    return pd.Series(results)
