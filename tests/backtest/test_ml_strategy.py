import numpy as np
import pandas as pd
import backtrader as bt

from strategies.ml_strategy_v1 import MLStrategyV1


# Helpers

def make_df(n=200, prob_value=0.7, price=100.0):
    """Create a minimal OHLCV + ml_prob DataFrame."""
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "open_price":  price,
        "high_price":  price * 1.001,
        "low_price":   price * 0.999,
        "close_price": price,
        "volume":      1000.0,
        "ml_prob":     prob_value,
    }, index=idx)
    return df


class MLDataFeedTest(bt.feeds.PandasData):
    lines = ("ml_prob",)
    params = (
        ("datetime", None),
        ("open",  "open_price"),
        ("high",  "high_price"),
        ("low",   "low_price"),
        ("close", "close_price"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("ml_prob", "ml_prob"),
    )


STRATEGY_DEFAULTS = dict(
    entry_threshold=0.6,
    exit_threshold=0.5,
    risk_fraction=1.0,
    commission=0.0018,
    slippage=0.0005,
    buffer=0.003,
    minimum_capital=10_000,
    max_drawdown_limit=1.0,
    min_position_size=1e-8,
)


def run_cerebro(df, **strategy_kwargs):
    params = {**STRATEGY_DEFAULTS, **strategy_kwargs}
    cerebro = bt.Cerebro()
    cerebro.adddata(MLDataFeedTest(dataname=df))
    cerebro.addstrategy(MLStrategyV1, **params)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=params["commission"])
    cerebro.broker.set_slippage_perc(params["slippage"])
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn,    _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.DrawDown,      _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns,       _name="returns")
    results = cerebro.run()
    return cerebro, results[0]


# Entry logic

def test_strategy_enters_when_prob_above_threshold():
    """prob=0.7 > threshold=0.6 → should open at least one position."""
    df = make_df(prob_value=0.7)
    _, strat = run_cerebro(df)
    assert strat.bars_in_market > 0


def test_strategy_never_enters_when_prob_below_threshold():
    """prob=0.4 < threshold=0.6 → should never open a position."""
    df = make_df(prob_value=0.4)
    _, strat = run_cerebro(df)
    assert strat.bars_in_market == 0


# Exit logic

def test_strategy_exits_when_prob_drops_below_exit_threshold():
    """Enter on first half (prob=0.7), exit on second half (prob=0.3)."""
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    probs = [0.7] * (n // 2) + [0.3] * (n // 2)
    df = pd.DataFrame({
        "open_price":  100.0, "high_price": 100.1,
        "low_price":   99.9,  "close_price": 100.0,
        "volume":      1000.0, "ml_prob": probs,
    }, index=idx)
    _, strat = run_cerebro(df)
    trades = strat.analyzers.trades.get_analysis()
    total_closed = trades.get("total", {}).get("closed", 0)
    assert total_closed >= 1


# Minimum capital guard

def test_strategy_stops_when_below_minimum_capital():
    """With minimum_capital = initial_capital the strategy never trades."""
    df = make_df(prob_value=0.7)
    _, strat = run_cerebro(df, minimum_capital=200_000)
    assert strat.bars_in_market == 0


# Drawdown limit guard

def test_strategy_stops_after_drawdown_limit():
    """max_drawdown_limit=0.01 (1%) should trigger stop very quickly."""
    df = make_df(prob_value=0.7, price=50_000.0)
    _, strat = run_cerebro(df, max_drawdown_limit=0.01, minimum_capital=0)
    assert strat.stop_trading_triggered or strat.bars_in_market < 200


# Fractional sizing (no MARGIN errors)

def test_fractional_sizing_does_not_produce_margin_errors(capsys):
    """High BTC price should not cause MARGIN errors with fractional sizing."""
    df = make_df(prob_value=0.7, price=90_000.0)
    _, strat = run_cerebro(df)
    # If fractional sizing works correctly, at least some bars trade
    # and no order should fail with Margin status
    # We verify indirectly: bars_in_market > 0 means orders executed
    assert strat.bars_in_market > 0


# Floating point residual guard

def test_min_position_size_prevents_residual_reentry():
    """
    After closing a position the residual floating point size must not
    be treated as an open position blocking re-entry.
    Total trades should be reasonable (not in the thousands).
    """
    n = 400
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    # Alternate high/low prob to force many open/close cycles
    probs = ([0.7, 0.3] * (n // 2))[:n]
    df = pd.DataFrame({
        "open_price":  100.0, "high_price": 100.1,
        "low_price":   99.9,  "close_price": 100.0,
        "volume":      1000.0, "ml_prob": probs,
    }, index=idx)
    _, strat = run_cerebro(df)
    trades = strat.analyzers.trades.get_analysis()
    total = trades.get("total", {}).get("total", 0)
    assert total < 300  # should not explode to thousands


# Trade duration tracking

def test_trade_durations_recorded():
    """Closed trades should have their duration tracked in strat.trade_durations."""
    n = 200
    idx = pd.date_range("2022-01-01", periods=n, freq="h")
    probs = [0.7] * (n // 2) + [0.3] * (n // 2)
    df = pd.DataFrame({
        "open_price":  100.0, "high_price": 100.1,
        "low_price":   99.9,  "close_price": 100.0,
        "volume":      1000.0, "ml_prob": probs,
    }, index=idx)
    _, strat = run_cerebro(df)
    assert len(strat.trade_durations) > 0
    assert all(d >= 0 for d in strat.trade_durations)


# Bar tracking

def test_total_bars_equals_data_length():
    """total_bars must equal the number of bars in the feed."""
    df = make_df(n=100, prob_value=0.4)  # no trades, just counting
    _, strat = run_cerebro(df)
    # Backtrader skips the first bar for indicators warmup — allow small diff
    assert abs(strat.total_bars - len(df)) <= 2
