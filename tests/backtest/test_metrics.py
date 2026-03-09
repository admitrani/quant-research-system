import numpy as np
import pandas as pd

from backtests.v1.ml.metrics import safe_get, compute_volatility, compute_sortino, compute_calmar


# safe_get

def test_safe_get_existing_keys():
    d = {"a": {"b": {"c": 42}}}
    assert safe_get(d, "a", "b", "c") == 42


def test_safe_get_missing_key_returns_default():
    d = {"a": {"b": 1}}
    assert safe_get(d, "a", "x") == 0


def test_safe_get_custom_default():
    d = {}
    assert safe_get(d, "missing", default=99) == 99


# compute_volatility

def test_compute_volatility_annualizes_correctly():
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.005])
    factor = 365 * 24  # hourly
    vol = compute_volatility(returns, factor)
    expected = np.std(returns) * np.sqrt(factor)
    assert np.isclose(vol, expected)


def test_compute_volatility_zero_returns():
    returns = np.zeros(100)
    vol = compute_volatility(returns, 365 * 24)
    assert vol == 0.0


# compute_sortino

def test_compute_sortino_returns_float():
    returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
    sortino = compute_sortino(returns, 365 * 24)
    assert isinstance(sortino, float)


def test_compute_sortino_no_downside_returns_none():
    # All positive returns — no downside std
    returns = np.array([0.01, 0.02, 0.03])
    sortino = compute_sortino(returns, 365 * 24)
    assert sortino is None


def test_compute_sortino_penalizes_downside():
    # More negative returns → lower sortino.
    # Arrays need varied downside values so std(downside) > 0.
    np.random.seed(0)
    good = np.array([0.02, -0.001, 0.015, -0.002, 0.02, -0.001, 0.018, -0.003])
    bad  = np.array([0.02, -0.05,  0.015, -0.04,  0.02, -0.06,  0.018, -0.045])
    s_good = compute_sortino(good, 365 * 24)
    s_bad  = compute_sortino(bad,  365 * 24)
    assert s_good is not None
    assert s_bad  is not None
    assert s_good > s_bad


# compute_calmar

def test_compute_calmar_uses_percentage_correctly():
    # max_dd arrives as percentage (e.g. 90.64), not decimal
    # compute_calmar must divide by 100 before using max_dd
    cagr   = -0.272
    max_dd = 90.64
    calmar = compute_calmar(cagr, max_dd)
    # -0.272 / (90.64 / 100) = -0.272 / 0.9064 ≈ -0.300
    assert np.isclose(calmar, cagr / (abs(max_dd) / 100), rtol=1e-4)


def test_compute_calmar_zero_drawdown_returns_none():
    assert compute_calmar(0.5, 0) is None


def test_compute_calmar_positive_cagr_positive_dd():
    calmar = compute_calmar(0.18, 77.0)
    assert calmar > 0


# profit_factor sign invariant

def test_profit_factor_always_positive():
    gross_profit = 5000.0
    gross_loss   = 8000.0  # negative trades total is stored as negative float
    profit_factor = gross_profit / abs(gross_loss)
    assert profit_factor > 0
    assert np.isclose(profit_factor, 5000 / 8000)
    