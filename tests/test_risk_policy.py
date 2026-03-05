from models.utils import get_risk_policy, validate_risk_policy


def test_risk_policy_load():

    policy = get_risk_policy()

    assert "initial_capital" in policy
    assert "risk_fraction" in policy
    assert "max_drawdown_limit" in policy


def test_risk_policy_validation():

    policy = get_risk_policy()

    # Should not raise
    validate_risk_policy(policy)

    assert policy["initial_capital"] > 0
    assert 0 < policy["risk_fraction"] <= 1
    assert 0 < policy["max_drawdown_limit"] < 1
    