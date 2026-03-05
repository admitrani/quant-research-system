from config.config_loader import load_config
from models.model_factory import get_model


def get_annualization_factor():

    config = load_config()
    interval = config["market_data"]["intervals"][0]

    if interval.endswith("h"):
        hours = int(interval.replace("h", ""))
        return (365 * 24) / hours

    elif interval.endswith("d"):
        return 365

    else:
        raise ValueError(f"Unsupported interval: {interval}")


def get_baseline_model_config(model_name):

    config = load_config()

    model = get_model(model_name)

    return {
        "threshold": config["system"]["modeling"]["threshold"],
        "max_depth": model.get_params().get("max_depth")
    }


def get_risk_policy():

    config = load_config()
    risk_config = config["system"]["risk"]

    return {
        "initial_capital": risk_config["initial_capital"],
        "position_sizing": risk_config["position_sizing"],
        "risk_fraction": risk_config["risk_fraction"],
        "compounding": risk_config["compounding"],
        "max_positions": risk_config["max_positions"],
        "max_drawdown_limit": risk_config["max_drawdown_limit"],
        "minimum_capital": risk_config["minimum_capital"],
    }


def validate_risk_policy(policy):

    if policy["initial_capital"] <= 0:
        raise ValueError("Initial capital must be positive.")
    
    if not 0 < policy["risk_fraction"] <= 1:
        raise ValueError("Risk fraction must be between 0 and 1.")
    
    if policy["max_drawdown_limit"] <= 0 or policy["max_drawdown_limit"] >= 1:
        raise ValueError("Max drawdown limit must be between 0 and 1.")
    
    if policy["minimum_capital"] <= 0:
        raise ValueError("Minimum capital must be positive.")
    