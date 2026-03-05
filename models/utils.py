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
