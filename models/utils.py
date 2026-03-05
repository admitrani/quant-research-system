from config.config_loader import load_config


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
    