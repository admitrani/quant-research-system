import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_combined_equity():
    """
    Load equity CSVs from all three backtests and produce a single
    combined equity curve chart saved to backtests/v1/results/.
    """

    ml_path  = Path("backtests/v1/ml/results/equity_ml_v1.csv")
    ema_path = Path("backtests/v1/baseline/results/equity_ema_v1.csv")
    bh_path  = Path("backtests/v1/baseline/results/equity_buy_hold_v1.csv")

    missing = [p for p in [ml_path, ema_path, bh_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot build equity plot — missing files: {[str(p) for p in missing]}. "
            "Run all three backtest stages first."
        )

    # Load
    ml_df  = pd.read_csv(ml_path,  parse_dates=["datetime"])
    ema_df = pd.read_csv(ema_path, parse_dates=["datetime"])
    bh_df  = pd.read_csv(bh_path,  parse_dates=["datetime"])

    ml_df  = ml_df.set_index("datetime").rename(columns={"equity_curve": "ML Strategy"})
    ema_df = ema_df.set_index("datetime").rename(columns={"equity_curve": "EMA Baseline"})
    bh_df  = bh_df.set_index("datetime").rename(columns={"equity_curve": "Buy & Hold"})

    # ML equity is hourly — resample to daily last value for a clean chart
    ml_daily = ml_df["ML Strategy"].resample("D").last().ffill()

    # Align all three on a common daily index
    combined = pd.concat([
        ml_daily,
        ema_df["EMA Baseline"],
        bh_df["Buy & Hold"],
    ], axis=1).ffill()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(combined.index, combined["ML Strategy"],  label="ML Strategy",  color="#e74c3c", linewidth=1.5)
    ax.plot(combined.index, combined["EMA Baseline"], label="EMA Baseline", color="#e67e22", linewidth=1.5)
    ax.plot(combined.index, combined["Buy & Hold"],   label="Buy & Hold",   color="#2ecc71", linewidth=1.5)

    # Minimum capital reference line
    ax.axhline(y=10_000, color="grey", linestyle="--", linewidth=0.8, alpha=0.6, label="Min capital ($10k)")

    ax.set_title("v1 Equity Curve Comparison — ML vs EMA vs Buy & Hold", fontsize=13, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path("backtests/v1/results")
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / "equity_comparison_v1.png"
    plt.savefig(out_file, dpi=150)
    plt.close()

    logger.info(f"Combined equity plot saved to: {out_file}")

    return combined
