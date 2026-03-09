import matplotlib.pyplot as plt
from pathlib import Path

def plot_equity_curve(equity):

    plt.figure(figsize=(12, 6))
    equity.plot()
    plt.title("ML Strategy Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.grid()
    plt.tight_layout()

    output_path = Path(__file__).parent / "results"
    output_path.mkdir(exist_ok=True)
    plt.savefig(output_path / "equity_ml_v1.png")
    plt.close()
    