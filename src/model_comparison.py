
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    metrics_path = RESULTS_DIR / "model_metrics.csv"

    results = pd.read_csv(metrics_path)
    results = results.sort_values("RMSE")

    print("\nModel comparison, sorted by RMSE:")
    print(results.to_string(index=False))

    results.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    metrics_to_plot = ["MAE", "RMSE", "R2"]

    for metric in metrics_to_plot:
        plot_data = results.sort_values(metric, ascending=(metric != "R2"))

        plt.figure(figsize=(10, 5))
        plt.bar(plot_data["model"], plot_data[metric])
        plt.title(f"Model comparison: {metric}")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / f"comparison_{metric.lower()}.png",
            dpi=150
        )
        plt.show()

    print("\nComparison saved in the results folder.")


if __name__ == "__main__":
    main()
