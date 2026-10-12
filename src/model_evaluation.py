
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.data_preprocessing import load_and_split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split_data()

    model_names = [
        "linear_regression",
        "decision_tree",
        "random_forest",
        "gradient_boosting",
    ]

    results = []

    for name in model_names:
        model_path = MODELS_DIR / f"{name}.joblib"

        model = joblib.load(model_path)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        results.append({
            "model": name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })

        print(f"\nModel: {name}")
        print(f"MAE:  {mae:.2f}")
        print(f"MSE:  {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2:   {r2:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULTS_DIR / "model_metrics.csv",
        index=False
    )

    print("\nResults saved to results/model_metrics.csv")


if __name__ == "__main__":
    main()
