
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.data_preprocessing import (
    load_and_split_data,
    build_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def get_models():
    return {
        "linear_regression": LinearRegression(),

        "decision_tree": DecisionTreeRegressor(
            max_depth=20,
            min_samples_leaf=5,
            random_state=42,
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_split_data()
    models = get_models()

    for name, estimator in models.items():
        print(f"\nTraining model: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        output_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, output_path)

        print(f"Saved model to: {output_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
