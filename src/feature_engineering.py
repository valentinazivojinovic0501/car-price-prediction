
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for price prediction."""
    df = df.copy()

    current_year = 2026

    df["car_age"] = (current_year - df["year"]).clip(lower=0)

    df["mileage_per_year"] = (
        df["mileage(kilometers)"] /
        df["car_age"].clip(lower=1)
    )

    df["engine_volume_liters"] = df["volume(cm3)"] / 1000

    df["is_newer_car"] = (df["year"] >= 2015).astype(int)
    df["is_high_mileage"] = (
        df["mileage(kilometers)"] >= 200000
    ).astype(int)

    df["brand_model"] = (
        df["make"].astype(str) + "_" + df["model"].astype(str)
    )

    return df
