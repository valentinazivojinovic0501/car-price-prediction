
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw used-car dataset."""
    df = df.copy()

    df = df.drop_duplicates()

    df = df.dropna(subset=["priceUSD"])
    df = df[df["priceUSD"] > 0]

    df["volume(cm3)"] = df["volume(cm3)"].fillna(
        df["volume(cm3)"].median()
    )

    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment",
    ]

    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown").astype(str)

    return df.reset_index(drop=True)
