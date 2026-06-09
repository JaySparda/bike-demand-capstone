from __future__ import annotations

from typing import List

import pandas as pd
import numpy as np

LEAKAGE_COLUMNS: List[str] = ["casual", "registered", "instant"]

TARGET: str = "cnt"


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dteday"] = pd.to_datetime(df["dteday"])
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns], errors="ignore")
    df["hum"] = df["hum"].fillna(df["hum"].median())
    df["windspeed"] = df["windspeed"].fillna(df["windspeed"].median())
    df = df.drop_duplicates()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofweek"] = df["dteday"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    return df
