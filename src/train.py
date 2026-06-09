from __future__ import annotations

import os
from typing import Optional

import joblib
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .data_prep import TARGET, add_features, load_and_clean

CHAMPION_PATH: str = "models/champion.pkl"

FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
    "dayofweek", "is_weekend", "hr_sin", "hr_cos", "mnth_sin", "mnth_cos",
]


def train_and_log(df: pd.DataFrame, champion_path: str = CHAMPION_PATH) -> str:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=25, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    best_rmse = float("inf")
    best_model = None
    best_name = ""

    os.makedirs(os.path.dirname(champion_path) or ".", exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
            mlflow.sklearn.log_model(model, artifact_path="model")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_name = name

    joblib.dump(best_model, champion_path, compress=3)
    print(f"Champion: {best_name} (RMSE={best_rmse:.2f}) saved to {champion_path}")
    return champion_path


_DEFAULT_DATA = str(Path(__file__).resolve().parent.parent.parent / "data" / "bike_sharing_hourly_sample.csv")


def main(data_path: Optional[str] = None) -> str:
    path = data_path or _DEFAULT_DATA
    df = add_features(load_and_clean(path))
    return train_and_log(df)


if __name__ == "__main__":
    print(f"Champion saved to: {main()}")
