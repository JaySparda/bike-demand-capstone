import pytest

from src.data_prep import LEAKAGE_COLUMNS, load_and_clean

import os

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "bike_sharing_hourly_sample.csv")


def test_no_leakage_columns_after_cleaning():
    df = load_and_clean(SAMPLE)
    for col in LEAKAGE_COLUMNS:
        assert col not in df.columns, f"leakage column {col!r} survived cleaning"


def test_prediction_is_non_negative_float():
    from src.predict import load_model, predict

    model = load_model()
    value = predict(model, {
        "hr": 8, "weekday": 1, "weathersit": 1,
        "temp": 0.5, "hum": 0.5, "windspeed": 0.2,
        "season": 1, "yr": 0, "mnth": 1,
        "holiday": 0, "workingday": 1, "atemp": 0.4,
    })
    assert isinstance(value, float)
    assert value >= 0
