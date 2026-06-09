from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from .train import CHAMPION_PATH


@lru_cache(maxsize=1)
def load_model(path: str = CHAMPION_PATH) -> Any:
    return joblib.load(path)


def predict(model: Any, inputs: Dict[str, Any]) -> float:
    df = pd.DataFrame([{
        "season": inputs["season"],
        "yr": inputs["yr"],
        "mnth": inputs["mnth"],
        "hr": inputs["hr"],
        "holiday": inputs["holiday"],
        "weekday": inputs["weekday"],
        "workingday": inputs["workingday"],
        "weathersit": inputs["weathersit"],
        "temp": inputs["temp"],
        "atemp": inputs["atemp"],
        "hum": inputs["hum"],
        "windspeed": inputs["windspeed"],
        "dayofweek": inputs["weekday"],
        "is_weekend": 1 if inputs["weekday"] in (5, 6) else 0,
        "hr_sin": np.sin(2 * np.pi * inputs["hr"] / 24),
        "hr_cos": np.cos(2 * np.pi * inputs["hr"] / 24),
        "mnth_sin": np.sin(2 * np.pi * inputs["mnth"] / 12),
        "mnth_cos": np.cos(2 * np.pi * inputs["mnth"] / 12),
    }])
    val = float(model.predict(df)[0])
    return max(val, 0.0)
