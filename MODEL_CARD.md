# Model Card — Bike Demand Forecaster

## What it does

Predicts hourly bike-share rental demand (`cnt`) from weather and calendar features. Designed for a city bike-share operator to rebalance stations ahead of peak hours.

## Training data

- **Source:** [UCI Bike Sharing dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (`hour.csv`)
- **Rows:** 17,379 hourly records (2011–2012)
- **Features used:** season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, dayofweek, is_weekend, plus cyclical sin/cos encodings of hr and mnth
- **Leakage handled:** `casual`, `registered`, `instant` dropped before training

## Models compared

| Model            | RMSE      | MAE       | R²        |
| ---------------- | --------- | --------- | --------- |
| **RandomForest** | **41.70** | **24.62** | **0.945** |
| GradientBoosting | 63.77 | 43.45 | 0.872 |
| LinearRegression | 125.18 | 92.22 | 0.505 |

**Champion:** RandomForestRegressor (25 trees, ~30 MB — kept under GitHub's 100 MB cap with negligible accuracy loss vs 100 trees).

## MLflow tracking

![MLflow runs](./assets/mlflow_runs.png)

Three runs logged with params (e.g. n_estimators) and metrics (RMSE, MAE, R²) plus the serialised model artifact.

## Limitations

- Trained on 2011–2012 Washington D.C. data — may not generalise to other cities or current ridership patterns.
- No real-time weather feed; predictions use user-supplied values.
- Does not capture station-level demand (only city-wide aggregate).
- Temporal split not strictly enforced (random 80/20 split may leak future info into training).

## How I worked with the agent

I used opencode to implement the full pipeline from the scaffold contracts. The agent handled data cleaning (median imputation, dedup, cyclical features), training with MLflow tracking, and building the Streamlit dashboard. I had to correct relative file paths for the data directory and adjust the `.gitignore` to exclude the large `mlruns/` folder (120 MB). The main loop was: specify the next deliverable → agent writes code → I test → fix path/config issues → repeat. This worked well for keeping velocity high across the 5 phases.
