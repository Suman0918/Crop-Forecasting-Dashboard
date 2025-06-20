# Market/backend/src/generate_test_results.py

import os
import pandas as pd # type: ignore
import joblib # type: ignore
import numpy as np # type: ignore
from evaluate import compute_metrics
from data_loader import load_crop_data

from tensorflow.keras.models import load_model # type: ignore
from prophet import Prophet # type: ignore

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
YEARS_TO_EVALUATE = 3

def load_full_series(crop_name):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{crop_name}_price.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def main(crop_name):
    print(f"Generating test results for last {YEARS_TO_EVALUATE} years: {crop_name}")

    df = load_full_series(crop_name)
    cutoff_date = df["date"].max() - pd.DateOffset(years=YEARS_TO_EVALUATE)
    train_df = df[df["date"] < cutoff_date].copy().reset_index(drop=True)
    test_df = df[df["date"] >= cutoff_date].copy().reset_index(drop=True)

    actuals = test_df["price"].tolist()
    test_dates = test_df["date"].tolist()

    # You can re-use the same functions from earlier for each model block below…

    # Example: ARIMA
    from statsmodels.tsa.arima.model import ARIMA # type: ignore
    model = ARIMA(train_df["price"], order=(5,1,0)).fit()
    preds = model.forecast(len(test_df)).tolist()
    metrics = compute_metrics(actuals, preds)
    joblib.dump({
        "test_dates": [d.strftime("%Y-%m-%d") for d in test_dates],
        "predictions": preds,
        "metrics": metrics
    }, os.path.join(MODEL_DIR, f"arima_test_results_{crop_name}.pkl"))
    joblib.dump(metrics, os.path.join(MODEL_DIR, f"arima_metrics_{crop_name}.pkl"))
    print("✅ ARIMA saved.")

    # Repeat Prophet, LSTM, RF similarly using the same structure as in `train_all.py`

if __name__ == "__main__":
    main("rice")  # Or call dynamically via CLI
