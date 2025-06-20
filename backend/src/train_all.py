# Market/backend/src/train_all.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import argparse
import joblib # type: ignore
import pandas as pd # type: ignore

from data_loader import load_crop_data
from train_arima import train_arima_model
from train_prophet import train_prophet_model
from train_lstm import train_lstm_model
from train_rf import train_rf_model
from evaluate import compute_metrics

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main(crop_name):
    print(f"Loading data for '{crop_name}'…")
    df = load_crop_data(crop_name)

    # 🔁 Evaluate only on recent 3 years
    YEARS_TO_EVALUATE = 3
    cutoff_date = df["date"].max() - pd.DateOffset(years=YEARS_TO_EVALUATE)
    train_df = df[df["date"] < cutoff_date].copy().reset_index(drop=True)
    test_df = df[df["date"] >= cutoff_date].copy().reset_index(drop=True)

    print(f"Train: {train_df.shape}, Test (last {YEARS_TO_EVALUATE} years): {test_df.shape}")

    # ---------- 1) ARIMA ----------
    print("Training ARIMA…")
    arima_model, _ = train_arima_model(train_df, test_df)
    arima_metrics = compute_metrics(test_df["price"], arima_model.forecast(len(test_df)))
    print("ARIMA metrics:", arima_metrics)
    joblib.dump(arima_model, os.path.join(MODEL_DIR, f"arima_{crop_name}.pkl"))
    arima_test_results = {
        "test_dates": test_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "predictions": arima_model.forecast(len(test_df)).tolist(),
        "metrics": arima_metrics
    }
    joblib.dump(arima_test_results, os.path.join(MODEL_DIR, f"arima_test_results_{crop_name}.pkl"))
    joblib.dump(arima_metrics, os.path.join(MODEL_DIR, f"arima_metrics_{crop_name}.pkl"))

    # ---------- 2) Prophet ----------
    print("Training Prophet…")
    prophet_model, prophet_test_df = train_prophet_model(df)
    prophet_test_df = prophet_test_df[prophet_test_df["date"] >= cutoff_date].reset_index(drop=True)
    prophet_metrics = compute_metrics(prophet_test_df["actual"], prophet_test_df["predicted"])
    print("Prophet metrics:", prophet_metrics)
    joblib.dump(prophet_model, os.path.join(MODEL_DIR, f"prophet_{crop_name}.pkl"))
    prophet_test_results = {
        "test_dates": prophet_test_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "predictions": prophet_test_df["predicted"].tolist(),
        "metrics": prophet_metrics
    }
    joblib.dump(prophet_test_results, os.path.join(MODEL_DIR, f"prophet_test_results_{crop_name}.pkl"))
    joblib.dump(prophet_metrics, os.path.join(MODEL_DIR, f"prophet_metrics_{crop_name}.pkl"))

    # ---------- 3) LSTM ----------
    print("Training LSTM…")
    lstm_model, lstm_test_df, lstm_scaler = train_lstm_model(df)
    lstm_test_df = lstm_test_df[lstm_test_df["date"] >= cutoff_date].reset_index(drop=True)
    lstm_metrics = compute_metrics(lstm_test_df["actual"], lstm_test_df["predicted"])
    print("LSTM metrics:", lstm_metrics)
    lstm_model.save(os.path.join(MODEL_DIR, f"lstm_{crop_name}.h5"))
    joblib.dump(lstm_scaler, os.path.join(MODEL_DIR, f"lstm_scaler_{crop_name}.pkl"))
    lstm_test_results = {
        "test_dates": lstm_test_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "predictions": lstm_test_df["predicted"].tolist(),
        "metrics": lstm_metrics
    }
    joblib.dump(lstm_test_results, os.path.join(MODEL_DIR, f"lstm_test_results_{crop_name}.pkl"))
    joblib.dump(lstm_metrics, os.path.join(MODEL_DIR, f"lstm_metrics_{crop_name}.pkl"))

    # ---------- 4) Random Forest ----------
    print("Training Random Forest…")
    rf_model, rf_test_df = train_rf_model(df)
    rf_test_df = rf_test_df[rf_test_df["date"] >= cutoff_date].reset_index(drop=True)
    rf_metrics = compute_metrics(rf_test_df["actual"], rf_test_df["predicted"])
    print("RF metrics:", rf_metrics)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, f"rf_{crop_name}.pkl"))
    rf_test_results = {
        "test_dates": rf_test_df["date"].dt.strftime("%Y-%m-%d").tolist(),
        "predictions": rf_test_df["predicted"].tolist(),
        "metrics": rf_metrics
    }
    joblib.dump(rf_test_results, os.path.join(MODEL_DIR, f"rf_test_results_{crop_name}.pkl"))
    joblib.dump(rf_metrics, os.path.join(MODEL_DIR, f"rf_metrics_{crop_name}.pkl"))

    print(f"✅ Models and metrics saved for crop: {crop_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models on recent years only.")
    parser.add_argument("--crop", type=str, required=True, help="Crop name (e.g. rice)")
    args = parser.parse_args()
    main(args.crop)
