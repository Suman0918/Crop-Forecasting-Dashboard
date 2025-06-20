# Market/backend/src/api.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO and WARNING from TensorFlow
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import datetime
from tensorflow.keras.models import load_model # type: ignore
from prophet import Prophet
import os

app = Flask(__name__)
CORS(app)

# ------------------ Directories ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../models"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data"))

# ------------------ Load the full CSV ------------------
def load_full_series(crop_name):
    file_path = os.path.join(DATA_DIR, f"{crop_name}_price.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file for crop '{crop_name}' not found at {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ------------------ Forecast helpers ------------------
def forecast_arima(crop_name, target_date):
    df = load_full_series(crop_name)
    last_date = df["date"].iloc[-1].date()
    delta_days = (target_date - last_date).days
    if delta_days < 1:
        return float(df["price"].iloc[-1])

    arima_path = os.path.join(MODEL_DIR, f"arima_{crop_name}.pkl")
    if not os.path.exists(arima_path):
        raise FileNotFoundError(f"ARIMA model not found: {arima_path}")
    arima_model = joblib.load(arima_path)

    forecast_obj = arima_model.get_forecast(steps=delta_days)
    pred_series = forecast_obj.predicted_mean
    return float(pred_series.iloc[-1])

def forecast_prophet(crop_name, target_date):
    df = load_full_series(crop_name)
    last_date = df["date"].iloc[-1].date()
    if target_date <= last_date:
        return float(df["price"].iloc[-1])

    prophet_path = os.path.join(MODEL_DIR, f"prophet_{crop_name}.pkl")
    if not os.path.exists(prophet_path):
        raise FileNotFoundError(f"Prophet model not found: {prophet_path}")
    prophet_model = joblib.load(prophet_path)

    future_df = pd.DataFrame({"ds": [pd.to_datetime(target_date)]})
    forecast = prophet_model.predict(future_df)
    return float(forecast["yhat"].iloc[0])

def forecast_lstm(crop_name, target_date):
    df = load_full_series(crop_name)
    last_date = df["date"].iloc[-1].date()
    delta_days = (target_date - last_date).days
    if delta_days < 1:
        return float(df["price"].iloc[-1])

    scaler_path = os.path.join(MODEL_DIR, f"lstm_scaler_{crop_name}.pkl")
    model_path = os.path.join(MODEL_DIR, f"lstm_{crop_name}.h5")
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"LSTM model or scaler not found for {crop_name}")
    scaler = joblib.load(scaler_path)
    lstm_model = load_model(model_path)

    n_steps = 10
    full_prices = df["price"].values.reshape(-1, 1)
    full_scaled = scaler.transform(full_prices).reshape(-1)

    seq = full_scaled[-n_steps:].copy()
    for _ in range(delta_days):
        x_input = seq.reshape((1, n_steps, 1))
        yhat_scaled = lstm_model.predict(x_input, verbose=0)[0][0]
        seq = np.append(seq[1:], yhat_scaled)
    pred_unscaled = scaler.inverse_transform(np.array([[seq[-1]]]))[0][0]
    return float(pred_unscaled)

def forecast_rf(crop_name, target_date):
    df = load_full_series(crop_name)
    last_date = df["date"].iloc[-1].date()
    delta_days = (target_date - last_date).days
    if delta_days < 1:
        return float(df["price"].iloc[-1])

    rf_path = os.path.join(MODEL_DIR, f"rf_{crop_name}.pkl")
    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"Random Forest model not found: {rf_path}")
    rf_model = joblib.load(rf_path)

    day_of_year = target_date.timetuple().tm_yday
    month = target_date.month
    last_price = float(df["price"].iloc[-1])

    feat_df = pd.DataFrame({
        "day_of_year": [day_of_year],
        "month": [month],
        "price_lag_1": [last_price]
    })

    pred = rf_model.predict(feat_df)[0]
    return float(pred)

# ------------------ Load metrics ------------------
def load_metrics(crop_name, model_name):
    metrics_path = os.path.join(MODEL_DIR, f"{model_name}_metrics_{crop_name}.pkl")
    if not os.path.exists(metrics_path):
        return {"MAE": float("inf"), "RMSE": float("inf"), "R2": float("-inf")}
    return joblib.load(metrics_path)

# ------------------ /predict (historical test set) ------------------
@app.route("/predict")
def predict_all():
    crop = request.args.get("crop")
    model_arg = request.args.get("model", "all")

    if crop is None:
        return jsonify({"error": "Missing crop parameter"}), 400
    if model_arg not in ("all", "arima", "prophet", "lstm", "rf"):
        return jsonify({"error": "Invalid model parameter"}), 400

    results = {}
    models_to_run = ["arima", "prophet", "lstm", "rf"] if model_arg == "all" else [model_arg]

    for m in models_to_run:
        test_path = os.path.join(MODEL_DIR, f"{m}_test_results_{crop}.pkl")
        if not os.path.exists(test_path):
            return jsonify({"error": f"Test results for {m} not found at {test_path}"}), 404
        test_results = joblib.load(test_path)
        results[m] = test_results

    return jsonify({"crop": crop, "model": model_arg, "results": results})

# ------------------ /predict_future (single‐date + choose best) ------------------
@app.route("/predict_future")
def predict_future():
    crop = request.args.get("crop")
    date_str = request.args.get("date")
    model_arg = request.args.get("model", "all")

    if crop is None or date_str is None:
        return jsonify({"error": "Missing crop or date parameter"}), 400
    if model_arg not in ("all", "arima", "prophet", "lstm", "rf"):
        return jsonify({"error": "Invalid model parameter"}), 400

    try:
        future_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    models_to_run = ["arima", "prophet", "lstm", "rf"] if model_arg == "all" else [model_arg]
    predictions = {}

    for m in models_to_run:
        try:
            if m == "arima":
                predictions["arima"] = forecast_arima(crop, future_date)
            elif m == "prophet":
                predictions["prophet"] = forecast_prophet(crop, future_date)
            elif m == "lstm":
                predictions["lstm"] = forecast_lstm(crop, future_date)
            elif m == "rf":
                predictions["rf"] = forecast_rf(crop, future_date)
        except Exception as e:
             import traceback
             traceback.print_exc()
             return jsonify({"error": f"Failed to forecast {m}: {str(e)}"}), 500

    # Find best model by lowest RMSE from historical metrics
    metrics_dict = {}
    for m in ["arima", "prophet", "lstm", "rf"]:
        mdata = load_metrics(crop, m)
        metrics_dict[m] = mdata.get("RMSE", float("inf"))

    filtered_metrics = {m: metrics_dict[m] for m in models_to_run}
    if filtered_metrics:
        best_model = min(filtered_metrics, key=lambda x: filtered_metrics[x])
        best_prediction = predictions.get(best_model)
    else:
        best_model = None
        best_prediction = None

    return jsonify({
        "crop": crop,
        "date": date_str,
        "predictions": predictions,
        "best_model": best_model,
        "best_prediction": best_prediction
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
