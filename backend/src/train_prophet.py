# Market/backend/src/train_prophet.py

import pandas as pd
from prophet import Prophet # type: ignore


def train_prophet_model(df, crop_name=None):
    """
    1) Splits df into 80% train / 20% test by row‐count.
    2) Fits a Prophet model on train DataFrame (with columns "ds", "y").
    3) Uses the fitted model to predict on test dates.
    4) Returns (fitted_prophet_model, test_df) where test_df has columns ["date", "actual", "predicted"].
    """

    # 80/20 split
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Prophet expects columns "ds" and "y"
    prophet_df = pd.DataFrame({
        "ds": train_df["date"],
        "y": train_df["price"]
    })

    m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    m.fit(prophet_df)

    # Forecast on test set
    future = pd.DataFrame({"ds": test_df["date"]})
    forecast = m.predict(future)
    preds = forecast["yhat"].tolist()

    out_df = pd.DataFrame({
        "date": test_df["date"],
        "actual": test_df["price"],
        "predicted": [float(x) for x in preds]
    })

    return m, out_df
