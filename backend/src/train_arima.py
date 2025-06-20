# Market/backend/src/train_arima.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA # type: ignore


def train_arima_model(df, crop_name=None):
    """
    1) Splits df into 80% train / 20% test by row‐count.
    2) Fits an ARIMA(p,d,q) on train["price"].
    3) Generates one‐step‐ahead forecasts on the test set, rolling forward.
    4) Returns (fitted_arima_model, test_df) where test_df has columns ["date", "actual", "predicted"].
    """

    # 80/20 split
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Fit ARIMA on training “price” series.  (You can tune the (p,d,q) order as needed.)
    model = ARIMA(train_df["price"], order=(5, 1, 0))
    fitted = model.fit()

    # One‐step ahead rolling forecast for each row in test_df
    preds = []
    for i in range(len(test_df)):
        steps = i + 1
        forecast_obj = fitted.get_forecast(steps=steps)
        # Take the very last predicted mean
        yhat = forecast_obj.predicted_mean.iloc[-1]
        preds.append(float(yhat))

    # Build a DataFrame with “date”, “actual”, “predicted”
    out_df = pd.DataFrame({
        "date": test_df["date"],
        "actual": test_df["price"],
        "predicted": preds
    })

    return fitted, out_df
