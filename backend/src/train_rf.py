# Market/backend/src/train_rf.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_rf_model(df, crop_name=None):
    """
    1) Splits df into 80% train / 20% test by row‐count.
    2) Constructs these three features for every row:
         - day_of_year (int)
         - month       (int)
         - price_lag_1 (previous‐day price)
    3) Trains a RandomForestRegressor on the train portion.
    4) Predicts on the test portion.
    5) Returns (fitted_rf_model, test_df) where test_df has columns ["date","actual","predicted"].
    """

    # 80/20 split
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Combine train+test to compute the lag_1 feature easily
    combined = pd.concat([train_df, test_df], ignore_index=True).reset_index(drop=True)
    combined["day_of_year"] = combined["date"].dt.dayofyear
    combined["month"] = combined["date"].dt.month
    combined["price_lag_1"] = combined["price"].shift(1)

    # Slice out the train portion for feature‐matrix / label
    train_feats = combined.iloc[:split_idx].copy().dropna().reset_index(drop=True)
    test_feats = combined.iloc[split_idx:].copy().reset_index(drop=True)

    X_train = train_feats[["day_of_year", "month", "price_lag_1"]]
    y_train = train_feats["price"]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Now predict on test portion
    X_test = test_feats[["day_of_year", "month", "price_lag_1"]]
    preds = rf.predict(X_test).tolist()

    out_df = pd.DataFrame({
        "date": test_feats["date"],
        "actual": test_feats["price"],
        "predicted": [float(x) for x in preds]
    })

    return rf, out_df
