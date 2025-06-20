# src/preprocess.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def fill_missing_values(df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    Handles missing price values by forward‐fill or interpolation.
    """
    if method == 'ffill':
        return df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        return df.interpolate().fillna(method='bfill')
    else:
        raise ValueError("Unsupported method for missing values: choose 'ffill' or 'interpolate'")

def create_lag_features(df: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    """
    For RandomForest (or other non‐time‐series‐native models), create lag features.
    Example: price_{t-1}, price_{t-2}, ..., price_{t-n_lags}
    """
    df_with_lags = df.copy()
    for lag in range(1, n_lags + 1):
        df_with_lags[f'lag_{lag}'] = df_with_lags['price'].shift(lag)
    df_with_lags = df_with_lags.dropna().reset_index(drop=True)
    return df_with_lags

from typing import Tuple

def train_test_split_time_series(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and testing, preserving time order.
    test_size: fraction (e.g., 0.2 for 20% holdout)
    """
    n = len(df)
    split_idx = int((1 - test_size) * n)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df

def scale_series(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fits a MinMaxScaler on the train['price'] and transforms both train and test.
    Returns scaled arrays and the scaler (for inverse transform later).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_values = train['price'].values.reshape(-1, 1)
    test_values = test['price'].values.reshape(-1, 1)
    scaler.fit(train_values)
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_values)
    return train_scaled, test_scaled, scaler

def create_lstm_sequences(series: np.ndarray, n_steps: int = 10):
    """
    For LSTM, convert a univariate series array into input/output sequences.
    e.g., X.shape = (samples, n_steps, 1), y.shape = (samples, )
    """
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i: i + n_steps])
        y.append(series[i + n_steps])
    X_arr = np.array(X)
    y_arr = np.array(y)
    # Reshape X to (samples, n_steps, features) where features=1
    X_arr = X_arr.reshape((X_arr.shape[0], X_arr.shape[1], 1))
    return X_arr, y_arr
