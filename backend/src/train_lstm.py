import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore


def create_lstm_sequences(series, n_steps):
    """
    Given a 1D NumPy array `series` of length N,
    returns X (shape (N-n_steps, n_steps, 1)) and y (shape (N-n_steps,)).
    """
    X, y = [], []
    for i in range(len(series) - n_steps):
        seq_x = series[i: i + n_steps]
        seq_y = series[i + n_steps]
        X.append(seq_x)
        y.append(seq_y)
    X = np.array(X).reshape(-1, n_steps, 1)
    y = np.array(y)
    return X, y


def train_lstm_model(df, crop_name=None, n_steps=10, epochs=20):
    """
    1) Splits df into 80% train / 20% test by row‐count.
    2) Scales the "price" column in [0, 1] using MinMaxScaler.
    3) Creates LSTM sequences of length n_steps for train.
    4) Trains a simple LSTM on the train portion.
    5) Generates rolling one‐step‐ahead forecasts on the test portion.
    6) Returns (fitted_lstm_model, test_df, scaler).
       test_df has columns ["date", "actual", "predicted"].
    """

    # Split 80/20
    n = len(df)
    split_idx = int(n * 0.8)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    # Scale the "price" series
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_prices = train_df["price"].values.reshape(-1, 1)
    scaler.fit(train_prices)

    train_scaled = scaler.transform(train_prices).reshape(-1)
    full_series = df["price"].values.reshape(-1, 1)
    full_scaled = scaler.transform(full_series).reshape(-1)

    # Prepare training sequences (n_steps)
    X_train, y_train = create_lstm_sequences(train_scaled, n_steps)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation="relu", input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=16,
        validation_split=0.1,
        shuffle=False,
        callbacks=[es],
        verbose=0
    )

    # Generate rolling one‐step‐ahead forecasts on the test set
    preds = []
    window = full_scaled[: split_idx]  # initial n_steps from training
    for t in range(len(test_df)):
        last_seq = window[-n_steps:]
        x_input = last_seq.reshape((1, n_steps, 1))
        yhat_scaled = model.predict(x_input, verbose=0)[0][0]

        # ✅ Guard against bad predictions
        if np.isnan(yhat_scaled) or np.isinf(yhat_scaled):
            print(f"⚠️ Invalid LSTM prediction at step {t}: {yhat_scaled}")
            break

        # ✅ Clamp between 0–1
        yhat_scaled = np.clip(yhat_scaled, 0, 1)

        # ✅ Inverse scale and store
        yhat = scaler.inverse_transform(np.array([[yhat_scaled]]))[0][0]
        preds.append(float(yhat))

        # Update window for next step
        window = np.append(window, yhat_scaled)

    out_df = pd.DataFrame({
        "date": test_df["date"][:len(preds)],
        "actual": test_df["price"][:len(preds)],
        "predicted": preds
    })

    return model, out_df, scaler
