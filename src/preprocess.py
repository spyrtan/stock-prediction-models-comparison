import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

def prepare_data(ticker, start, end, interval="1d", window_size=30):
    """
    Download and preprocess historical stock price data using a sliding window approach.

    Returns:
        X_train, y_train, X_test, y_test, scaler, original DataFrame (df)
    """
    df = yf.download(ticker, start=start, end=end, interval=interval)

    if df.empty:
        raise ValueError("Failed to download data. Check the ticker or dates.")

    close_prices = df["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    split_index = int(len(X) * 0.9)  # changed from 0.8 to 0.9
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    # Save scaler to disk
    scaler_path = Path("models") / f"{ticker}_scaler.save"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)

    return X_train, y_train, X_test, y_test, scaler, df

def prepare_from_series(close_prices, window_size=30):
    """
    Preprocess a given series of closing prices (without downloading),
    using a sliding window approach.

    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    split_index = int(len(X) * 0.9)  # changed from 0.8 to 0.9
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test, scaler
