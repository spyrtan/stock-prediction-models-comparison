import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(ticker, start, end, interval="1d", window_size=30):
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

    split_index = int(len(X) * 0.8)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    return X_train, y_train, X_test, y_test, scaler, df  # <--- ZWRACA CAŁY df

def prepare_from_series(close_prices, window_size=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    split_index = int(len(X) * 0.8)
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:], scaler
