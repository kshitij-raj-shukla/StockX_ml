import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import os
import json
from dotenv import load_dotenv

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)

def fetch_data(ticker):
    cache_file = f"ml_model/cache/{ticker}_5y.json"
    os.makedirs("ml_model/cache", exist_ok=True)
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
    else:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        response = requests.get(url)
        if response.status_code != 200 or "Time Series (Daily)" not in response.json():
            raise ValueError("Failed to fetch data from Alpha Vantage")
        data = response.json()
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    time_series = data["Time Series (Daily)"]
    dates = sorted(time_series.keys())
    prices = [float(time_series[date]["4. close"]) for date in dates[-1825:]]
    return np.array(prices).reshape(-1, 1)

def preprocess_data(data, lookback=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler

def build_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(ticker="AAPL"):
    data = fetch_data(ticker)
    X_train, y_train, X_test, y_test, scaler = preprocess_data(data)
    
    model = build_model()
    lr_scheduler = LearningRateScheduler(scheduler)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[lr_scheduler], verbose=1)
    
    os.makedirs("ml_model/models", exist_ok=True)
    model.save("ml_model/models/lstm_model.h5")
    np.save("ml_model/models/scaler.npy", scaler)
    
    return model, scaler

if __name__ == "__main__":
    if not ALPHA_VANTAGE_API_KEY:
        raise ValueError("ALPHA_VANTAGE_API_KEY not set in .env")
    train_model()