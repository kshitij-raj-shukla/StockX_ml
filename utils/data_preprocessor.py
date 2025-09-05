import requests
import numpy as np
from datetime import datetime, timedelta
import os
import json
from dotenv import load_dotenv

load_dotenv()
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def prepare_prediction_data(ticker, lookback=60, future_days=7):
    cache_file = f"ml_model/cache/{ticker}_1y.json"
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
    prices = [float(time_series[date]["4. close"]) for date in dates[-365:]]
    prices = np.array(prices).reshape(-1, 1)
    
    scaler = np.load("ml_model/models/scaler.npy", allow_pickle=True).item()
    scaled_data = scaler.transform(prices)
    
    last_sequence = scaled_data[-lookback:]
    X = np.array([last_sequence])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    historical_dates = dates[-lookback:]
    historical_prices = scaler.inverse_transform(scaled_data[-lookback:]).flatten().tolist()
    
    last_date = datetime.strptime(historical_dates[-1], '%Y-%m-%d')
    future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(future_days)]
    
    return X, scaler, historical_dates, historical_prices, future_dates