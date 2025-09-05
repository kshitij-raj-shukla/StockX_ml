from flask import Flask, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from utils.data_preprocessor import prepare_prediction_data
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model("ml_model/models/lstm_model.h5")

@app.route('/api/predict/<ticker>', methods=['GET'])
def predict(ticker):
    try:
        X, scaler, historical_dates, historical_prices, predicted_dates = prepare_prediction_data(ticker)

        predicted_prices = []
        current_sequence = X.copy()

        for _ in range(7):
            pred = model.predict(current_sequence, verbose=0)
            predicted_prices.append(pred[0, 0])
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0, 0]

        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten().tolist()

        return jsonify({
            'historical_dates': historical_dates,
            'historical_prices': historical_prices,
            'predicted_dates': predicted_dates,
            'predicted_prices': predicted_prices
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(port=5001, debug=True)
