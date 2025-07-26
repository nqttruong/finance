import torch
import torch.nn as nn
import numpy as np
from pymongo import MongoClient
import datetime
import joblib
from model.train_lstm import LSTM

SEQ_LEN = 60

def load_data():
    client = MongoClient('mongodb://localhost:27017/')
    collection = client['realtime']['btc_prices']
    docs = list(collection.find().sort('timestamp', -1).limit(SEQ_LEN))[::-1]
    return np.array([d['price'] for d in docs])

# dự đoán cho giờ tiếp theo
def predict_next_hour(model_path='model/lstm_btc.pt', scaler_path='model/scaler.pkl'):
    model = LSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prices = load_data()
    scaler = joblib.load(scaler_path)
    norm = scaler.transform(prices.reshape(-1, 1)).flatten()

    x = torch.tensor(norm[-60:].reshape(1, -1, 1), dtype=torch.float32)

    with torch.no_grad():
        y = model(x).item()

    pred_price = scaler.inverse_transform([[y]])[0][0]
    return round(pred_price, 2)