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


def predict_btc_prices(only_one=False, index=1, model_path='model/lstm_multi.pt', scaler_path='model/scaler.pkl'):
    """
    Dự đoán giá BTC tương lai.

    Args:
        only_one (bool): Nếu True → trả về 1 giá trị duy nhất (tại vị trí `index`)
        index (int): Vị trí thời điểm cần lấy nếu only_one=True. (0=15p, 1=1h, 2=3h, 3=1d)

    Returns:
        float nếu only_one=True
        List[float] nếu only_one=False
    """
    model = LSTM(output_size=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prices = load_data()
    scaler = joblib.load(scaler_path)
    norm = scaler.transform(prices.reshape(-1, 1)).flatten()

    x = torch.tensor(norm[-60:].reshape(1, -1, 1), dtype=torch.float32)

    with torch.no_grad():
        y = model(x).detach().cpu().numpy().flatten()  # shape: (4,)

    preds = scaler.inverse_transform(y.reshape(-1, 1)).flatten()  # shape: (4,)

    if only_one:
        return round(preds[index], 2)
    return [round(p, 2) for p in preds]

