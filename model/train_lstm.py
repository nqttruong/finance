# model/train_lstm.py
import torch
import torch.nn as nn
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
import joblib

# ⚙️ Cấu hình
SEQ_LEN = 60  # 60 phút gần nhất để dự đoán
BATCH_SIZE = 16
EPOCHS = 32
LR = 0.001


# 🧠 Mô hình LSTM đơn giản
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# 📥 Tải dữ liệu từ MongoDB
def load_data():
    client = MongoClient('mongodb://localhost:27017/')
    collection = client['realtime']['btc_prices']
    docs = list(collection.find().sort('timestamp', 1))
    prices = np.array([d['price'] for d in docs], dtype=np.float32)
    return prices


# ✂️ Tạo tập huấn luyện từ chuỗi thời gian
def create_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


# 🚂 Huấn luyện mô hình
def train():
    prices = load_data()

    if len(prices) <= SEQ_LEN:
        raise ValueError(f"Chỉ có {len(prices)} điểm dữ liệu. Cần ít nhất {SEQ_LEN + 1} để dự đoán.")

    # Chuẩn hóa
    scaler = MinMaxScaler()
    norm_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    X, y = create_sequences(norm_prices)
    X = torch.tensor(X[:, :, None], dtype=torch.float32)
    y = torch.tensor(y[:, None], dtype=torch.float32)

    # ✅ Thiết bị (GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Đưa dữ liệu lên GPU nếu có
    X = X.to(device)
    y = y.to(device)

    previous_losses = []
    stable_epochs = 0
    tolerance = 1e-6

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X.size(0))
        losses = []

        for i in range(0, X.size(0), BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = X[indices], y[indices]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_losses = np.mean(losses)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_losses:.6f}")

        if previous_losses and abs(avg_losses - previous_losses[-1] < tolerance):
            stable_epochs += 1
        else:
            stable_epochs = 0
        previous_losses.append(avg_losses)

        if stable_epochs > 3:
            break

    # 💾 Lưu mô hình và scaler
    torch.save(model.state_dict(), 'model/lstm_btc.pt')
    joblib.dump(scaler, 'model/scaler.pkl')
    print("✅ Mô hình và scaler đã được lưu.")


if __name__ == '__main__':
    train()
