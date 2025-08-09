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
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=4):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # ⬅️ Dự đoán nhiều bước

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # lấy output tại bước cuối cùng
        return self.fc(out)



# 📥 Tải dữ liệu từ MongoDB
def load_data():
    client = MongoClient('mongodb://localhost:27017/')
    collection = client['realtime']['btc_prices']
    docs = list(collection.find().sort('timestamp', 1))
    prices = np.array([d['price'] for d in docs], dtype=np.float32)
    return prices


# ✂️ Tạo tập huấn luyện từ chuỗi thời gian
def create_sequences(data, seq_len=60):
    X = []
    y = []
    for i in range(len(data) - seq_len - 1440):  # cần đủ xa cho t+1440
        seq_x = data[i:i+seq_len]
        # Tạo nhãn tại các bước: +15p, +1h, +3h, +1d
        seq_y = [
            data[i + seq_len + 15],
            data[i + seq_len + 60],
            data[i + seq_len + 180],
            data[i + seq_len + 1440]
        ]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# 🚂 Huấn luyện mô hình
def train():
    prices = load_data()

    if len(prices) <= SEQ_LEN + 1440:
        raise ValueError(f"Chỉ có {len(prices)} điểm dữ liệu. Cần ít nhất {SEQ_LEN + 1440} để dự đoán tới 1 ngày.")

    # 📉 Chuẩn hóa dữ liệu
    scaler = MinMaxScaler()
    norm_prices = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # 🎯 Tạo tập dữ liệu huấn luyện với nhiều nhãn đầu ra
    X, y = create_sequences(norm_prices)  # y shape: (samples, 4)
    X = torch.tensor(X[:, :, None], dtype=torch.float32)   # (samples, seq_len, 1)
    y = torch.tensor(y, dtype=torch.float32)               # (samples, 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTM(output_size=4).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
            output = model(batch_x)  # (batch_size, 4)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        avg_losses = np.mean(losses)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_losses:.6f}")

        if previous_losses and abs(avg_losses - previous_losses[-1]) < tolerance:
            stable_epochs += 1
        else:
            stable_epochs = 0
        previous_losses.append(avg_losses)

        if stable_epochs > 3:
            break

    # 💾 Lưu mô hình và scaler
    torch.save(model.state_dict(), 'model/lstm_multi.pt')
    joblib.dump(scaler, 'model/scaler.pkl')
    print("✅ Mô hình nhiều bước và scaler đã được lưu.")



if __name__ == '__main__':
    train()
