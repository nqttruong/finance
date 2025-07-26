# 📈 Dự đoán giá Bitcoin theo thời gian thực bằng LSTM | Real-time Bitcoin Price Forecasting with LSTM

## 🇻🇳 Giới thiệu

Đây là một dự án học máy sử dụng mô hình **LSTM (Long Short-Term Memory)** để dự đoán giá Bitcoin theo thời gian thực. Dữ liệu được thu thập từ **Binance API** thông qua **Kafka Producer**, lưu trữ trong **MongoDB** và được hiển thị bằng **Streamlit Dashboard**.

## 🔧 Thành phần hệ thống

- 🐍 Python
- 🧠 PyTorch (LSTM)
- 🧪 scikit-learn (chuẩn hóa dữ liệu)
- 🗃 MongoDB (lưu trữ dữ liệu)
- ⚡ Kafka (truyền dữ liệu thời gian thực)
- 📊 Streamlit (giao diện người dùng)
- ⛓ Binance API (nguồn dữ liệu)

## 🚀 Cách sử dụng

### 1. Cài đặt môi trường

```bash```
python3 -m venv final <br>
source final/bin/activate <br>
pip install -r requirements.txt <br>

### 2. Chạy hệ thống
chmod +x run.sh <br>
./run.sh
