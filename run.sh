#!/bin/bash

# Tự động tắt docker-compose khi Ctrl+C
trap 'echo "⛔ Dừng hệ thống..."; docker compose down; exit' INT

echo "🚀 Khởi động Kafka, Zookeeper, MongoDB..."
docker compose up -d

sleep 10

echo "📡 Chạy Kafka Producer (lấy dữ liệu từ Binance)..."
gnome-terminal -- bash -c "python kafka_producer/producer.py; exec bash"

sleep 2

echo "🗃️ Chạy Kafka Consumer (lưu vào MongoDB)..."
gnome-terminal -- bash -c "python kafka_producer/consumer.py; exec bash"

sleep 10

echo "🧠 Huấn luyện mô hình LSTM ..."
python model/train_lstm.py

sleep 2

echo "📊 Mở Streamlit Dashboard..."
streamlit run streamlit/dashboard.py

# Khi streamlit kết thúc (hoặc Ctrl+C), docker sẽ bị tắt do trap
