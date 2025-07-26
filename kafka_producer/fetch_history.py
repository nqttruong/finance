# fetch_history.py
import requests
import time
import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
collection = client["realtime"]["btc_prices"]

def fetch_klines(symbol="BTCUSDT", interval="1m", limit=1000, end_time=None):
    url = ""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if end_time:
        params["endTime"] = end_time

    response = requests.get(url, params=params)

    # Kiểm tra status
    if response.status_code != 200:
        print(f"❌ Lỗi API: {response.status_code}, nội dung: {response.text}")
        return []

    data = response.json()

    # Kiểm tra dữ liệu có hay không
    if not data:
        print("⚠️ Không có dữ liệu trả về từ Binance.")
    else:
        print(f"✅ Nhận được {len(data)} dòng dữ liệu.")

    return data

def insert_data():
    now = int(time.time() * 1000)
    total = 43200  # 30 ngày * 24h * 60p
    step = 1000
    end_time = now

    for _ in range(total // step):
        klines = fetch_klines(limit=step, end_time=end_time)
        docs = [
            {
                "timestamp": int(k[0] / 1000),
                "price": float(k[4])  # close price
            }
            for k in klines
        ]
        if not docs:
            break
        collection.insert_many(docs)
        end_time = klines[0][0] - 1  # lùi lại để lấy tiếp
        time.sleep(0.5)

insert_data()