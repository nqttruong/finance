# streamlit/dashboard.py
import streamlit as st
import pandas as pd
from pymongo import MongoClient
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.model_predict import predict_next_hour
import plotly.express as px

st.title("⏱️ Dự đoán giá BTC theo thời gian thực")
# Tự động làm mới mỗi 1 phút
st_autorefresh(interval=600 * 1000, key="data_refresh")

client = MongoClient('mongodb://localhost:27017/')
collection = client['realtime']['btc_prices']
docs = list(collection.find().sort('timestamp', -1).limit(100))[::-1]

df = pd.DataFrame(docs)
df['time'] = pd.to_datetime(df['timestamp'], unit='s')

fig = px.line(df, x='time', y='price', title='Biểu đồ giá BTC theo thời gian thực(1 phút)')
fig.update_layout(width=1000, height=600)  # Tăng kích thước
st.plotly_chart(fig)

# 🕒 Lấy thời gian và giá hiện tại (dòng cuối cùng)
latest = df.iloc[-1]
current_time = latest['time'].strftime("%Y-%m-%d %H:%M:%S")
current_price = latest['price']

# 🖥️ Hiển thị giá và giờ hiện tại
st.info(f"🕒 **Thời gian hiện tại**: {current_time}")
st.info(f"💰 **Giá BTC hiện tại**: **${current_price}**")

if st.button("📈 Dự đoán 1 giờ tiếp theo"):
    try:
        pred = predict_next_hour()
        st.success(f"Giá BTC sau 1 giờ (dự đoán): **${pred}**")
    except ValueError as e:
        st.error(f"Không thể dự đoán, chưa đủ dữ liệu huấn luyện")
    except Exception as e:
        st.error(f"Lỗi không xác định: {str(e)}")
