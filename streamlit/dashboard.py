import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import plotly.express as px
import sys
import os
from pymongo import MongoClient

# 📦 Thêm đường dẫn tới thư mục model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 📈 Hàm dự đoán nhiều bước (multi-step)
from model.model_predict import predict_btc_prices

# 🚀 Giao diện chính
st.title("⏱️ Dự đoán giá BTC theo thời gian thực")

# 🔁 Tự động làm mới mỗi 60 giây
st_autorefresh(interval=120 * 1000, key="data_refresh")

# 🌐 Kết nối MongoDB và lấy dữ liệu
client = MongoClient('mongodb://localhost:27017/')
collection = client['realtime']['btc_prices']
docs = list(collection.find().sort('timestamp', -1).limit(100))[::-1]

# 📊 Tạo DataFrame và biểu đồ
df = pd.DataFrame(docs)
df['time'] = pd.to_datetime(df['timestamp'], unit='s')

fig = px.line(df, x='time', y='price', title='Biểu đồ giá BTC theo thời gian thực (1 phút)')
fig.update_layout(width=1000, height=600)
st.plotly_chart(fig)

# 🕒 Thông tin giá hiện tại
latest = df.iloc[-1]
current_time = latest['time'].strftime("%Y-%m-%d %H:%M:%S")
current_price = latest['price']

st.info(f"🕒 **Thời gian hiện tại**: {current_time}")
st.info(f"💰 **Giá BTC hiện tại**: **${current_price}**")

# 📈 Dự đoán nhiều mốc thời gian tương lai
if st.button("📈 Dự đoán nhiều mốc thời gian"):
    try:
        preds = predict_btc_prices()
        intervals = ["15 phút", "1 giờ", "3 giờ", "1 ngày"]
        future_times = [latest['time'] + pd.Timedelta(minutes=m) for m in [15, 60, 180, 1440]]

        for i in range(4):
            st.success(f"📅 Dự đoán giá BTC vào lúc {future_times[i].strftime('%Y-%m-%d %H:%M:%S')} ({intervals[i]}): **${preds[i]}**")
    except Exception as e:
        st.error(f"⚠️ Lỗi khi dự đoán: {str(e)}")
