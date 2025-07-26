from kafka import KafkaProducer
import requests
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda  v: json.dumps(v).encode('utf-8')
)

TOPIC = 'btc_price'

def fetch_price():
    url = ''
    res = requests.get(url).json()
    print('res',res)
    return {'timestamp': time.time(), 'price': float(res['price'])}

while True:
    data = fetch_price()
    print("data",data)
    producer.send(TOPIC, value=data)
    print("Produced: ",data)
    time.sleep(60) # mỗi phút một lần
