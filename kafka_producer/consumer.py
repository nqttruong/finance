from kafka import KafkaConsumer
from pymongo import MongoClient
import json

consumer = KafkaConsumer(
	'btc_price',
	bootstrap_servers='localhost:9092',
	value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

mongo_client = MongoClient('mongodb://localhost:27017/')
print("mongo_client: ",mongo_client)

db = mongo_client['realtime']
collection = db['btc_prices']

for message in consumer:
	data = message.value
	collection.insert_one(data)
	print("Saves to MongoDB:", data)