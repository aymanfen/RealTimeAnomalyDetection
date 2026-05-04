import json
import time
import uuid
import random
import signal
import logging
import argparse
from datetime import datetime, timezone
from faker import Faker
from kafka import KafkaProducer

'''
Usage :
python3 kafkascripts/CBSProducer.py --max-transactions 1000 --tps 100

'''


TRANSACTION_TYPES = ["PAYMENT", "WITHDRAWAL", "TRANSFER", "REFUND"]
CHANNELS = ["ATM", "POS", "WEB", "MOBILE"]
CARD_TYPES = ["VISA", "MASTERCARD", "AMEX"]
MERCHANT_GROUPS = ["GROCERY", "TECH", "TRAVEL", "RESTAURANT"]
COUNTRIES = ["MA", "FR", "ES", "US"]
CITIES = ["Casablanca", "Rabat", "Paris", "Madrid", "New York"]

fake = Faker()
def GenerateTransaction():
    amount = round(random.uniform(10, 5000), 2)
    return {
        "TransactionID":str(uuid.uuid4()),
        "Time":datetime.now(timezone.utc).isoformat(),
        "AccountNumber":f"ACC{random.randint(100000, 999999)}",
        "CardNumber":f"****-****-****-{random.randint(1000,9999)}",
        "TransactionType":random.choice(TRANSACTION_TYPES),
        "Channel":random.choice(CHANNELS),
        "TransactionAmount":amount,
        "MerchandCode":f"MER{random.randint(1000,9999)}",
        "MerchandGroup":random.choice(MERCHANT_GROUPS),
        "Country":random.choice(COUNTRIES),
        "City":random.choice(CITIES),
        "Country2":random.choice(COUNTRIES),
        "City2":random.choice(CITIES),
        "CardType":random.choice(CARD_TYPES),
        "Age":random.randint(18, 80),
        "Gender":random.choice(["M", "F"]),
        "Bank":"CDM"
    }


p = argparse.ArgumentParser(description="Parametric banking transaction Kafka producer")
p.add_argument("--brokers", default="172.31.24.87:9092",
                   help="Comma-separated Kafka bootstrap servers")
p.add_argument("--topic",   default="transactions",
                   help="Target Kafka topic")
p.add_argument("--tps", type=float, help="Transactions per second (e.g. 500)")
 
p.add_argument("--max-transactions", type=int, default=0,
                   help="Stop after N messages (default: run forever)")

args=p.parse_args()


producer=KafkaProducer(
    bootstrap_servers=args.brokers,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda k: k.encode("utf-8") if k else None
)

count=0
rate=args.tps
interval=1.0/rate
maxtransactions=args.max_transactions

nextsend=time.time()

while True:
    if maxtransactions and count >= maxtransactions:
        break
    
    tx=GenerateTransaction()
    key=tx['AccountNumber']

    producer.send(args.topic,key=key,value=tx)
    count+=1
    
    nextsend+=interval
    sleeptime=nextsend-time.time()
    if sleeptime > 0:
        time.sleep(sleeptime)

producer.flush()
producer.close()







