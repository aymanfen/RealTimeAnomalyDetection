import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",  # or "latest"
    group_id="terminal-printer",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    key_deserializer=lambda k: k.decode("utf-8") if k else None,
)

print("Listening on topic 'transactions'...")

try:
    for msg in consumer:
        print("-" * 80)
        print(f"Partition: {msg.partition}")
        print(f"Offset:    {msg.offset}")
        print(f"Key:       {msg.key}")
        print("Value:")
        print(json.dumps(msg.value, indent=2))
except KeyboardInterrupt:
    print("\nStopping consumer...")
finally:
    consumer.close()