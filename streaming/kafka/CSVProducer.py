import json
import time
import csv
import argparse
from datetime import datetime, timezone
from kafka import KafkaProducer

'''
Usage:
python3 kafkascripts/CBSProducerCSV.py --csv transactions.csv --tps 100
python3 kafkascripts/CBSProducerCSV.py --csv transactions.csv --tps 50 --loop
python3 kafkascripts/CBSProducerCSV.py --csv transactions.csv --tps 100 --max-transactions 500

Expected CSV columns (same fields as the generated transactions):
TransactionID, Time, AccountNumber, CardNumber, TransactionType, Channel,
TransactionAmount, MerchandCode, MerchandGroup, Country, City, Country2,
City2, CardType, Age, Gender, Bank

Notes:
- If --tps is not set, rows are sent as fast as possible.
- If --loop is set, the producer loops back to the start of the file when
  it reaches the end, useful for continuous load testing.
- If --max-transactions is set, the producer stops after N messages
  regardless of how many rows the CSV contains.
- The "Time" field in the CSV is overwritten with the current UTC timestamp
  at send time so events always appear fresh in downstream consumers.
'''

def iter_csv(path: str, loop: bool):
    """Yield rows from a CSV file, optionally looping forever."""
    while True:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield dict(row)
        if not loop:
            break


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

p = argparse.ArgumentParser(
    description="CSV-based banking transaction Kafka producer"
)
p.add_argument(
    "--brokers", default="localhost:9092",
    help="Comma-separated Kafka bootstrap servers (default: localhost:9092)"
)
p.add_argument(
    "--topic", default="transactions",
    help="Target Kafka topic (default: transactions)"
)
p.add_argument(
    "--csv", required=True, metavar="FILE",
    help="Path to the CSV file containing transactions"
)
p.add_argument(
    "--tps", type=float, default=None,
    help="Transactions per second; omit to send as fast as possible"
)
p.add_argument(
    "--max-transactions", type=int, default=0,
    help="Stop after N messages (default: send all rows)"
)
p.add_argument(
    "--loop", action="store_true",
    help="Loop back to the start of the CSV when the end is reached"
)
p.add_argument(
    "--no-timestamp-override", action="store_true",
    help="Keep the original 'Time' value from the CSV instead of overwriting it"
)

args = p.parse_args()

# ---------------------------------------------------------------------------
# Producer setup
# ---------------------------------------------------------------------------

producer = KafkaProducer(
    bootstrap_servers=args.brokers,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda k: k.encode("utf-8") if k else None,
)

interval = 1.0 / args.tps if args.tps else None
max_tx = args.max_transactions
count = 0
next_send = time.time()

print(f"Reading from: {args.csv}")
print(f"Publishing to topic '{args.topic}' on {args.brokers}")
if args.tps:
    print(f"Rate: {args.tps} TPS  (interval: {interval*1000:.1f} ms)")
else:
    print("Rate: unlimited")
if max_tx:
    print(f"Stopping after {max_tx} messages")
if args.loop:
    print("Loop mode enabled — will restart CSV when exhausted")

# ---------------------------------------------------------------------------
# Main send loop
# ---------------------------------------------------------------------------

try:
    for row in iter_csv(args.csv, loop=args.loop):
        if max_tx and count >= max_tx:
            break

        
        tx = row
        tx["Time"] = datetime.strptime(tx["Time"], "%m/%d/%Y %H:%M").isoformat()
        tx["TransactionAmount"] = float(tx["TransactionAmount"])
        tx["Age"] = int(tx["Age"])
        key = tx.get("AccountNumber")

        producer.send(args.topic, key=key, value=tx)
        count += 1

        if count % 1000 == 0:
            print(f"  Sent {count} messages…")

        if interval:
            next_send += interval
            sleep_time = next_send - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    producer.flush()
    producer.close()
    print(f"Done. Total messages sent: {count}")