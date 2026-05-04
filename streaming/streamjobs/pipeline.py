from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, from_json

from Features import ComputeTimeFeatures, ComputeAllFreqs, ComputeAllEntropy
from Scoring import ComputeScores

'''
Usage :
opt/spark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.2 --py-files sparkjobs/Scoring.py sparkjobs/pipeline.py

'''

spark = SparkSession.builder \
    .appName("StreamPipeline") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.lake", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.lake.type", "hadoop") \
    .config("spark.sql.catalog.lake.warehouse", "hdfs://172.31.28.178:9000/iceberg") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("TransactionID",     StringType()),
    StructField("Time",              TimestampType()),
    StructField("AccountNumber",     StringType()),
    StructField("CardNumber",        StringType()),
    StructField("TransactionType",   StringType()),
    StructField("Channel",           StringType()),
    StructField("TransactionAmount", DoubleType()),
    StructField("MerchandCode",      StringType()),
    StructField("MerchandGroup",     StringType()),
    StructField("Country",           StringType()),
    StructField("City",              StringType()),
    StructField("Country2",          StringType()),
    StructField("City2",             StringType()),
    StructField("CardType",          StringType()),
    StructField("Age",               IntegerType()),
    StructField("Gender",            StringType()),
    StructField("Bank",              StringType()),
])

# ── 1. read ────────────────────────────────────────────────────────────────────
kafkadf = spark.readStream \
         .format("kafka") \
         .option("kafka.bootstrap.servers", "172.31.24.87:9092") \
         .option("subscribe",       "transactions") \
         .option("startingOffsets", "latest") \
         .load()

# ── 2. parse ───────────────────────────────────────────────────────────────────
df = kafkadf.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")



# ── 3. write functions ─────────────────────────────────────────────────────────
def IcebergBronzeWrite(batch_df, batch_id):
    batch_df.writeTo("lake.bronze.transactions").append()

def IcebergSilverWrite(batch_df, batch_id):
    batch_df = batch_df.repartition("AccountNumber")
    batch_df = ComputeTimeFeatures(batch_df)
    batch_df = ComputeAllFreqs(batch_df)
    batch_df = ComputeAllEntropy(batch_df)
    batch_df.writeTo("lake.silver.features").append()

def IcebergGoldWrite(batch_df, batch_id):
    batch_df = ComputeScores(batch_df)
    batch_df.writeTo("lake.gold.scores").append()


# ── 4. queries ─────────────────────────────────────────────────────────────────
bronzequery = df.writeStream \
      .foreachBatch(IcebergBronzeWrite) \
      .option("checkpointLocation", "/home/ec2-user/checkpoints/bronze") \
      .outputMode("append") \
      .start()

silverquery = df.writeStream \
      .foreachBatch(IcebergSilverWrite) \
      .option("checkpointLocation", "/home/ec2-user/checkpoints/silver") \
      .outputMode("append") \
      .start()

goldquery = spark.readStream.format("iceberg") \
    .load("lake.silver.features") \
    .writeStream.foreachBatch(IcebergGoldWrite) \
    .option("checkpointLocation", "/home/ec2-user/checkpoints/gold") \
    .start()

spark.streams.awaitAnyTermination()

# silverquery = (
#     df.writeStream
#       .foreachBatch(IcebergSilverWrite)
#       .option("checkpointLocation", "/home/ec2-user/checkpoints/silver")
#       .outputMode("append")
#       .trigger(processingTime="1 minute")
#       .start()
# )