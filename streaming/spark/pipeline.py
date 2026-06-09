from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, from_json,current_timestamp,explode,map_keys,map_values,lit

from Features import ComputeTimeFeatures, ComputeAllFreqs, ComputeAllEntropy
from Scoring import ComputeScores, TRANSACTION_COLS, FEATURE_COLS

'''
Usage :
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.2 --py-files Scoring.py pipeline.py

'''

ANOMALY_THRESHOLD=3.5
CRITICAL_THRESHOLD=5.25

KAFKA_BOOTSTRAP   = "192.168.1.14:9092"
KAFKA_TOPIC       = "transactions"
CHECKPOINT_BASE   = "hdfs://192.168.1.14:9000/checkpoints/lakehouse"
WAREHOUSE         = "hdfs://192.168.1.14:9000/iceberg"

spark = SparkSession.builder \
    .appName("StreamPipeline") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.lake", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.lake.type", "hadoop") \
    .config("spark.sql.catalog.lake.warehouse", WAREHOUSE) \
    .config("spark.sql.shuffle.partitions", "2")\
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
         .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
         .option("subscribe",       KAFKA_TOPIC) \
         .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
         .load()

# ── 2. parse ───────────────────────────────────────────────────────────────────
df = kafkadf.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")



# ── 3. write functions ─────────────────────────────────────────────────────────
def LakehouseWrite(batch_df, batch_id):
    bronze_df = batch_df.withColumn("bronzetimestamp",current_timestamp())
    bronze_df.writeTo("lake.bronze.transactions").append()

    featureddf=batch_df.transform(ComputeTimeFeatures)\
        .transform(ComputeAllFreqs)\
        .transform(ComputeAllEntropy)

    scoreddf=featureddf.transform(ComputeScores)

    silver_df=scoreddf.withColumnRenamed("som_score","modelscore")\
        .withColumnRenamed("som_explain","modelexplain")\
        .withColumn("silvertimestamp",current_timestamp())
    
    silver_df.writeTo("lake.silver.features").append()

    txnfactdf=silver_df.withColumn("IsAnomaly",col("modelscore")>lit(ANOMALY_THRESHOLD))\
        .withColumn("IsCritical",col("modelscore")>lit(CRITICAL_THRESHOLD))\
        .withColumn("goldtimestamp",current_timestamp())\
        .select( "TransactionID",
            "Time",
            "AccountNumber",
            "MerchandCode",
            "TransactionType",
            "Channel",
            "TransactionAmount",
            "Country",
            "City",
            "modelscore",
            "IsAnomaly",
            "IsCritical",
            "goldtimestamp") 
    txnfactdf.writeTo("lake.gold.TransactionFact").append()
    
    explainfactdf=silver_df.select("TransactionID","Time","modelexplain")\
        .select("TransactionID","Time",
        explode(col("modelexplain")).alias("FeatureName","FeatureValue"))
    
    explainfactdf.writeTo("lake.gold.ExplainFact").append()



# ── 4. queries ─────────────────────────────────────────────────────────────────
query=df.writeStream\
    .foreachBatch(LakehouseWrite)\
    .option("checkpointLocation", "/home/aymanfen/checkpoints/lakehouse") \
    .outputMode("append") \
    .start()
    #.trigger(processingTime="5 seconds") \


query.awaitTermination()

