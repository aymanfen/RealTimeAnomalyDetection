from pyspark.sql import SparkSession

'''
spark-submit --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.2 TableCreate.py

'''

spark = SparkSession.builder \
    .appName("StreamPipeline") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.lake", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.lake.type", "hadoop") \
    .config("spark.sql.catalog.lake.warehouse", "hdfs://192.168.1.14:9000/iceberg") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.bronze")
spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.silver")
spark.sql("CREATE NAMESPACE IF NOT EXISTS lake.gold")


spark.sql("DROP TABLE IF EXISTS lake.bronze.transactions")
spark.sql("DROP TABLE IF EXISTS lake.silver.features")
spark.sql("DROP TABLE IF EXISTS lake.gold.TimeDim")
spark.sql("DROP TABLE IF EXISTS lake.gold.ClientDim")
spark.sql("DROP TABLE IF EXISTS lake.gold.MerchandDim")
spark.sql("DROP TABLE IF EXISTS lake.gold.TransactionFact")
spark.sql("DROP TABLE IF EXISTS lake.gold.ExplainFact")

#Bronze Table
spark.sql("""
CREATE TABLE IF NOT EXISTS lake.bronze.transactions (
    TransactionID STRING,
    Time TIMESTAMP,
    AccountNumber STRING,
    CardNumber STRING,
    TransactionType STRING,
    Channel STRING,
    TransactionAmount DOUBLE,
    MerchandCode STRING,
    MerchandGroup STRING,
    Country STRING,
    City STRING,
    Country2 STRING,
    City2 STRING,
    CardType STRING,
    Age INT,
    Gender STRING,
    Bank STRING,

    -- Pipeline timestamp
    bronzetimestamp TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(Time))
""")


#Silver Table
spark.sql("""
CREATE TABLE IF NOT EXISTS lake.silver.features (
    -- =========================
    -- Original transaction data
    -- =========================
    TransactionID STRING,
    Time TIMESTAMP,
    AccountNumber STRING,
    CardNumber STRING,
    TransactionType STRING,
    Channel STRING,
    TransactionAmount DOUBLE,
    MerchandCode STRING,
    MerchandGroup STRING,
    Country STRING,
    City STRING,
    Country2 STRING,
    City2 STRING,
    CardType STRING,
    Age INT,
    Gender STRING,
    Bank STRING,

    -- =========================
    -- Time-based features
    -- =========================
    LogAmount DOUBLE,
    MovingAvg DOUBLE,
    MovingStd DOUBLE,
    AmountZScore DOUBLE,
    LogTimeDiff DOUBLE,
    HourSin DOUBLE,
    HourCos DOUBLE,

    -- =========================
    -- Frequency features
    -- =========================
    TransactionTypeFreq DOUBLE,
    ChannelFreq DOUBLE,
    CardTypeFreq DOUBLE,
    MerchandFreq DOUBLE,
    CountryFreq DOUBLE,
    CityFreq DOUBLE,

    -- =========================
    -- Entropy features
    -- =========================
    TransactionTypeEntropy DOUBLE,
    ChannelEntropy DOUBLE,
    CardTypeEntropy DOUBLE,
    MerchandEntropy DOUBLE,
    CountryEntropy DOUBLE,
    CityEntropy DOUBLE,

    -- =========================
    -- Model Scores
    -- =========================
    modelscore DOUBLE,
    modelexplain MAP<STRING, DOUBLE>,

    -- =========================
    -- Pipeline timestamp
    -- =========================
    silvertimestamp TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(Time))
""")

#Gold Table
# ── Dimensions ──────────────────────────────────────────
spark.sql("""
CREATE TABLE IF NOT EXISTS lake.gold.TimeDim (
    Time          TIMESTAMP,
    Date          DATE,
    Year          INT,
    Quarter       INT,
    Month         INT,
    Day           INT,
    Hour          INT,
    Minute        INT,
    Second        INT
)
USING iceberg
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS lake.gold.ClientDim (
    AccountNumber STRING,       
    CardNumber    STRING,
    Age           INT,
    Gender        STRING,
    Bank          STRING,
    CardType      STRING,
    AvgTxnAmount  DOUBLE,       
    UsualChannel  STRING        
)
USING iceberg
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS lake.gold.MerchandDim (
    MerchandCode  STRING,
    MerchandGroup STRING,
    Country      STRING,
    City         STRING
)
USING iceberg
""")

# ── Fact Tables ─────────────────────────────────────────
spark.sql("""
CREATE TABLE IF NOT EXISTS lake.gold.TransactionFact (
    TransactionID   STRING,
    Time            TIMESTAMP,         
    AccountNumber   STRING,      
    MerchandCode    STRING,     
    TransactionType STRING,
    Channel         STRING,
    TransactionAmount DOUBLE,
    Country         STRING,
    City            STRING,
    modelscore       DOUBLE,
    IsAnomaly       BOOLEAN,     -- som_score < threshold, precomputed
    IsCritical      BOOLEAN,     -- som_score < critical_threshold
    goldtimestamp   TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(Time))  
""")

spark.sql("""
CREATE TABLE IF NOT EXISTS lake.gold.ExplainFact (
    TransactionID   STRING,
    Time            TIMESTAMP,         
    FeatureName     STRING,
    FeatureValue    DOUBLE
)
USING iceberg
PARTITIONED BY (days(Time))
""")
