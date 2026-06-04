from pyspark.sql import SparkSession

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
spark.sql("DROP TABLE IF EXISTS lake.gold.scores")


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

    -- Pipeline timestamp
    silvertimestamp TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(Time))
""")

#Gold Table
spark.sql("""
CREATE TABLE IF NOT EXISTS lake.gold.scores (
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

    -- Model scores
    som_score DOUBLE,
    som_explain MAP<STRING, DOUBLE>,

    -- Pipeline timestamp
    goldtimestamp TIMESTAMP
)
USING iceberg
PARTITIONED BY (days(Time))
""")
