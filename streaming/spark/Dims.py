"""
dimensions.py  —  Dimension extraction + upsert functions

Provides:
  - BuildTimeDim(target_year)        → generates minute-grain TimeDim rows
  - BuildClientDimStaging(spark)     → computes ClientDim staging from full silver
  - BuildMerchandDimStaging(spark)   → computes MerchandDim staging from silver

  - RefreshClientDim(spark)          → staging → MERGE INTO lake.gold.ClientDim
  - RefreshMerchandDim(spark)        → staging → MERGE INTO lake.gold.MerchandDim
  - PopulateTimeDim(spark, year)     → writes TimeDim for the full target year

Run as a standalone batch job (every 15 min via Airflow / cron):

  spark-submit \
    --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.4.2 \
    dimensions.py [--bootstrap]

  --bootstrap : also (re)generates TimeDim for the current year (run once on deploy)
"""

import argparse
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    col, avg, count, lit, explode, sequence,
    year, quarter, month, dayofmonth,
    hour, minute, second, dayofweek,
    to_date, row_number, desc,
)
from pyspark.sql.types import TimestampType

WAREHOUSE      = "hdfs://192.168.1.14:9000/iceberg"
LOOKBACK_HOURS = 1   # how far back to scan silver to detect active accounts


# ══════════════════════════════════════════════════════════════════════════════
# TimeDim
# ══════════════════════════════════════════════════════════════════════════════
def BuildTimeDim(spark: SparkSession, target_year: int) -> DataFrame:
    """
    Generates one row per minute for the entire target_year.
    Returns a DataFrame shaped for lake.gold.TimeDim.
    """
    start_ts = f"{target_year}-01-01 00:00:00"
    end_ts   = f"{target_year}-12-31 23:59:00"

    seq_df = spark.range(1).select(
        explode(
            sequence(
                lit(start_ts).cast(TimestampType()),
                lit(end_ts).cast(TimestampType()),
                lit("1 minute").cast("interval"),
            )
        ).alias("Time")
    )

    return (
        seq_df
        .withColumn("Date",      to_date(col("Time")))
        .withColumn("Year",      year(col("Time")))
        .withColumn("Quarter",   quarter(col("Time")))
        .withColumn("Month",     month(col("Time")))
        .withColumn("Day",       dayofmonth(col("Time")))
        .withColumn("Hour",      hour(col("Time")))
        .withColumn("Minute",    minute(col("Time")))
        .withColumn("Second",    second(col("Time")))
        # dayofweek: 1 = Sunday … 7 = Saturday
    )


def PopulateTimeDim(spark: SparkSession, target_year: int) -> None:
    """
    Writes TimeDim for target_year. Overwrites the year partition — idempotent.
    """
    print(f"[TimeDim] Generating minute-grain rows for {target_year} …")
    df = BuildTimeDim(spark, target_year)
    df.writeTo("lake.gold.TimeDim").overwritePartitions()
    print(f"[TimeDim] ✅  Done.")


# ══════════════════════════════════════════════════════════════════════════════
# ClientDim
# ══════════════════════════════════════════════════════════════════════════════
def BuildClientDim(spark: SparkSession) -> DataFrame:
    """
    Builds ClientDim from the full silver history.
    """

    silver = spark.table("lake.silver.features")

    # Latest profile per account
    profile_window = (
        Window.partitionBy("AccountNumber")
        .orderBy(desc("silvertimestamp"))
    )

    profile_df = (
        silver
        .withColumn("rn", row_number().over(profile_window))
        .filter(col("rn") == 1)
        .select(
            "AccountNumber",
            "CardNumber",
            "Age",
            "Gender",
            "Bank",
            "CardType",
        )
    )

    # Average transaction amount over all history
    avg_amount_df = (
        silver
        .groupBy("AccountNumber")
        .agg(avg("TransactionAmount").alias("AvgTxnAmount"))
    )

    # Most frequent channel over all history
    channel_window = (
        Window.partitionBy("AccountNumber")
        .orderBy(desc("ChannelCount"))
    )

    usual_channel_df = (
        silver
        .groupBy("AccountNumber", "Channel")
        .agg(count("*").alias("ChannelCount"))
        .withColumn("rn", row_number().over(channel_window))
        .filter(col("rn") == 1)
        .select(
            "AccountNumber",
            col("Channel").alias("UsualChannel"),
        )
    )

    return (
        profile_df
        .join(avg_amount_df, "AccountNumber")
        .join(usual_channel_df, "AccountNumber")
    )

def PopulateClientDim(spark: SparkSession):
    print("[ClientDim] Building full dimension...")

    df = BuildClientDim(spark)

    (
        df.writeTo("lake.gold.ClientDim")
        .overwritePartitions()
    )

    print(f"[ClientDim] ✅ Wrote {df.count()} rows.")


# ══════════════════════════════════════════════════════════════════════════════
# MerchandDim
# ══════════════════════════════════════════════════════════════════════════════
def BuildMerchandDim(spark: SparkSession) -> DataFrame:
    """
    Builds MerchandDim from the full silver history.
    """

    merchand_window = (
        Window.partitionBy("MerchandCode")
        .orderBy(desc("silvertimestamp"))
    )

    return (
        spark.table("lake.silver.features")
        .withColumn("rn", row_number().over(merchand_window))
        .filter(col("rn") == 1)
        .select(
            "MerchandCode",
            "MerchandGroup",
            "Country2",
            "City2",
        )
    )


def PopulateMerchandDim(spark: SparkSession):
    print("[MerchandDim] Building full dimension...")

    df = BuildMerchandDim(spark)

    (
        df.writeTo("lake.gold.MerchandDim")
        .overwritePartitions()
    )

    print(f"[MerchandDim] ✅ Wrote {df.count()} rows.")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Also (re)generate TimeDim for the current year. Run once on first deploy.",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("CDM_Dimensions")
        .config("spark.sql.extensions",
                "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
        .config("spark.sql.catalog.lake",           "org.apache.iceberg.spark.SparkCatalog")
        .config("spark.sql.catalog.lake.type",      "hadoop")
        .config("spark.sql.catalog.lake.warehouse", WAREHOUSE)
        .config("spark.sql.shuffle.partitions",     "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    if args.bootstrap:
        PopulateTimeDim(spark, datetime.now().year)

    PopulateClientDim(spark)
    PopulateMerchandDim(spark)

    print("✅ All dimensions rebuilt.")

    print("✅  All dimensions refreshed.")
    spark.stop()