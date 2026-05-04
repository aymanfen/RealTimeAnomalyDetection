import math
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, log, avg, stddev, lag, lit, sin, cos, when,
    unix_timestamp, hour, count, sum as spark_sum
)
from pyspark.sql.window import Window


# ─────────────────────────────────────────────────────────────
# Shared windows (IMPORTANT: reuse them everywhere)
# ─────────────────────────────────────────────────────────────
def get_windows():
    roll = (
        Window.partitionBy("AccountNumber")
              .orderBy("Time")
              .rowsBetween(-6, 0)
    )

    order = (
        Window.partitionBy("AccountNumber")
              .orderBy("Time")
    )

    return roll, order


# ─────────────────────────────────────────────────────────────
# Time features (already good, just reuse windows)
# ─────────────────────────────────────────────────────────────
def ComputeTimeFeatures(df: DataFrame) -> DataFrame:
    roll, order = get_windows()

    return (
        df
        .withColumn("Hour", hour(col("Time")))
        .withColumn("HourSin", sin(lit(2 * math.pi) * col("Hour") / lit(24.0)))
        .withColumn("HourCos", cos(lit(2 * math.pi) * col("Hour") / lit(24.0)))
        .withColumn("LogAmount", log(col("TransactionAmount").cast("double")))

        .withColumn("MovingAvg", avg("LogAmount").over(roll))
        .withColumn("MovingStd", stddev("LogAmount").over(roll))
        .withColumn(
            "AmountZScore",
            (col("LogAmount") - col("MovingAvg")) / (col("MovingStd") + lit(1e-6))
        )

        .withColumn("PrevTime", lag("Time", 1).over(order))
        .withColumn(
            "TimeDiff",
            (unix_timestamp("Time") - unix_timestamp("PrevTime")) / lit(3600.0)
        )
        .withColumn(
            "LogTimeDiff",
            log(when(col("TimeDiff") > 1e-6, col("TimeDiff")).otherwise(lit(1e-6)))
        )

        .drop("Hour", "PrevTime", "TimeDiff")
    )


# ─────────────────────────────────────────────────────────────
# Frequency (optimized: reuse counts)
# ─────────────────────────────────────────────────────────────
def ComputeAllFreqs(df: DataFrame) -> DataFrame:
    roll, _ = get_windows()

    total_count = count("*").over(roll)

    for cat_col, feat in [
        ("TransactionType", "TransactionTypeFreq"),
        ("Channel",         "ChannelFreq"),
        ("CardType",        "CardTypeFreq"),
        ("MerchandGroup",   "MerchandFreq"),
        ("Country",         "CountryFreq"),
        ("City",            "CityFreq"),
    ]:
        value_window = (
            Window.partitionBy("AccountNumber", cat_col)
                  .orderBy("Time")
                  .rowsBetween(-6, 0)
        )

        value_count = count("*").over(value_window)

        df = df.withColumn(
            feat,
            value_count.cast("double") / total_count.cast("double")
        )

    return df


# ─────────────────────────────────────────────────────────────
# Entropy (NO groupBy, NO join, fully window-based)
# ─────────────────────────────────────────────────────────────
def ComputeAllEntropy(df: DataFrame) -> DataFrame:
    roll, _ = get_windows()

    total_count = count("*").over(roll)

    for cat_col, feat in [
        ("TransactionType", "TransactionTypeEntropy"),
        ("Channel",         "ChannelEntropy"),
        ("CardType",        "CardTypeEntropy"),
        ("MerchandGroup",   "MerchandEntropy"),
        ("Country",         "CountryEntropy"),
        ("City",            "CityEntropy"),
    ]:
        value_window = (
            Window.partitionBy("AccountNumber", cat_col)
                  .orderBy("Time")
                  .rowsBetween(-6, 0)
        )

        value_count = count("*").over(value_window)

        p = value_count.cast("double") / total_count.cast("double")

        df = df.withColumn(
            feat,
            -spark_sum(p * log(p)).over(roll)
        )

    return df


# ─────────────────────────────────────────────────────────────
def ComputeAllFeatures(df: DataFrame) -> DataFrame:
    df = ComputeTimeFeatures(df)
    df = ComputeAllFreqs(df)
    df = ComputeAllEntropy(df)
    return df