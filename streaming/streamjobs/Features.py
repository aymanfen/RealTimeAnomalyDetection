import math
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, log, avg, stddev, lag, lit, sin, cos, when,
    unix_timestamp, hour, count, sum as spark_sum, max as spark_max, dense_rank
)
from pyspark.sql.window import Window


# ─────────────────────────────────────────────────────────────
# Shared windows
# ─────────────────────────────────────────────────────────────
def get_windows():
    roll = Window.partitionBy("AccountNumber").orderBy("Time").rowsBetween(-6, 0)   # last 7 rows (current + 6 preceding)
    order = Window.partitionBy("AccountNumber").orderBy("Time")
    
    return roll, order


# ─────────────────────────────────────────────────────────────
# Time features
# ─────────────────────────────────────────────────────────────
def ComputeTimeFeatures(df: DataFrame) -> DataFrame:
    roll, order = get_windows()

    return (
        df
        .withColumn("Hour",      hour(col("Time")))
        .withColumn("HourSin",   sin(lit(2 * math.pi) * col("Hour") / lit(24.0)))
        .withColumn("HourCos",   cos(lit(2 * math.pi) * col("Hour") / lit(24.0)))
        .withColumn("LogAmount", log(col("TransactionAmount").cast("double")))

        .withColumn("MovingAvg", avg("LogAmount").over(roll))
        .withColumn("MovingStd", stddev("LogAmount").over(roll))
        .withColumn("AmountZScore",(col("LogAmount") - col("MovingAvg")) / (col("MovingStd") + lit(1e-6)))

        .withColumn("PrevTime", lag("Time", 1).over(order))
        .withColumn("TimeDiff",(unix_timestamp("Time") - unix_timestamp("PrevTime")) / lit(3600.0))
        .withColumn("LogTimeDiff",log(when(col("TimeDiff").isNotNull() & (col("TimeDiff").cast("double") > lit(1e-6)),col("TimeDiff")).otherwise(lit(1e-6))))

        .drop("Hour", "PrevTime", "TimeDiff")
    )


# ─────────────────────────────────────────────────────────────
# Frequency — proportion of each category value in the 7-row window
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
        value_window = Window.partitionBy("AccountNumber", cat_col).orderBy("Time").rowsBetween(-6, 0)
        value_count = count("*").over(value_window)

        df = df.withColumn(feat,value_count.cast("double") / total_count.cast("double"))

    return df


# ─────────────────────────────────────────────────────────────
# Entropy — normalised Shannon entropy over the 7-row window
#
# Fix 1: divide (p * log(p)) by value_count so each category
#         contributes exactly once regardless of how many rows
#         share that value — avoiding the overcounting bug.
#
# Fix 2: normalise by log(n_unique) to match the pandas version,
#         clamped to 1 unique value (returns 0, same as pandas).
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
        value_window = Window.partitionBy("AccountNumber", cat_col).orderBy("Time").rowsBetween(-6, 0)
 
        # dense_rank over (AccountNumber, cat_col value) — unbounded, no rowsBetween,
        # so rank is stable across the whole partition. spark_max over the roll window
        # then picks the highest rank seen in the last 7 rows = n_unique in window.
        rank_window = Window.partitionBy("AccountNumber").orderBy(cat_col)
        
 
        value_count = count("*").over(value_window)
        n_unique    = spark_max(dense_rank().over(rank_window)).over(roll)
 
        p = value_count.cast("double") / total_count.cast("double")
 
        raw_entropy = -spark_sum((p * log(p)) / value_count.cast("double")).over(roll)
 
        df = df.withColumn(feat,when(n_unique > lit(1), raw_entropy / log(n_unique.cast("double"))).otherwise(lit(0.0)))

    return df


# ─────────────────────────────────────────────────────────────
def ComputeAllFeatures(df: DataFrame) -> DataFrame:
    df = ComputeTimeFeatures(df)
    df = ComputeAllFreqs(df)
    df = ComputeAllEntropy(df)
    return df