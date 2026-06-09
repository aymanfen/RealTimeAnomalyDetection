"""
modelling.py  —  Gold fact extraction functions

Given a silver_df (already feature-engineered + scored), provides:
  - BuildTransactionFact(silver_df)  → TransactionFact-shaped DataFrame
  - BuildExplainFact(silver_df)      → ExplainFact-shaped DataFrame (long format)

Imported and called by pipeline.py inside foreachBatch.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, current_timestamp, explode, lit

# ── Anomaly thresholds — tune to your SOM score distribution ──────────────────
ANOMALY_THRESHOLD  = -0.10   # somscore < this → IsAnomaly  = True
CRITICAL_THRESHOLD = -0.15   # somscore < this → IsCritical = True


def BuildTransactionFact(silver_df: DataFrame) -> DataFrame:
    """
    Extracts TransactionFact columns from silver_df.

    Adds:
      - IsAnomaly    : somscore < ANOMALY_THRESHOLD
      - IsCritical   : somscore < CRITICAL_THRESHOLD
      - goldtimestamp: current pipeline timestamp

    Returns a DataFrame ready to append to lake.gold.TransactionFact.
    """
    return (
        silver_df
        .withColumn("IsAnomaly",     col("somscore") < lit(ANOMALY_THRESHOLD))
        .withColumn("IsCritical",    col("somscore") < lit(CRITICAL_THRESHOLD))
        .withColumn("goldtimestamp", current_timestamp())
        .select(
            "TransactionID",
            "Time",
            "AccountNumber",
            "MerchandCode",
            "TransactionType",
            "Channel",
            "TransactionAmount",
            "Country",
            "City",
            "somscore",
            "IsAnomaly",
            "IsCritical",
            "goldtimestamp",
        )
    )


def BuildExplainFact(silver_df: DataFrame) -> DataFrame:
    """
    Explodes the somexplain MAP<STRING, DOUBLE> into long format.

    One row per (TransactionID × FeatureName).
    Time is carried over for partition-local scans in Dremio
    without needing to join TransactionFact.

    Returns a DataFrame ready to append to lake.gold.ExplainFact.
    """
    return (
        silver_df
        .select("TransactionID", "Time", "somexplain")
        .select(
            "TransactionID",
            "Time",
            explode(col("somexplain")).alias("FeatureName", "FeatureValue"),
        )
    )