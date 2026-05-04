# Scoring.py
import numpy as np
import pandas as pd
import requests
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import DoubleType


FEATURE_COLS = ["Age","LogAmount","AmountZScore", "MovingAvg", "MovingStd",
    "LogTimeDiff","HourSin", "HourCos",
    "TransactionTypeFreq", "ChannelFreq", "CardTypeFreq",
    "MerchandFreq", "CountryFreq", "CityFreq",
    "TransactionTypeEntropy", "ChannelEntropy", "CardTypeEntropy",
    "MerchandEntropy", "CountryEntropy", "CityEntropy",
]

MODEL_SERVER = "http://localhost:8080"


def _prep(*cols) -> list:
    X = pd.concat(list(cols), axis=1)
    X.columns = FEATURE_COLS
    X = X.fillna(0.0)
    return X.values.tolist()           # JSON-serialisable


def _call(endpoint: str, instances: list) -> pd.Series:
    response = requests.post(
        f"{MODEL_SERVER}{endpoint}",
        json={"instances": instances},
        timeout=30
    )
    response.raise_for_status()
    return pd.Series(response.json()["scores"], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# UDFs — no model loading, just HTTP calls
# ──────────────────────────────────────────────────────────────────────────────
@pandas_udf(DoubleType())
def if_score_udf(*cols: pd.Series) -> pd.Series:
    return _call("/if/score", _prep(*cols))


@pandas_udf(DoubleType())
def ae_score_udf(*cols: pd.Series) -> pd.Series:
    return _call("/ae/score", _prep(*cols))


@pandas_udf(DoubleType())
def som_score_udf(*cols: pd.Series) -> pd.Series:
    return _call("/som/score", _prep(*cols))


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
TRANSACTION_COLS = [
    "TransactionID", "Time", "AccountNumber", "CardNumber",
    "TransactionType", "Channel", "TransactionAmount",
    "MerchandCode", "MerchandGroup", "Country", "City",
    "Country2", "City2", "CardType", "Age", "Gender", "Bank",
]

def ComputeScores(df: DataFrame) -> DataFrame:
    feature_args = [col(c) for c in FEATURE_COLS]
    return (
        df
        .withColumn("isoscore",  if_score_udf(*feature_args))
        .withColumn("aescore",  ae_score_udf(*feature_args))
        .withColumn("somscore", som_score_udf(*feature_args))
        .select(
            *[col(c) for c in TRANSACTION_COLS],
            col("isoscore"),
            col("aescore"),
            col("somscore"),
        )
    )