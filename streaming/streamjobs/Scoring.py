# Scoring.py
import pandas as pd
import requests

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType,StructField,DoubleType,StringType,MapType

FEATURE_COLS = [
    "Age","LogAmount","AmountZScore","MovingAvg","MovingStd",
    "LogTimeDiff","HourSin","HourCos",
    "TransactionTypeFreq","ChannelFreq","CardTypeFreq","MerchandFreq","CountryFreq","CityFreq",
    "TransactionTypeEntropy","ChannelEntropy","CardTypeEntropy","MerchandEntropy","CountryEntropy","CityEntropy",
]

MODEL_SERVER = "http://localhost:8080"

ENSEMBLE_SCHEMA = StructType([
    StructField("if_score", DoubleType()),
    StructField("if_explain", MapType(StringType(), DoubleType())),

    StructField("ae_score", DoubleType()),
    StructField("ae_explain", MapType(StringType(), DoubleType())),

    StructField("som_score", DoubleType()),
    StructField("som_explain", MapType(StringType(), DoubleType())),
])

MODEL_OUTPUT_SCHEMA=StructType([
    StructField("som_score", DoubleType()),
    StructField("som_explain", MapType(StringType(), DoubleType())),
])

def _prep(*cols):
    X = pd.concat(list(cols), axis=1)
    X.columns = FEATURE_COLS
    X = X.fillna(0.0)

    return X.values.tolist()


def _call(endpoint, instances):
    response = requests.post(
        f"{MODEL_SERVER}{endpoint}",
        json={"instances": instances},
        timeout=60
    )

    response.raise_for_status()
    payload = response.json()
    scores = payload["scores"]
    explanations = payload["explanations"]

    return pd.DataFrame({
        "score": scores,
        "explain": explanations
    })


# ──────────────────────────────────────────────────────────────────────────────
# UDFs — no model loading, just HTTP calls
# ──────────────────────────────────────────────────────────────────────────────
@pandas_udf(MODEL_OUTPUT_SCHEMA)
def if_udf(*cols):
    return _call(
        "/if/explain",
        _prep(*cols)
    )

@pandas_udf(MODEL_OUTPUT_SCHEMA)
def ae_udf(*cols):

    return _call(
        "/ae/explain",
        _prep(*cols)
    )


@pandas_udf(MODEL_OUTPUT_SCHEMA)
def som_udf(*cols):
    df = _call("/som/explain", _prep(*cols))

    return pd.DataFrame({
        "som_score": df["score"],
        "som_explain": df["explain"],
    })

@pandas_udf(ENSEMBLE_SCHEMA)
def ensemble_udf(*cols):

    instances = _prep(*cols)

    response = requests.post(
        f"{MODEL_SERVER}/ensemble/explain",
        json={"instances": instances},
        timeout=120
    )

    response.raise_for_status()

    payload = response.json()

    return pd.DataFrame({
        "if_score": payload["if_scores"],
        "if_explain": payload["if_explanations"],

        "ae_score": payload["ae_scores"],
        "ae_explain": payload["ae_explanations"],

        "som_score": payload["som_scores"],
        "som_explain": payload["som_explanations"],
    })

# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────
TRANSACTION_COLS = [
    "TransactionID", "Time", "AccountNumber",
    "CardNumber", "TransactionType", "Channel",
    "TransactionAmount", "MerchandCode",
    "MerchandGroup", "Country", "City",
    "Country2", "City2", "CardType",
    "Age", "Gender", "Bank",
]

def ComputeScores(df: DataFrame) -> DataFrame:

    feature_args = [col(c) for c in FEATURE_COLS]

    df = df.withColumn("som",som_udf(*feature_args))
    
    return df.select(*[col(c) for c in TRANSACTION_COLS],
            # SOM
            col("som.som_score").alias("som_score"),
            col("som.som_explain").alias("som_explain"),
        )
    
