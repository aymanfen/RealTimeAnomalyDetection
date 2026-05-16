import cloudpickle
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import uvicorn
import sys

sys.path.insert(0, "/home/ec2-user/")


app = FastAPI(
    title="Anomaly Detection Model Server"
)


IF_MODEL_PATH  = "models/IF.pkl"
AE_MODEL_PATH  = "models/AE.pkl"
SOM_MODEL_PATH = "models/SOM.pkl"


FEATURE_COLS = [
    "Age","LogAmount","AmountZScore","MovingAvg","MovingStd",
    "LogTimeDiff","HourSin","HourCos",
    "TransactionTypeFreq","ChannelFreq","CardTypeFreq","MerchandFreq","CountryFreq","CityFreq",
    "TransactionTypeEntropy","ChannelEntropy","CardTypeEntropy","MerchandEntropy","CountryEntropy","CityEntropy",
]


class ModelStore:

    if_model = None
    ae_model = None
    som_model = None


store = ModelStore()


@app.on_event("startup")
def load_models():

    import keras

    with open(IF_MODEL_PATH, "rb") as f:
        store.if_model = cloudpickle.load(f)

    with open(AE_MODEL_PATH, "rb") as f:
        store.ae_model = cloudpickle.load(f)

    with open(SOM_MODEL_PATH, "rb") as f:
        store.som_model = cloudpickle.load(f)

    print("All models loaded.")


class ScoreRequest(BaseModel):

    instances: list[list[float]]


class ExplainResponse(BaseModel):

    scores: list[float]

    explanations: list[dict[str, float]]

class EnsembleExplainResponse(BaseModel):

    if_scores: list[float]
    if_explanations: list[dict[str, float]]

    ae_scores: list[float]
    ae_explanations: list[dict[str, float]]

    som_scores: list[float]
    som_explanations: list[dict[str, float]]


def _prep(instances):

    X = np.array(instances, dtype=np.float32)

    X = np.nan_to_num(X)

    if X.shape[1] != len(FEATURE_COLS):

        raise ValueError(
            f"Expected {len(FEATURE_COLS)} "
            f"features, got {X.shape[1]}"
        )

    return X


def _predict(model, X):

    scores, explanations = model.explain(
        X,
        feature_names=FEATURE_COLS
    )

    return ExplainResponse(
        scores=scores.tolist(),
        explanations=explanations
    )

def _predict_raw(model, X):

    scores, explanations = model.explain(
        X,
        feature_names=FEATURE_COLS
    )

    return scores.tolist(), explanations


@app.post(
    "/if/explain",
    response_model=ExplainResponse
)
def if_explain(req: ScoreRequest):

    if store.if_model is None:

        raise HTTPException(
            status_code=503,
            detail="IF model not loaded"
        )

    X = _prep(req.instances)

    return _predict(store.if_model, X)


@app.post(
    "/ae/explain",
    response_model=ExplainResponse
)
def ae_explain(req: ScoreRequest):

    if store.ae_model is None:

        raise HTTPException(
            status_code=503,
            detail="AE model not loaded"
        )

    X = _prep(req.instances)

    return _predict(store.ae_model, X)


@app.post(
    "/som/explain",
    response_model=ExplainResponse
)
def som_explain(req: ScoreRequest):

    if store.som_model is None:

        raise HTTPException(
            status_code=503,
            detail="SOM model not loaded"
        )

    X = _prep(req.instances)

    return _predict(store.som_model, X)

@app.post(
    "/ensemble/explain",
    response_model=EnsembleExplainResponse
)
def ensemble_explain(req: ScoreRequest):

    if (
        store.if_model is None or
        store.ae_model is None or
        store.som_model is None
    ):

        raise HTTPException(
            status_code=503,
            detail="One or more models not loaded"
        )

    X = _prep(req.instances)

    # Isolation Forest
    if_scores, if_explanations = _predict_raw(
        store.if_model,
        X
    )

    # AutoEncoder
    ae_scores, ae_explanations = _predict_raw(
        store.ae_model,
        X
    )

    # SOM
    som_scores, som_explanations = _predict_raw(
        store.som_model,
        X
    )

    return EnsembleExplainResponse(
        if_scores=if_scores,
        if_explanations=if_explanations,

        ae_scores=ae_scores,
        ae_explanations=ae_explanations,

        som_scores=som_scores,
        som_explanations=som_explanations,
    )


@app.get("/health")
def health():

    return {
        "if_model": store.if_model is not None,
        "ae_model": store.ae_model is not None,
        "som_model": store.som_model is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8080)