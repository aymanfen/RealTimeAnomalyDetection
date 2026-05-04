# model_server.py
import cloudpickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

import sys
sys.path.insert(0, "/home/ec2-user/")  # parent of src, not src itself

app = FastAPI(title="Anomaly Detection Model Server")

# ── Model paths ────────────────────────────────────────────────────────────────
IF_MODEL_PATH  = "models/IF.pkl"
AE_MODEL_PATH  = "models/AE.pkl"
SOM_MODEL_PATH = "models/SOM.pkl"

FEATURE_COLS = [
    "Age","HourSin", "HourCos","LogAmount",
    "AmountZScore", "MovingAvg", "MovingStd", "LogTimeDiff",
    "TransactionTypeFreq", "ChannelFreq", "CardTypeFreq",
    "MerchandFreq", "CountryFreq", "CityFreq",
    "TransactionTypeEntropy", "ChannelEntropy", "CardTypeEntropy",
    "MerchandEntropy", "CountryEntropy", "CityEntropy",
]

# ── Singleton model store ──────────────────────────────────────────────────────
class ModelStore:
    if_model  = None
    ae_model  = None
    som_model = None
    ae_max_err  = None
    som_max_err = None

store = ModelStore()


# ── Load all models at startup — once, in the server process ──────────────────
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


# ── Request / Response schemas ─────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    instances: list[list[float]]   # shape (n_rows, n_features)

class ScoreResponse(BaseModel):
    scores: list[float]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _prep(instances: list[list[float]]) -> np.ndarray:
    X = np.array(instances, dtype=np.float32)
    X = np.nan_to_num(X)
    if X.shape[1] != len(FEATURE_COLS):
        raise ValueError(
            f"Expected {len(FEATURE_COLS)} features, got {X.shape[1]}"
        )
    return X


def _normalise(scores: np.ndarray, max_attr: str) -> np.ndarray:
    current_max = getattr(store, max_attr, None)
    batch_max   = float(scores.max()) if scores.max() > 0 else 1.0
    new_max     = max(current_max, batch_max) if current_max else batch_max
    setattr(store, max_attr, new_max)
    return scores / new_max


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/if/score", response_model=ScoreResponse)
def if_score(req: ScoreRequest):
    if store.if_model is None:
        raise HTTPException(status_code=503, detail="IF model not loaded")
    X      = _prep(req.instances)
    scores = store.if_model.score(X)
    scores = _normalise(scores, "ae_max_err")
    return ScoreResponse(scores=scores.tolist())


@app.post("/ae/score", response_model=ScoreResponse)
def ae_score(req: ScoreRequest):
    if store.ae_model is None:
        raise HTTPException(status_code=503, detail="AE model not loaded")
    X      = _prep(req.instances)
    scores = store.ae_model.score(X)
    scores = _normalise(scores, "ae_max_err")
    return ScoreResponse(scores=scores.tolist())


@app.post("/som/score", response_model=ScoreResponse)
def som_score(req: ScoreRequest):
    if store.som_model is None:
        raise HTTPException(status_code=503, detail="SOM model not loaded")
    X      = _prep(req.instances)
    scores = store.som_model.score(X)
    scores = _normalise(scores, "som_max_err")
    return ScoreResponse(scores=scores.tolist())


@app.get("/health")
def health():
    return {
        "if_model":  store.if_model  is not None,
        "ae_model":  store.ae_model  is not None,
        "som_model": store.som_model is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)