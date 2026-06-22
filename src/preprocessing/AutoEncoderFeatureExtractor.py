"""
AutoEncoderFeatureExtractor
---------------------------
Wraps a trained AutoEncoderModel so it can be used as a preprocessing
step: fit on training features, then transform (encode) any feature
matrix into the autoencoder's latent space.

This is intentionally NOT a BaseAnomalyModel. The autoencoder here is
not being asked to flag anomalies on its own -- it is a dimensionality
reduction step that downstream detectors (IsolationForestModel,
SOMModel, ...) consume. Treating it as preprocessing keeps the
model layer "one model = one anomaly score" and keeps this
transformation alongside the other preprocessing steps (ClientScale /
ClientNorm) rather than blurring it into the model layer.

The underlying AutoEncoderModel and its hyperparameters are still
fully logged (see training/train.py) under a
"feature_extractor" / "feature_extractor_params" tag, so the fact
that an autoencoder sits upstream of the detector is never silently
lost -- it's just logged as a preprocessing choice rather than as the
model being evaluated.
"""

import numpy as np
from src.models.AutoEncoder import AutoEncoderModel


class AutoEncoderFeatureExtractor:

    name = "autoencoder"

    def __init__(self, inputdim, latentdim=6, lr=0.001,
                 epochs=20, batchsize=256, validationsplit=0.1):

        self.params = {
            "inputdim": inputdim,
            "latentdim": latentdim,
            "lr": lr,
            "epochs": epochs,
            "batchsize": batchsize,
            "validationsplit": validationsplit,
        }

        self._autoencoder = AutoEncoderModel(
            inputdim=inputdim,
            latentdim=latentdim,
            lr=lr,
            epochs=epochs,
            batchsize=batchsize,
            validationsplit=validationsplit,
        )
        self._fitted = False

    def fit(self, X, **fit_kwargs):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        self._autoencoder.fit(X, **fit_kwargs)
        self._fitted = True
        return self

    def transform(self, X):
        if not self._fitted:
            raise RuntimeError(
                "AutoEncoderFeatureExtractor must be fit() before transform()."
            )
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        return self._autoencoder.encode(X)

    def fit_transform(self, X, **fit_kwargs):
        self.fit(X, **fit_kwargs)
        return self.transform(X)

    def get_params_for_logging(self):
        return dict(self.params)