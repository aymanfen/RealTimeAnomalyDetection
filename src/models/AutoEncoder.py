from src.models.BaseClass import BaseAnomalyModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


def build_autoencoder(inputdim, latentdim, lr):
    # Input layers
    inputs = Input(shape=(inputdim,))

    # Encoder
    x = Dense(12, activation="relu")(inputs)
    latent = Dense(latentdim, activation="relu")(x)

    # Decoder
    x = Dense(12, activation="relu")(latent)
    output = Dense(inputdim, activation="linear")(x)

    # Autoencoder model
    autoencoder = Model(inputs=inputs, outputs=output)
    encoder = Model(inputs=inputs, outputs=latent)
    autoencoder.compile(optimizer='adam', loss="mse")

    return autoencoder, encoder


class AutoEncoderModel(BaseAnomalyModel):
    def __init__(self,
                 inputdim=18, latentdim=6, lr=0.001,
                 epochs=10, batchsize=64, validationsplit=0.1):

        self.params = {
            "inputdim": inputdim,
            "latentdim": latentdim,
            "lr": lr,
            "epochs": epochs,
            "batchsize": batchsize,
            "validationsplit": validationsplit,
        }
        self.model, self.encoder = build_autoencoder(inputdim, latentdim, lr)
        self.epochs = epochs
        self.batch_size = batchsize
        self.validation_split = validationsplit
        self.feature_names = None

    def fit(self, X, **fit_kwargs):
        callbacks = fit_kwargs.get("callbacks", [])
        earlystop = EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True, verbose=0)
        callbacks.append(earlystop)

        history = self.model.fit(
            X, X,
            epochs=self.epochs, batch_size=self.batch_size,
            validation_split=self.validation_split, callbacks=callbacks,
            shuffle=True, verbose=0,
        )
        return history

    def reconstructionerror(self, X):
        recon = self.model.predict(X, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)

    def score(self, X):
        return self.reconstructionerror(X)

    def encode(self, X):
        """
        Returns the latent representation of X. Used by the
        AutoEncoderFeatureExtractor preprocessing step so the encoder
        can feed downstream detectors (IsolationForest, SOM, ...).
        """
        return self.encoder.predict(X, verbose=0)

    def explain(self, X, feature_names=None):

        recon = self.model.predict(X, verbose=0)

        errors = (X - recon) ** 2

        scores = np.mean(errors, axis=1)

        contributions = (
            errors /
            (np.sum(errors, axis=1, keepdims=True) + 1e-8)
        )

        if feature_names is None:
            feature_names = [
                f"feature_{i}"
                for i in range(X.shape[1])
            ]

        explanations = []

        for row in contributions:

            explanations.append({
                feature_names[i]: float(row[i])
                for i in range(len(feature_names))
            })

        return scores, explanations