from src.models.BaseClass import BaseAnomalyModel
from minisom import MiniSom
import numpy as np


class SOMModel(BaseAnomalyModel):
    def __init__(self, x=10, y=10, inputlen=18,
                 sigma=0.8, learning_rate=0.4, iterations=1000):

        self.params = {
            "x": x,
            "y": y,
            "inputlen": inputlen,
            "sigma": sigma,
            "learning_rate": learning_rate,
            "iterations": iterations,
        }
        self.x = x
        self.y = y
        self.iterations = iterations
        self.model = MiniSom(
            self.x, self.y, inputlen,
            sigma=sigma,
            learning_rate=learning_rate,
        )
        self.feature_names = None

    def fit(self, X, **fit_kwargs):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        self.model.random_weights_init(X)
        self.model.train_random(X, self.iterations)

    def bmudistance(self, sample):
        winner = self.model.winner(sample)
        weights = self.model.get_weights()[winner]

        return np.linalg.norm(sample - weights)

    def score(self, X):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        distances = np.array([self.bmudistance(x) for x in X])
        return distances

    def explain(self, X, feature_names=None):

        if not isinstance(X, np.ndarray):
            X = X.to_numpy()

        scores = self.score(X)

        if feature_names is None:
            feature_names = [
                f"feature_{i}"
                for i in range(X.shape[1])
            ]

        explanations = []

        for sample in X:

            winner = self.model.winner(sample)

            prototype = self.model.get_weights()[winner]

            deviations = np.abs(sample - prototype)

            contributions = (
                deviations /
                (np.sum(deviations) + 1e-8)
            )

            explanations.append({
                feature_names[i]: float(contributions[i])
                for i in range(len(feature_names))
            })

        return scores, explanations