from src.models.BaseClass import BaseAnomalyModel
from sklearn.ensemble import IsolationForest
import numpy as np


class IsolationForestModel(BaseAnomalyModel):
    def __init__(self, nestimators=100, maxsamples='auto', contamination=0.001, random_state=42):

        self.params = {
            "n_estimators": nestimators,
            "max_samples": maxsamples,
            "contamination": contamination,
            "random_state": random_state,
        }

        self.model = IsolationForest(
            n_estimators=nestimators,
            max_samples=maxsamples,
            contamination=contamination,
            random_state=random_state,
        )
        self.feature_names = None

    def fit(self, X, **fit_kwargs):
        return self.model.fit(X)

    def score(self, X):
        return -self.model.decision_function(X)

    def explain(self, X, feature_names=None):

        scores = self.score(X)

        if feature_names is None:
            feature_names = [
                f"feature_{i}"
                for i in range(X.shape[1])
            ]

        explanations = []

        for sample in X:

            feature_importance = np.zeros(len(feature_names))

            for tree in self.model.estimators_:

                node_indicator = tree.decision_path(
                    sample.reshape(1, -1)
                )

                node_index = node_indicator.indices

                features = tree.tree_.feature

                for depth, node_id in enumerate(node_index):

                    feature_id = features[node_id]

                    if feature_id >= 0:

                        contribution = 1 / (depth + 1)

                        feature_importance[feature_id] += contribution

            feature_importance /= (
                np.sum(feature_importance) + 1e-8
            )

            explanations.append({
                feature_names[i]: float(feature_importance[i])
                for i in range(len(feature_names))
            })

        return scores, explanations