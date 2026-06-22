from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import skew
import pickle


class BaseAnomalyModel(ABC):

    @abstractmethod
    def fit(self, X, **fit_kwargs):
        pass

    @abstractmethod
    def score(self, X):
        """
        Must return anomaly scores
        Higher = more anomalous
        """
        pass

    def evaluate(self, X):
        """
        Unsupervised score-distribution statistics, computed from the
        model's own anomaly scores on X (no labels required).
        Used for training-time logging where ground truth isn't available.
        """
        scores = self.score(X)

        return {
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_std": float(np.std(scores)),
            "score_skew": float(skew(scores)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
            "tailsep": float(np.percentile(scores, 99) - np.percentile(scores, 90)),
        }

    @abstractmethod
    def explain(self, X):
        pass

    def save(self, model_name=None):
        if model_name is None:
            model_name = self.__class__.__name__

        filename = f"{model_name}.pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def get_params_for_logging(self):
        """
        Returns the model's hyperparameters as a flat dict suitable for
        mlflow.log_params(). Relies on the convention (used by every model
        in this package) that __init__ stores its hyperparameters in
        self.params.
        """
        return dict(getattr(self, "params", {}))