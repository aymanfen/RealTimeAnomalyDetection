from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import skew
import mlflow
import pickle
from src.models.MLFlowWrapper import AnomalyModelWrapper

class BaseAnomalyModel(ABC):

    @abstractmethod
    def fit(self, X,**fit_kwargs):
        pass

    @abstractmethod
    def score(self, X):
        """
        Must return anomaly scores
        Higher = more anomalous
        """
        pass

    def evaluate(self, X):
        scores = self.score(X)

        return {
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_skew": float(skew(scores)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
            "tailsep": float(np.percentile(scores,99)-np.percentile(scores,90))
        }
    
    def normalize(self,scores):
        return (scores - scores.mean()) / scores.std()

    def save(self, artifact_path="model",model_name=None):

        if model_name is None:
            model_name = self.__class__.__name__

        filename = f"{model_name}.pkl"

        with open(filename, "wb") as f:
            pickle.dump(self, f)

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=AnomalyModelWrapper(),
            artifacts={"model": filename}
        )

        mlflow.log_params(self.params)