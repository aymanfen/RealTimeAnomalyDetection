from abc import ABC, abstractmethod
import numpy as np


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
            "score_p95": float(np.percentile(scores, 95)),
        }
    
    def normalize(self,scores):
        return (scores - scores.mean()) / scores.std()