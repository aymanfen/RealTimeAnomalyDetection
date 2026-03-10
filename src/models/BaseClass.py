from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import skew

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
            "tailsep": float(np.percentile(scores,99)-np.mean(scores))
        }
    
    def normalize(self,scores):
        return (scores - scores.mean()) / scores.std()