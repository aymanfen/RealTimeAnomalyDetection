from src.models.BaseClass import BaseAnomalyModel
from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestModel(BaseAnomalyModel):
    def __init__(self,nestimators=200,maxsamples=0.8,contamination=0.05,random_state=42):

        self.params={
            "n_estimators":nestimators,
            "max_samples":maxsamples,
            "contamination":contamination
            }
        
        self.model=IsolationForest(
            n_estimators=nestimators,
            max_samples=maxsamples,
            contamination=contamination,
            random_state=random_state
            )
        

    def fit(self,X):
        return self.model.fit(X)
    
    def score(self,X):
        return -self.model.decision_function(X)