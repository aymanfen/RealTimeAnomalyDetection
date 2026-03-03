from src.models.BaseClass import BaseAnomalyModel
from minisom import MiniSom
import numpy as np



class SOMModel(BaseAnomalyModel):
    framework='unknown'
    def __init__(self,x=10,y=10,inputlen=18,
                 sigma=1.0,learning_rate=0.5,iterations=1000):
        
        self.params=locals()
        self.x=x
        self.y=y
        self.iterations=iterations
        self.model=MiniSom(
            self.x,self.y,inputlen,
            sigma=self.params['sigma'],
            learning_rate=self.params['learning_rate']
        )

    def fit(self,X):
        X=X.to_numpy()

        self.model.random_weights_init(X)
        self.model.train_random(X,self.iterations)


    def bmudistance(self, sample):
        winner=self.model.winner(sample)
        weights=self.model.get_weights()[winner]

        return np.linalg.norm(sample-weights)

    def score(self,X):
        X=X.to_numpy()
        distances=np.array([self.bmudistance(x) for x in X])
        return distances