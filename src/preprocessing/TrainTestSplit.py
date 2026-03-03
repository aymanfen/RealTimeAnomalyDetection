import numpy as np
import pandas as pd

def split(df):

    traindf=df[df['rank']<=0.8].drop(columns='rank')
    testdf=df[df['rank']>0.8].drop(columns='rank')  

    traindf = traindf.replace([np.inf, -np.inf], np.nan).fillna(0)
    testdf = testdf.replace([np.inf, -np.inf], np.nan).fillna(0)

    return traindf,testdf
