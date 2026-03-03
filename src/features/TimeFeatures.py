import numpy as np
import pandas as pd

def ComputeTimeFeatures(df):
    df['LogAmount']=np.log(df['Transaction Amount'])
    df['MovingAvg']=df.groupby("Account Number")['LogAmount'].transform(lambda x : x.rolling(window=7).mean())
    df['MovingStd']=df.groupby("Account Number")['LogAmount'].transform(lambda x : x.rolling(window=7).std())
    df['TimeDiff']=df.groupby("Account Number")['Time'].diff().dt.total_seconds()/3600
    df['LogTimeDiff']=np.log(df['TimeDiff'])

    return df