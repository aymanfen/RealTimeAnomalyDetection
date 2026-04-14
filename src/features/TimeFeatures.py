import numpy as np
import pandas as pd

def ComputeTimeFeatures(df):
    df['LogAmount']=np.log(df['Transaction Amount'])
    df['MovingAvg']=df.groupby("Account Number")['LogAmount'].transform(lambda x : x.rolling(window=7).mean())
    df['MovingStd']=df.groupby("Account Number")['LogAmount'].transform(lambda x : x.rolling(window=7).std())
    df['AmountZScore'] = (df['LogAmount'] - df['MovingAvg']) / (df['MovingStd'] + 1e-6)

    df['TimeDiff']=df.groupby("Account Number")['Time'].diff().dt.total_seconds()/3600
    df['LogTimeDiff']=np.log(df['TimeDiff'])
    
    df['Hoursin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hourcos'] = np.cos(2 * np.pi * df['Hour'] / 24)

    return df