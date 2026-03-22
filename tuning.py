import numpy as np
import pandas as pd

df=pd.read_csv("./data/dataset_PFE_CDM_complet.csv")

df['Time']=pd.to_datetime(df['Time'])
df['Date']=df['Time'].dt.date
df['Hour']=df['Time'].dt.hour
df['Age']=df['Age'].astype(int)

df.sort_values(['Account Number','Time'],inplace=True)
df['rank']=df.groupby("Account Number")['Time'].rank(method='first',pct=True)

from src.features.TimeFeatures import ComputeTimeFeatures
from src.features.CatEntropy import ComputeCatEntropy
from src.features.CatFreq import ComputeCatFreq
from src.preprocessing.TrainTestSplit import split

df=ComputeTimeFeatures(df)
df=ComputeCatEntropy(df)
df=ComputeCatFreq(df)

df=df[['Age','rank',
       'LogAmount', 'MovingAvg','MovingStd', 'LogTimeDiff','Hour',
       'TransactionTypeEntropy', 'ChannelEntropy','CardTypeEntropy', 'MerchandEntropy', 'CountryEntropy', 'CityEntropy',
       'TransactionTypeFreq','ChannelFreq','CardTypeFreq', 'MerchandFreq', 'CountryFreq','CityFreq']]

traindf,testdf=split(df)

from src.models.IsolationForest import IsolationForestModel
from src.models.SelfOrganizingMap import SOMModel
from src.models.AutoEncoder import AutoEncoderModel

from src.training.seachspaces import ifsearchspace,somsearchspace,aesearchspace
from src.training.tune import tunemodel
 

# aestudy=tunemodel(AutoEncoderModel,aesearchspace,
#                 traindf,ntrials=20,experimentname='AETuningMeanMin',
#                 target='score_mean',dir='minimize')
