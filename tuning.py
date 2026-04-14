import numpy as np
import pandas as pd

df=pd.read_csv("./data/dataset.csv")

df['Time']=pd.to_datetime(df['Time'])
df['Account Creation Date']=pd.to_datetime(df['Account Creation Date'])
df['Date']=df['Time'].dt.date
df['Hour']=df['Time'].dt.hour
df['Age']=df['Age'].astype(int)

df.sort_values(['Account Number','Time'],inplace=True)
df['rank']=df.groupby("Account Number")['Time'].rank(method='first',pct=True)

from src.features.TimeFeatures import ComputeTimeFeatures
from src.features.CatEntropy import ComputeCatEntropy
from src.features.CatFreq import ComputeCatFreq
from src.preprocessing.TrainTestSplit import split
from sklearn.preprocessing import StandardScaler


df=ComputeTimeFeatures(df)
df=ComputeCatEntropy(df)
df=ComputeCatFreq(df)

df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
df=df[['rank','Age', 
       'LogAmount', 'MovingAvg','MovingStd', 'LogTimeDiff','Hour',
       'TransactionTypeEntropy', 'ChannelEntropy','CardTypeEntropy', 'MerchandEntropy', 'CountryEntropy', 'CityEntropy',
       'TransactionTypeFreq','ChannelFreq','CardTypeFreq', 'MerchandFreq', 'CountryFreq','CityFreq']]

traindf,testdf=split(df)

scaler=StandardScaler()

traindfscaled=scaler.fit_transform(traindf)
testdfscaled=scaler.transform(testdf)

from src.models.IsolationForest import IsolationForestModel
from src.models.SelfOrganizingMap import SOMModel
from src.models.AutoEncoder import AutoEncoderModel
from src.training.train import trainmodel

results=testdf.copy()

#standard data
iso=IsolationForestModel(nestimators=300,maxsamples='auto',contamination=0.01)
ae=AutoEncoderModel(inputdim=traindf.shape[1],latentdim=6,lr=0.001,batchsize=256,epochs=50)
som=SOMModel(inputlen=traindf.shape[1],x=20,y=20,sigma=0.8,learning_rate=0.4)

iso.fit(traindfscaled)
ae.fit(traindfscaled)
som.fit(traindfscaled)

results['scaledisoscores']=iso.score(testdfscaled)
results['scaledaescores']=ae.score(testdfscaled)
results['scaledsomscores']=som.score(testdfscaled)

from sklearn.preprocessing import minmax_scale

results['scaledisoscores']=minmax_scale(results['scaledisoscores'])
results['scaledsomscores']=minmax_scale(results['scaledsomscores'])
results['scaledaescores']=minmax_scale(results['scaledaescores'])

iso.save(model_name='IF')
ae.save(model_name='AE')
som.save(model_name='SOM')

results.to_csv("./data/results.csv")