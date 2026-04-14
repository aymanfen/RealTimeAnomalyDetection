import numpy as np
import pandas as pd
from src.features.TimeFeatures import ComputeTimeFeatures
from src.features.CatEntropy import ComputeCatEntropy
from src.features.CatFreq import ComputeCatFreq
from src.preprocessing.TrainTestSplit import split
from src.preprocessing.PreProcessing import ClientScale

df=pd.read_csv("./data/dataset_PFE_CDM_complet.csv")
df.sort_values(['Account Number','Time'],inplace=True)
df['rank']=df.groupby("Account Number")['Time'].rank(method='first',pct=True)

df['Time']=pd.to_datetime(df['Time'])
df['Date']=df['Time'].dt.date
df['Hour']=df['Time'].dt.hour
df['Age']=df['Age'].astype(int)

traindf,testdf=split(df)

traindf=ComputeTimeFeatures(traindf)
traindf=ComputeCatEntropy(traindf)
traindf=ComputeCatFreq(traindf)

traindf=traindf[['Account Number','Age', 'LogAmount','AmountZScore', 'MovingAvg','MovingStd', 'LogTimeDiff','Hoursin','Hourcos',
       'TransactionTypeEntropy', 'ChannelEntropy','CardTypeEntropy', 'MerchandEntropy', 'CountryEntropy', 'CityEntropy',
       'TransactionTypeFreq','ChannelFreq','CardTypeFreq', 'MerchandFreq', 'CountryFreq','CityFreq']]

traindf=ClientScale(traindf,'Account Number')

from src.models.IsolationForest import IsolationForestModel
from src.models.SelfOrganizingMap import SOMModel
from src.models.AutoEncoder import AutoEncoderModel

#standard data
iso=IsolationForestModel(nestimators=100,maxsamples=256,contamination=0.001)
ae=AutoEncoderModel(inputdim=traindf.shape[1],latentdim=4,lr=0.001,batchsize=256,epochs=50)
som=SOMModel(inputlen=traindf.shape[1],x=20,y=20,sigma=0.8,learning_rate=0.4)

iso.fit(traindf)
ae.fit(traindf)
som.fit(traindf)

testdf=ComputeTimeFeatures(testdf)
testdf=ComputeCatEntropy(testdf)
testdf=ComputeCatFreq(testdf)

testdf=testdf[['Account Number','Age', 'LogAmount','AmountZScore', 'MovingAvg','MovingStd', 'LogTimeDiff','Hoursin','Hourcos',
       'TransactionTypeEntropy', 'ChannelEntropy','CardTypeEntropy', 'MerchandEntropy', 'CountryEntropy', 'CityEntropy',
       'TransactionTypeFreq','ChannelFreq','CardTypeFreq', 'MerchandFreq', 'CountryFreq','CityFreq']]

testdf=ClientScale(testdf,'Account Number')

results=testdf.copy()
results['isoscores']=iso.score(testdf)
results['aescores']=ae.score(testdf)
results['somscores']=som.score(testdf)

from sklearn.preprocessing import minmax_scale

results['isoscores']=minmax_scale(results['isoscores'])
results['somscores']=minmax_scale(results['somscores'])
results['aescores']=minmax_scale(results['aescores'])

iso.save(model_name='IF')
ae.save(model_name='AE')
som.save(model_name='SOM')

results.to_csv("./data/results.csv")




