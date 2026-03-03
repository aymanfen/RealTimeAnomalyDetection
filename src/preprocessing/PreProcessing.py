import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder

def AccNumEncode(traindf,testdf):
    le=LabelEncoder()
    traindf['Account Number']=le.fit_transform(traindf['Account Number'])
    testdf['Account Number']=le.transform(testdf['Account Number'])

    Xtrain=traindf.drop(columns='Account Number').values
    trainAccs=traindf['Account Number'].values

    testdf['Account Number']=le.transform(testdf['Account Number'])
    Xtest=testdf.drop(columns='Account Number').values
    testAccs=testdf['Account Number'].values

    return Xtrain,trainAccs,Xtest,testAccs