from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def CatEncoding(df):
    encoder=OneHotEncoder(sparse_output=False)
    encoded=encoder.fit_transform(df[['Account Type','Client Type']])
    encoded=pd.DataFrame(encoded,columns=encoder.get_feature_names_out(['Account Type','Client Type']))
    df=pd.concat([df,encoded],axis=1)
    return df