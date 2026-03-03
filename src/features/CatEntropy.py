import numpy as np
import pandas as pd


def normalentropy(series):
    p=series.value_counts(normalize=True)
    h=-(p*np.log(p)).sum()
    return h/np.log(len(p)) if len(p) > 1 else 0

def ComputeCatEntropy(df):
    transactiontypeentropy=df.groupby("Account Number")['Transaction Type'].apply( normalentropy ).reset_index(name="TransactionTypeEntropy")
    channelentropy=df.groupby("Account Number")['Channel'].apply( normalentropy ).reset_index(name="ChannelEntropy")
    cardtypeentropy=df.groupby("Account Number")['Card Type'].apply( normalentropy ).reset_index(name="CardTypeEntropy")
    merchandentropy=df.groupby("Account Number")['Merchand Group'].apply( normalentropy ).reset_index(name="MerchandEntropy")
    countryentropy=df.groupby("Account Number")['Country'].apply( normalentropy ).reset_index(name="CountryEntropy")
    cityentropy=df.groupby("Account Number")['City'].apply( normalentropy ).reset_index(name="CityEntropy")

    df=df.merge(transactiontypeentropy,on='Account Number',how='left')
    df=df.merge(channelentropy,on='Account Number',how='left')
    df=df.merge(cardtypeentropy,on='Account Number',how='left')
    df=df.merge(merchandentropy,on='Account Number',how='left')
    df=df.merge(countryentropy,on='Account Number',how='left')
    df=df.merge(cityentropy,on='Account Number',how='left')

    return df