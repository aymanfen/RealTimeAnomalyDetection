import pandas as pd
import numpy as np

def ComputeCatFreq(df):
    df['TransactionTypeFreq']=(df.groupby(['Account Number','Transaction Type']).cumcount()+1)/(df.groupby("Account Number").cumcount()+1)
    df['ChannelFreq']=(df.groupby(['Account Number','Channel']).cumcount()+1)/(df.groupby("Account Number").cumcount()+1)
    df['CardTypeFreq']=(df.groupby(['Account Number','Card Type']).cumcount()+1)/(df.groupby("Account Number").cumcount()+1)
    df['MerchandFreq']=(df.groupby(['Account Number','Merchand Group']).cumcount()+1)/(df.groupby("Account Number").cumcount()+1)
    df['CountryFreq']=(df.groupby(['Account Number','Country']).cumcount()+1)/(df.groupby("Account Number").cumcount()+1)
    df['CityFreq']=(df.groupby(['Account Number','City']).cumcount()+1)/(df.groupby("Account Number").cumcount()+1)

    return df