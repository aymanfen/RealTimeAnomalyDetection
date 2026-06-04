import numpy as np
import pandas as pd

def running_entropy(series):
    """
    For each position in the series, compute normalized entropy
    using only values seen up to and including that position.
    """
    counts = {}
    entropies = []
    
    for i, val in enumerate(series, start=1):
        counts[val] = counts.get(val, 0) + 1
        
        total = i  # number of transactions seen so far
        n_unique = len(counts)
        
        if n_unique == 1:
            entropies.append(0.0)  # only one category seen → zero entropy
            continue
        
        # compute entropy from running counts
        probs = np.array(list(counts.values())) / total
        h = -(probs * np.log(probs)).sum()
        h_norm = h / np.log(n_unique)  # normalize to [0, 1]
        entropies.append(h_norm)
    
    return pd.Series(entropies, index=series.index)


def ComputeCatEntropy(df):
    cat_cols = {
        'Transaction Type': 'TransactionTypeEntropy',
        'Channel':          'ChannelEntropy',
        'Card Type':        'CardTypeEntropy',
        'Merchand Group':   'MerchandEntropy',
        'Country':          'CountryEntropy',
        'City':             'CityEntropy',
    }
    
    for col, feat_name in cat_cols.items():
        df[feat_name] = (
            df.groupby("Account Number")[col]
            .transform(running_entropy)
        )
    
    return df