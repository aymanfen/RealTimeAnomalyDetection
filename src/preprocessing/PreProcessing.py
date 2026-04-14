import pandas as pd

def ClientScale(df, client_col, cols_to_scale):
    scaled_df = df.copy()
    
    scaled_df[cols_to_scale] = (
        df.groupby(client_col)[cols_to_scale]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    )
    
    return scaled_df.drop(columns=client_col)

def ClientNorm(df, client_col, cols_to_scale):
    normed_df = df.copy()
    
    normed_df[cols_to_scale] = (
        df.groupby(client_col)[cols_to_scale]
        .transform(lambda x: (x - x.min()) / (x.max()-x.min()))
    )
    
    return normed_df.drop(columns=client_col)