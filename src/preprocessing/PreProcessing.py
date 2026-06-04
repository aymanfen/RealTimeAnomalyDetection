import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def ClientScale(df, client_col, per_client_cols, global_cols):
    scaled_df = df.copy()
    
    # 1. Per-Client Scaling (Standardization)
    if per_client_cols:
        scaled_df[per_client_cols] = (
            df.groupby(client_col)[per_client_cols]
            .transform(lambda x: (x - x.mean()) /( x.std() + 1e-6) )
        )
    
    # 2. Global Scaling (Standardization)
    if global_cols:
        for col in global_cols:
            mean = df[col].mean()
            std = df[col].std()
            scaled_df[col] = (df[col] - mean) / (std+1e-6)
            
    return scaled_df

def ClientNorm(df, client_col, per_client_cols, global_cols):
    normed_df = df.copy()
    
    # 1. Per-Client Normalization (Min-Max)
    if per_client_cols:
        normed_df[per_client_cols] = (
            df.groupby(client_col)[per_client_cols]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6))
        )
    
    # 2. Global Normalization (Min-Max)
    if global_cols:
        for col in global_cols:
            c_min = df[col].min()
            c_max = df[col].max()
            normed_df[col] = (df[col] - c_min) / (c_max - c_min + 1e-6)
            
    return normed_df


