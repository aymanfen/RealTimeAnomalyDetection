import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def ClientScale(df, client_col, per_client_cols, global_cols):
    scaled_df = df.copy()
    
    # 1. Per-Client Scaling (Standardization)
    if per_client_cols:
        scaled_df[per_client_cols] = (
            df.groupby(client_col)[per_client_cols]
            .transform(lambda x: (x - x.mean()) / x.std())
        )
    
    # 2. Global Scaling (Standardization)
    if global_cols:
        for col in global_cols:
            mean = df[col].mean()
            std = df[col].std()
            scaled_df[col] = (df[col] - mean) / std
            
    return scaled_df

def ClientNorm(df, client_col, per_client_cols, global_cols):
    normed_df = df.copy()
    
    # 1. Per-Client Normalization (Min-Max)
    if per_client_cols:
        normed_df[per_client_cols] = (
            df.groupby(client_col)[per_client_cols]
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )
    
    # 2. Global Normalization (Min-Max)
    if global_cols:
        for col in global_cols:
            c_min = df[col].min()
            c_max = df[col].max()
            normed_df[col] = (df[col] - c_min) / (c_max - c_min)
            
    return normed_df



class ClientScaler(BaseEstimator, TransformerMixin):
    def __init__(self, client_col, cols_to_scale, add_new_client_flag=True):
        self.client_col = client_col
        self.cols = cols_to_scale
        self.add_new_client_flag = add_new_client_flag

        self.client_stats_ = None
        self.global_stats_ = None

    def fit(self, X, y=None):
        df = X.copy()

        # --- Per-client stats ---
        client_stats = (
            df.groupby(self.client_col)[self.cols]
            .agg(['mean', 'std'])
        )
        client_stats.columns = [f"{col}_{stat}" for col, stat in client_stats.columns]

        # --- Global stats (fallback) ---
        global_stats = df[self.cols].agg(['mean', 'std'])

        self.client_stats_ = client_stats
        self.global_stats_ = global_stats

        return self

    def transform(self, X):
        df = X.copy()

        # merge client stats
        df = df.merge(
            self.client_stats_,
            left_on=self.client_col,
            right_index=True,
            how='left'
        )

        # apply scaling with fallback
        for col in self.cols:
            mean_col = f"{col}_mean"
            std_col = f"{col}_std"

            df[mean_col] = df[mean_col].fillna(self.global_stats_.loc['mean', col])
            df[std_col]  = df[std_col].fillna(self.global_stats_.loc['std', col])

            df[col] = (df[col] - df[mean_col]) / (df[std_col] + 1e-6)

        # cleanup helper columns
        drop_cols = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
        df.drop(columns=drop_cols, inplace=True)

        # drop client column (optional — depends on your model)
        df.drop(columns=self.client_col, inplace=True)

        return df