"""
main.py
------------------
Iterates over standardization methods × standalone models and
standardization methods × (Autoencoder encoder + anomaly model),
then prints AUC, PR-AUC, and max-F1 for each combination,
evaluated on a labeled validation set.

Usage:
    python main.py \
        --train  ../data/dataset_PFE_CDM_complet.csv \
        --val    ../data/validation_dataset.csv

Adjust paths as needed.
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from itertools import product

# ── sklearn ────────────────────────────────────────────────────────────────
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# ── tensorflow / keras ─────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# ── minisom (pip install minisom) ──────────────────────────────────────────
from minisom import MiniSom

from features.TimeFeatures import ComputeTimeFeatures
from features.CatEntropy import ComputeCatEntropy
from features.CatFreq import ComputeCatFreq
from preprocessing.PreProcessing import ClientScale,ClientNorm

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


# ═══════════════════════════════════════════════════════════════════════════
# 1.  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'Age', 'LogAmount', 'AmountZScore', 'MovingAvg', 'MovingStd',
    'LogTimeDiff', 'Hoursin', 'Hourcos',
    'TransactionTypeEntropy', 'ChannelEntropy', 'CardTypeEntropy',
    'MerchandEntropy', 'CountryEntropy', 'CityEntropy',
    'TransactionTypeFreq', 'ChannelFreq', 'CardTypeFreq',
    'MerchandFreq', 'CountryFreq', 'CityFreq',
]

GLOBAL_COLS = [
    'Age', 'LogAmount', 'LogTimeDiff',
    'TransactionTypeEntropy', 'ChannelEntropy', 'CardTypeEntropy',
    'MerchandEntropy', 'CountryEntropy', 'CityEntropy',
]

CLIENT_COLS = [
    'AmountZScore', 'MovingAvg', 'MovingStd',
    'TransactionTypeFreq', 'ChannelFreq', 'CardTypeFreq',
    'MerchandFreq', 'CountryFreq', 'CityFreq',
]

CAT_ENTROPY_COLS = {
    'TransactionTypeEntropy': 'Transaction Type',
    'ChannelEntropy':         'Channel',
    'CardTypeEntropy':        'Card Type',
    'MerchandEntropy':        'Merchand Group',
    'CountryEntropy':         'Country',
    'CityEntropy':            'City',
}

CAT_FREQ_COLS = {
    'TransactionTypeFreq': 'Transaction Type',
    'ChannelFreq':         'Channel',
    'CardTypeFreq':        'Card Type',
    'MerchandFreq':        'Merchand Group',
    'CountryFreq':         'Country',
    'CityFreq':            'City',
}


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = ComputeTimeFeatures(df)
    df = ComputeCatEntropy(df)
    df = ComputeCatFreq(df)
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_train(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.sort_values(['Account Number', 'Time'], inplace=True)
    df['rank'] = df.groupby('Account Number')['Time'].rank(method='first', pct=True)
    df['Time'] = pd.to_datetime(df['Time'])
    df['Date'] = df['Time'].dt.date
    df['Hour'] = df['Time'].dt.hour
    df['Age'] = df['Age'].astype(int)
    df['Transaction Amount'] = df['Transaction Amount'].astype(str).str.replace(',', '').astype(float)
    
    return df


def load_val(path: str):
    df = pd.read_csv(path)
    df.drop(columns=['ClientID', 'ClientName'], inplace=True, errors='ignore')
    anomalies = df.pop('AnomalyType') if 'AnomalyType' in df.columns else None
    columns = [
        "TranasctionID", "Time", "Account Number", "Card Number",
        "Transaction Type", "Channel", "Transaction Amount",
        "Merchand Group", "Country", "Country2", "City2", "Card Type",
        "Age", "Gender", "Bank", "City", "Merchand Code", "isAnomaly",
    ]
    df.columns = columns
    df.sort_values(['Account Number', 'Time'], inplace=True)
    df['rank'] = df.groupby('Account Number')['Time'].rank(method='first', pct=True)
    df['Time'] = pd.to_datetime(df['Time'])
    df['Date'] = df['Time'].dt.date
    df['Hour'] = df['Time'].dt.hour
    df['Age'] = df['Age'].astype(int)
    df['Transaction Amount'] = (
        df['Transaction Amount'].astype(str).str.replace(',', '').astype(float)
    )
    y_true = df['isAnomaly'].values
    return df, y_true


# ═══════════════════════════════════════════════════════════════════════════
# 3.  STANDARDIZATION
# ═══════════════════════════════════════════════════════════════════════════

def apply_scaling(method: str,
                  train_feat: pd.DataFrame,
                  val_feat: pd.DataFrame,
                  train_raw: pd.DataFrame,
                  val_raw: pd.DataFrame):
    """
    Returns (X_train, X_val) as numpy arrays, scaled according to `method`.
    """
    if method == 'none':
        return train_feat, val_feat

    elif method == 'global_minmax':
        sc = MinMaxScaler()
        X_tr = pd.DataFrame(sc.fit_transform(train_feat[FEATURE_COLS]),columns=FEATURE_COLS)
        X_va = pd.DataFrame(sc.transform(val_feat[FEATURE_COLS]),columns=FEATURE_COLS)
        return X_tr, X_va

    elif method == 'global_standard':
        sc = StandardScaler()
        X_tr = pd.DataFrame(sc.fit_transform(train_feat[FEATURE_COLS]),columns=FEATURE_COLS)
        X_va = pd.DataFrame(sc.transform(val_feat[FEATURE_COLS]),columns=FEATURE_COLS)
        return X_tr, X_va

    elif method == 'global_minmax_client_minmax':

        train_g = ClientNorm(train_feat,'Account Number',CLIENT_COLS,GLOBAL_COLS)
        val_g   = ClientNorm(val_feat,'Account Number',CLIENT_COLS,GLOBAL_COLS)
        return train_g, val_g

    elif method == 'global_standard_client_standard':

        train_g = ClientScale(train_feat,'Account Number',CLIENT_COLS,GLOBAL_COLS)
        val_g   = ClientScale(val_feat,'Account Number',CLIENT_COLS,GLOBAL_COLS)
        return train_g, val_g

    else:
        raise ValueError(f"Unknown scaling method: {method}")


SCALING_METHODS = [
    'none',
    'global_minmax',
    'global_standard',
    'global_minmax_client_minmax',
    'global_standard_client_standard',
]

SCALING_LABELS = {
    'none':                             'No Scaling',
    'global_minmax':                    'Global MinMax',
    'global_standard':                  'Global Standard',
    'global_minmax_client_minmax':      'Global MinMax + Client MinMax',
    'global_standard_client_standard':  'Global Standard + Client Standard',
}


# ═══════════════════════════════════════════════════════════════════════════
# 4.  MODELS
# ═══════════════════════════════════════════════════════════════════════════

def clean(arr: np.ndarray) -> np.ndarray:
    arr = np.where(np.isinf(arr), np.nan, arr)
    col_means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_means, inds[1])
    return arr


# ── Isolation Forest ───────────────────────────────────────────────────────

def fit_predict_if(X_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    model = IsolationForest(n_estimators=100, max_samples='auto',
                            contamination=0.001, random_state=42)
    model.fit(X_train)
    # decision_function: higher = more normal → negate for anomaly score
    return -model.decision_function(X_val)


# ── Local Outlier Factor ───────────────────────────────────────────────────

def fit_predict_lof(X_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.001)
    model.fit(X_train)
    return -model.decision_function(X_val)


# ── Self-Organizing Map ────────────────────────────────────────────────────

def fit_predict_som(X_train: np.ndarray, X_val: np.ndarray,
                    input_len: int) -> np.ndarray:
    som = MiniSom(10, 10, input_len, sigma=0.8, learning_rate=0.4,
                  random_seed=42)
    som.train_batch(X_train, num_iteration=5000, verbose=False)

    # anomaly score = quantisation error (distance to best matching unit)
    scores = np.array([som.quantization_error(X_val[i:i+1])
                       for i in range(len(X_val))])
    return scores


# ── Autoencoder ────────────────────────────────────────────────────────────

def build_autoencoder(input_dim: int, latent_dim: int = 8):
    inputs = Input(shape=(input_dim,))
    x      = Dense(12, activation='relu')(inputs)
    latent = Dense(latent_dim, activation='relu')(x)
    x      = Dense(12, activation='relu')(latent)
    outputs = Dense(input_dim, activation='linear')(x)

    autoencoder = Model(inputs, outputs)
    encoder     = Model(inputs, latent)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder


def fit_predict_ae(X_train: np.ndarray, X_val: np.ndarray) -> np.ndarray:
    ae, _ = build_autoencoder(X_train.shape[1])
    ae.fit(X_train, X_train, epochs=20, batch_size=256,
           validation_split=0.1, verbose=0)
    recon = ae.predict(X_val, verbose=0)
    # reconstruction error = anomaly score
    return np.mean((X_val - recon) ** 2, axis=1)


def fit_encoder(X_train: np.ndarray, latent_dim: int = 6):
    ae, encoder = build_autoencoder(X_train.shape[1], latent_dim)
    ae.fit(X_train, X_train, epochs=20, batch_size=256,
           validation_split=0.1, verbose=0)
    return encoder


# ═══════════════════════════════════════════════════════════════════════════
# 5.  METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    auc    = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return {'AUC': auc, 'PR_AUC': pr_auc, 'Max_F1': float(np.max(f1))}


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

STANDALONE_MODELS = ['IsolationForest', 'SOM', 'AutoEncoder']
META_MODELS       = ['IsolationForest', 'SOM']


def run_standalone(model_name: str,
                   X_train: np.ndarray,
                   X_val: np.ndarray,
                   y_true: np.ndarray) -> dict:
    if model_name == 'IsolationForest':
        scores = fit_predict_if(X_train, X_val)
    elif model_name == 'SOM':
        scores = fit_predict_som(X_train, X_val, X_train.shape[1])
    elif model_name == 'AutoEncoder':
        scores = fit_predict_ae(X_train, X_val)
    else:
        raise ValueError(model_name)

    if scores is None:
        return None
    return compute_metrics(y_true, scores)


def run_meta(meta_model_name: str,
             encoder: Model,
             X_train: np.ndarray,
             X_val: np.ndarray,
             y_true: np.ndarray) -> dict:
    Z_train = encoder.predict(X_train, verbose=0)
    Z_val   = encoder.predict(X_val,   verbose=0)

    if meta_model_name == 'IsolationForest':
        scores = fit_predict_if(Z_train, Z_val)
    elif meta_model_name == 'SOM':
        scores = fit_predict_som(Z_train, Z_val, Z_train.shape[1])
    else:
        raise ValueError(meta_model_name)

    if scores is None:
        return None
    return compute_metrics(y_true, scores)


def main(train_path: str, val_path: str):
    # ── Load & engineer ────────────────────────────────────────────────────
    print("Loading training data …")
    train_raw = load_train(train_path)
    train_eng = feature_engineer(train_raw)
    # train_feat = train_eng[FEATURE_COLS].reset_index(drop=True)
    train_raw  = train_raw.reset_index(drop=True)

    print("Loading validation data …")
    val_raw, y_true = load_val(val_path)
    val_eng  = feature_engineer(val_raw)
    # val_feat = val_eng[FEATURE_COLS].reset_index(drop=True)
    val_raw  = val_raw.reset_index(drop=True)

    rows = []

    for scaling in SCALING_METHODS:
        print(f"\n{'═'*60}")
        print(f"  Scaling: {SCALING_LABELS[scaling]}")
        print(f"{'═'*60}")

        X_train, X_val = apply_scaling(scaling, train_eng, val_eng,train_raw, val_raw)
        
        X_train = X_train[FEATURE_COLS].reset_index(drop=True)
        X_val   = X_val[FEATURE_COLS].reset_index(drop=True)
        
        X_train = clean(X_train)
        X_val   = clean(X_val)

        # ── Standalone models ──────────────────────────────────────────────
        print("\n  [Standalone models]")
        for model_name in STANDALONE_MODELS:
            try:
                metrics = run_standalone(model_name, X_train, X_val, y_true)
                tag = f"Standalone/{model_name}"
                print(f"    {model_name:20s}  "
                      f"AUC={metrics['AUC']:.4f}  "
                      f"PR-AUC={metrics['PR_AUC']:.4f}  "
                      f"MaxF1={metrics['Max_F1']:.4f}")
                rows.append({'Scaling': SCALING_LABELS[scaling],
                             'Pipeline': tag, **metrics})
            except Exception as e:
                print(f"    {model_name:20s}  ERROR: {e}")

        # ── Train shared encoder ───────────────────────────────────────────
        print("\n  [Encoder + meta-model]")
        try:
            encoder = fit_encoder(X_train, latent_dim=8)
        except Exception as e:
            print(f"    Encoder training failed: {e}")
            continue

        for meta_name in META_MODELS:
            try:
                metrics = run_meta(meta_name, encoder, X_train, X_val, y_true)
                tag = f"AE+{meta_name}"
                print(f"    AE+{meta_name:18s}  "
                      f"AUC={metrics['AUC']:.4f}  "
                      f"PR-AUC={metrics['PR_AUC']:.4f}  "
                      f"MaxF1={metrics['Max_F1']:.4f}")
                rows.append({'Scaling': SCALING_LABELS[scaling],
                             'Pipeline': tag, **metrics})
            except Exception as e:
                print(f"    AE+{meta_name:18s}  ERROR: {e}")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'═'*80}")
    print("SUMMARY")
    print(f"{'═'*80}")
    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        pd.set_option('display.max_rows', 200)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(results_df.to_string(index=False))

        out = 'results.csv'
        results_df.to_csv(out, index=False)
        print(f"\nResults saved to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True,
                        help='Path to training CSV (dataset_PFE_CDM_complet.csv)')
    parser.add_argument('--val',   required=True,
                        help='Path to validation CSV (validation_dataset.csv)')
    args = parser.parse_args()
    main(args.train, args.val)