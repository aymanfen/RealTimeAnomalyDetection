"""
main.py
------------------
Iterates over standardization methods x models (IsolationForest, SOM,
AutoEncoder, and AutoEncoder-encoded features feeding IsolationForest
or SOM), training and evaluating each combination via the model-agnostic,
MLflow-integrated functions in src/training/train.py.

Every (scaling x model) combination is logged as its own MLflow run
during training and its own run during evaluation, tagged with the
model's class + hyperparameters, the scaling method, and (when
relevant) the autoencoder feature extractor used upstream.

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
import tensorflow as tf

from src.features.TimeFeatures import ComputeTimeFeatures
from src.features.CatEntropy import ComputeCatEntropy
from src.features.CatFreq import ComputeCatFreq

from src.preprocessing.PreProcessing import ClientScale, ClientNorm
from src.preprocessing.AutoEncoderFeatureExtractor import AutoEncoderFeatureExtractor

from src.models.IsolationForest import IsolationForestModel
from src.models.SelfOrganizingMap import SOMModel
from src.models.AutoEncoder import AutoEncoderModel

from src.training.train import train_and_log, evaluate_and_log

from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    columns = [
        "TranasctionID", "Time", "Account Number", "Card Number",
        "Transaction Type", "Channel", "Transaction Amount",
        "Merchand Group", "Country", "Country2", "City2", "Card Type",
        "Age", "Gender", "Bank", "City", "Merchand Code", "isAnomaly"
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
                   val_feat: pd.DataFrame):
    """
    Returns (X_train, X_val) as DataFrames over FEATURE_COLS, scaled
    according to `method`.
    """
    if method == 'none':
        return train_feat, val_feat

    elif method == 'global_minmax':
        sc = MinMaxScaler()
        X_tr = pd.DataFrame(sc.fit_transform(train_feat[FEATURE_COLS]), columns=FEATURE_COLS)
        X_va = pd.DataFrame(sc.transform(val_feat[FEATURE_COLS]), columns=FEATURE_COLS)
        return X_tr, X_va

    elif method == 'global_standard':
        sc = StandardScaler()
        X_tr = pd.DataFrame(sc.fit_transform(train_feat[FEATURE_COLS]), columns=FEATURE_COLS)
        X_va = pd.DataFrame(sc.transform(val_feat[FEATURE_COLS]), columns=FEATURE_COLS)
        return X_tr, X_va

    elif method == 'global_minmax_client_minmax':
        train_g = ClientNorm(train_feat, 'Account Number', CLIENT_COLS, GLOBAL_COLS)
        val_g = ClientNorm(val_feat, 'Account Number', CLIENT_COLS, GLOBAL_COLS)
        return train_g, val_g

    elif method == 'global_standard_client_standard':
        train_g = ClientScale(train_feat, 'Account Number', CLIENT_COLS, GLOBAL_COLS)
        val_g = ClientScale(val_feat, 'Account Number', CLIENT_COLS, GLOBAL_COLS)
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
    'none': 'No Scaling',
    'global_minmax': 'Global MinMax',
    'global_standard': 'Global Standard',
    'global_minmax_client_minmax': 'Global MinMax + Client MinMax',
    'global_standard_client_standard': 'Global Standard + Client Standard',
}


def clean(arr) -> np.ndarray:
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy()
    arr = arr.astype(float)
    arr = np.where(np.isinf(arr), np.nan, arr)
    col_means = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    arr[inds] = np.take(col_means, inds[1])
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# 4.  GRID DEFINITION
# ═══════════════════════════════════════════════════════════════════════════
#
# "Standalone" models score the (scaled) features directly.
# "Meta" combos run an autoencoder as a preprocessing/feature-extraction
# step first (see AutoEncoderFeatureExtractor), then score the resulting
# latent features with IsolationForest or SOM.

STANDALONE_MODEL_FACTORIES = {
    'IsolationForest': lambda input_dim: IsolationForestModel(),
    'SOM': lambda input_dim: SOMModel(inputlen=input_dim),
    'AutoEncoder': lambda input_dim: AutoEncoderModel(inputdim=input_dim),
}

META_DETECTOR_FACTORIES = {
    'IsolationForest': lambda latent_dim: IsolationForestModel(),
    'SOM': lambda latent_dim: SOMModel(inputlen=latent_dim),
}

AE_LATENT_DIM = 6


# ═══════════════════════════════════════════════════════════════════════════
# 5.  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_standalone(model_name: str, scaling: str,
                    X_train: np.ndarray, X_val: np.ndarray, y_true: np.ndarray):

    model = STANDALONE_MODEL_FACTORIES[model_name](X_train.shape[1])
    run_tag = f"{model_name}_{scaling}"

    train_and_log(
        model, X_train,
        scaling_method=SCALING_LABELS[scaling],
        run_name=f"train_{run_tag}",
    )

    metrics = evaluate_and_log(
        model, X_val, y_true,
        scaling_method=SCALING_LABELS[scaling],
        run_name=f"eval_{run_tag}",
    )
    return metrics


def run_meta(meta_name: str, scaling: str,
             X_train: np.ndarray, X_val: np.ndarray, y_true: np.ndarray):

    # ── Encoder as a preprocessing/feature-extraction step ──────────────
    extractor = AutoEncoderFeatureExtractor(
        inputdim=X_train.shape[1], latentdim=AE_LATENT_DIM,
    )
    Z_train = extractor.fit_transform(X_train)
    Z_val = extractor.transform(X_val)

    detector = META_DETECTOR_FACTORIES[meta_name](Z_train.shape[1])
    run_tag = f"AE_{meta_name}_{scaling}"

    train_and_log(
        detector, Z_train,
        scaling_method=SCALING_LABELS[scaling],
        feature_extractor=extractor,
        run_name=f"train_{run_tag}",
    )

    metrics = evaluate_and_log(
        detector, Z_val, y_true,
        scaling_method=SCALING_LABELS[scaling],
        feature_extractor=extractor,
        run_name=f"eval_{run_tag}",
    )
    return metrics


def main(train_path: str, val_path: str):
    print("Loading training data …")
    train_raw = load_train(train_path)
    train_eng = feature_engineer(train_raw).reset_index(drop=True)

    print("Loading validation data …")
    val_raw, y_true = load_val(val_path)
    val_eng = feature_engineer(val_raw).reset_index(drop=True)

    rows = []

    for scaling in SCALING_METHODS:
        print(f"\n{'═'*60}")
        print(f"  Scaling: {SCALING_LABELS[scaling]}")
        print(f"{'═'*60}")

        X_train_df, X_val_df = apply_scaling(scaling, train_eng, val_eng)
        X_train = clean(X_train_df[FEATURE_COLS])
        X_val = clean(X_val_df[FEATURE_COLS])

        print("\n  [Standalone models]")
        for model_name in STANDALONE_MODEL_FACTORIES:
            try:
                metrics = run_standalone(model_name, scaling, X_train, X_val, y_true)
                print(f"    {model_name:20s}  "
                      f"AUC={metrics['roc_auc']:.4f}  "
                      f"PR-AUC={metrics['pr_auc']:.4f}  "
                      f"F1={metrics['f1']:.4f}")
                rows.append({'Scaling': SCALING_LABELS[scaling],
                             'Pipeline': f"Standalone/{model_name}", **metrics})
            except Exception as e:
                print(f"    {model_name:20s}  ERROR: {e}")

        print("\n  [Encoder + meta-model]")
        for meta_name in META_DETECTOR_FACTORIES:
            try:
                metrics = run_meta(meta_name, scaling, X_train, X_val, y_true)
                print(f"    AE+{meta_name:18s}  "
                      f"AUC={metrics['roc_auc']:.4f}  "
                      f"PR-AUC={metrics['pr_auc']:.4f}  "
                      f"F1={metrics['f1']:.4f}")
                rows.append({'Scaling': SCALING_LABELS[scaling],
                             'Pipeline': f"AE+{meta_name}", **metrics})
            except Exception as e:
                print(f"    AE+{meta_name:18s}  ERROR: {e}")

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
        print("Full per-run params/metrics are in MLflow "
              "(experiments: anomaly-detection-training, anomaly-detection-evaluation)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True,
                         help='Path to training CSV (dataset_PFE_CDM_complet.csv)')
    parser.add_argument('--val', required=True,
                         help='Path to validation CSV (validation_dataset.csv)')
    args = parser.parse_args()
    main(args.train, args.val)