"""
training/train.py
------------------
Model-agnostic training and evaluation functions that work with any
BaseAnomalyModel subclass, with self-contained MLflow logging.

Each public function (train_and_log, evaluate_and_log) opens and closes
its own MLflow run, so a single call is a complete, traceable unit of
work: every run records which model was used, its hyperparameters,
which scaling method was applied, and (if relevant) which feature
extractor sat upstream of the model.

Usage pattern (see main.py for the full grid loop):

    model = IsolationForestModel(contamination=0.01)
    train_and_log(model, X_train, scaling_method="global_standard")

    metrics = evaluate_and_log(
        model, X_val, y_true,
        scaling_method="global_standard",
        run_name="IsolationForest_global_standard",
    )
"""

import mlflow
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)

from src.models.BaseClass import BaseAnomalyModel


# ═══════════════════════════════════════════════════════════════════════════
# Logging helpers
# ═══════════════════════════════════════════════════════════════════════════

def _log_model_identity(model: BaseAnomalyModel,
                         scaling_method: str = None,
                         feature_extractor=None):
    """
    Logs everything needed to reproduce *what produced these scores*:
    the model class, its hyperparameters, the scaling method applied
    upstream, and (if any) the feature extractor (e.g. an autoencoder
    used for dimensionality reduction) applied before the model.
    """
    mlflow.set_tag("model_class", model.__class__.__name__)
    mlflow.log_params(model.get_params_for_logging())

    if scaling_method is not None:
        mlflow.set_tag("scaling_method", scaling_method)
        mlflow.log_param("scaling_method", scaling_method)

    if feature_extractor is not None:
        mlflow.set_tag("feature_extractor", getattr(feature_extractor, "name", feature_extractor.__class__.__name__))
        fe_params = {
            f"feature_extractor__{k}": v
            for k, v in feature_extractor.get_params_for_logging().items()
        }
        mlflow.log_params(fe_params)


def _best_f1_threshold_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """
    Sweeps the precision-recall curve to find the threshold that
    maximizes F1, and returns precision/recall/f1 at that point
    alongside ROC-AUC and PR-AUC (which are threshold-independent).
    """
    auc = roc_auc_score(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-10)

    best_idx = int(np.argmax(f1s))
    best_f1 = float(f1s[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])

    return {
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training (unsupervised, no labels)
# ═══════════════════════════════════════════════════════════════════════════

def train_and_log(model: BaseAnomalyModel,
                   X_train,
                   scaling_method: str = None,
                   feature_extractor=None,
                   experiment_name: str = "AnomalyScoringTraining",
                   run_name: str = None,
                   fit_kwargs: dict = None,
                   nested: bool = False):
    """
    Fits `model` on X_train and logs, in a self-contained MLflow run:
      - model class + hyperparameters
      - scaling method used to produce X_train
      - feature extractor used upstream (if any, e.g. an autoencoder)
      - training-time score distribution statistics (mean, median,
        std, skew, p95, p99, tailsep), computed by scoring the model's
        own training data right after fit().

    Returns the fitted model.
    """
    fit_kwargs = fit_kwargs or {}

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, nested=nested):
        _log_model_identity(model, scaling_method=scaling_method,
                             feature_extractor=feature_extractor)

        model.fit(X_train, **fit_kwargs)

        train_score_stats = model.evaluate(X_train)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_score_stats.items()})

    return model


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation (supervised, against a labeled validation/test set)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_and_log(model: BaseAnomalyModel,
                      X_val,
                      y_true: np.ndarray,
                      scaling_method: str = None,
                      feature_extractor=None,
                      experiment_name: str = "AnomalyScoringEvaluation",
                      run_name: str = None,
                      nested: bool = False) -> dict:
    """
    Scores `model` on X_val and logs, in a self-contained MLflow run:
      - model class + hyperparameters
      - scaling method used to produce X_val
      - feature extractor used upstream (if any)
      - labeled evaluation metrics: f1, precision, recall, roc_auc, pr_auc
      - the same unsupervised score-distribution stats as training,
        computed on X_val, for drift / sanity comparison against the
        training-time distribution

    Returns the metrics dict.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name, nested=nested):
        _log_model_identity(model, scaling_method=scaling_method,
                             feature_extractor=feature_extractor)

        scores = model.score(X_val)

        labeled_metrics = _best_f1_threshold_metrics(y_true, scores)
        mlflow.log_metrics(labeled_metrics)

        val_score_stats = model.evaluate(X_val)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_score_stats.items()})

        return {**labeled_metrics, **{f"val_{k}": v for k, v in val_score_stats.items()}}