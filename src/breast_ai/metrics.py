from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: Iterable[int], y_prob: Iterable[float], n_bins: int = 10) -> float:
    """Compute expected calibration error for binary probabilities."""
    y_true = np.asarray(list(y_true)).astype(int)
    y_prob = np.asarray(list(y_prob)).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1 else y_prob <= hi)
        if mask.any():
            confidence = y_prob[mask].mean()
            accuracy = y_true[mask].mean()
            ece += mask.mean() * abs(accuracy - confidence)
    return float(ece)


def binary_classification_report(y_true, y_prob, threshold: float = 0.5) -> Dict[str, float]:
    """Return a metric dictionary for binary malignant classification."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece_10_bins": expected_calibration_error(y_true, y_prob, n_bins=10),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_auc_ci(y_true, y_prob, n_bootstraps: int = 1000, seed: int = 42, alpha: float = 0.95) -> Dict[str, float]:
    """Bootstrap ROC-AUC confidence interval."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not scores:
        return {"auc_ci_low": float("nan"), "auc_ci_high": float("nan")}
    lo = (1 - alpha) / 2
    hi = 1 - lo
    return {"auc_ci_low": float(np.quantile(scores, lo)), "auc_ci_high": float(np.quantile(scores, hi))}


def threshold_sweep(y_true, y_prob, step: float = 0.01) -> pd.DataFrame:
    """Evaluate metrics across thresholds from 0 to 1."""
    rows = []
    for threshold in np.round(np.arange(0.0, 1.0 + step, step), 4):
        row = binary_classification_report(y_true, y_prob, float(threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold_from_validation(y_true, y_prob, sensitivity_floor: float = 0.90) -> Dict[str, float]:
    """Select an operating threshold on validation data without touching test labels.

    Primary rule: among thresholds reaching the desired sensitivity floor, choose
    the one with the highest specificity, then balanced accuracy. Fallback rule:
    choose the threshold with the highest Youden index.
    """
    sweep = threshold_sweep(y_true, y_prob)
    feasible = sweep[sweep["recall_sensitivity"] >= sensitivity_floor].copy()
    if len(feasible):
        chosen = feasible.sort_values(["specificity", "balanced_accuracy", "threshold"], ascending=[False, False, False]).iloc[0]
        rule = f"max_specificity_given_sensitivity_at_least_{sensitivity_floor:.2f}"
    else:
        tmp = sweep.assign(youden=sweep["recall_sensitivity"] + sweep["specificity"] - 1.0)
        chosen = tmp.sort_values(["youden", "balanced_accuracy"], ascending=False).iloc[0]
        rule = "max_youden_index_fallback"
    return {"threshold": float(chosen["threshold"]), "threshold_rule": rule}


def summarize_repeated_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Create mean/std summaries across seeds for each model."""
    numeric_cols = [
        "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "precision",
        "recall_sensitivity", "specificity", "npv", "f1", "brier_score", "ece_10_bins",
        "threshold", "best_epoch", "stopped_epoch"
    ]
    available = [c for c in numeric_cols if c in metrics_df.columns]
    grouped = metrics_df.groupby("model", dropna=False)[available].agg(["mean", "std", "min", "max"])
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    grouped = grouped.reset_index()
    return grouped.sort_values(["roc_auc_mean", "balanced_accuracy_mean"], ascending=False)
