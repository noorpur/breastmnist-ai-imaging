from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.calibration import calibration_curve


def _save(fig, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_training_curves(history: pd.DataFrame, path: str | Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="train loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training dynamics")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_metric_table(metrics_df: pd.DataFrame, path: str | Path):
    """Render a compact metric table for either per-run or summary metrics."""
    if "roc_auc_mean" in metrics_df.columns:
        display_cols = [
            "model", "roc_auc_mean", "roc_auc_std", "balanced_accuracy_mean",
            "recall_sensitivity_mean", "specificity_mean", "f1_mean"
        ]
    else:
        display_cols = [
            "model", "seed", "accuracy", "balanced_accuracy", "roc_auc",
            "average_precision", "recall_sensitivity", "specificity", "f1"
        ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    small = metrics_df[display_cols].copy()
    for col in small.columns:
        if col not in {"model", "seed"}:
            small[col] = small[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    fig, ax = plt.subplots(figsize=(12, max(2.8, 0.55 * len(small) + 1)))
    ax.axis("off")
    table = ax.table(cellText=small.values, colLabels=small.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1, 1.35)
    ax.set_title("Experiment comparison on BreastMNIST test split", pad=15)
    return _save(fig, path)

def plot_confusion(y_true, y_prob, threshold: float, path: str | Path):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["non-malignant", "malignant"], ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix at threshold={threshold:.2f}")
    return _save(fig, path)


def plot_roc_pr(y_true, y_prob, path_roc: str | Path, path_pr: str | Path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("ROC curve for malignant classification")
    roc_path = _save(fig, path_roc)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Precision-recall curve for malignant classification")
    pr_path = _save(fig, path_pr)
    return roc_path, pr_path


def plot_calibration(y_true, y_prob, path: str | Path, n_bins: int = 8):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect calibration")
    ax.set_xlabel("Mean predicted malignant probability")
    ax.set_ylabel("Observed malignant fraction")
    ax.set_title("Calibration curve")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, path)


def plot_probability_histogram(y_true, y_prob, path: str | Path):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(y_prob[y_true == 0], bins=12, alpha=0.65, label="non-malignant")
    ax.hist(y_prob[y_true == 1], bins=12, alpha=0.65, label="malignant")
    ax.set_xlabel("Predicted malignant probability")
    ax.set_ylabel("Case count")
    ax.set_title("Risk score distribution")
    ax.legend()
    return _save(fig, path)


def plot_dataset_examples(loader, class_names: Tuple[str, str], path: str | Path, max_per_class: int = 5):
    examples = {0: [], 1: []}
    for x, y in loader:
        for image, label in zip(x, y):
            label_int = int(label.item())
            if len(examples[label_int]) < max_per_class:
                examples[label_int].append(image.squeeze().numpy() * 0.5 + 0.5)
        if all(len(v) >= max_per_class for v in examples.values()):
            break
    fig, axes = plt.subplots(2, max_per_class, figsize=(max_per_class * 1.8, 4))
    for row, label in enumerate([0, 1]):
        for col in range(max_per_class):
            ax = axes[row, col]
            ax.imshow(examples[label][col], cmap="gray")
            ax.axis("off")
            ax.set_title(class_names[label], fontsize=8)
    fig.suptitle("BreastMNIST examples after normalization reversal", y=1.02)
    return _save(fig, path)


def plot_error_gallery(images, labels, probs, path: str | Path, threshold: float = 0.5, max_items: int = 12):
    preds = (np.asarray(probs) >= threshold).astype(int)
    misses = np.where(preds != np.asarray(labels))[0][:max_items]
    if len(misses) == 0:
        return None
    cols = min(6, len(misses))
    rows = int(np.ceil(len(misses) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, misses):
        img = images[idx].squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(f"true={labels[idx]} pred={preds[idx]}\np={probs[idx]:.2f}", fontsize=8)
    fig.suptitle("Misclassified test cases", y=1.02)
    return _save(fig, path)


def plot_threshold_sweep(sweep_df: pd.DataFrame, chosen_threshold: float, path: str | Path):
    """Plot sensitivity, specificity, and balanced accuracy across thresholds."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(sweep_df["threshold"], sweep_df["recall_sensitivity"], label="sensitivity")
    ax.plot(sweep_df["threshold"], sweep_df["specificity"], label="specificity")
    ax.plot(sweep_df["threshold"], sweep_df["balanced_accuracy"], label="balanced accuracy")
    ax.axvline(chosen_threshold, linestyle="--", linewidth=1.2, label=f"selected threshold={chosen_threshold:.2f}")
    ax.set_xlabel("Decision threshold for malignant class")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.set_title("Threshold sensitivity analysis")
    ax.grid(alpha=0.25)
    ax.legend()
    return _save(fig, path)


def plot_repeated_metric_summary(summary_df: pd.DataFrame, path: str | Path):
    """Plot mean ROC-AUC with seed-to-seed standard deviation."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = summary_df.sort_values("roc_auc_mean", ascending=True)
    x = np.arange(len(order))
    ax.barh(x, order["roc_auc_mean"], xerr=order["roc_auc_std"].fillna(0.0), capsize=4)
    ax.set_yticks(x)
    ax.set_yticklabels(order["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC-AUC, mean ± SD across seeds")
    ax.set_title("Repeated-run model comparison")
    ax.grid(axis="x", alpha=0.25)
    return _save(fig, path)
