from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def _fmt_metric(row, key: str) -> str:
    value = row.get(key)
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def render_markdown_report(
    metrics_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
    dataset_summary: Dict,
    config: Dict,
    figure_paths: List[str],
    output_path: str | Path,
) -> Path:
    """Generate an experiment analysis from actual run artifacts."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_summary = metrics_summary_df.sort_values(["roc_auc_mean", "balanced_accuracy_mean"], ascending=False).iloc[0]
    best_model = best_summary["model"]
    best_run = metrics_df[metrics_df["model"] == best_model].sort_values(["roc_auc", "balanced_accuracy"], ascending=False).iloc[0]
    seeds = config.get("seeds") or [config.get("seed")]
    lines = [
        "# Experiment Analysis: AI for Women's Health Imaging",
        "",
        "## Study framing",
        "This project evaluates malignant versus non-malignant classification on BreastMNIST, a standardized breast ultrasound benchmark. The emphasis is not a single pretty score; the emphasis is a reproducible evaluation loop with repeated seeds, threshold selection on validation data, classical baselines, calibration, error analysis, and interpretability artifacts.",
        "",
        "## Dataset",
        f"- Dataset: {config.get('dataset_name', 'breastmnist')}",
        f"- Image size: {config.get('image_size', 28)}x{config.get('image_size', 28)} grayscale",
        f"- Split sizes: {dataset_summary.get('dataset_sizes', {})}",
        f"- Label distributions: {dataset_summary.get('label_distribution', {})}",
        "- Clinical note: this benchmark supports methodological exploration only. It is not a clinical diagnostic system.",
        "",
        "## Experiment design",
        f"- Seeds: {seeds}",
        f"- Epoch budget: {config.get('epochs')} with early stopping patience={config.get('patience')}",
        f"- Batch size: {config.get('batch_size')}",
        f"- Optimizer for CNNs: AdamW with learning rate {config.get('learning_rate')} and weight decay {config.get('weight_decay')}",
        f"- Decision threshold rule: selected on validation predictions, targeting sensitivity >= {config.get('sensitivity_floor')} when feasible",
        "- Primary endpoint: mean ROC-AUC across repeated test runs",
        "- Secondary endpoints: balanced accuracy, sensitivity, specificity, F1, average precision, Brier score, and expected calibration error",
        "- Baseline comparators: class-balanced logistic regression and random forest on flattened pixels",
        "",
        "## Repeated-run model summary",
        metrics_summary_df.to_markdown(index=False),
        "",
        "## Per-run metrics",
        metrics_df.to_markdown(index=False),
        "",
        "## Main finding",
        f"The strongest aggregate model was **{best_model}**, with mean ROC-AUC={_fmt_metric(best_summary, 'roc_auc_mean')} ± {_fmt_metric(best_summary, 'roc_auc_std')} and mean balanced accuracy={_fmt_metric(best_summary, 'balanced_accuracy_mean')} across seeds.",
        f"The representative best run used seed {int(best_run['seed'])}, threshold={float(best_run['threshold']):.2f}, ROC-AUC={float(best_run['roc_auc']):.3f}, sensitivity={float(best_run['recall_sensitivity']):.3f}, and specificity={float(best_run['specificity']):.3f}.",
        "",
        "## Interpretation",
        "The validation-selected threshold makes the test evaluation cleaner than tuning directly on test labels. The threshold sweep should be read as an operating-point analysis, not as a license to cherry-pick. A high ROC-AUC indicates useful ranking behavior, while sensitivity, specificity, calibration, and error-gallery behavior decide whether the model is practically attractive for any downstream workflow.",
        "",
        "Grad-CAM panels are included for qualitative auditing of CNN attention patterns. They are useful smoke signals, not proof of causal reasoning or clinical correctness.",
        "",
        "## Limitations",
        "- BreastMNIST is deliberately low-resolution, which makes it ideal for reproducible benchmarking but less representative of full-resolution ultrasound review.",
        "- Dataset size is modest, so repeated seeds and confidence intervals matter more than a single headline score.",
        "- The binary formulation merges normal and benign into non-malignant, which is appropriate for this benchmark but not a complete clinical taxonomy.",
        "- No patient-level external validation is performed in this notebook.",
        "- This repository should be presented as a benchmark research prototype, not a deployed diagnostic system.",
        "",
        "## Saved figures",
    ]
    for path in figure_paths:
        lines.append(f"- `{path}`")
    lines.extend([
        "",
        "## Reproducibility notes",
        "All tables, predictions, figures, model checkpoints, and this analysis file are generated by the notebook and written to the repository folders. Re-running the notebook refreshes the artifacts with the current execution outputs.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
