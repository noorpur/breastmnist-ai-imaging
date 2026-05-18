from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader

from breast_ai.data import ArrayDataset
from breast_ai.interpretability import gradcam_heatmap
from breast_ai.metrics import binary_classification_report, bootstrap_auc_ci, threshold_sweep, choose_threshold_from_validation
from breast_ai.models import build_model
from breast_ai.train import predict_proba, train_model
from breast_ai.visualization import plot_confusion, plot_roc_pr, plot_training_curves


def test_training_metrics_figures_and_gradcam(tmp_path):
    rng = np.random.default_rng(7)
    x_train = rng.integers(0, 255, size=(64, 28, 28), dtype=np.uint8)
    y_train = rng.integers(0, 2, size=(64, 1), dtype=np.int64)
    x_val = rng.integers(0, 255, size=(24, 28, 28), dtype=np.uint8)
    y_val = rng.integers(0, 2, size=(24, 1), dtype=np.int64)
    x_test = rng.integers(0, 255, size=(20, 28, 28), dtype=np.uint8)
    y_test = np.array([[0], [1]] * 10, dtype=np.int64)

    train_loader = DataLoader(ArrayDataset(x_train, y_train, augment=True), batch_size=16)
    val_loader = DataLoader(ArrayDataset(x_val, y_val), batch_size=16)
    test_loader = DataLoader(ArrayDataset(x_test, y_test), batch_size=10)

    model = build_model("enhanced_cnn")
    result = train_model(model, train_loader, val_loader, torch.device("cpu"), epochs=1, patience=1, use_class_weights=True)
    y_true, y_prob = predict_proba(result.model, test_loader, torch.device("cpu"))
    metrics = binary_classification_report(y_true, y_prob)
    ci = bootstrap_auc_ci(y_true, y_prob, n_bootstraps=10)

    assert "roc_auc" in metrics
    assert "auc_ci_low" in ci
    assert len(result.history) == 1
    sweep = threshold_sweep(y_true, y_prob)
    chosen = choose_threshold_from_validation(y_true, y_prob, sensitivity_floor=0.8)
    assert len(sweep) > 50
    assert 0.0 <= chosen["threshold"] <= 1.0

    plot_training_curves(result.history, tmp_path / "training.png")
    plot_confusion(y_true, y_prob, 0.5, tmp_path / "cm.png")
    plot_roc_pr(y_true, y_prob, tmp_path / "roc.png", tmp_path / "pr.png")
    assert (tmp_path / "training.png").exists()
    assert (tmp_path / "cm.png").exists()
    assert (tmp_path / "roc.png").exists()
    assert (tmp_path / "pr.png").exists()

    sample, _ = next(iter(test_loader))
    heatmap = gradcam_heatmap(result.model, sample[0], target_class=1)
    assert heatmap.shape == (28, 28)
    assert np.isfinite(heatmap).all()
