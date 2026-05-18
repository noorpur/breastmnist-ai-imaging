from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .data import make_dataloaders, summarize_label_distribution
from .interpretability import save_gradcam_panel
from .metrics import (
    binary_classification_report,
    bootstrap_auc_ci,
    choose_threshold_from_validation,
    summarize_repeated_metrics,
    threshold_sweep,
)
from .models import build_model
from .reporting import render_markdown_report
from .train import loader_to_flat_arrays, predict_proba, train_model
from .utils import resolve_device, save_json, set_seed
from .visualization import (
    plot_calibration,
    plot_confusion,
    plot_dataset_examples,
    plot_metric_table,
    plot_probability_histogram,
    plot_repeated_metric_summary,
    plot_roc_pr,
    plot_threshold_sweep,
    plot_training_curves,
)


def _fit_classical_baselines(bundle, seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit transparent non-deep baselines on flattened 28x28 pixels."""
    x_train, y_train = loader_to_flat_arrays(bundle.train_loader)
    x_val, y_val = loader_to_flat_arrays(bundle.val_loader)
    x_test, y_test = loader_to_flat_arrays(bundle.test_loader)

    baselines = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }
    outputs: Dict[str, Dict[str, np.ndarray]] = {}
    for name, estimator in baselines.items():
        estimator.fit(x_train, y_train)
        outputs[name] = {
            "val_true": y_val,
            "val_prob": estimator.predict_proba(x_val)[:, 1],
            "test_true": y_test,
            "test_prob": estimator.predict_proba(x_test)[:, 1],
        }
    return outputs


def run_full_experiment(config: ExperimentConfig, model_names: List[str] | None = None) -> Dict:
    """Run the complete BreastMNIST experiment and write all artifacts.

    The test threshold is selected from validation predictions only, then locked
    before test evaluation. This avoids sneaky test-set threshold tuning.
    """
    paths = config.paths()
    device = resolve_device(config.device)
    model_names = model_names or ["tiny_cnn", "enhanced_cnn", "residual_cnn"]
    seeds = list(config.seeds or [config.seed])

    # Load the dataset once for the summary and example figure.
    set_seed(seeds[0])
    bundle_preview = make_dataloaders(
        root=str(paths["data"]),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=seeds[0],
        download=True,
    )
    dataset_summary = {
        "dataset_sizes": bundle_preview.dataset_sizes,
        "class_names": list(bundle_preview.class_names),
        "label_distribution": {
            "train": summarize_label_distribution(bundle_preview.train_loader, bundle_preview.class_names),
            "val": summarize_label_distribution(bundle_preview.val_loader, bundle_preview.class_names),
            "test": summarize_label_distribution(bundle_preview.test_loader, bundle_preview.class_names),
        },
    }
    save_json(dataset_summary, paths["results"] / "dataset_summary.json")

    figure_paths: List[str] = []
    figure_paths.append(str(plot_dataset_examples(bundle_preview.train_loader, bundle_preview.class_names, paths["figures"] / "dataset_examples.png")))

    metrics_rows = []
    prediction_frames = []
    validation_frames = []
    trained_models = {}
    best_single_run = {"score": -np.inf, "model_key": None, "model": None, "threshold": config.threshold}

    for seed in seeds:
        set_seed(seed)
        bundle = make_dataloaders(
            root=str(paths["data"]),
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=seed,
            download=True,
        )

        # Classical baselines create a sanity-check floor. They are run for each
        # seed so the experiment table has the same repeated-evaluation shape.
        for base_name, preds in _fit_classical_baselines(bundle, seed).items():
            threshold_info = choose_threshold_from_validation(preds["val_true"], preds["val_prob"], config.sensitivity_floor)
            threshold = threshold_info["threshold"]
            metrics = binary_classification_report(preds["test_true"], preds["test_prob"], threshold=threshold)
            metrics.update(bootstrap_auc_ci(preds["test_true"], preds["test_prob"], n_bootstraps=config.bootstrap_iterations, seed=seed))
            metrics.update({
                "model": base_name,
                "seed": seed,
                "best_epoch": np.nan,
                "stopped_epoch": np.nan,
                "best_val_loss": np.nan,
                "threshold_rule": threshold_info["threshold_rule"],
                "model_family": "classical_ml",
            })
            metrics_rows.append(metrics)
            prediction_frames.append(pd.DataFrame({
                "model": base_name,
                "seed": seed,
                "case_index": range(len(preds["test_true"])),
                "y_true": preds["test_true"],
                "p_malignant": preds["test_prob"],
                "threshold": threshold,
            }))
            validation_frames.append(pd.DataFrame({
                "model": base_name,
                "seed": seed,
                "case_index": range(len(preds["val_true"])),
                "y_true": preds["val_true"],
                "p_malignant": preds["val_prob"],
                "threshold": threshold,
            }))

        for name in model_names:
            set_seed(seed)
            model = build_model(name)
            model_key = f"{name}_seed{seed}"
            model_path = paths["models"] / f"{model_key}_best.pt"
            result = train_model(
                model,
                bundle.train_loader,
                bundle.val_loader,
                device,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                patience=config.patience,
                min_delta=config.min_delta,
                model_path=model_path,
                use_class_weights=config.use_class_weights,
            )
            result.history.assign(model=name, seed=seed).to_csv(paths["results"] / f"history_{model_key}.csv", index=False)
            trained_models[model_key] = result.model
            figure_paths.append(str(plot_training_curves(result.history, paths["figures"] / f"training_curves_{model_key}.png")))

            y_val, p_val = predict_proba(result.model, bundle.val_loader, device)
            threshold_info = choose_threshold_from_validation(y_val, p_val, config.sensitivity_floor)
            threshold = threshold_info["threshold"]
            y_true, y_prob = predict_proba(result.model, bundle.test_loader, device)

            metrics = binary_classification_report(y_true, y_prob, threshold=threshold)
            metrics.update(bootstrap_auc_ci(y_true, y_prob, n_bootstraps=config.bootstrap_iterations, seed=seed))
            metrics.update({
                "model": name,
                "seed": seed,
                "best_epoch": result.best_epoch,
                "stopped_epoch": result.stopped_epoch,
                "best_val_loss": result.best_val_loss,
                "threshold_rule": threshold_info["threshold_rule"],
                "model_family": "cnn",
            })
            metrics_rows.append(metrics)
            prediction_frames.append(pd.DataFrame({
                "model": name,
                "seed": seed,
                "case_index": range(len(y_true)),
                "y_true": y_true,
                "p_malignant": y_prob,
                "threshold": threshold,
            }))
            validation_frames.append(pd.DataFrame({
                "model": name,
                "seed": seed,
                "case_index": range(len(y_val)),
                "y_true": y_val,
                "p_malignant": p_val,
                "threshold": threshold,
            }))
            # Pick the strongest individual CNN for interpretability panels.
            score = metrics["roc_auc"] + 0.05 * metrics["balanced_accuracy"]
            if score > best_single_run["score"]:
                best_single_run = {"score": score, "model_key": model_key, "model": result.model, "threshold": threshold}

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["roc_auc", "balanced_accuracy"], ascending=False)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    validation_df = pd.concat(validation_frames, ignore_index=True)
    metrics_summary_df = summarize_repeated_metrics(metrics_df)

    metrics_df.to_csv(paths["results"] / "metrics_by_run.csv", index=False)
    metrics_summary_df.to_csv(paths["results"] / "metrics_summary_by_model.csv", index=False)
    # Backward-compatible name used by earlier notebook cells and README text.
    metrics_df.to_csv(paths["results"] / "metrics.csv", index=False)
    predictions_df.to_csv(paths["results"] / "predictions_test.csv", index=False)
    validation_df.to_csv(paths["results"] / "predictions_validation.csv", index=False)

    best_model_name = str(metrics_summary_df.iloc[0]["model"])
    best_seed_row = metrics_df[metrics_df["model"] == best_model_name].sort_values(["roc_auc", "balanced_accuracy"], ascending=False).iloc[0]
    best_seed = int(best_seed_row["seed"])
    best_preds = predictions_df.query("model == @best_model_name and seed == @best_seed")
    y_true = best_preds["y_true"].to_numpy()
    y_prob = best_preds["p_malignant"].to_numpy()
    best_threshold = float(best_preds["threshold"].iloc[0])

    sweep_df = threshold_sweep(y_true, y_prob)
    sweep_df.to_csv(paths["results"] / "threshold_sweep_best.csv", index=False)

    figure_paths.append(str(plot_metric_table(metrics_summary_df, paths["figures"] / "experiment_metric_table.png")))
    figure_paths.append(str(plot_repeated_metric_summary(metrics_summary_df, paths["figures"] / "repeated_metric_summary.png")))
    figure_paths.append(str(plot_threshold_sweep(sweep_df, best_threshold, paths["figures"] / "threshold_sweep_best.png")))
    figure_paths.append(str(plot_confusion(y_true, y_prob, best_threshold, paths["figures"] / "confusion_matrix_best.png")))
    roc_path, pr_path = plot_roc_pr(y_true, y_prob, paths["figures"] / "roc_curve_best.png", paths["figures"] / "precision_recall_curve_best.png")
    figure_paths.extend([str(roc_path), str(pr_path)])
    figure_paths.append(str(plot_calibration(y_true, y_prob, paths["figures"] / "calibration_curve_best.png")))
    figure_paths.append(str(plot_probability_histogram(y_true, y_prob, paths["figures"] / "risk_score_histogram_best.png")))

    test_images = []
    for x, _y in bundle_preview.test_loader:
        test_images.extend([(im.squeeze().numpy() * 0.5 + 0.5) for im in x])
    from .visualization import plot_error_gallery
    error_path = plot_error_gallery(test_images, y_true, y_prob, paths["figures"] / "error_gallery_best.png", threshold=best_threshold)
    if error_path is not None:
        figure_paths.append(str(error_path))

    # Grad-CAM is meaningful only for a CNN. If the aggregate winner is a
    # classical baseline, use the strongest individual CNN run instead.
    gradcam_model = best_single_run["model"]
    if gradcam_model is not None:
        figure_paths.append(str(save_gradcam_panel(gradcam_model, bundle_preview.test_loader, paths["figures"] / "gradcam_panel_best_cnn.png")))

    summary = {
        "config": config.to_dict(),
        "device": str(device),
        "best_model_by_mean_auc": best_model_name,
        "best_seed_for_visuals": best_seed,
        "best_operating_threshold": best_threshold,
        "dataset_summary": dataset_summary,
        "metrics_path": str(paths["results"] / "metrics_by_run.csv"),
        "metrics_summary_path": str(paths["results"] / "metrics_summary_by_model.csv"),
        "predictions_path": str(paths["results"] / "predictions_test.csv"),
        "validation_predictions_path": str(paths["results"] / "predictions_validation.csv"),
        "threshold_sweep_path": str(paths["results"] / "threshold_sweep_best.csv"),
        "figures": figure_paths,
    }
    save_json(summary, paths["results"] / "experiment_summary.json")
    report_path = render_markdown_report(
        metrics_df,
        metrics_summary_df,
        dataset_summary,
        config.to_dict(),
        figure_paths,
        paths["analysis"] / "experiment_report.md",
    )
    summary["analysis_report"] = str(report_path)
    save_json(summary, paths["results"] / "experiment_summary.json")
    return {"metrics": metrics_df, "metrics_summary": metrics_summary_df, "summary": summary, "dataset_summary": dataset_summary}
