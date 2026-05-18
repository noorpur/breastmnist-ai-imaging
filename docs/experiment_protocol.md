# Experiment Protocol

## Objective

Evaluate malignant versus non-malignant classification on BreastMNIST using a reproducible benchmark workflow with classical baselines, repeated CNN training, validation-selected thresholds, and full artifact export.

## Dataset

BreastMNIST from MedMNIST v2 is used. The benchmark is based on a breast ultrasound dataset and formulates the task as binary classification: non-malignant versus malignant.

## Models

The experiment suite includes:

1. Class-balanced logistic regression on flattened pixels.
2. Class-balanced random forest on flattened pixels.
3. Tiny CNN baseline.
4. Enhanced CNN with regularization and adaptive pooling.
5. Residual CNN comparator.

## Training

CNN models use AdamW, class-weighted cross entropy, cosine learning-rate scheduling, gradient clipping, and early stopping. The default epoch budget is 200 with patience of 30 epochs. Multiple random seeds are used to report seed-to-seed variability instead of relying on a single run.

## Thresholding

Decision thresholds are selected from validation predictions only. The default rule chooses the highest-specificity threshold that still reaches the configured sensitivity floor when feasible. The selected threshold is then locked before test evaluation.

## Evaluation

The notebook exports per-run metrics, model-level mean/std summaries, test predictions, validation predictions, threshold sweeps, ROC/PR curves, calibration plots, confusion matrices, risk histograms, error galleries, and Grad-CAM panels.

## Interpretation boundary

This is a benchmark research prototype. It is not a clinical diagnostic system and should not be represented as patient-facing evidence.
