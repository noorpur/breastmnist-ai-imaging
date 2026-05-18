# Model Card

## Model family

This repository evaluates lightweight machine learning and convolutional neural network models for BreastMNIST malignant versus non-malignant classification.

## Intended use

The intended use is reproducible benchmarking, research demonstration, and portfolio-level analysis of medical imaging classification workflows.

## Not intended for

The models are not intended for clinical deployment, diagnosis, triage, patient counseling, or replacement of radiologist interpretation.

## Dataset

BreastMNIST from MedMNIST v2 is used. The dataset is small and intentionally low-resolution at 28x28 grayscale pixels.

## Evaluation approach

The notebook reports repeated runs across seeds, classical baselines, validation-selected thresholds, test-set metrics, bootstrap ROC-AUC confidence intervals, calibration, error analysis, and Grad-CAM visualizations.

## Key limitations

- Low image resolution.
- Small dataset size.
- Binary label simplification merges normal and benign cases into non-malignant.
- No external patient-level validation.
- Grad-CAM provides qualitative explanation only, not proof of medical reasoning.
