# Data Statement

## Dataset

BreastMNIST from MedMNIST v2.

## Source modality

Breast ultrasound imaging.

## Original source

Al-Dhabyani et al., *Dataset of breast ultrasound images*, Data in Brief, 2020.

## Benchmark transformation

MedMNIST v2 standardizes the source dataset by resizing images to 28x28 grayscale and defining a binary classification task. Normal and benign cases are combined into non-malignant; malignant cases form the positive class.

## Usage boundary

The dataset is appropriate for lightweight benchmarking and reproducibility studies. It is not a substitute for full-resolution clinical imaging datasets or external validation cohorts.
