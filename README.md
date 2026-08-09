# EEEM068 Applied Machine Learning — LSA Melanoma Classification

Individual LSA coursework for EEEM068 Applied Machine Learning.

## Project objective

This project investigates melanoma classification using the
SIIM-ISIC 2020 dataset under severe class imbalance.

The study includes:

1. Historical unweighted Logistic Regression lower bound.
2. Class-weighted Logistic Regression from the original assessment.
3. A new ResNet18 deep-learning baseline.
4. A new ConvNeXt-Tiny image classifier.
5. A ConvNeXt-Tiny model incorporating clinical metadata.

The revised methodology uses patient-aware validation and includes
controlled ablation studies, model failure analysis, subgroup
analysis, and model explainability.

## Dataset

SIIM-ISIC Melanoma Classification dataset.

Raw image files are not stored in this Git repository.

## Repository structure

- `src/` — reusable implementation
- `configs/` — experiment configurations
- `notebooks/` — data audit and analysis notebooks
- `logs/` — experiment and training logs
- `outputs/` — predictions, figures, and model checkpoints
- `tests/` — basic tests for data and validation behaviour
- `report/` — IEEE coursework report

## Reproducibility

Installation and execution instructions will be expanded as the
implementation progresses.