# Patient-Aware Melanoma Classification under Severe Class Imbalance: Deep Visual Learning, Metadata Fusion and Model Failure Analysis

## 1. Overview

This EEEM068 Applied Machine Learning project studies binary melanoma
classification on the SIIM-ISIC 2020 data under severe class imbalance. It
uses fixed patient-aware validation and compares historical Logistic
Regression baselines (H0/H1), an image-only ResNet18 baseline (B0), an
image-only ConvNeXt-Tiny model (M1), and leakage-safe ConvNeXt-Tiny metadata
fusion (M2). Post-hoc analyses examine operating thresholds, model
disagreement, subgroups, failure cases, uncertainty, Grad-CAM attribution, and
exact-content integrity. The work is an internal experimental evaluation, not
a clinical-deployment study.

## 2. Final model hierarchy

| ID | Model | Role and training scope |
|---|---|---|
| H0 | Unweighted Logistic Regression | Historical lower bound using flattened 64x64 grayscale pixels and a 5,000-image training subset |
| H1 | Weighted Logistic Regression | Historical weighting comparison using the same representation and 5,000-image subset |
| B0 | ResNet18 with weighted BCE | Deep baseline using the full 26,499-image training partition |
| M1 | ConvNeXt-Tiny with weighted BCE | Image-only architecture comparison using the full training partition |
| M2 | ConvNeXt-Tiny with leakage-safe metadata fusion and weighted BCE | Metadata-fusion system ablation using the full training partition |

H0/H1 use a different representation and training scope from B0/M1/M2. Their
comparison with the deep models is therefore historical and system-level,
whereas B0 to M1 is a matched architecture-family comparison and M1 to M2 is
a metadata-fusion system ablation.

## 3. Repository structure

| Path | Contents |
|---|---|
| `configs/` | Versioned experiment configurations |
| `data/` | Metadata and fixed patient-aware fold manifest; raw images are Git-ignored |
| `logs/` | Resolved configurations, histories, metrics, and training logs |
| `notebooks/` | Data audit and final results-analysis notebooks |
| `outputs/` | Predictions, analyses, final figures/tables, manifests, and local checkpoints |
| `report/` | Assessment mapping and evidence-oriented methodology/results notes |
| `src/` | Data, modelling, training, evaluation, analysis, and publication-asset code |
| `tests/` | Unit and integration tests |

## 4. Dataset

The audited dataset contains 33,126 images from 2,056 patients: 32,542 benign
and 584 melanoma. The permanent Fold-0 split contains 26,499 training images
(467 melanoma) and 6,627 validation images (117 melanoma), with zero patient
overlap.

The integrity audit found 433 exact byte-duplicate image groups. Every group
remained within the same patient, split, and target, so no exact-content
training-validation leakage was found. This byte-level audit does not exclude
perceptual near-duplicates.

## 5. Environment

The audited experiment environment was:

- Python 3.12.10
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- CUDA 12.8
- NVIDIA RTX 3080 Laptop GPU

Relevant pinned packages are listed in `requirements.txt`. A generic
`pip install -r requirements.txt` does not by itself guarantee the audited
`+cu128` PyTorch build. Select and install the appropriate official PyTorch
CUDA build/index for the target environment first, then install the remaining
requirements as appropriate.

## 6. Data placement

Place the training metadata at `data/train.csv` and JPEGs at
`data/train_images/<image_name>.jpg`. The fixed patient-aware mapping is stored
in `data/train_folds.csv`. Raw image directories are intentionally excluded
from Git.

## 7. Running tests

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Platform-neutral equivalent when the active Python environment has all
dependencies:

```bash
python -m pytest -q
```

## 8. Final analysis notebook

`notebooks/02_results_analysis.ipynb` is the executed final analysis notebook.
It consumes the frozen J1 evidence bundle without training and consolidates
model results, comparison boundaries, and reliability analyses.

## 9. Core experiment configurations

The deep experiments are defined by `configs/resnet18.yaml`,
`configs/convnext_image.yaml`, and `configs/convnext_metadata.yaml`. Shared
settings are: 224x224 images, batch size 32, AdamW, learning rate `1e-4`,
weight decay `1e-4`, weighted BCE, training-derived
`pos_weight = 55.74304068522484`, maximum 10 epochs, patience 3, validation
ROC-AUC checkpoint selection, and primary threshold 0.5.

Observed training duration and selected checkpoints were:

| Model | Epochs executed | Best checkpoint epoch |
|---|---:|---:|
| B0 | 7 | 4 |
| M1 | 7 | 4 |
| M2 | 10 | 8 |

M2 completed all 10 configured epochs; epoch 8 was the selected checkpoint,
not an early-stopping epoch.

## 10. Hyperparameter-selection note

- No systematic grid search, systematic random search, or Bayesian search was performed.
- Deep settings were deliberately matched to reduce optimisation-policy confounding.
- `pos_weight` was derived from training-partition class counts.
- Early stopping and validation checkpointing controlled effective training duration.
- Phase I thresholds from 0.1 to 0.9 were post-hoc behavioural analysis, not tuning.
- Future systematic tuning would require patient-grouped inner validation or nested cross-validation.

## 11. Evaluation metrics

Evaluation reports ROC-AUC, Average Precision (AP), accuracy, balanced
accuracy, precision, sensitivity, specificity, F1, and confusion matrices.
Threshold-dependent primary results use the common threshold 0.5.

## 12. Key final results

Rounded display values below come only from
`outputs/final/tables/main_model_results.csv`.

| Model | ROC-AUC | AP | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| H0 | 0.657 | 0.039 | 0.009 | 0.998 | 0.016 |
| H1 | 0.624 | 0.036 | 0.145 | 0.951 | 0.076 |
| B0 | 0.864 | 0.119 | 0.795 | 0.768 | 0.108 |
| M1 | 0.901 | 0.169 | 0.974 | 0.628 | 0.086 |
| M2 | 0.897 | 0.165 | 0.897 | 0.736 | 0.108 |

## 13. Reliability analyses

The final evidence includes threshold behaviour, M1/M2 disagreement,
exploratory subgroup analysis, paired patient-level bootstrap resampling,
failure review, Grad-CAM, and exact-content integrity analysis. Bootstrap
results quantify internal patient-level uncertainty, not external validation.
Subgroup results do not establish fairness conclusions. M2 Grad-CAM is
image-branch melanoma-logit attribution conditioned on metadata; it does not
explain the metadata contribution or establish causality.

## 14. Checkpoint and large-file policy

Large neural checkpoints may remain Git-ignored. Tracked reproducibility
evidence includes experiment configurations, resolved configs, training
histories, metrics, predictions, and checkpoint hashes/manifests. No remote
checkpoint download location is asserted.

## 15. Reproducibility limitations

- Results use one fixed patient-aware validation fold.
- Fold 0 was used for checkpoint selection and final reported metrics.
- No independent test cohort was evaluated.
- No external validation was performed.
- No final cross-validation estimate was produced.
- No systematic hyperparameter search was performed.
- Deterministic seeds and configuration improve reproducibility but do not guarantee bitwise determinism.
