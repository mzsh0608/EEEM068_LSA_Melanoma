# Patient-Aware Melanoma Classification under Severe Class Imbalance: Deep Visual Learning, Metadata Fusion and Reliability Analysis

## Project Overview

This EEEM068 Applied Machine Learning project investigates binary melanoma
classification on the SIIM-ISIC 2020 dataset under severe class imbalance. It
uses a fixed patient-aware validation split, a controlled ResNet18 versus
ConvNeXt-Tiny comparison, a ConvNeXt metadata-fusion ablation, and post-hoc
reliability analysis. The model hierarchy is an experimental progression with
explicit comparison boundaries, not a five-model architecture leaderboard.

## Experimental Hierarchy

| ID | Model | Role |
|---|---|---|
| H0 | Unweighted Logistic Regression | Historical lower bound |
| H1 | Class-weighted Logistic Regression | Logistic Regression weighting comparison |
| B0 | ResNet18 + weighted BCE | Deep residual CNN baseline |
| M1 | ConvNeXt-Tiny + weighted BCE | Matched architecture-family comparison |
| M2 | ConvNeXt-Tiny + metadata + weighted BCE | Metadata-fusion system ablation |

H0 and H1 use a stratified 5,000-image sample from the training partition,
64x64 grayscale inputs, and flattened 4,096-dimensional features. B0, M1, and
M2 use the full 26,499-image training partition and 224x224 RGB network inputs.
Consequently, H0/H1 to B0 is only a historical system-level progression, not a
controlled architecture comparison.

- H0 -> H1 isolates Logistic Regression class weighting.
- B0 -> M1 is the cleanest architecture-family comparison.
- M1 -> M2 is the metadata-fusion system ablation.

## Dataset

The project uses SIIM-ISIC 2020 data with this audited composition:

- 33,126 source images, represented locally as 512x512 JPEG files
- 32,542 benign images
- 584 melanoma images
- 2,056 patients

The local source dataset remains at 512x512. During preprocessing, H0/H1
resize images to 64x64 grayscale, while B0/M1/M2 resize images to 224x224 RGB.

The raw dataset is not included in this repository. Obtain it from the
official SIIM-ISIC 2020 source and place the training metadata and images at:

```text
data/train.csv
data/train_images/<image_name>.jpg
```

The repository does not implement or claim an automated dataset-download
command. Raw metadata, images, DICOM files, and dataset archives are ignored by
Git. The tracked `data/train_folds.csv` and `data/lr_train_subset.csv` are
derived reproducibility manifests, not copies of the raw dataset.

## Patient-Aware Validation

`StratifiedGroupKFold` uses `patient_id` as the grouping variable and `target`
for stratification. The saved five-fold mapping uses Fold 0 as the fixed
validation fold and folds 1-4 as training.

| Partition | Total | Benign | Melanoma |
|---|---:|---:|---:|
| Training | 26,499 | 26,032 | 467 |
| Validation (Fold 0) | 6,627 | 6,510 | 117 |

Known-patient overlap between the two partitions is zero. Fold 0 is a
validation fold, not an independent test set or external cohort.

## Data Integrity Audit

The exact-content audit computed SHA-256 over the bytes of every source JPEG:

- 33,126 records hashed
- 32,693 unique hashes
- 433 duplicate hash groups
- 866 records involved in exact duplicate pairs

All duplicate groups belong to the same patient, split, and target. There are
zero exact train-validation duplicate groups and zero conflicting-target
duplicate groups. This audit detects byte-identical files only and does not
rule out perceptual near-duplicates.

## Preprocessing and Training Configuration

B0, M1, and M2 share the following controlled protocol:

- source: 512x512 JPEG
- network preprocessing: 224x224 RGB
- train-only augmentation: horizontal flip `p=0.5`, vertical flip `p=0.5`,
  rotation up to 15 degrees, and brightness/contrast jitter of 0.10
- validation: deterministic resize and ImageNet normalisation
- batch size 32
- ImageNet `IMAGENET1K_V1` pretrained weights and full fine-tuning
- AdamW, learning rate `1e-4`, weight decay `1e-4`, and no LR scheduler
- weighted BCE with `pos_weight = 55.74304068522484`
- maximum 10 epochs, patience 3, and validation ROC-AUC checkpoint selection
- automatic mixed precision (AMP) and seed 42

These are shared controlled experimental settings. They were not obtained
through systematic grid, random, or Bayesian hyperparameter optimisation. The
positive-class weight is derived only from training counts:
`26,032 / 467 = 55.74304068522484`.

Threshold 0.5 is the common primary evaluation threshold. The threshold grid
from 0.1 to 0.9 is post-hoc behavioural analysis, not threshold tuning.

## Model Rationale

ResNet18 is the conventional residual CNN baseline. ConvNeXt-Tiny is a
modernised convolutional architecture. Keeping the data, transforms, loss,
optimiser, training policy, and evaluation fixed makes B0 -> M1 more
interpretable than changing architecture and imbalance handling together.

M1 and M2 share the ConvNeXt backbone design; M2 adds only a small metadata
branch.

| Model | Parameters |
|---|---:|
| B0 | 11,177,025 |
| M1 | 27,820,897 |
| M2 | 27,821,313 |

M2 therefore adds 416 parameters relative to M1.

## Metadata Fusion

M2 uses `age_approx`, `sex`, and `anatom_site_general_challenge` while
excluding leakage-sensitive identifiers and outcome fields. The metadata
preprocessor is fitted only on the training partition: age receives median
imputation and standardisation, categorical fields are encoded, and unknown
categories are handled without changing the learned feature dimension. The
fitted transformation is then applied to both training and validation data.

The processed metadata vector has 11 dimensions. Its branch is
`11 -> 32 -> GELU -> Dropout(0.20)`. The resulting 32-dimensional embedding is
concatenated with the 768-dimensional ConvNeXt image feature to form an
800-dimensional fused vector, followed by an `800 -> 1` raw-logit head.

M2 starts independently from ImageNet-pretrained ConvNeXt weights. It does not
start from the trained M1 checkpoint.

## Evaluation

Primary and summary metrics are ROC-AUC, Average Precision (AP), sensitivity,
specificity, precision, F1, and confusion matrices. Reliability analyses cover
the 0.1-0.9 threshold sweep, M1/M2 disagreement, paired patient-level
bootstrap, subgroup behaviour, failure cases, and Grad-CAM.

M2 Grad-CAM targets the raw melanoma logit in the image branch while supplying
metadata to the forward pass. It is therefore image-branch attribution
conditioned on metadata. It is not a causal explanation and does not explain
the metadata branch.

## Main Results

The frozen primary-threshold results are:

| Model | ROC-AUC | AP | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|---:|
| H0 | 0.657022 | 0.038592 | 0.008547 | 0.998464 | 0.015625 |
| H1 | 0.624392 | 0.036316 | 0.145299 | 0.951459 | 0.075556 |
| B0 | 0.864276 | 0.119273 | 0.794872 | 0.767588 | 0.107951 |
| M1 | 0.900848 | 0.169406 | 0.974359 | 0.627650 | 0.085876 |
| M2 | 0.897446 | 0.165292 | 0.897436 | 0.735945 | 0.108192 |

Threshold-dependent metrics use the primary threshold 0.5. Weighting improves
Logistic Regression minority detection but not its ranking metrics. Under the
matched deep protocol, M1 improves ranking and sensitivity over B0. M2 does
not improve the observed ROC-AUC or AP over M1, but at threshold 0.5 it has
higher specificity and lower sensitivity. No model is universally best across
all metrics and operating requirements.

## Reproducibility

### 1. Environment

From the repository root on Windows with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The audited environment used PyTorch `2.11.0+cu128` and torchvision
`0.26.0+cu128`. `requirements.txt` pins direct dependencies but does not encode
the CUDA wheel index; select the appropriate official PyTorch build for the
target hardware rather than treating `+cu128` as portable.

### 2. Dataset placement and split preparation

Place the external files at `data/train.csv` and `data/train_images/*.jpg`,
then start Jupyter:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

Run `notebooks/00_data_audit.ipynb` to reproduce dataset checks, figures, and
the patient-aware fold manifest. This workflow requires the external dataset.

### 3. H0/H1 baseline workflow

Run `notebooks/01_logistic_regression.ipynb` after the data-audit notebook.
It uses `configs/logistic_unweighted.yaml` and
`configs/logistic_weighted.yaml`, the tracked fold/subset manifests, and the
external JPEGs. There is no separate Logistic Regression CLI.

### 4. Deep training and evaluation

Each `--fit` command trains the configured model, selects the checkpoint by
validation ROC-AUC, evaluates Fold 0, and saves its logs, metrics, predictions,
and figures. These commands require the external dataset and suitable compute.

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs/resnet18.yaml --fit
.\.venv\Scripts\python.exe -m src.train --config configs/convnext_image.yaml --fit
.\.venv\Scripts\python.exe -m src.train --config configs/convnext_metadata.yaml --fit
```

### 5. Reliability analysis and final notebook

After the deep predictions and checkpoints exist, the implemented order is:

```powershell
.\.venv\Scripts\python.exe -m src.analysis --project-root . --bootstrap-iterations 1000 --bootstrap-seed 42
.\.venv\Scripts\python.exe -m src.explainability --project-root . --cases-per-category 3
.\.venv\Scripts\python.exe -m src.technical_audit --project-root .
.\.venv\Scripts\python.exe -m src.final_tables --project-root .
.\.venv\Scripts\python.exe -m src.results_notebook --output notebooks/02_results_analysis.ipynb
.\.venv\Scripts\python.exe -m src.publication_assets --root .
```

These commands regenerate tracked outputs and should be used only for a full
reproduction, not routine inspection of the frozen submission. Analysis and
Grad-CAM require the external JPEGs; Grad-CAM and notebook hash verification
also require the matching ignored checkpoints.

A clean clone can inspect the tracked configs, logs, predictions, tables,
figures, executed notebooks, and report without the dataset. The current
`notebooks/02_results_analysis.ipynb` is already executed and stores the final
analysis outputs. Running `src.results_notebook` recreates its unexecuted
structure and should not overwrite the submitted notebook during inspection.

Tests use synthetic fixtures and do not require the SIIM-ISIC files:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

## Repository Structure

```text
configs/      Experiment configurations for H0, H1, B0, M1, and M2
data/         Tracked split/subset manifests and external-data instructions
logs/         Frozen experiment configurations, histories, metrics, and logs
notebooks/    Executed data-audit, Logistic Regression, and results notebooks
outputs/      Predictions, analyses, figures, tables, and provenance manifests
report/       Final report PDF and supporting evidence notes
src/          Splitting, preprocessing, modelling, training, analysis, and QC code
tests/        Unit and integration tests using synthetic fixtures
```

## Final Report

The submitted report is
[`report/EEEM068_LSA_Melanoma_Final_Report_Revised.pdf`](report/EEEM068_LSA_Melanoma_Final_Report_Revised.pdf).
No complete LaTeX report source was supplied; the `.tex` files under
`outputs/final/tables/publication/` are generated table fragments only.

## Limitations

- Fold 0 is a fixed patient-aware validation fold, not an independent test set.
- Fold 0 is also used for checkpoint selection and final reported metrics.
- No external validation or full cross-validation performance estimate exists.
- No systematic hyperparameter optimisation was performed.
- Some subgroup analyses contain very few melanoma cases.
- SHA-256 detects exact byte duplicates only, not perceptual near-duplicates.
- Grad-CAM is non-causal, image-branch attribution conditioned on metadata.
- Seed control improves reproducibility but does not guarantee bitwise-identical retraining.
