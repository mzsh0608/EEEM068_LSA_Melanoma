# Pre-Phase-J Technical Audit

This document is technical source material for Phase J. It does not begin the
final report or reopen model training. The frozen H0, H1, B0, M1, and M2
hierarchy, Fold-0 manifest, prediction probabilities, historical metrics, and
checkpoint weights were not changed.

## PASS

### Bootstrap multiplicity

`paired_patient_bootstrap` groups row indices by patient, samples group
positions with replacement, and concatenates every selected index array in
draw order. If a patient is selected twice, all of that patient's rows occur
twice. M1 and M2 probabilities are indexed by the same expanded row array.

A regression test forces patient draws `[A, A, B]` where A has two images and
B has one. The resulting sample contains five rows and verifies paired M1/M2
metrics. The implementation was already correct, so the Phase I bootstrap
samples, intervals, and figure were not regenerated.

### Validation terminology

Fold 0 is consistently treated as the fixed patient-aware validation fold.
No positive scientific claim describes it as an independent test set. Negative
statements explaining that no independent test set exists are intentional.

### B0 versus M1 comparison

B0 and M1 share the external data, image, transform, loss, optimizer,
checkpoint, and threshold protocol. This is the cleanest architecture-family
comparison in the repository, but multiple internal architecture properties
change together. No depthwise-convolution, LayerNorm, GELU, stochastic-depth,
or individual-block effect is isolated causally.

### Hyperparameter claims

No grid, random, or Bayesian search was performed, and no repository text
claims systematic optimization or optimal hyperparameters. The defensible
description is a predeclared matched configuration with early stopping used to
select effective training duration.

### Checkpoints

B0, M1, and M2 checkpoints exist locally and remain ignored by
`outputs/checkpoints/*.pt`. For each experiment, config, environment, history,
metrics, training log, and predictions are tracked. Exact bytes and hashes are
recorded in `outputs/analysis/technical_audit/checkpoint_manifest.json`.

## FIXED

### Qualitative content duplication

The Phase I FN sample contained byte-identical `ISIC_0149568` and
`ISIC_9022005`. Quantitative evaluation still contains both validation rows.
For qualitative review only, `ISIC_9022005` was replaced by the next-ranked
content-unique FN, `ISIC_4967586`. The failure manifest, visual review, montage,
Phase I summary, analysis notes, and viva notes were updated. There are six
unique JPEG hashes per confusion category. All 12 Grad-CAM cases were already
content-unique, so no CAM was regenerated.

### Average Precision terminology

The implementation calls `sklearn.metrics.average_precision_score`. The final
term is Average Precision (AP), not a separately computed trapezoidal area.
The legacy keys `pr_auc_average_precision` and `val_pr_auc` remain unchanged
for compatibility with frozen metrics and histories. Their AP semantics are
now explicit in evaluator documentation, methodology, notebook interpretation,
and the log index.

### Comparison boundaries

H0/H1 versus B0 is now explicitly a historical system-level progression.
H0/H1 used 64x64 flattened grayscale pixels and a 5,000-image subset; B0 used
224x224 RGB images, all 26,499 training rows, ImageNet initialization,
augmentation, and a CNN. The performance difference cannot be assigned to
architecture alone.

M1 versus M2 remains the principal metadata-fusion system ablation, but both
models were independently initialized and trained from ImageNet. Per-sample
probability differences describe changed predictions, not causal age, sex, or
site contributions.

### Grad-CAM interpretation

Grad-CAM still targets the raw melanoma logit at `features.7.2`. Exact
metadata transformed by the saved Phase H preprocessor is supplied during the
M2 forward pass, and the preprocessor is not refitted. It explains coarse
image-branch attribution conditioned on metadata, not metadata contribution or
clinical reasoning.

A TN CAM also targets the positive melanoma logit. It shows spatial regions
contributing toward that logit despite a low overall score; it does not
directly explain why the prediction was benign.

### Timing labels

| Model | Epochs | Parameters | Sum epoch s | Mean epoch s | Median epoch s | Reported fit s |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 7 | 11,177,025 | 582.85 | 83.26 | 68.53 | 593.76 |
| M1 | 7 | 27,820,897 | 751.71 | 107.39 | 98.95 | 767.50 |
| M2 | 10 | 27,821,313 | 1,064.32 | 106.43 | 100.50 | 1,080.50 |

Per-epoch seconds time one training and validation pass. Reported fit duration
starts inside `fit_model` before fit setup and ends after best-checkpoint reload
and final validation; it excludes final prediction/metric/environment writes.
The small difference between each sum and reported total is therefore expected.
M1's 107.39-second mean and 767.50-second total use different scopes.

M2 ran three more epochs than M1 while mean and median epoch times were
similar. The 313-second total-duration difference is not direct metadata-branch
overhead.

### Requirements

The former requirements file was unpinned and omitted `joblib` despite a
direct source import. It now pins verified direct runtime dependencies plus
notebook/test tooling. `timm`, OpenCV, and Albumentations were removed because
tracked code and notebooks do not import them. `pip check` reports no broken
requirements. No environment-wide freeze or redundant lock file was added.

The audited environment used `torch 2.11.0+cu128` and
`torchvision 0.26.0+cu128`. CUDA wheel selection still requires the official
PyTorch selector/index; no unverified index URL was placed in requirements.

### Pytest on Windows

A fixed reusable `.pytest_tmp` still failed when pytest attempted to delete a
previous Windows long-path directory. `pytest.ini` now disables only the cache
provider, and root `conftest.py` assigns a fresh unique directory beneath the
ignored repository-local `.pytest_tmp` for each invocation. Genuine warnings
remain visible. The standard verified command is:

```powershell
python -m pytest -v
```

The complete run collected 99 tests: 99 passed, 0 failed, 0 warnings.

## DOCUMENTATION-ONLY

### Exact image-content duplicates

All 33,126 local training JPEGs were hashed by exact file bytes. There are
32,693 unique hashes and 433 duplicate groups covering 866 records. Every
group contains exactly two images from the same patient, same split role, and
same target. No hash crosses training and validation, no duplicate uses a
different patient ID, and no target conflict exists.

This is a dataset-quality and within-patient-dependence issue, not train/Fold-0
content leakage under the current patient-aware split. All affected records
are listed in `outputs/analysis/data_integrity/exact_image_duplicates.csv`.

### Training-history metrics

B0, M1, and M2 histories do not contain `train_auc`. No train AUC was computed,
reconstructed, or claimed. Available training-behaviour evidence is train
loss, validation loss, validation ROC-AUC, validation AP, and validation
threshold metrics.

### Determinism

Experiment setup seeds Python, NumPy, torch CPU, and all CUDA devices. The
training DataLoader generator uses seed 42, and worker initialization seeds
NumPy and Python from `torch.initial_seed()`. During experiment setup, cuDNN
benchmarking is disabled and cuDNN deterministic mode is requested.

`torch.use_deterministic_algorithms` is not enabled, so exact bitwise identity
is not guaranteed across hardware, CUDA/PyTorch builds, or all operators. The
appropriate claim is a seed-controlled/reproducible protocol, not bitwise
deterministic training. Frozen training semantics were not changed.

### Validation-design limitations

1. Fold 0 is validation data, not an independent test cohort.
2. Fold 0 was used for epoch validation, checkpoint selection, and reported metrics.
3. Threshold, subgroup, failure, and Grad-CAM analyses reuse Fold 0 exploratorily.
4. There is no external validation cohort.
5. There is no final full-cross-validation performance estimate.
6. The paired bootstrap is internal patient-level resampling, not external validation.
7. No new training is being added to repair these limitations at this stage.

## CRITICAL

None. There are no cross-split exact-content duplicates, conflicting duplicate
targets, changed frozen artifacts, or issues requiring retraining.

## NOT APPLICABLE

- Bootstrap correction and interval regeneration: implementation was already correct.
- Grad-CAM regeneration: all selected Grad-CAM cases were content-unique.
- Train-AUC reconstruction: the metric was not recorded and will not be fabricated.
- Model retraining, fold reassignment, record deletion, threshold tuning, and new experiments: prohibited and unnecessary.
- Repository invitations: not sent; the user retains responsibility for them.
