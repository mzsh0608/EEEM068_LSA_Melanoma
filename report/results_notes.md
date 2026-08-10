# Results notes

## B0 - ResNet18 with weighted BCE

### Experimental setup

- Fixed validation partition: patient-aware Fold 0
- Training samples: 26,499 (26,032 benign; 467 melanoma)
- Validation samples: 6,627 (6,510 benign; 117 melanoma)
- Known patient overlap: 0
- Architecture: ResNet18 with `IMAGENET1K_V1` weights
- Input: 224x224 RGB images
- Fine-tuning: all model parameters
- Training augmentation: horizontal and vertical flips, rotation up to
  15 degrees, and brightness/contrast jitter of 0.10
- Validation preprocessing: deterministic resize and ImageNet normalization
- Loss: `BCEWithLogitsLoss` with training-only `pos_weight`
- `pos_weight`: 26,032 / 467 = 55.74304068522484
- Optimizer: AdamW, learning rate 0.0001, weight decay 0.0001
- AMP: enabled on CUDA
- Maximum epochs: 10
- Early stopping: patience 3, selected by maximum validation ROC-AUC
- Actual epochs: 7; early stopping triggered
- Best epoch: 4

### Best-checkpoint validation results at threshold 0.5

| Metric | B0 |
|---|---:|
| ROC-AUC | 0.8642758675 |
| Average Precision | 0.1192731307 |
| Accuracy | 0.7680700166 |
| Balanced accuracy | 0.7812300603 |
| Precision | 0.0579078456 |
| Sensitivity | 0.7948717949 |
| Specificity | 0.7675883257 |
| F1 | 0.1079512478 |
| TN | 4,997 |
| FP | 1,513 |
| FN | 24 |
| TP | 93 |

The saved prediction CSV contains all 6,627 Fold 0 samples. Its image names
and targets match the H0 and H1 validation predictions exactly. Independent
metric recomputation from its probabilities reproduced `metrics.json`.

### Training behaviour

Training loss generally decreased from 1.0615 at epoch 1 to 0.8120 at epoch
7, with a small increase at epoch 6. Validation ROC-AUC peaked at epoch 4
(0.8643), then declined over epochs 5-7. Validation loss rose from 1.0283 at
the selected epoch to 1.2573 at epoch 7. Average Precision varied across
epochs and reached its highest observed value at epoch 6 (0.1329), but model
selection followed the predeclared ROC-AUC criterion. The widening loss trend
and deterioration in ROC-AUC after epoch 4 are consistent with overfitting or
training instability. They are not evidence that the fluctuations are
statistically significant. Early stopping restored the epoch 4 checkpoint.

### Comparison with H0 and H1

| Model | ROC-AUC | Average Precision | Accuracy | Balanced accuracy | Precision | Sensitivity | Specificity | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 | 0.657022 | 0.038592 | 0.980987 | 0.503505 | 0.090909 | 0.008547 | 0.998464 | 0.015625 | 1 | 10 | 116 | 6,500 |
| H1 | 0.624392 | 0.036316 | 0.937226 | 0.548379 | 0.051051 | 0.145299 | 0.951459 | 0.075556 | 17 | 316 | 100 | 6,194 |
| B0 | 0.864276 | 0.119273 | 0.768070 | 0.781230 | 0.057908 | 0.794872 | 0.767588 | 0.107951 | 93 | 1,513 | 24 | 4,997 |

B0 substantially improved ranking performance over both flattened-grayscale
Logistic Regression baselines: ROC-AUC increased by 0.2073 relative to H0 and
0.2399 relative to H1, while Average Precision was approximately three times
the baseline values. At the fixed 0.5 threshold, B0 detected 93 of 117
melanomas, compared with 1 for H0 and 17 for H1, so it did not collapse to the
majority class. This gain came with 1,513 false positives and specificity of
0.7676. Its precision remained low at 0.0579 because melanoma prevalence is
very low and the weighted loss strongly penalises missed positives.

The evidence supports the value of a pretrained hierarchical image
representation over flattened grayscale pixels on this fixed fold. It does
not establish clinical utility or statistical significance. The remaining
sensitivity-specificity trade-off, low precision, single-fold evaluation, and
lack of uncertainty estimates remain important limitations.
