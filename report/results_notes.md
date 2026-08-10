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

## M1 - ConvNeXt-Tiny + weighted BCE

### Experimental setup

- Architecture: torchvision ConvNeXt-Tiny with `IMAGENET1K_V1` weights
- Image-only model with one raw binary logit and full fine-tuning
- Training partition: the same 26,499 images used by B0
- Validation partition: the same fixed 6,627-sample patient-aware Fold 0
- Input and transforms: the same 224x224 RGB pipeline used by B0
- Loss: weighted `BCEWithLogitsLoss`
- Training-only `pos_weight`: 26,032 / 467 = 55.74304068522484
- Optimizer: AdamW, learning rate 0.0001, weight decay 0.0001
- Scheduler: none; AMP enabled on CUDA
- Maximum epochs: 10; early-stopping patience: 3
- Checkpoint selection: maximum validation ROC-AUC
- Actual epochs: 7; best epoch: 4; early stopping triggered

### Validation results at threshold 0.5

| Metric | M1 |
|---|---:|
| ROC-AUC | 0.9008481363 |
| Average Precision | 0.1694056107 |
| Accuracy | 0.6337709371 |
| Balanced accuracy | 0.8010043720 |
| Precision | 0.0449172577 |
| Sensitivity | 0.9743589744 |
| Specificity | 0.6276497696 |
| F1 | 0.0858757062 |
| TN | 4,086 |
| FP | 2,424 |
| FN | 3 |
| TP | 114 |

The authoritative prediction file contains all 6,627 validation samples and
matches the image names and targets in H0, H1, B0, and the permanent Fold 0
manifest. Independent metric recomputation reproduced `metrics.json`.

### Training behaviour

Training loss fell from 1.1163 at epoch 1 to 0.7766 at epoch 6, then rose to
0.8844 at epoch 7. Validation loss reached its minimum at epoch 4 (0.8010),
spiked at epoch 5, and recovered to 0.8066 by epoch 7. ROC-AUC and Average
Precision both peaked at epoch 4 (0.9008 and 0.1694). ROC-AUC remained between
0.8912 and 0.8924 over epochs 5-7, below the selected checkpoint. Early
stopping therefore triggered after epoch 7 and restored epoch 4. The loss and
metric fluctuations indicate some instability, but the post-peak ROC-AUC and
validation loss were more stable than B0's later-epoch deterioration.

### Controlled comparison with B0

| Metric | B0 | M1 | M1 - B0 |
|---|---:|---:|---:|
| ROC-AUC | 0.8642758675 | 0.9008481363 | +0.0365722688 |
| Average Precision | 0.1192731307 | 0.1694056107 | +0.0501324800 |
| Accuracy | 0.7680700166 | 0.6337709371 | -0.1342990795 |
| Balanced accuracy | 0.7812300603 | 0.8010043720 | +0.0197743117 |
| Precision | 0.0579078456 | 0.0449172577 | -0.0129905879 |
| Sensitivity | 0.7948717949 | 0.9743589744 | +0.1794871795 |
| Specificity | 0.7675883257 | 0.6276497696 | -0.1399385561 |
| F1 | 0.1079512478 | 0.0858757062 | -0.0220755416 |
| TP | 93 | 114 | +21 |
| FP | 1,513 | 2,424 | +911 |
| FN | 24 | 3 | -21 |
| TN | 4,997 | 4,086 | -911 |

Under the matched external protocol, changing from ResNet18 to
ConvNeXt-Tiny was associated with better ranking, balanced accuracy, and
melanoma sensitivity. At the fixed threshold, that gain came from a much more
positive prediction policy: M1 found 21 additional melanomas but produced 911
additional false positives. Consequently, accuracy, specificity, precision,
and F1 were lower than B0.

### Computational/model-size comparison

| Property | B0 ResNet18 | M1 ConvNeXt-Tiny |
|---|---:|---:|
| Total parameters | 11,177,025 | 27,820,897 |
| Trainable parameters | 11,177,025 | 27,820,897 |
| Actual epochs | 7 | 7 |
| Best epoch | 4 | 4 |
| Total duration (seconds) | 593.76 | 767.50 |
| Mean epoch duration after epoch 1 (seconds) | 68.03 | 98.94 |

M1 had 2.49 times as many parameters and took approximately 29% longer
overall. Its post-startup epochs were approximately 45% longer than B0's.

### Interpretation

ConvNeXt-Tiny achieved a meaningful ranking improvement under the matched
budget, but it did not improve every threshold-dependent metric. The larger,
more modern architecture improved ROC-AUC, Average Precision, balanced
accuracy, and minority-class detection while worsening the false-positive
burden. The defensible conclusion is that changing the image architecture
under the fixed protocol was associated with these performance differences;
the experiment cannot attribute them to one internal ConvNeXt component.

### Limitations

The comparison uses one fixed validation fold and has no confidence intervals,
external validation, or architecture-specific hyperparameter optimisation.
Threshold 0.5 was deliberately fixed rather than tuned. Severe class imbalance
kept precision low, and M1's 2,424 false positives show that stronger ranking
does not imply an acceptable operating point or clinical utility.
