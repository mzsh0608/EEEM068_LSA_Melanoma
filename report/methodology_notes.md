# Methodology notes

## Dataset

- SIIM-ISIC 2020
- 33,126 images
- 32,542 benign
- 584 melanoma
- melanoma prevalence approximately 1.763%
- 2,056 unique patients
- 512x512 JPEG source

## Validation

- StratifiedGroupKFold
- 5 folds generated
- random seed 42
- patient_id used for grouping
- target used for stratification
- Fold 0 used as fixed held-out validation set
- train: 26,499
- validation: 6,627
- patient overlap: 0
- this is NOT full five-fold cross-validation

## Historical Logistic Regression

- 5,000 stratified training observations sampled only from
  the training partition
- grayscale
- 64x64
- scaled to [0,1]
- flattened to 4,096 features

H0:
- unweighted Logistic Regression

H1:
- class_weight="balanced"

Evaluation:
- common threshold = 0.5
- historical H1 threshold 0.3 analysed separately
- ROC-AUC
- PR-AUC / Average Precision
- balanced accuracy
- sensitivity
- specificity
- precision
- F1
- confusion matrix

## B0 ResNet18 methodology

- architecture: ResNet18
- initial weights: ImageNet `IMAGENET1K_V1`
- classifier replacement: one output logit with no model-side sigmoid
- input: 224x224 RGB
- fine-tuning: the complete network
- validation: the same fixed patient-aware Fold 0 used by H0 and H1
- training augmentation: horizontal and vertical flips, rotation up to
  15 degrees, and brightness/contrast jitter of 0.10
- validation preprocessing: deterministic resize only
- normalization: ImageNet mean and standard deviation
- loss: `BCEWithLogitsLoss` with positive-class weighting
- training-only `pos_weight`: 26,032 / 467 = 55.74304068522484
- optimizer: AdamW
- learning rate: 0.0001
- weight decay: 0.0001
- scheduler: none
- CUDA automatic mixed precision: enabled
- maximum epochs: 10
- early stopping: patience 3
- checkpoint selection: maximum validation ROC-AUC
- authoritative evaluation: reloaded best checkpoint on all 6,627 Fold 0
  samples at the fixed threshold of 0.5

Fold 0 labels were used for validation and checkpoint selection only. They
were not used to calculate `pos_weight`, sample training data, or update model
parameters.
