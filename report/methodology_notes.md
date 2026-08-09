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