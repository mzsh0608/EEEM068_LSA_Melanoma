# Data Directory

Only small derived reproducibility manifests are tracked here:

- `train_folds.csv` maps every `image_name` to one of five patient-aware folds.
  Fold 0 is validation; folds 1-4 are training.
- `lr_train_subset.csv` records the stratified 5,000-image training subset used
  by the H0/H1 Logistic Regression baselines.

The external SIIM-ISIC 2020 source data is intentionally not tracked. For a
dataset-dependent run, place it at:

```text
data/
|-- train.csv
`-- train_images/
    `-- <image_name>.jpg
```

The expected source images are 512x512 JPEG files. Raw images, original input
metadata, DICOM files, dataset archives, and large array formats are excluded
through `.gitignore`.
