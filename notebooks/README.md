# Notebooks

The repository contains three completed notebooks:

- `00_data_audit.ipynb` audits the source metadata/images, derives the
  patient-aware folds, and records dataset/integrity summaries.
- `01_logistic_regression.ipynb` implements and evaluates the H0/H1 historical
  Logistic Regression baselines.
- `02_results_analysis.ipynb` consolidates the frozen final tables,
  comparison boundaries, reliability analyses, and integrity checks without
  training models.

All three submitted notebooks contain executed code cells and stored outputs
with no recorded notebook errors. Their outputs are point-in-time evidence;
seed control and deterministic validation do not guarantee bitwise-identical
re-execution across platforms.

The first two notebooks require the external SIIM-ISIC dataset. The final
results notebook can be inspected as submitted from a clean clone, while full
execution also verifies local ignored checkpoint hashes.
