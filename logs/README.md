# Experiment Logs

The H0, H1, B0, M1, and M2 directories contain completed, frozen experiment
evidence.

- `config.json` records the resolved experiment configuration.
- `metrics.json` records Fold-0 validation metrics at the primary threshold
  0.5; H1 also retains its separate historical threshold-0.3 metrics.
- `history.csv` records epoch-level loss, validation metrics, learning rate,
  and timing for B0/M1/M2.
- `environment.json` records the audited deep-training software and hardware
  environment.
- `training.log` or `experiment.log` records the completed run narrative.
- M2 additionally stores the fitted metadata preprocessor and its summary.

Frozen per-image prediction files are stored under `outputs/predictions/`.
Historical fields named `val_pr_auc` or `pr_auc_average_precision` contain
scikit-learn Average Precision (AP), not a separately integrated trapezoidal
PR-AUC.
