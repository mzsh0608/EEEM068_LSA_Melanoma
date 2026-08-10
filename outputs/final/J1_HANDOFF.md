# J1 Handoff

Status: J1 evidence audit, consolidation, and final verification are complete.

## Authoritative manifests

- `outputs/final/manifests/authoritative_sources.json`
- `outputs/final/manifests/comparison_boundaries.json`
- `outputs/final/manifests/J1A_evidence_audit.json`
- `outputs/final/manifests/J1B_tables_manifest.json`
- `outputs/final/manifests/dataset_summary_sources.json`
- `outputs/final/manifests/hyperparameter_strategy.json`
- `outputs/final/manifests/bootstrap_interpretation.json`
- `outputs/final/manifests/J1_COMPLETE.json`

## Tables for J2

- `outputs/final/tables/dataset_summary.csv`
- `outputs/final/tables/main_model_results.csv`
- `outputs/final/tables/deep_model_protocol.csv`
- `outputs/final/tables/hyperparameter_selection.csv`
- `outputs/final/tables/training_summary.csv`
- `outputs/final/tables/B0_M1_architecture_comparison.csv`
- `outputs/final/tables/M1_M2_metadata_ablation.csv`
- `outputs/final/tables/threshold_summary.csv`
- `outputs/final/tables/bootstrap_summary.csv`
- `outputs/final/tables/subgroup_summary.csv`
- `outputs/final/tables/failure_summary.csv`
- `outputs/final/tables/integrity_summary.csv`

J2 should use these tables as its primary machine-readable inputs. Source provenance and SHA-256 values are in `J1B_tables_manifest.json`.

## Files J2 must not modify

- `data/train_folds.csv`
- `outputs/predictions/H0_logistic_unweighted.csv`
- `outputs/predictions/H1_logistic_weighted.csv`
- `outputs/predictions/B0_resnet18.csv`
- `outputs/predictions/M1_convnext_image.csv`
- `outputs/predictions/M2_convnext_metadata.csv`
- `outputs/checkpoints/B0_resnet18_best.pt`
- `outputs/checkpoints/M1_convnext_image_best.pt`
- `outputs/checkpoints/M2_convnext_metadata_best.pt`
- Frozen experiment evidence under `logs/H0_logistic_unweighted/` through `logs/M2_convnext_metadata/`
- Existing Phase I evidence under `outputs/analysis/`
- J1 tables and manifests under `outputs/final/`

Do not retrain, run `--fit`, tune thresholds, or regenerate authoritative metrics. J2 may create the final results-analysis notebook and publication/review figures from the frozen J1 evidence bundle.

## Required terminology

- Use **Average Precision (AP)** for the precision-recall summary metric.
- Call Fold 0 the **fixed patient-aware validation fold**.
- Treat Phase I threshold, subgroup, failure, bootstrap, and Grad-CAM analyses as exploratory reuse of the same validation fold.

## Comparison boundaries

- H0 to H1: controlled historical Logistic Regression class-weighting comparison.
- H0/H1 to B0: historical system-level progression, not an isolated architecture effect.
- B0 to M1: matched external-protocol architecture-family comparison; individual ConvNeXt mechanisms are not isolated.
- M1 to M2: metadata-fusion system ablation.
- M1/M2 per-sample probability differences describe model disagreement, not causal effects of age, sex, or anatomical site.
- Bootstrap intervals are paired internal patient-level resampling, not external validation.
- Grad-CAM is image-branch positive-logit attribution conditioned on metadata, not metadata attribution or causal reasoning.

## Hyperparameter boundaries

- No systematic grid, random, or Bayesian search was performed.
- Deep-model settings were predeclared and matched to reduce optimization-policy confounding.
- `pos_weight` was derived from training class counts.
- Early stopping and validation ROC-AUC checkpointing selected effective training duration.
- Threshold `0.5` is the predeclared common primary threshold.
- The Phase I threshold grid is post-hoc behavioural analysis, not optimization.
- M2 metadata embedding dimension and dropout are predeclared architecture settings, not tuned optima.
