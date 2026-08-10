# Assessment Compliance Matrix

Final submission evidence map. The report, executed notebook, publication figures, tables, and frozen experiment evidence are present.

## TECHNICAL REPORT

| Criterion | Final submission evidence | Code/artifact evidence | Current status | Scope or limitation |
|---|---|---|---|---|
| Abstract | Dataset, method hierarchy, principal validation results, and limitations | report/EEEM068_LSA_Melanoma_Final_Report_Revised.pdf; main_model_results.csv | complete | Final report and frozen evidence are present |
| Introduction | Clinical classification context, imbalance, patient grouping, and project aims | final report; outputs/dataset_summary.json; report/methodology_notes.md | complete | Claims remain scoped to internal validation |
| Literature review | Relevant melanoma classification, transfer learning, imbalance, metadata fusion, and explainability literature | report/EEEM068_LSA_Melanoma_Final_Report_Revised.pdf | complete | Literature synthesis is contained in the submitted report |
| Methodology | Patient-aware split, preprocessing, models, loss, optimization, evaluation, bootstrap, and Grad-CAM | final report; deep_model_protocol.csv; hyperparameter_selection.csv; src/; configs/ | complete | Fold 0 is validation, not an independent test set |
| Experiments/results | Main metrics, training behaviour, comparisons, thresholds, subgroups, failures, and uncertainty | final report; outputs/final/tables/; outputs/final/figures/; frozen predictions | complete | Comparison boundaries and threshold 0.5 are retained |
| Conclusion/future work | Evidence-bounded conclusions and external/nested-validation recommendations | final report; comparison_boundaries.json; bootstrap_summary.csv; integrity_summary.csv | complete | No clinical or external-validation claim |
| Creativity | Metadata fusion, patient-level bootstrap, exact-content audit, subgroup/failure analysis, and metadata-conditioned Grad-CAM | src/metadata.py; src/analysis.py; src/explainability.py; Phase I outputs | complete | No individual-metadata causality claim |

## PROJECT-SPECIFIC OBSERVATIONS

| Criterion | Final submission evidence | Code/artifact evidence | Current status | Scope or limitation |
|---|---|---|---|---|
| Hyperparameter discussion | Predeclared matched settings, data-derived pos_weight, checkpoint selection, and no systematic search | hyperparameter_selection.csv; hyperparameter_strategy.json | documented | No grid, random, or Bayesian search was performed |
| Training behaviour | Train/validation loss, validation ROC-AUC/AP, early stopping, and timing | training_summary.csv; deep_model_protocol.csv | complete | Checkpoint selection and effective duration are distinguished |
| Performance metrics | ROC-AUC, AP, threshold metrics, sensitivity/specificity, and F1 | main_model_results.csv; threshold_summary.csv | complete | Average Precision is the final-facing term |
| Confusion matrices | TN, FP, FN, and TP at the shared threshold 0.5 | main_model_results.csv; frozen prediction CSVs | complete | Threshold 0.5 is the primary common operating point |
| Visualisations | Training, ROC/PR, threshold, subgroup, bootstrap, failure, and Grad-CAM evidence | outputs/final/figures/; outputs/figures/; outputs/analysis/ | complete | Approved PNG/PDF publication assets are present |

## FUNCTIONALITY

| Criterion | Final submission evidence | Code/artifact evidence | Current status | Scope or limitation |
|---|---|---|---|---|
| Dataset/DataLoader | Image loading, targets, optional metadata, and batched iteration | src/dataset.py; tests/test_dataset.py | implemented_verified | External source JPEGs are intentionally not tracked |
| Transformations/augmentation | RGB resize, flips, rotation, brightness/contrast, and ImageNet normalization | src/transforms.py; frozen deep configs | implemented_verified | Augmentation is training-only; validation is deterministic |
| Patient-aware split | Permanent StratifiedGroupKFold manifest with zero patient overlap | src/splits.py; data/train_folds.csv; tests/test_splits.py | implemented_verified | Fold 0 is validation only |
| Model design | LR, ResNet18, ConvNeXt-Tiny, and ConvNeXt metadata fusion | src/models.py; frozen configs; tests/test_models.py | implemented_verified | H0/H1 to B0 is historical, not a controlled comparison |
| Training | Weighted BCE, AdamW, AMP, checkpointing, and early stopping | src/train.py; src/losses.py; histories and logs | implemented_verified | Full retraining requires the external dataset and compute |
| Evaluation | Shared binary metrics, AP semantics, predictions, and threshold handling | src/evaluate.py; tests/test_evaluate.py | implemented_verified | AP and validation terminology verified |
| Metadata inference pipeline | Training-only preprocessing, serialized transformer reuse, and metadata-conditioned M2 forward pass | src/metadata.py; metadata_summary.json; metadata_preprocessor.joblib | implemented_verified | Grad-CAM does not explain the metadata branch |
| Analysis pipeline | Threshold, subgroup, disagreement, bootstrap, failure, and Grad-CAM analyses | src/analysis.py; src/explainability.py; outputs/analysis/ | implemented_verified | Post-hoc analysis is not hyperparameter tuning |

## CODE QUALITY

| Criterion | Final submission evidence | Code/artifact evidence | Current status | Scope or limitation |
|---|---|---|---|---|
| Modularity | Separated dataset, transforms, models, metadata, training, evaluation, and analysis modules | src/ | implemented_verified | Marker-facing structure documented in README.md |
| Configs | Versioned experiment definitions and frozen resolved configurations | configs/; logs/*/config.json | implemented_verified | Five final experiment configs are retained |
| Documentation | Method, results, analysis, audit, and viva technical notes | README.md; report/; viva_notes.md | complete | J1 manifests remain historical provenance snapshots |
| Testing | Unit and integration coverage across split, data, models, training, evaluation, metadata, analysis, and audits | tests/; pytest.ini | implemented_verified | Final documentation suite: 108 passed, 0 failed, 0 warnings |
| Reproducibility | Seeds, worker seeding, configs, histories, predictions, hashes, and source manifests | src/utils.py; logs/; outputs/final/manifests/ | implemented_with_documented_bitwise_limit | Seed-controlled does not imply bitwise-identical retraining |
| Test suite | Complete standard pytest workflow with no failures | Final documentation suite: 108 passed, 0 failed, 0 warnings | implemented_verified | Review-package-only tests were removed with their generator |
| Requirements | Pinned direct runtime, notebook, and test dependencies | requirements.txt | implemented_verified | CUDA wheel index remains platform-specific |
| Logging | Per-experiment configuration, environment, history, metrics, and training logs | logs/H0_logistic_unweighted through logs/M2_convnext_metadata | implemented_verified | Predictions are retained under outputs/predictions/ |

## ORAL/VIVA

| Criterion | Final submission evidence | Code/artifact evidence | Current status | Scope or limitation |
|---|---|---|---|---|
| Concepts represented in viva_notes.md | Data leakage, imbalance, metrics, architecture comparisons, metadata ablation, uncertainty, failures, and Grad-CAM limitations | viva_notes.md | retained_final_reference | Concise technical preparation notes; no review-package dependency |
