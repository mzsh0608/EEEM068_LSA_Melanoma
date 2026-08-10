# Assessment Compliance Matrix

Technical evidence map only. Final notebook, figures, IEEE prose, and submission packaging remain pending.

## TECHNICAL REPORT

| Criterion | Planned report evidence | Code/artifact evidence | Current status | Remaining Phase J work |
|---|---|---|---|---|
| Abstract | Dataset, method hierarchy, principal validation results, and limitations | main_model_results.csv; dataset_summary.csv; integrity_summary.csv | evidence_ready_report_pending | Draft and word-count review in the final IEEE report |
| Introduction | Clinical classification context, imbalance, patient grouping, and project aims | outputs/dataset_summary.json; report/methodology_notes.md | evidence_ready_report_pending | Write motivation and scoped research questions |
| Literature review | Relevant melanoma classification, transfer learning, imbalance, metadata fusion, and explainability literature | Repository contains model evidence but no completed literature synthesis | report_pending | Select and verify scholarly sources, then write synthesis |
| Methodology | Patient-aware split, preprocessing, models, loss, optimization, evaluation, bootstrap, and Grad-CAM | deep_model_protocol.csv; hyperparameter_selection.csv; src/; configs/ | evidence_ready_report_pending | Convert technical evidence into concise report methodology |
| Experiments/results | Main metrics, training behaviour, comparisons, thresholds, subgroups, failures, and uncertainty | outputs/final/tables/*.csv; frozen predictions and metrics | evidence_ready_report_pending | Generate and manually review J2 figures/notebook, then write results |
| Conclusion/future work | Evidence-bounded conclusions and external/nested-validation recommendations | comparison_boundaries.json; bootstrap_summary.csv; integrity_summary.csv | evidence_ready_report_pending | Write conclusion after final results synthesis |
| Creativity | Metadata fusion, patient-level bootstrap, exact-content audit, subgroup/failure analysis, and metadata-conditioned Grad-CAM | src/metadata.py; src/analysis.py; src/explainability.py; Phase I outputs | evidence_ready_report_pending | Explain novelty without overstating causality or clinical value |

## PROJECT-SPECIFIC OBSERVATIONS

| Criterion | Planned report evidence | Code/artifact evidence | Current status | Remaining Phase J work |
|---|---|---|---|---|
| Hyperparameter discussion | Predeclared matched settings, data-derived pos_weight, checkpoint selection, and no systematic search | hyperparameter_selection.csv; hyperparameter_strategy.json | evidence_ready | Discuss limitations and stronger nested-validation design |
| Training behaviour | Train/validation loss, validation ROC-AUC/AP, early stopping, and timing | training_summary.csv; deep_model_protocol.csv | evidence_ready | Create J2 training-behaviour presentation |
| Performance metrics | ROC-AUC, AP, threshold metrics, sensitivity/specificity, and F1 | main_model_results.csv; threshold_summary.csv | evidence_ready | Select report table/figure presentation |
| Confusion matrices | TN, FP, FN, and TP at the shared threshold 0.5 | main_model_results.csv; frozen prediction CSVs | evidence_ready | Generate and visually verify J2 figure |
| Visualisations | Training, ROC/PR, threshold, subgroup, bootstrap, failure, and Grad-CAM evidence | outputs/figures/ and outputs/analysis/ | source_evidence_ready_final_figures_pending | Generate publication figures in J2 and perform manual review |

## FUNCTIONALITY

| Criterion | Planned report evidence | Code/artifact evidence | Current status | Remaining Phase J work |
|---|---|---|---|---|
| Dataset/DataLoader | Image loading, targets, optional metadata, and batched iteration | src/dataset.py; tests/test_dataset.py | implemented_verified | Summarize in methodology |
| Transformations/augmentation | RGB resize, flips, rotation, brightness/contrast, and ImageNet normalization | src/transforms.py; frozen deep configs | implemented_verified | Summarize exact protocol |
| Patient-aware split | Permanent StratifiedGroupKFold manifest with zero patient overlap | src/splits.py; data/train_folds.csv; tests/test_splits.py | implemented_verified | State validation-only limitation |
| Model design | LR, ResNet18, ConvNeXt-Tiny, and ConvNeXt metadata fusion | src/models.py; frozen configs; tests/test_models.py | implemented_verified | Present hierarchy and comparison boundaries |
| Training | Weighted BCE, AdamW, AMP, checkpointing, and early stopping | src/train.py; src/losses.py; histories and logs | implemented_verified | Report frozen protocol and observed durations |
| Evaluation | Shared binary metrics, AP semantics, predictions, and threshold handling | src/evaluate.py; tests/test_evaluate.py | implemented_verified | Use final-facing AP and validation terminology |
| Metadata inference pipeline | Training-only preprocessing, serialized transformer reuse, and metadata-conditioned M2 forward pass | src/metadata.py; metadata_summary.json; metadata_preprocessor.joblib | implemented_verified | Describe leakage controls and attribution boundary |
| Analysis pipeline | Threshold, subgroup, disagreement, bootstrap, failure, and Grad-CAM analyses | src/analysis.py; src/explainability.py; outputs/analysis/ | implemented_verified | Consume consolidated evidence in J2 |

## CODE QUALITY

| Criterion | Planned report evidence | Code/artifact evidence | Current status | Remaining Phase J work |
|---|---|---|---|---|
| Modularity | Separated dataset, transforms, models, metadata, training, evaluation, and analysis modules | src/ | implemented_verified | Reference concise architecture in report or README |
| Configs | Versioned experiment definitions and frozen resolved configurations | configs/; logs/*/config.json | implemented_verified | Use protocol table as final source |
| Documentation | Method, results, analysis, audit, and viva technical notes | README.md; report/; viva_notes.md | technical_documentation_ready_final_report_pending | Complete final notebook/report documentation |
| Testing | Unit and integration coverage across split, data, models, training, evaluation, metadata, analysis, and audits | tests/; pytest.ini | implemented_verified | Run final suite after each Phase J integration step |
| Reproducibility | Seeds, worker seeding, configs, histories, predictions, hashes, and source manifests | src/utils.py; logs/; outputs/final/manifests/ | implemented_with_documented_bitwise_limit | State seed-controlled rather than bitwise-deterministic claim |
| 99+ tests | Complete standard pytest workflow with no failures | J1B full suite: 102 passed, 0 failed, 0 warnings | implemented_verified | Continue the standard suite after later Phase J integration steps |
| Requirements | Pinned direct runtime, notebook, and test dependencies | requirements.txt | implemented_verified | Retain official PyTorch CUDA wheel-index note |
| Logging | Per-experiment configuration, environment, history, metrics, and training logs | logs/H0_logistic_unweighted through logs/M2_convnext_metadata | implemented_verified | Reference evidence rather than duplicating logs in report |

## ORAL/VIVA

| Criterion | Planned report evidence | Code/artifact evidence | Current status | Remaining Phase J work |
|---|---|---|---|---|
| Concepts represented in viva_notes.md | Data leakage, imbalance, metrics, architecture comparisons, metadata ablation, uncertainty, failures, and Grad-CAM limitations | viva_notes.md | evidence_ready_viva_pending | Final consistency pass after J2/report completion |
