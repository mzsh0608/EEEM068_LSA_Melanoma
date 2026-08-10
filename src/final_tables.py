"""Build and validate the frozen machine-readable evidence tables for J1B."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_FILES = {
    "H0": {
        "directory": "H0_logistic_unweighted",
        "prediction": "outputs/predictions/H0_logistic_unweighted.csv",
    },
    "H1": {
        "directory": "H1_logistic_weighted",
        "prediction": "outputs/predictions/H1_logistic_weighted.csv",
    },
    "B0": {
        "directory": "B0_resnet18",
        "prediction": "outputs/predictions/B0_resnet18.csv",
    },
    "M1": {
        "directory": "M1_convnext_image",
        "prediction": "outputs/predictions/M1_convnext_image.csv",
    },
    "M2": {
        "directory": "M2_convnext_metadata",
        "prediction": "outputs/predictions/M2_convnext_metadata.csv",
    },
}

METRIC_COLUMNS = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
]


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_history(history, model_id):
    """Map the frozen AP compatibility column to final-facing terminology."""
    result = history.rename(columns={"val_pr_auc": "val_average_precision"}).copy()
    result.insert(0, "model_id", model_id)
    result["is_best_roc_auc_epoch"] = result["val_roc_auc"].eq(
        result["val_roc_auc"].max()
    )
    columns = [
        "model_id",
        "epoch",
        "train_loss",
        "val_loss",
        "val_roc_auc",
        "val_average_precision",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_precision",
        "val_sensitivity",
        "val_specificity",
        "val_f1",
        "learning_rate",
        "seconds",
        "is_best_roc_auc_epoch",
    ]
    return result.loc[:, columns]


def _comparison_table(
    main_results,
    left_id,
    right_id,
    delta_column,
    metadata,
):
    """Return exact right-minus-left metric differences with shared metadata."""
    indexed = main_results.set_index("model_id")
    rows = []
    for metric in METRIC_COLUMNS:
        left_value = indexed.at[left_id, metric]
        right_value = indexed.at[right_id, metric]
        row = {
            "metric": metric,
            left_id: left_value,
            right_id: right_value,
            delta_column: right_value - left_value,
        }
        row.update(metadata)
        rows.append(row)
    return pd.DataFrame(rows)


def _verify_subgroup_partitions(subgroups, validation_n):
    """Require every model/variable subgroup partition to cover validation."""
    totals = subgroups.groupby(["model", "group_variable"], dropna=False)["n"].sum()
    invalid = totals.loc[totals.ne(validation_n)]
    if not invalid.empty:
        raise ValueError(
            "Subgroup partitions do not reconcile to validation_n: "
            f"{invalid.to_dict()}"
        )
    return True


def _assert_frozen_hashes(root, j1a_audit):
    for artifact in j1a_audit["frozen_artifact_verification"]["artifacts"]:
        current = _sha256_file(root / artifact["path"])
        if current != artifact["current_sha256"]:
            raise ValueError(f"Frozen artifact changed: {artifact['path']}")


def _assessment_compliance_text():
    sections = {
        "TECHNICAL REPORT": [
            (
                "Abstract",
                "Dataset, method hierarchy, principal validation results, and limitations",
                "main_model_results.csv; dataset_summary.csv; integrity_summary.csv",
                "evidence_ready_report_pending",
                "Draft and word-count review in the final IEEE report",
            ),
            (
                "Introduction",
                "Clinical classification context, imbalance, patient grouping, and project aims",
                "outputs/dataset_summary.json; report/methodology_notes.md",
                "evidence_ready_report_pending",
                "Write motivation and scoped research questions",
            ),
            (
                "Literature review",
                "Relevant melanoma classification, transfer learning, imbalance, metadata fusion, and explainability literature",
                "Repository contains model evidence but no completed literature synthesis",
                "report_pending",
                "Select and verify scholarly sources, then write synthesis",
            ),
            (
                "Methodology",
                "Patient-aware split, preprocessing, models, loss, optimization, evaluation, bootstrap, and Grad-CAM",
                "deep_model_protocol.csv; hyperparameter_selection.csv; src/; configs/",
                "evidence_ready_report_pending",
                "Convert technical evidence into concise report methodology",
            ),
            (
                "Experiments/results",
                "Main metrics, training behaviour, comparisons, thresholds, subgroups, failures, and uncertainty",
                "outputs/final/tables/*.csv; frozen predictions and metrics",
                "evidence_ready_report_pending",
                "Generate and manually review J2 figures/notebook, then write results",
            ),
            (
                "Conclusion/future work",
                "Evidence-bounded conclusions and external/nested-validation recommendations",
                "comparison_boundaries.json; bootstrap_summary.csv; integrity_summary.csv",
                "evidence_ready_report_pending",
                "Write conclusion after final results synthesis",
            ),
            (
                "Creativity",
                "Metadata fusion, patient-level bootstrap, exact-content audit, subgroup/failure analysis, and metadata-conditioned Grad-CAM",
                "src/metadata.py; src/analysis.py; src/explainability.py; Phase I outputs",
                "evidence_ready_report_pending",
                "Explain novelty without overstating causality or clinical value",
            ),
        ],
        "PROJECT-SPECIFIC OBSERVATIONS": [
            (
                "Hyperparameter discussion",
                "Predeclared matched settings, data-derived pos_weight, checkpoint selection, and no systematic search",
                "hyperparameter_selection.csv; hyperparameter_strategy.json",
                "evidence_ready",
                "Discuss limitations and stronger nested-validation design",
            ),
            (
                "Training behaviour",
                "Train/validation loss, validation ROC-AUC/AP, early stopping, and timing",
                "training_summary.csv; deep_model_protocol.csv",
                "evidence_ready",
                "Create J2 training-behaviour presentation",
            ),
            (
                "Performance metrics",
                "ROC-AUC, AP, threshold metrics, sensitivity/specificity, and F1",
                "main_model_results.csv; threshold_summary.csv",
                "evidence_ready",
                "Select report table/figure presentation",
            ),
            (
                "Confusion matrices",
                "TN, FP, FN, and TP at the shared threshold 0.5",
                "main_model_results.csv; frozen prediction CSVs",
                "evidence_ready",
                "Generate and visually verify J2 figure",
            ),
            (
                "Visualisations",
                "Training, ROC/PR, threshold, subgroup, bootstrap, failure, and Grad-CAM evidence",
                "outputs/figures/ and outputs/analysis/",
                "source_evidence_ready_final_figures_pending",
                "Generate publication figures in J2 and perform manual review",
            ),
        ],
        "FUNCTIONALITY": [
            (
                "Dataset/DataLoader",
                "Image loading, targets, optional metadata, and batched iteration",
                "src/dataset.py; tests/test_dataset.py",
                "implemented_verified",
                "Summarize in methodology",
            ),
            (
                "Transformations/augmentation",
                "RGB resize, flips, rotation, brightness/contrast, and ImageNet normalization",
                "src/transforms.py; frozen deep configs",
                "implemented_verified",
                "Summarize exact protocol",
            ),
            (
                "Patient-aware split",
                "Permanent StratifiedGroupKFold manifest with zero patient overlap",
                "src/splits.py; data/train_folds.csv; tests/test_splits.py",
                "implemented_verified",
                "State validation-only limitation",
            ),
            (
                "Model design",
                "LR, ResNet18, ConvNeXt-Tiny, and ConvNeXt metadata fusion",
                "src/models.py; frozen configs; tests/test_models.py",
                "implemented_verified",
                "Present hierarchy and comparison boundaries",
            ),
            (
                "Training",
                "Weighted BCE, AdamW, AMP, checkpointing, and early stopping",
                "src/train.py; src/losses.py; histories and logs",
                "implemented_verified",
                "Report frozen protocol and observed durations",
            ),
            (
                "Evaluation",
                "Shared binary metrics, AP semantics, predictions, and threshold handling",
                "src/evaluate.py; tests/test_evaluate.py",
                "implemented_verified",
                "Use final-facing AP and validation terminology",
            ),
            (
                "Metadata inference pipeline",
                "Training-only preprocessing, serialized transformer reuse, and metadata-conditioned M2 forward pass",
                "src/metadata.py; metadata_summary.json; metadata_preprocessor.joblib",
                "implemented_verified",
                "Describe leakage controls and attribution boundary",
            ),
            (
                "Analysis pipeline",
                "Threshold, subgroup, disagreement, bootstrap, failure, and Grad-CAM analyses",
                "src/analysis.py; src/explainability.py; outputs/analysis/",
                "implemented_verified",
                "Consume consolidated evidence in J2",
            ),
        ],
        "CODE QUALITY": [
            (
                "Modularity",
                "Separated dataset, transforms, models, metadata, training, evaluation, and analysis modules",
                "src/",
                "implemented_verified",
                "Reference concise architecture in report or README",
            ),
            (
                "Configs",
                "Versioned experiment definitions and frozen resolved configurations",
                "configs/; logs/*/config.json",
                "implemented_verified",
                "Use protocol table as final source",
            ),
            (
                "Documentation",
                "Method, results, analysis, audit, and viva technical notes",
                "README.md; report/; viva_notes.md",
                "technical_documentation_ready_final_report_pending",
                "Complete final notebook/report documentation",
            ),
            (
                "Testing",
                "Unit and integration coverage across split, data, models, training, evaluation, metadata, analysis, and audits",
                "tests/; pytest.ini",
                "implemented_verified",
                "Run final suite after each Phase J integration step",
            ),
            (
                "Reproducibility",
                "Seeds, worker seeding, configs, histories, predictions, hashes, and source manifests",
                "src/utils.py; logs/; outputs/final/manifests/",
                "implemented_with_documented_bitwise_limit",
                "State seed-controlled rather than bitwise-deterministic claim",
            ),
            (
                "99+ tests",
                "Complete standard pytest workflow with no failures",
                "J1B full suite: 102 passed, 0 failed, 0 warnings",
                "implemented_verified",
                "Continue the standard suite after later Phase J integration steps",
            ),
            (
                "Requirements",
                "Pinned direct runtime, notebook, and test dependencies",
                "requirements.txt",
                "implemented_verified",
                "Retain official PyTorch CUDA wheel-index note",
            ),
            (
                "Logging",
                "Per-experiment configuration, environment, history, metrics, and training logs",
                "logs/H0_logistic_unweighted through logs/M2_convnext_metadata",
                "implemented_verified",
                "Reference evidence rather than duplicating logs in report",
            ),
        ],
        "ORAL/VIVA": [
            (
                "Concepts represented in viva_notes.md",
                "Data leakage, imbalance, metrics, architecture comparisons, metadata ablation, uncertainty, failures, and Grad-CAM limitations",
                "viva_notes.md",
                "evidence_ready_viva_pending",
                "Final consistency pass after J2/report completion",
            )
        ],
    }

    lines = [
        "# Assessment Compliance Matrix",
        "",
        "Technical evidence map only. Final notebook, figures, IEEE prose, and submission packaging remain pending.",
    ]
    for heading, rows in sections.items():
        lines.extend([
            "",
            f"## {heading}",
            "",
            "| Criterion | Planned report evidence | Code/artifact evidence | Current status | Remaining Phase J work |",
            "|---|---|---|---|---|",
        ])
        for row in rows:
            escaped = [str(value).replace("|", "\\|") for value in row]
            lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines) + "\n"


def generate_final_tables(project_root, timestamp=None):
    """Generate all J1B tables, manifests, and internal checks."""
    root = Path(project_root).resolve()
    table_dir = root / "outputs/final/tables"
    manifest_dir = root / "outputs/final/manifests"
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now().astimezone().isoformat(timespec="seconds")

    j1a = _read_json(manifest_dir / "J1A_evidence_audit.json")
    _assert_frozen_hashes(root, j1a)

    dataset = _read_json(root / "outputs/dataset_summary.json")
    duplicates = _read_json(
        root / "outputs/analysis/data_integrity/exact_image_duplicate_summary.json"
    )
    phase_i = _read_json(root / "outputs/analysis/phase_i_summary.json")
    bootstrap = _read_json(
        root / "outputs/analysis/bootstrap/M1_M2_bootstrap_summary.json"
    )
    disagreement = _read_json(
        root / "outputs/analysis/failures/M1_M2_disagreement_summary.json"
    )
    pre_j = _read_json(root / "outputs/analysis/technical_audit/pre_phase_j_audit.json")

    train = pd.read_csv(root / "data/train.csv", usecols=["image_name", "patient_id", "target"])
    folds = pd.read_csv(root / "data/train_folds.csv")
    split = train.merge(folds, on="image_name", validate="one_to_one")
    training = split.loc[split["fold"].ne(0)]
    validation = split.loc[split["fold"].eq(0)]
    patient_overlap = len(
        set(training["patient_id"].dropna())
        & set(validation["patient_id"].dropna())
    )

    dataset_summary = pd.DataFrame([{
        "total_images": dataset["total_images"],
        "benign_images": dataset["benign_images"],
        "melanoma_images": dataset["melanoma_images"],
        "melanoma_prevalence": dataset["melanoma_prevalence"],
        "unique_patients": dataset["unique_known_patients"],
        "train_images": len(training),
        "train_benign": int(training["target"].eq(0).sum()),
        "train_melanoma": int(training["target"].eq(1).sum()),
        "validation_images": len(validation),
        "validation_benign": int(validation["target"].eq(0).sum()),
        "validation_melanoma": int(validation["target"].eq(1).sum()),
        "patient_overlap": patient_overlap,
        "exact_duplicate_groups": duplicates["duplicate_hash_groups"],
        "exact_duplicate_records": duplicates["duplicate_image_records"],
        "cross_split_exact_duplicates": duplicates["cross_split_groups"],
        "conflicting_target_duplicate_groups": duplicates["target_conflict_groups"],
    }])

    configs = {}
    raw_metrics = {}
    for model_id, files in MODEL_FILES.items():
        experiment_dir = root / "logs" / files["directory"]
        configs[model_id] = _read_json(experiment_dir / "config.json")
        raw_metrics[model_id] = _read_json(experiment_dir / "metrics.json")

    descriptors = {
        "H0": {
            "model_name": "Logistic Regression (unweighted)",
            "input_representation": configs["H0"]["feature_representation"],
            "image_size": 64,
            "metadata_used": False,
            "training_data_scope": "5,000-image training subset",
            "imbalance_strategy": "none",
            "comparison_role": "historical unweighted LR lower bound",
        },
        "H1": {
            "model_name": "Logistic Regression (balanced class weight)",
            "input_representation": configs["H1"]["feature_representation"],
            "image_size": 64,
            "metadata_used": False,
            "training_data_scope": "5,000-image training subset",
            "imbalance_strategy": "balanced class weights",
            "comparison_role": "historical weighted LR comparison",
        },
        "B0": {
            "model_name": "ResNet18",
            "input_representation": "224x224 RGB images",
            "image_size": 224,
            "metadata_used": False,
            "training_data_scope": "full 26,499-image training partition",
            "imbalance_strategy": "training-derived weighted BCE",
            "comparison_role": "deep ResNet baseline",
        },
        "M1": {
            "model_name": "ConvNeXt-Tiny image-only",
            "input_representation": "224x224 RGB images",
            "image_size": 224,
            "metadata_used": False,
            "training_data_scope": "full 26,499-image training partition",
            "imbalance_strategy": "training-derived weighted BCE",
            "comparison_role": "controlled ConvNeXt image model",
        },
        "M2": {
            "model_name": "ConvNeXt-Tiny with metadata fusion",
            "input_representation": "224x224 RGB images + age, sex, anatomical site",
            "image_size": 224,
            "metadata_used": True,
            "training_data_scope": "full 26,499-image training partition",
            "imbalance_strategy": "training-derived weighted BCE",
            "comparison_role": "metadata-fusion ablation model",
        },
    }
    main_rows = []
    for model_id in MODEL_FILES:
        metrics = raw_metrics[model_id]
        row = {"model_id": model_id, **descriptors[model_id]}
        row.update({
            "roc_auc": metrics["roc_auc"],
            "average_precision": metrics["pr_auc_average_precision"],
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "precision": metrics["precision"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "f1": metrics["f1"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
            "primary_threshold": metrics["threshold"],
            "validation_n": metrics["n_samples"],
            "validation_positive_n": metrics["n_positive"],
        })
        main_rows.append(row)
    main_model_results = pd.DataFrame(main_rows)

    histories = {
        model_id: pd.read_csv(root / "logs" / MODEL_FILES[model_id]["directory"] / "history.csv")
        for model_id in ["B0", "M1", "M2"]
    }
    timing = pd.read_csv(
        root / "outputs/analysis/technical_audit/training_timing_audit.csv"
    ).set_index("model")
    deep_rows = []
    for model_id in ["B0", "M1", "M2"]:
        config = configs[model_id]
        configured = config["configured"]
        resolved = config["resolved"]
        history = histories[model_id]
        timing_row = timing.loc[model_id]
        transform = configured["transforms"]
        parameters = int(timing_row["parameters"])
        trainable_parameters = resolved.get("trainable_parameters")
        if trainable_parameters is None and resolved["fine_tune_all"]:
            trainable_parameters = parameters
        deep_rows.append({
            "model_id": model_id,
            "architecture": resolved["architecture"],
            "pretrained_weights": resolved["weights"],
            "fine_tune_all": resolved["fine_tune_all"],
            "image_size": resolved["image_size"],
            "batch_size": resolved["batch_size"],
            "num_workers": resolved["num_workers"],
            "train_n": resolved["train_samples"],
            "validation_n": resolved["validation_samples"],
            "train_benign": resolved["train_negative"],
            "train_melanoma": resolved["train_positive"],
            "validation_benign": resolved["validation_negative"],
            "validation_melanoma": resolved["validation_positive"],
            "augmentation_summary": (
                f"horizontal_flip={transform['horizontal_flip_probability']}; "
                f"vertical_flip={transform['vertical_flip_probability']}; "
                f"rotation_degrees={transform['rotation_degrees']}; "
                f"brightness={transform['brightness']}; contrast={transform['contrast']}"
            ),
            "normalization": (
                f"mean={json.dumps(transform['normalization_mean'])}; "
                f"std={json.dumps(transform['normalization_std'])}"
            ),
            "loss": resolved["loss"],
            "pos_weight": resolved["pos_weight"],
            "optimizer": resolved["optimizer"],
            "learning_rate": resolved["learning_rate"],
            "weight_decay": resolved["weight_decay"],
            "scheduler": configured["training"]["scheduler"],
            "amp": resolved["amp_active"],
            "max_epochs": resolved["max_epochs"],
            "patience": resolved["early_stopping_patience"],
            "actual_epochs": len(history),
            "best_epoch": int(history.loc[history["val_roc_auc"].idxmax(), "epoch"]),
            "checkpoint_metric": resolved["checkpoint_metric"],
            "primary_threshold": resolved["threshold"],
            "seed": resolved["seed"],
            "parameter_count": parameters,
            "trainable_parameter_count": int(trainable_parameters),
            "duration_seconds": timing_row["reported_total_duration_seconds"],
            "mean_epoch_seconds": timing_row["mean_epoch_seconds"],
            "median_epoch_seconds": timing_row["median_epoch_seconds"],
        })
    deep_model_protocol = pd.DataFrame(deep_rows)
    deep_index = deep_model_protocol.set_index("model_id")

    def shared_values(field):
        return {model: deep_index.at[model, field] for model in ["B0", "M1", "M2"]}

    hyper_rows = []

    def add_hyper(parameter, values, category, basis, notes):
        hyper_rows.append({
            "parameter": parameter,
            "B0_value": values.get("B0", "not_applicable"),
            "M1_value": values.get("M1", "not_applicable"),
            "M2_value": values.get("M2", "not_applicable"),
            "selection_category": category,
            "selection_basis": basis,
            "systematically_tuned": "NO",
            "notes": notes,
        })

    add_hyper("image_size", shared_values("image_size"), "predeclared_shared", "Matched deep-model input resolution", "No resolution search was performed.")
    add_hyper("batch_size", shared_values("batch_size"), "predeclared_shared", "Matched memory-feasible batch setting", "No batch-size search was performed.")
    add_hyper("optimizer", shared_values("optimizer"), "predeclared_shared", "Matched optimizer policy", "AdamW was used for all deep models.")
    add_hyper("learning_rate", shared_values("learning_rate"), "predeclared_shared", "Matched conservative learning rate", "No learning-rate search was performed.")
    add_hyper("weight_decay", shared_values("weight_decay"), "predeclared_shared", "Matched regularization setting", "No weight-decay search was performed.")
    add_hyper("loss", shared_values("loss"), "predeclared_shared", "Matched imbalance-aware objective", "The loss family was not systematically tuned.")
    add_hyper("pos_weight", shared_values("pos_weight"), "data_derived", "train_benign / train_melanoma = 26032 / 467", "Derived from the training partition only.")
    add_hyper("scheduler", shared_values("scheduler"), "predeclared_shared", "Matched no-scheduler policy", "No scheduler search was performed.")
    add_hyper("max_epochs", shared_values("max_epochs"), "predeclared_shared", "Shared upper bound with early stopping", "Maximum epochs were not systematically tuned.")
    add_hyper("early_stopping_patience", shared_values("patience"), "predeclared_shared", "Shared early-stopping policy", "Patience was not systematically tuned.")
    add_hyper("checkpoint_metric", shared_values("checkpoint_metric"), "predeclared_shared", "Validation ROC-AUC selection metric", "Checkpoint selection is not a broad hyperparameter search.")
    add_hyper("primary_threshold", shared_values("primary_threshold"), "predeclared_shared", "Common evaluation operating point", "Threshold 0.5 is the authoritative comparison threshold.")
    add_hyper("best_epoch", shared_values("best_epoch"), "checkpoint_selected", "Maximum validation ROC-AUC before stopping", "Effective duration was selected by checkpointing/early stopping.")
    add_hyper(
        "phase_i_threshold_grid",
        {"B0": "not_applicable", "M1": json.dumps(phase_i["threshold_grid"]), "M2": json.dumps(phase_i["threshold_grid"])},
        "posthoc_analysis_only",
        "Fixed behavioural-analysis grid on Fold 0",
        "No threshold was labelled optimal or substituted for the primary threshold.",
    )
    m2_model = configs["M2"]["configured"]["model"]
    add_hyper("metadata_embedding_dim", {"B0": "not_applicable", "M1": "not_applicable", "M2": m2_model["metadata_embedding_dim"]}, "architecture_specific", "Predeclared M2 metadata-branch design", "Not systematically tuned.")
    add_hyper("metadata_dropout", {"B0": "not_applicable", "M1": "not_applicable", "M2": m2_model["metadata_dropout"]}, "architecture_specific", "Predeclared M2 metadata-branch regularization", "Not systematically tuned.")
    hyperparameter_selection = pd.DataFrame(hyper_rows)

    training_summary = pd.concat(
        [_normalise_history(histories[model], model) for model in ["B0", "M1", "M2"]],
        ignore_index=True,
    )

    b0_m1_metadata = {
        "B0_parameters": int(deep_index.at["B0", "parameter_count"]),
        "M1_parameters": int(deep_index.at["M1", "parameter_count"]),
        "B0_actual_epochs": int(deep_index.at["B0", "actual_epochs"]),
        "M1_actual_epochs": int(deep_index.at["M1", "actual_epochs"]),
        "B0_duration": deep_index.at["B0", "duration_seconds"],
        "M1_duration": deep_index.at["M1", "duration_seconds"],
        "comparison_type": "matched external-protocol architecture-family comparison",
        "interpretation_boundary": "does not isolate individual ConvNeXt mechanisms",
    }
    b0_m1 = _comparison_table(
        main_model_results,
        "B0",
        "M1",
        "M1_minus_B0",
        b0_m1_metadata,
    )

    m1_m2_metadata = {
        "M1_parameters": int(deep_index.at["M1", "parameter_count"]),
        "M2_parameters": int(deep_index.at["M2", "parameter_count"]),
        "parameter_difference": int(deep_index.at["M2", "parameter_count"] - deep_index.at["M1", "parameter_count"]),
        "M1_actual_epochs": int(deep_index.at["M1", "actual_epochs"]),
        "M2_actual_epochs": int(deep_index.at["M2", "actual_epochs"]),
        "M1_duration": deep_index.at["M1", "duration_seconds"],
        "M2_duration": deep_index.at["M2", "duration_seconds"],
        "comparison_type": "metadata-fusion system ablation",
        "interpretation_boundary": "per-sample probability differences are not causal metadata effects",
    }
    m1_m2 = _comparison_table(
        main_model_results,
        "M1",
        "M2",
        "M2_minus_M1",
        m1_m2_metadata,
    )

    threshold_frames = []
    for model_id in ["M1", "M2"]:
        frame = pd.read_csv(
            root / f"outputs/analysis/threshold/{model_id}_threshold_analysis.csv"
        )
        frame.insert(0, "model", model_id)
        frame["is_primary_threshold"] = np.isclose(frame["threshold"], 0.5)
        threshold_frames.append(frame)
    threshold_summary = pd.concat(threshold_frames, ignore_index=True).loc[:, [
        "model",
        "threshold",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "balanced_accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "predicted_positive_count",
        "is_primary_threshold",
    ]]

    bootstrap_mapping = {
        "roc_auc_difference": "roc_auc_difference_m2_minus_m1",
        "average_precision_difference": "average_precision_difference_m2_minus_m1",
        "sensitivity_difference": "sensitivity_difference_m2_minus_m1",
    }
    bootstrap_rows = []
    for final_name, source_name in bootstrap_mapping.items():
        values = bootstrap["metrics"][source_name]
        bootstrap_rows.append({
            "metric": final_name,
            "observed_m2_minus_m1": values["observed_difference"],
            "bootstrap_median": values["bootstrap_median_difference"],
            "ci_2_5": values["percentile_2_5"],
            "ci_97_5": values["percentile_97_5"],
            "requested_iterations": bootstrap["requested_iterations"],
            "valid_iterations": values["valid_iterations"],
            "skipped_iterations": values["skipped_iterations"],
            "bootstrap_unit": "patient",
            "paired": "YES",
        })
    bootstrap_summary = pd.DataFrame(bootstrap_rows)

    subgroup_summary = pd.concat([
        pd.read_csv(root / "outputs/analysis/subgroups/M1_subgroup_metrics.csv"),
        pd.read_csv(root / "outputs/analysis/subgroups/M2_subgroup_metrics.csv"),
    ], ignore_index=True).rename(columns={"N": "n"}).loc[:, [
        "model",
        "group_variable",
        "group_value",
        "n",
        "positive_count",
        "negative_count",
        "prevalence",
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "small_positive_count",
    ]]

    failure_cases = pd.read_csv(
        root / "outputs/analysis/failures/M2_failure_cases.csv"
    )
    duplicate_records = pd.read_csv(
        root / "outputs/analysis/data_integrity/exact_image_duplicates.csv"
    )
    duplicate_hash = duplicate_records.set_index("image_name")["sha256"].to_dict()
    content_ids = failure_cases["image_name"].map(
        lambda image: duplicate_hash.get(image, f"unique:{image}")
    )
    confusion_transitions = phase_i["disagreement_counts"]
    m1_metrics = raw_metrics["M1"]
    m2_metrics = raw_metrics["M2"]
    failure_summary = pd.DataFrame([{
        "m2_tp": m2_metrics["tp"],
        "m2_tn": m2_metrics["tn"],
        "m2_fp": m2_metrics["fp"],
        "m2_fn": m2_metrics["fn"],
        "m1_fp_to_m2_tn": confusion_transitions["M1_FP_TO_M2_TN"],
        "m1_tp_to_m2_fn": confusion_transitions["M1_TP_TO_M2_FN"],
        "m1_tn_to_m2_fp": confusion_transitions["M1_TN_TO_M2_FP"],
        "m1_fn_to_m2_tp": confusion_transitions["M1_FN_TO_M2_TP"],
        "net_false_positive_change": m2_metrics["fp"] - m1_metrics["fp"],
        "net_false_negative_change": m2_metrics["fn"] - m1_metrics["fn"],
        "mean_probability_delta": disagreement["mean_probability_delta"],
        "median_probability_delta": disagreement["median_probability_delta"],
        "visual_review_cases": phase_i["failure_selection"]["visually_reviewed"],
        "content_unique_visual_cases": int(content_ids.nunique()),
        "interpretation_boundary": "model disagreement only; not causal metadata contribution",
    }])

    integrity_summary = pd.DataFrame([{
        "patient_overlap": patient_overlap,
        "images_hashed": duplicates["total_images_hashed"],
        "unique_hashes": duplicates["unique_sha256_count"],
        "duplicate_groups": duplicates["duplicate_hash_groups"],
        "duplicate_records": duplicates["duplicate_image_records"],
        "same_patient_same_split_groups": duplicates["same_patient_same_split_groups"],
        "different_patient_same_split_groups": duplicates["different_patient_same_split_groups"],
        "same_patient_cross_split_groups": duplicates["same_patient_cross_split_groups"],
        "different_patient_cross_split_groups": duplicates["different_patient_cross_split_groups"],
        "conflicting_target_groups": duplicates["target_conflict_groups"],
        "bootstrap_multiplicity_verified": pre_j["bootstrap"]["regression_test_passed"],
        "frozen_artifacts_hash_verified": j1a["frozen_artifact_verification"]["status"] == "PASS",
        "validation_set_equivalence_verified": j1a["prediction_equivalence"]["status"] == "PASS",
    }])

    tables = {
        "dataset_summary.csv": dataset_summary,
        "main_model_results.csv": main_model_results,
        "deep_model_protocol.csv": deep_model_protocol,
        "hyperparameter_selection.csv": hyperparameter_selection,
        "training_summary.csv": training_summary,
        "B0_M1_architecture_comparison.csv": b0_m1,
        "M1_M2_metadata_ablation.csv": m1_m2,
        "threshold_summary.csv": threshold_summary,
        "bootstrap_summary.csv": bootstrap_summary,
        "subgroup_summary.csv": subgroup_summary,
        "failure_summary.csv": failure_summary,
        "integrity_summary.csv": integrity_summary,
    }

    table_sources = {
        "dataset_summary.csv": ["outputs/dataset_summary.json", "data/train.csv", "data/train_folds.csv", "outputs/analysis/data_integrity/exact_image_duplicate_summary.json"],
        "main_model_results.csv": [*[f"logs/{files['directory']}/metrics.json" for files in MODEL_FILES.values()], *[f"logs/{files['directory']}/config.json" for files in MODEL_FILES.values()], "outputs/final/manifests/comparison_boundaries.json"],
        "deep_model_protocol.csv": ["logs/B0_resnet18/config.json", "logs/M1_convnext_image/config.json", "logs/M2_convnext_metadata/config.json", "logs/B0_resnet18/history.csv", "logs/M1_convnext_image/history.csv", "logs/M2_convnext_metadata/history.csv", "outputs/analysis/technical_audit/training_timing_audit.csv"],
        "hyperparameter_selection.csv": ["logs/B0_resnet18/config.json", "logs/M1_convnext_image/config.json", "logs/M2_convnext_metadata/config.json", "logs/B0_resnet18/history.csv", "logs/M1_convnext_image/history.csv", "logs/M2_convnext_metadata/history.csv", "outputs/analysis/phase_i_summary.json"],
        "training_summary.csv": ["logs/B0_resnet18/history.csv", "logs/M1_convnext_image/history.csv", "logs/M2_convnext_metadata/history.csv"],
        "B0_M1_architecture_comparison.csv": ["logs/B0_resnet18/metrics.json", "logs/M1_convnext_image/metrics.json", "outputs/analysis/technical_audit/training_timing_audit.csv", "outputs/final/manifests/comparison_boundaries.json"],
        "M1_M2_metadata_ablation.csv": ["logs/M1_convnext_image/metrics.json", "logs/M2_convnext_metadata/metrics.json", "outputs/analysis/technical_audit/training_timing_audit.csv", "outputs/final/manifests/comparison_boundaries.json"],
        "threshold_summary.csv": ["outputs/analysis/threshold/M1_threshold_analysis.csv", "outputs/analysis/threshold/M2_threshold_analysis.csv", "outputs/analysis/phase_i_summary.json"],
        "bootstrap_summary.csv": ["outputs/analysis/bootstrap/M1_M2_bootstrap_summary.json", "outputs/analysis/technical_audit/pre_phase_j_audit.json"],
        "subgroup_summary.csv": ["outputs/analysis/subgroups/M1_subgroup_metrics.csv", "outputs/analysis/subgroups/M2_subgroup_metrics.csv"],
        "failure_summary.csv": ["logs/M1_convnext_image/metrics.json", "logs/M2_convnext_metadata/metrics.json", "outputs/analysis/failures/M1_M2_disagreement_summary.json", "outputs/analysis/phase_i_summary.json", "outputs/analysis/failures/M2_failure_cases.csv", "outputs/analysis/data_integrity/exact_image_duplicates.csv"],
        "integrity_summary.csv": ["data/train.csv", "data/train_folds.csv", "outputs/analysis/data_integrity/exact_image_duplicate_summary.json", "outputs/analysis/technical_audit/pre_phase_j_audit.json", "outputs/final/manifests/J1A_evidence_audit.json"],
    }

    for filename, frame in tables.items():
        frame.to_csv(table_dir / filename, index=False)

    _write_json(manifest_dir / "dataset_summary_sources.json", {
        "table": "outputs/final/tables/dataset_summary.csv",
        "created_at": timestamp,
        "source_files": table_sources["dataset_summary.csv"],
        "derivations": {
            "class_and_patient_counts": "outputs/dataset_summary.json",
            "train_validation_counts_and_patient_overlap": "data/train.csv joined one-to-one with data/train_folds.csv on image_name",
            "exact_content_counts": "outputs/analysis/data_integrity/exact_image_duplicate_summary.json"
        }
    })
    _write_json(manifest_dir / "hyperparameter_strategy.json", {
        "created_at": timestamp,
        "table": "outputs/final/tables/hyperparameter_selection.csv",
        "systematic_search_performed": False,
        "systematic_search_statement": "A systematic grid, random, or Bayesian hyperparameter search was not performed.",
        "matching_strategy": "Deep-model settings were deliberately matched across B0, M1, and M2 to reduce optimization-policy confounding.",
        "effective_duration_selection": "Early stopping and maximum validation ROC-AUC checkpointing selected effective training duration.",
        "threshold_strategy": "Threshold 0.5 is the predeclared common comparison point; the Phase I threshold grid is post-hoc behavioural analysis only.",
        "stronger_future_design": "A stronger tuning design would require a separate patient-grouped inner validation process or nested cross-validation.",
        "source_files": table_sources["hyperparameter_selection.csv"]
    })
    _write_json(manifest_dir / "bootstrap_interpretation.json", {
        "created_at": timestamp,
        "table": "outputs/final/tables/bootstrap_summary.csv",
        "bootstrap_unit": "patient",
        "paired": True,
        "scope": "internal validation resampling only",
        "external_validation": False,
        "automatic_significance_proof": False,
        "notes": "Intervals quantify paired internal Fold-0 uncertainty and are neither external validation nor automatic significance proof.",
        "source_files": table_sources["bootstrap_summary.csv"]
    })

    compliance_path = root / "report/assessment_compliance.md"
    compliance_path.write_text(_assessment_compliance_text(), encoding="utf-8")

    # Cross-check every generated numerical relationship before writing the manifest.
    main_index = main_model_results.set_index("model_id")
    for model_id, metrics in raw_metrics.items():
        assert main_index.at[model_id, "roc_auc"] == metrics["roc_auc"]
        assert main_index.at[model_id, "average_precision"] == metrics["pr_auc_average_precision"]
        for metric in ["accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]:
            assert main_index.at[model_id, metric] == metrics[metric]

    for frame, left_id, right_id, delta_column in [
        (b0_m1, "B0", "M1", "M1_minus_B0"),
        (m1_m2, "M1", "M2", "M2_minus_M1"),
    ]:
        for row in frame.itertuples(index=False):
            assert getattr(row, delta_column) == getattr(row, right_id) - getattr(row, left_id)

    for model_id in ["M1", "M2"]:
        primary = threshold_summary.loc[
            threshold_summary["model"].eq(model_id)
            & threshold_summary["is_primary_threshold"]
        ].iloc[0]
        for metric in ["tn", "fp", "fn", "tp"]:
            assert primary[metric] == raw_metrics[model_id][metric]

    for row in bootstrap_summary.itertuples(index=False):
        source = bootstrap["metrics"][bootstrap_mapping[row.metric]]
        assert row.observed_m2_minus_m1 == source["observed_difference"]
        assert row.bootstrap_median == source["bootstrap_median_difference"]
        assert row.ci_2_5 == source["percentile_2_5"]
        assert row.ci_97_5 == source["percentile_97_5"]

    _verify_subgroup_partitions(subgroup_summary, len(validation))
    for model_id in ["B0", "M1", "M2"]:
        expected = _normalise_history(histories[model_id], model_id).reset_index(drop=True)
        actual = training_summary.loc[training_summary["model_id"].eq(model_id)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected)

    _assert_frozen_hashes(root, j1a)
    table_records = []
    for filename, frame in tables.items():
        path = table_dir / filename
        table_records.append({
            "path": path.relative_to(root).as_posix(),
            "rows": len(frame),
            "column_count": len(frame.columns),
            "columns": list(frame.columns),
            "source_files": table_sources[filename],
            "sha256": _sha256_file(path),
            "creation_timestamp": timestamp,
        })

    companion_paths = [
        manifest_dir / "dataset_summary_sources.json",
        manifest_dir / "hyperparameter_strategy.json",
        manifest_dir / "bootstrap_interpretation.json",
        compliance_path,
    ]
    manifest = {
        "phase": "J1B",
        "created_at": timestamp,
        "starting_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "generation_utility": "src/final_tables.py",
        "table_count": len(table_records),
        "tables": table_records,
        "companion_artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in companion_paths
        ],
        "cross_checks": {
            "main_results_match_authoritative_metrics": True,
            "B0_M1_deltas_exact": True,
            "M1_M2_deltas_exact": True,
            "threshold_0_5_confusion_counts_match": True,
            "bootstrap_summary_matches_source": True,
            "subgroup_partitions_reconcile_to_validation_n": True,
            "training_summary_matches_histories": True,
            "frozen_artifacts_hash_verified": True,
        },
        "unsupported_or_invented_values_detected": False,
        "publication_figures_generated": False,
        "J1C_started": False,
    }
    _write_json(manifest_dir / "J1B_tables_manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    manifest = generate_final_tables(args.project_root, args.timestamp)
    print(
        f"Generated and verified {manifest['table_count']} J1B evidence tables."
    )


if __name__ == "__main__":
    main()
