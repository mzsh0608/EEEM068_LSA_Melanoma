"""Build the frozen-evidence final results-analysis notebook."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


TABLE_NAMES = (
    "dataset_summary",
    "main_model_results",
    "deep_model_protocol",
    "hyperparameter_selection",
    "training_summary",
    "B0_M1_architecture_comparison",
    "M1_M2_metadata_ablation",
    "threshold_summary",
    "bootstrap_summary",
    "subgroup_summary",
    "failure_summary",
    "integrity_summary",
)

SECTION_HEADINGS = (
    "# Final Results Analysis — EEEM068 LSA Melanoma Classification",
    "## 1. Dataset and validation summary",
    "## 2. Main model hierarchy",
    "## 3. H0 to H1 imbalance effect",
    "## 4. B0 to M1 controlled architecture comparison",
    "## 5. M1 to M2 metadata ablation",
    "## 6. Training behaviour",
    "## 7. Hyperparameter selection",
    "## 8. Threshold behaviour",
    "## 9. Bootstrap uncertainty",
    "## 10. M1/M2 disagreement summary",
    "## 11. Subgroup summary",
    "## 12. Failure case summary",
    "## 13. Grad-CAM overview",
    "## 14. Data integrity summary",
    "## 15. Submission artifact scope",
)


def _source(text: str) -> str:
    return dedent(text).strip()


def build_notebook(output_path: Path) -> Path:
    """Create the final notebook without changing experiment artifacts."""
    output_path = Path(output_path)

    cells = [
        new_markdown_cell(_source(
            """
            # Final Results Analysis — EEEM068 LSA Melanoma Classification

            All core models and scientific artifacts are frozen. This notebook performs no training; it reads the authoritative evidence bundle for final submission analysis. Fold 0 is the fixed patient-aware validation fold. AP denotes Average Precision.
            """
        )),
        new_code_cell(_source(
            f"""
            from pathlib import Path
            import hashlib
            import json
            import subprocess

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import Markdown, display
            from PIL import Image as PILImage

            pd.set_option("display.max_columns", 30)
            pd.set_option("display.width", 160)
            plt.rcParams.update({{"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25}})

            def find_repo_root(start: Path) -> Path:
                for candidate in (start, *start.parents):
                    if (candidate / "outputs" / "final" / "tables").is_dir():
                        return candidate
                raise FileNotFoundError("Could not locate the repository root")

            ROOT = find_repo_root(Path.cwd().resolve())
            TABLE_DIR = ROOT / "outputs" / "final" / "tables"
            MANIFEST_DIR = ROOT / "outputs" / "final" / "manifests"

            table_names = {TABLE_NAMES!r}
            tables = {{name: pd.read_csv(TABLE_DIR / f"{{name}}.csv") for name in table_names}}

            def read_json(path: Path):
                return json.loads(path.read_text(encoding="utf-8"))

            j1_complete = read_json(MANIFEST_DIR / "J1_COMPLETE.json")
            j1_table_manifest = read_json(MANIFEST_DIR / "J1B_tables_manifest.json")
            authoritative_sources = read_json(MANIFEST_DIR / "authoritative_sources.json")
            comparison_boundaries = read_json(MANIFEST_DIR / "comparison_boundaries.json")
            hyperparameter_strategy = read_json(MANIFEST_DIR / "hyperparameter_strategy.json")
            j1_handoff = (ROOT / "outputs" / "final" / "J1_HANDOFF.md").read_text(encoding="utf-8")

            boundary_names = {{item["comparison"] for item in comparison_boundaries["boundaries"]}}
            required_boundaries = {{"H0_to_H1", "H0_H1_to_B0", "B0_to_M1", "M1_to_M2"}}
            source_categories = {{item["claim_category"] for item in authoritative_sources["sources"]}}
            assert required_boundaries.issubset(boundary_names)
            assert {{"precision_recall_metric_definition", "validation_set_definition"}}.issubset(source_categories)
            assert "Average Precision (AP)" in j1_handoff
            assert "fixed patient-aware validation fold" in j1_handoff

            git_head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            assert j1_complete["status"] == "COMPLETE"
            display(pd.DataFrame({{
                "Item": ["Repository", "Current Git HEAD", "J1 status", "Tables loaded"],
                "Value": [ROOT.name, git_head, j1_complete["status"], len(tables)],
            }}))
            """
        )),
        new_markdown_cell(_source(
            """
            ## 1. Dataset and validation summary

            Evidence: `dataset_summary.csv` and `integrity_summary.csv`. The compact view reports the frozen dataset, fixed Fold-0 split, patient separation, and exact-content audit.
            """
        )),
        new_code_cell(_source(
            """
            dataset = tables["dataset_summary"].iloc[0]
            integrity = tables["integrity_summary"].iloc[0]

            dataset_view = pd.DataFrame({
                "Measure": [
                    "Total images", "Benign", "Melanoma", "Melanoma prevalence", "Patients",
                    "Training N", "Validation N", "Training positives", "Validation positives",
                    "Patient overlap", "Exact duplicate groups", "Cross-split exact duplicates",
                ],
                "Value": [
                    int(dataset.total_images), int(dataset.benign_images), int(dataset.melanoma_images),
                    f"{float(dataset.melanoma_prevalence):.3%}", int(dataset.unique_patients),
                    int(dataset.train_images), int(dataset.validation_images), int(dataset.train_melanoma),
                    int(dataset.validation_melanoma), int(dataset.patient_overlap),
                    int(dataset.exact_duplicate_groups), int(dataset.cross_split_exact_duplicates),
                ],
            })
            display(dataset_view)
            """
        )),
        new_markdown_cell(_source(
            """
            The dataset is severely class-imbalanced. The split is patient-aware, and the exact-content audit found no cross-split duplicate leakage.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 2. Main model hierarchy

            Evidence: `main_model_results.csv`. Values are rounded only in the display below; the loaded table retains full precision.
            """
        )),
        new_code_cell(_source(
            """
            main_results = tables["main_model_results"].copy()
            expected_models = ["H0", "H1", "B0", "M1", "M2"]
            assert main_results["model_id"].tolist() == expected_models

            hierarchy = main_results[[
                "model_id", "input_representation", "metadata_used", "imbalance_strategy",
                "roc_auc", "average_precision", "sensitivity", "specificity", "f1", "accuracy",
            ]].rename(columns={
                "model_id": "Model", "input_representation": "Representation",
                "metadata_used": "Metadata", "imbalance_strategy": "Imbalance strategy",
                "roc_auc": "ROC-AUC", "average_precision": "AP",
                "sensitivity": "Sensitivity", "specificity": "Specificity",
                "f1": "F1", "accuracy": "Accuracy",
            })
            display(hierarchy.round({
                "ROC-AUC": 4, "AP": 4, "Sensitivity": 4,
                "Specificity": 4, "F1": 4, "Accuracy": 4,
            }))
            """
        )),
        new_markdown_cell(_source(
            """
            Interpretation boundaries:

            - **H0 to H1:** historical Logistic Regression weighting comparison.
            - **H0/H1 to B0:** system-level historical progression, not an isolated architecture effect.
            - **B0 to M1:** matched training and evaluation protocol architecture-family comparison.
            - **M1 to M2:** metadata-fusion system ablation; individual metadata fields are not assigned causal effects.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 3. H0 to H1 imbalance effect

            Evidence: threshold-0.5 metrics and confusion counts in `main_model_results.csv`.
            """
        )),
        new_code_cell(_source(
            """
            imbalance_metrics = [
                "accuracy", "balanced_accuracy", "precision", "sensitivity", "specificity",
                "f1", "roc_auc", "average_precision", "tn", "fp", "fn", "tp",
            ]
            h0_h1 = (
                main_results.loc[main_results.model_id.isin(["H0", "H1"]), ["model_id", *imbalance_metrics]]
                .set_index("model_id")
                .T
                .rename(index={"roc_auc": "ROC-AUC", "average_precision": "AP"})
            )
            display(h0_h1.round(4))
            """
        )),
        new_markdown_cell(_source(
            """
            H0's high accuracy is majority-class dominated. H1 weighting increases melanoma sensitivity, while ranking metrics do not necessarily improve. This comparison shows why accuracy alone is misleading under severe imbalance; it does not isolate a deep-architecture effect.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 4. B0 to M1 controlled architecture comparison

            Evidence: `B0_M1_architecture_comparison.csv` and `deep_model_protocol.csv`.
            """
        )),
        new_code_cell(_source(
            """
            architecture_comparison = tables["B0_M1_architecture_comparison"]
            display(
                architecture_comparison[["metric", "B0", "M1", "M1_minus_B0"]]
                .rename(columns={"M1_minus_B0": "Delta (M1 - B0)"})
                .round(4)
            )

            protocol = tables["deep_model_protocol"].set_index("model_id")
            protocol_items = [
                ("Split", "Fixed Fold-0 validation"),
                ("Image size", "image_size"), ("Loss", "loss"), ("pos_weight", "pos_weight"),
                ("Optimizer", "optimizer"), ("Learning rate", "learning_rate"),
                ("Weight decay", "weight_decay"), ("Scheduler", "scheduler"), ("AMP", "amp"),
                ("Maximum epochs", "max_epochs"), ("Patience", "patience"),
                ("Checkpoint metric", "checkpoint_metric"), ("Primary threshold", "primary_threshold"),
                ("Parameter count", "parameter_count"), ("Actual epochs", "actual_epochs"),
            ]
            protocol_view = []
            for label, source in protocol_items:
                if label == "Split":
                    b0_value = m1_value = source
                else:
                    b0_value, m1_value = protocol.loc["B0", source], protocol.loc["M1", source]
                protocol_view.append({"Protocol item": label, "B0": b0_value, "M1": m1_value})
            display(pd.DataFrame(protocol_view))
            """
        )),
        new_markdown_cell(_source(
            """
            M1 improved the observed ROC-AUC, AP, and sensitivity, while specificity decreased at threshold 0.5. Under the documented matched external protocol, architecture family is the main externally controlled change; the difference cannot be assigned to one ConvNeXt mechanism.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 5. M1 to M2 metadata ablation

            Evidence: `M1_M2_metadata_ablation.csv` and `deep_model_protocol.csv`.
            """
        )),
        new_code_cell(_source(
            """
            metadata_ablation = tables["M1_M2_metadata_ablation"]
            display(
                metadata_ablation[["metric", "M1", "M2", "M2_minus_M1"]]
                .rename(columns={"M2_minus_M1": "Delta (M2 - M1)"})
                .round(4)
            )

            ablation_protocol = pd.DataFrame({
                "Measure": ["Parameters", "Parameter difference", "Actual epochs", "Best epoch", "Duration (s)"],
                "M1": [
                    int(protocol.loc["M1", "parameter_count"]), 0,
                    int(protocol.loc["M1", "actual_epochs"]), int(protocol.loc["M1", "best_epoch"]),
                    float(protocol.loc["M1", "duration_seconds"]),
                ],
                "M2": [
                    int(protocol.loc["M2", "parameter_count"]),
                    int(protocol.loc["M2", "parameter_count"] - protocol.loc["M1", "parameter_count"]),
                    int(protocol.loc["M2", "actual_epochs"]), int(protocol.loc["M2", "best_epoch"]),
                    float(protocol.loc["M2", "duration_seconds"]),
                ],
            })
            display(ablation_protocol.round(3))
            """
        )),
        new_markdown_cell(_source(
            """
            Observed ROC-AUC and AP changed only slightly. At threshold 0.5, M2 increased specificity and reduced false positives, but reduced sensitivity and increased false negatives. Metadata did not provide a clear ranking advantage on this fold. Per-sample M1/M2 differences are descriptive model disagreements, not causal metadata effects.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 6. Training behaviour

            Evidence: frozen epoch records in `training_summary.csv`. The checkpoint marker denotes the maximum validation ROC-AUC epoch; no training ranking metric is inferred.
            """
        )),
        new_code_cell(_source(
            """
            training = tables["training_summary"].copy()
            best_mask = training["is_best_roc_auc_epoch"].astype(str).str.lower().eq("true")
            best_rows = training.loc[best_mask, [
                "model_id", "epoch", "train_loss", "val_loss", "val_roc_auc", "val_average_precision",
            ]]
            display(best_rows.rename(columns={
                "model_id": "Model", "epoch": "Best ROC-AUC epoch", "train_loss": "Train loss",
                "val_loss": "Validation loss", "val_roc_auc": "Validation ROC-AUC",
                "val_average_precision": "Validation AP",
            }).round(4))

            colors = {"B0": "#2f6b9a", "M1": "#c44e52", "M2": "#3a8f5c"}
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            for model_id in ["B0", "M1", "M2"]:
                model_history = training.loc[training.model_id.eq(model_id)].sort_values("epoch")
                best = model_history.loc[
                    model_history["is_best_roc_auc_epoch"].astype(str).str.lower().eq("true")
                ].iloc[0]
                color = colors[model_id]
                axes[0].plot(model_history.epoch, model_history.train_loss, "--", color=color, label=f"{model_id} train")
                axes[0].plot(model_history.epoch, model_history.val_loss, "-", color=color, label=f"{model_id} validation")
                axes[0].scatter(best.epoch, best.val_loss, color=color, edgecolor="black", zorder=3)
                axes[1].plot(model_history.epoch, model_history.val_roc_auc, color=color, marker="o", label=model_id)
                axes[1].scatter(best.epoch, best.val_roc_auc, color=color, edgecolor="black", s=65, zorder=3)
                axes[2].plot(model_history.epoch, model_history.val_average_precision, color=color, marker="o", label=model_id)
                axes[2].scatter(best.epoch, best.val_average_precision, color=color, edgecolor="black", s=65, zorder=3)

            axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
            axes[1].set(title="Validation ROC-AUC", xlabel="Epoch", ylabel="ROC-AUC")
            axes[2].set(title="Validation Average Precision", xlabel="Epoch", ylabel="AP")
            for axis in axes:
                axis.legend(fontsize=8)
                axis.set_xticks(sorted(training.epoch.unique()))
            fig.suptitle("Frozen Fold-0 training histories; outlined markers identify checkpoint epochs", fontsize=11)
            fig.tight_layout()
            plt.show()
            """
        )),
        new_markdown_cell(_source(
            """
            B0 and M1 reached their best validation ROC-AUC at epoch 4; M2 reached it at epoch 8. B0 and M1 show lower ROC-AUC after the checkpoint epoch. M2 also varies after epoch 8, with AP peaking one epoch later than the ROC-AUC-selected checkpoint.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 7. Hyperparameter selection

            Evidence: `hyperparameter_selection.csv` and `hyperparameter_strategy.json`. This section documents experimental control and its validation-design boundary.
            """
        )),
        new_code_cell(_source(
            """
            hyperparameters = tables["hyperparameter_selection"][[
                "parameter", "B0_value", "M1_value", "M2_value", "selection_category", "systematically_tuned",
            ]].rename(columns={
                "parameter": "Parameter", "B0_value": "B0", "M1_value": "M1", "M2_value": "M2",
                "selection_category": "Selection category", "systematically_tuned": "Systematically tuned?",
            })
            display(hyperparameters)
            display(pd.DataFrame({
                "Evidence statement": [
                    hyperparameter_strategy["systematic_search_statement"],
                    hyperparameter_strategy["matching_strategy"],
                    hyperparameter_strategy["effective_duration_selection"],
                    hyperparameter_strategy["threshold_strategy"],
                    hyperparameter_strategy["stronger_future_design"],
                ]
            }))
            """
        )),
        new_markdown_cell(_source(
            """
            No systematic grid search, random search, or Bayesian tuning was performed. Common deep-model settings were deliberately matched; `pos_weight` was derived from training class counts. Early stopping and validation ROC-AUC checkpointing selected effective duration. The threshold grid is post-hoc behavioural analysis. A stronger future tuning design would use patient-grouped inner validation or nested cross-validation.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 8. Threshold behaviour

            Evidence: the fixed 0.1 to 0.9 grid in `threshold_summary.csv`. Threshold 0.5 is the predeclared common primary threshold; the grid is descriptive and post-hoc.
            """
        )),
        new_code_cell(_source(
            """
            threshold = tables["threshold_summary"].copy()
            threshold_metrics = ["sensitivity", "specificity", "precision", "f1"]
            threshold_view = pd.concat({
                model_id: threshold.loc[threshold.model.eq(model_id)].set_index("threshold")[threshold_metrics]
                for model_id in ["M1", "M2"]
            }, axis=1)
            display(threshold_view.round(4))

            labels = {
                "sensitivity": "Sensitivity", "specificity": "Specificity",
                "precision": "Precision", "f1": "F1",
            }
            fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
            for axis, metric in zip(axes.flat, threshold_metrics):
                for model_id in ["M1", "M2"]:
                    model_thresholds = threshold.loc[threshold.model.eq(model_id)].sort_values("threshold")
                    axis.plot(
                        model_thresholds.threshold, model_thresholds[metric], marker="o",
                        color=colors[model_id], label=model_id,
                    )
                axis.axvline(0.5, color="black", linestyle=":", linewidth=1.4, label="Primary threshold 0.5")
                axis.set(title=labels[metric], xlabel="Threshold", ylabel=labels[metric], ylim=(0, 1.03))
                axis.legend(fontsize=8)
            fig.suptitle("Fold-0 threshold behaviour", fontsize=11)
            fig.tight_layout()
            plt.show()
            """
        )),
        new_markdown_cell(_source(
            """
            M1 is generally more sensitive at equal numerical thresholds; M2 is generally more conservative and specific. These threshold-dependent metrics describe a different operating-point view from ROC-AUC and AP.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 9. Bootstrap uncertainty

            Evidence: `bootstrap_summary.csv`, derived from paired patient-level resampling of the fixed validation fold.
            """
        )),
        new_code_cell(_source(
            """
            bootstrap = tables["bootstrap_summary"][[
                "metric", "observed_m2_minus_m1", "bootstrap_median", "ci_2_5", "ci_97_5",
            ]].rename(columns={
                "metric": "Metric", "observed_m2_minus_m1": "Observed difference (M2 - M1)",
                "bootstrap_median": "Bootstrap median", "ci_2_5": "2.5 percentile", "ci_97_5": "97.5 percentile",
            })
            display(bootstrap.round(4))
            """
        )),
        new_markdown_cell(_source(
            """
            This is paired patient-level internal bootstrap uncertainty, not external validation. The ROC-AUC and AP difference intervals include zero. The frozen sensitivity-difference interval remains below zero.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 10. M1/M2 disagreement summary

            Evidence: transition and net-error counts in `failure_summary.csv`.
            """
        )),
        new_code_cell(_source(
            """
            failure = tables["failure_summary"].iloc[0]
            disagreement_view = pd.DataFrame({
                "Transition": ["M1 FP to M2 TN", "M1 TP to M2 FN", "M1 TN to M2 FP", "M1 FN to M2 TP"],
                "Count": [
                    int(failure.m1_fp_to_m2_tn), int(failure.m1_tp_to_m2_fn),
                    int(failure.m1_tn_to_m2_fp), int(failure.m1_fn_to_m2_tp),
                ],
            })
            net_view = pd.DataFrame({
                "Measure": ["Net false-positive change (M2 - M1)", "Net false-negative change (M2 - M1)"],
                "Value": [int(failure.net_false_positive_change), int(failure.net_false_negative_change)],
            })
            display(disagreement_view)
            display(net_view)
            """
        )),
        new_markdown_cell(_source(
            """
            M2 removed many M1 false positives but also lost some M1 true positives, explaining the threshold-0.5 trade-off. These transitions do not show that age, sex, or anatomical site individually caused a prediction change.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 11. Subgroup summary

            Evidence: `subgroup_summary.csv`. Sex, age, and anatomical-site views report support alongside performance so sparse-positive groups remain visible.
            """
        )),
        new_code_cell(_source(
            """
            subgroup = tables["subgroup_summary"].copy()
            subgroup_columns = [
                "model", "group_value", "n", "positive_count", "sensitivity", "specificity",
                "roc_auc", "average_precision", "small_positive_count",
            ]
            for group_variable, label in [("sex", "Sex"), ("age", "Age"), ("site", "Anatomical site")]:
                display(Markdown(f"**{label}**"))
                view = subgroup.loc[subgroup.group_variable.eq(group_variable), subgroup_columns].rename(columns={
                    "model": "Model", "group_value": "Group", "n": "N", "positive_count": "Positive count",
                    "sensitivity": "Sensitivity", "specificity": "Specificity", "roc_auc": "ROC-AUC",
                    "average_precision": "AP", "small_positive_count": "Small positive count",
                })
                display(view.round({"Sensitivity": 4, "Specificity": 4, "ROC-AUC": 4, "AP": 4}))
            """
        )),
        new_markdown_cell(_source(
            """
            Subgroup estimates can be unstable. Groups with very few melanoma cases should not support strong bias claims; this is exploratory internal-validation analysis.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 12. Failure case summary

            Evidence: `failure_summary.csv` plus the frozen manual review record `outputs/analysis/failures/M2_failure_review.csv`.
            """
        )),
        new_code_cell(_source(
            """
            failure_review_path = ROOT / "outputs" / "analysis" / "failures" / "M2_failure_review.csv"
            failure_review = pd.read_csv(failure_review_path)
            reviewed_counts = (
                failure_review.groupby("category", sort=False).size().reindex(["FN", "FP", "TP", "TN"])
                .rename("Reviewed cases").reset_index().rename(columns={"category": "Category"})
            )

            artifact_columns = {
                "hair_visible": "Hair visible",
                "ruler_or_marker_visible": "Ruler or marker visible",
                "dark_border_visible": "Dark border visible",
                "illumination_issue": "Illumination issue",
                "low_contrast": "Low contrast",
                "lesion_near_edge": "Lesion near edge",
                "blur_or_quality_issue": "Blur or quality issue",
            }
            artifact_rows = []
            for column, label in artifact_columns.items():
                counts = failure_review[column].astype(str).str.lower().value_counts()
                artifact_rows.append({
                    "Observed feature": label,
                    "Yes": int(counts.get("yes", 0)),
                    "No": int(counts.get("no", 0)),
                    "Unclear": int(counts.get("unclear", 0)),
                })

            display(reviewed_counts)
            display(pd.DataFrame(artifact_rows))
            display(pd.DataFrame({
                "Review integrity measure": ["Selected cases", "Content-unique selected cases"],
                "Count": [int(failure.visual_review_cases), int(failure.content_unique_visual_cases)],
            }))
            """
        )),
        new_markdown_cell(_source(
            """
            These observations are descriptive; no visual artifact was established as a causal failure mechanism. The qualitative selection contains one record per exact-content identity after the duplicate-content issue was corrected.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 13. Grad-CAM overview

            Evidence: the frozen Phase I summary and figure. Grad-CAM is displayed, not recomputed.
            """
        )),
        new_code_cell(_source(
            """
            gradcam_summary_path = ROOT / "outputs" / "analysis" / "gradcam" / "M2_gradcam_summary.json"
            gradcam_summary = read_json(gradcam_summary_path)
            gradcam_view = pd.DataFrame({
                "Item": ["Target", "Model", "Metadata during forward", "Interpretation", "Limitation"],
                "Value": [
                    "Raw melanoma logit", "M2", bool(gradcam_summary["metadata_included_during_forward"]),
                    "Image-branch positive-logit attribution conditioned on metadata",
                    "Does not explain metadata contribution or causal reasoning",
                ],
            })
            display(gradcam_view)

            gradcam_figure = ROOT / gradcam_summary["figure"].replace("\\\\", "/")
            assert gradcam_figure.is_file()
            with PILImage.open(gradcam_figure) as source_image:
                gradcam_image = source_image.convert("RGB")
                gradcam_image.thumbnail((1200, 1200))
                display(gradcam_image)
            """
        )),
        new_markdown_cell(_source(
            """
            The attribution target is the raw melanoma logit in M2, with metadata supplied during the forward pass. The result is image-branch positive-logit attribution conditioned on metadata; it does not explain the metadata contribution or causal reasoning.
            """
        )),
        new_markdown_cell(_source(
            """
            ## 14. Data integrity summary

            Evidence: `integrity_summary.csv`, `J1B_tables_manifest.json`, and `J1_COMPLETE.json`. The runtime checks below verify current bytes against the frozen SHA-256 records.
            """
        )),
        new_code_cell(_source(
            """
            integrity_view = pd.DataFrame({
                "Measure": [
                    "Patient overlap", "Images hashed", "Unique hashes", "Duplicate groups",
                    "Cross-split duplicate groups", "Conflicting-target duplicate groups",
                    "Bootstrap multiplicity verified", "Frozen artifact verification",
                ],
                "Value": [
                    int(integrity.patient_overlap), int(integrity.images_hashed), int(integrity.unique_hashes),
                    int(integrity.duplicate_groups),
                    int(integrity.same_patient_cross_split_groups + integrity.different_patient_cross_split_groups),
                    int(integrity.conflicting_target_groups), bool(integrity.bootstrap_multiplicity_verified),
                    bool(integrity.frozen_artifacts_hash_verified),
                ],
            })
            display(integrity_view)

            def sha256_file(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()

            frozen_checks = []
            for record in j1_complete["frozen_hash_verification"]["artifacts"]:
                path = ROOT / record["path"]
                frozen_checks.append(sha256_file(path) == record["sha256"])

            table_checks = []
            for record in j1_table_manifest["tables"]:
                path = ROOT / record["path"]
                table_checks.append(sha256_file(path) == record["sha256"])

            assert all(frozen_checks), "A frozen scientific artifact hash changed"
            assert all(table_checks), "A J1 authoritative table hash changed"
            display(pd.DataFrame({
                "Runtime verification": ["Frozen scientific artifacts", "J1 authoritative tables"],
                "Files checked": [len(frozen_checks), len(table_checks)],
                "Status": ["PASS", "PASS"],
            }))
            """
        )),
        new_markdown_cell(_source(
            """
            ## 15. Submission artifact scope

            This notebook creates no derivative data exports or publication figures. All displayed tables and plots are notebook outputs computed in memory from frozen evidence.
            """
        )),
        new_code_cell(_source(
            """
            display(pd.DataFrame({
                "Notebook artifact": ["Tracked executed notebook"],
                "Derivative exports": ["None"],
            }))
            """
        )),
    ]

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("notebooks/02_results_analysis.ipynb"),
        help="Repository-relative notebook output path.",
    )
    args = parser.parse_args()
    path = build_notebook(args.output)
    print(path)


if __name__ == "__main__":
    main()
