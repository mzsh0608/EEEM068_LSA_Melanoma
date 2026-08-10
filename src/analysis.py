"""Post-hoc behaviour analyses operating on saved Fold-0 predictions."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluate import calculate_binary_metrics


THRESHOLD_GRID = tuple(round(value / 10, 1) for value in range(1, 10))
BASE_PREDICTION_COLUMNS = [
    "image_name",
    "patient_id",
    "target",
    "probability",
    "prediction",
]
M2_METADATA_COLUMNS = [
    "age_approx",
    "sex",
    "anatom_site_general_challenge",
]
CONFUSION_CATEGORIES = ("FN", "FP", "TP", "TN")
SMALL_POSITIVE_THRESHOLD = 10


def load_prediction_file(path, required_extra_columns=None, expected_rows=None):
    """Load and strictly validate one saved prediction CSV."""
    frame = pd.read_csv(path)
    return validate_prediction_frame(
        frame,
        required_extra_columns=required_extra_columns,
        expected_rows=expected_rows,
    )


def validate_prediction_frame(
    frame,
    required_extra_columns=None,
    expected_rows=None,
):
    """Return a normalized prediction frame or raise on invalid evidence."""
    required = set(BASE_PREDICTION_COLUMNS)
    required.update(required_extra_columns or [])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if expected_rows is not None and len(frame) != int(expected_rows):
        raise ValueError(
            f"Expected {int(expected_rows)} prediction rows; received {len(frame)}."
        )
    if frame["image_name"].duplicated().any():
        raise ValueError("Prediction image_name values must be unique.")
    if frame[["image_name", "patient_id"]].isna().any().any():
        raise ValueError("image_name and patient_id must not be missing.")

    result = frame.copy()
    targets = pd.to_numeric(result["target"], errors="coerce")
    probabilities = pd.to_numeric(result["probability"], errors="coerce")
    predictions = pd.to_numeric(result["prediction"], errors="coerce")
    if targets.isna().any() or not targets.isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1.")
    if probabilities.isna().any() or not np.isfinite(probabilities).all():
        raise ValueError("probability must contain finite values.")
    if not probabilities.between(0.0, 1.0, inclusive="both").all():
        raise ValueError("probability must lie in [0, 1].")
    if predictions.isna().any() or not predictions.isin([0, 1]).all():
        raise ValueError("prediction must contain only 0 and 1.")
    expected_predictions = (probabilities.to_numpy() >= 0.5).astype(int)
    if not np.array_equal(predictions.to_numpy(dtype=int), expected_predictions):
        raise ValueError("prediction must use probability >= 0.5.")

    result["image_name"] = result["image_name"].astype(str)
    result["patient_id"] = result["patient_id"].astype(str)
    result["target"] = targets.astype(int)
    result["probability"] = probabilities.astype(float)
    result["prediction"] = predictions.astype(int)
    return result


def merge_model_predictions(m1_frame, m2_frame):
    """Merge M1 and M2 by image name and verify identity fields."""
    m1 = validate_prediction_frame(m1_frame)
    m2 = validate_prediction_frame(
        m2_frame,
        required_extra_columns=M2_METADATA_COLUMNS,
    )
    m1_columns = BASE_PREDICTION_COLUMNS
    m2_columns = [*BASE_PREDICTION_COLUMNS, *M2_METADATA_COLUMNS]
    merged = m1[m1_columns].merge(
        m2[m2_columns],
        on="image_name",
        how="outer",
        validate="one_to_one",
        suffixes=("_m1", "_m2"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("M1 and M2 image sets differ.")
    if not np.array_equal(merged["target_m1"], merged["target_m2"]):
        raise ValueError("M1 and M2 targets differ.")
    if not np.array_equal(
        merged["patient_id_m1"].astype(str),
        merged["patient_id_m2"].astype(str),
    ):
        raise ValueError("M1 and M2 patient IDs differ.")

    output = pd.DataFrame({
        "image_name": merged["image_name"].astype(str),
        "patient_id": merged["patient_id_m2"].astype(str),
        "target": merged["target_m2"].astype(int),
        "m1_probability": merged["probability_m1"].astype(float),
        "m1_prediction": merged["prediction_m1"].astype(int),
        "m2_probability": merged["probability_m2"].astype(float),
        "m2_prediction": merged["prediction_m2"].astype(int),
    })
    for column in M2_METADATA_COLUMNS:
        output[column] = merged[column]
    return output.sort_values("image_name").reset_index(drop=True)


def calculate_threshold_metrics(targets, probabilities, threshold):
    """Calculate one threshold row, including prediction counts."""
    metrics = calculate_binary_metrics(targets, probabilities, threshold)
    predicted_positive_count = int(metrics["tp"] + metrics["fp"])
    predicted_negative_count = int(metrics["tn"] + metrics["fn"])
    return {
        "threshold": metrics["threshold"],
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
        "predicted_positive_count": predicted_positive_count,
        "predicted_negative_count": predicted_negative_count,
        "predicted_positive_rate": predicted_positive_count / len(targets),
    }


def calculate_threshold_grid(
    targets,
    probabilities,
    thresholds=THRESHOLD_GRID,
):
    """Calculate the predeclared threshold grid without optimization."""
    thresholds = tuple(float(value) for value in thresholds)
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Thresholds must lie in [0, 1].")
    return pd.DataFrame([
        calculate_threshold_metrics(targets, probabilities, threshold)
        for threshold in thresholds
    ])


def assign_confusion_category(targets, probabilities, threshold=0.5):
    """Assign TP, TN, FP, or FN using probability >= threshold."""
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have equal shape.")
    calculate_binary_metrics(targets, probabilities, threshold)
    predictions = (probabilities >= float(threshold)).astype(int)
    return np.select(
        [
            (targets == 1) & (predictions == 1),
            (targets == 0) & (predictions == 0),
            (targets == 0) & (predictions == 1),
            (targets == 1) & (predictions == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="INVALID",
    )


def select_failure_cases(
    merged_frame,
    per_category=6,
    content_hash_by_image=None,
):
    """Select deterministic high-confidence, optionally unique-content cases."""
    frame = merged_frame.copy()
    frame["m2_category"] = assign_confusion_category(
        frame["target"], frame["m2_probability"], threshold=0.5
    )
    frame["m1_category"] = assign_confusion_category(
        frame["target"], frame["m1_probability"], threshold=0.5
    )
    ordering = {
        "FN": True,
        "FP": False,
        "TP": False,
        "TN": True,
    }
    selected = []
    for category in CONFUSION_CATEGORIES:
        subset = frame.loc[frame["m2_category"] == category].sort_values(
            ["m2_probability", "image_name"],
            ascending=[ordering[category], True],
            kind="mergesort",
        ).copy()
        if content_hash_by_image is not None:
            subset["content_sha256"] = subset["image_name"].map(
                content_hash_by_image
            )
            if subset["content_sha256"].isna().any():
                raise ValueError("Content hashes are missing for candidate cases.")
            subset = subset.drop_duplicates("content_sha256", keep="first")
        subset = subset.head(int(per_category)).copy()
        subset.insert(0, "rank", np.arange(1, len(subset) + 1))
        subset.insert(1, "category", category)
        selected.append(subset)
    result = pd.concat(selected, ignore_index=True)
    result["probability_delta_m2_minus_m1"] = (
        result["m2_probability"] - result["m1_probability"]
    )
    return result


def calculate_prediction_transitions(merged_frame):
    """Describe decision and confusion transitions from M1 to M2."""
    frame = merged_frame.copy()
    frame["probability_delta"] = (
        frame["m2_probability"] - frame["m1_probability"]
    )
    m1_label = np.where(frame["m1_prediction"] == 1, "POS", "NEG")
    m2_label = np.where(frame["m2_prediction"] == 1, "POS", "NEG")
    frame["transition_category"] = [
        f"M1_{left}_TO_M2_{right}"
        for left, right in zip(m1_label, m2_label)
    ]
    frame["m1_category"] = assign_confusion_category(
        frame["target"], frame["m1_probability"], threshold=0.5
    )
    frame["m2_category"] = assign_confusion_category(
        frame["target"], frame["m2_probability"], threshold=0.5
    )
    frame["confusion_transition"] = [
        f"M1_{left}_TO_M2_{right}"
        for left, right in zip(frame["m1_category"], frame["m2_category"])
    ]
    return frame


def build_disagreement_summary(transitions, largest_count=10):
    """Build a JSON-safe aggregate summary of all prediction transitions."""
    by_target = {}
    for target, subset in transitions.groupby("target", sort=True):
        by_target[str(int(target))] = {
            str(key): int(value)
            for key, value in subset["transition_category"].value_counts().items()
        }
    columns = [
        "image_name",
        "patient_id",
        "target",
        "m1_probability",
        "m2_probability",
        "probability_delta",
        "transition_category",
        "confusion_transition",
    ]
    positive = transitions.nlargest(largest_count, "probability_delta")[columns]
    negative = transitions.nsmallest(largest_count, "probability_delta")[columns]
    return {
        "rows": int(len(transitions)),
        "transition_counts": {
            str(key): int(value)
            for key, value in transitions["transition_category"]
            .value_counts()
            .items()
        },
        "confusion_transition_counts": {
            str(key): int(value)
            for key, value in transitions["confusion_transition"]
            .value_counts()
            .items()
        },
        "transition_counts_by_target": by_target,
        "mean_probability_delta": float(transitions["probability_delta"].mean()),
        "median_probability_delta": float(
            transitions["probability_delta"].median()
        ),
        "largest_positive_probability_changes": json.loads(
            positive.to_json(orient="records")
        ),
        "largest_negative_probability_changes": json.loads(
            negative.to_json(orient="records")
        ),
    }


def assign_age_band(age):
    """Assign one predeclared age band with explicit boundaries."""
    if pd.isna(age):
        return "unknown"
    age = float(age)
    if age < 40:
        return "<40"
    if age < 60:
        return "40-59"
    return "60+"


def add_subgroup_columns(frame):
    """Add normalized sex, age-band, and site labels."""
    result = frame.copy()
    result["sex_group"] = result["sex"].fillna("unknown").astype(str)
    result["age_band"] = result["age_approx"].map(assign_age_band)
    result["site_group"] = (
        result["anatom_site_general_challenge"]
        .fillna("unknown")
        .astype(str)
    )
    return result


def calculate_subgroup_metrics(frame, model_name, probability_column):
    """Calculate threshold-0.5 metrics for every observed subgroup."""
    grouped = add_subgroup_columns(frame)
    variables = {
        "sex": "sex_group",
        "age": "age_band",
        "site": "site_group",
    }
    rows = []
    for group_variable, column in variables.items():
        values = sorted(grouped[column].unique().tolist())
        if group_variable == "age":
            order = ["<40", "40-59", "60+", "unknown"]
            values = [value for value in order if value in values]
        for value in values:
            subset = grouped.loc[grouped[column] == value]
            metrics = calculate_binary_metrics(
                subset["target"], subset[probability_column], threshold=0.5
            )
            positive_count = int(metrics["n_positive"])
            rows.append({
                "model": model_name,
                "group_variable": group_variable,
                "group_value": value,
                "N": int(metrics["n_samples"]),
                "negative_count": int(metrics["n_negative"]),
                "positive_count": positive_count,
                "prevalence": positive_count / len(subset),
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["pr_auc_average_precision"],
                "metric_defined": metrics["roc_auc"] is not None,
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
                "small_positive_count": (
                    positive_count < SMALL_POSITIVE_THRESHOLD
                ),
            })
    return pd.DataFrame(rows)


def compare_subgroups(m1_metrics, m2_metrics):
    """Align M1/M2 subgroup estimates and calculate descriptive deltas."""
    keys = ["group_variable", "group_value"]
    metrics = [
        "N",
        "positive_count",
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "small_positive_count",
    ]
    left = m1_metrics[keys + metrics].rename(
        columns={column: f"M1_{column}" for column in metrics}
    )
    right = m2_metrics[keys + metrics].rename(
        columns={column: f"M2_{column}" for column in metrics}
    )
    merged = left.merge(right, on=keys, validate="one_to_one")
    if not np.array_equal(merged["M1_N"], merged["M2_N"]):
        raise ValueError("M1/M2 subgroup sample counts differ.")
    if not np.array_equal(
        merged["M1_positive_count"], merged["M2_positive_count"]
    ):
        raise ValueError("M1/M2 subgroup positive counts differ.")
    merged["N"] = merged.pop("M2_N")
    merged["positive_count"] = merged.pop("M2_positive_count")
    merged = merged.drop(columns=["M1_N", "M1_positive_count"])
    merged["small_positive_count"] = (
        merged.pop("M1_small_positive_count")
        | merged.pop("M2_small_positive_count")
    )
    for metric in [
        "roc_auc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]:
        merged[f"delta_{metric}"] = (
            merged[f"M2_{metric}"] - merged[f"M1_{metric}"]
        )
    return merged


def paired_patient_bootstrap(merged_frame, iterations=1000, seed=42):
    """Run a paired patient-level cluster bootstrap for M2 minus M1."""
    if int(iterations) < 1:
        raise ValueError("iterations must be positive.")
    frame = merged_frame.reset_index(drop=True)
    groups = [
        indexes.to_numpy(dtype=int)
        for _, indexes in frame.groupby("patient_id", sort=True).groups.items()
    ]
    if not groups:
        raise ValueError("No patient groups are available.")
    rng = np.random.default_rng(int(seed))
    rows = []
    for iteration in range(1, int(iterations) + 1):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        indexes = np.concatenate([groups[index] for index in sampled_groups])
        targets = frame.loc[indexes, "target"].to_numpy(dtype=int)
        m1_probability = frame.loc[
            indexes, "m1_probability"
        ].to_numpy(dtype=float)
        m2_probability = frame.loc[
            indexes, "m2_probability"
        ].to_numpy(dtype=float)
        both_classes = np.unique(targets).size == 2
        positives = targets == 1
        if both_classes:
            m1_auc = float(roc_auc_score(targets, m1_probability))
            m2_auc = float(roc_auc_score(targets, m2_probability))
            m1_ap = float(average_precision_score(targets, m1_probability))
            m2_ap = float(average_precision_score(targets, m2_probability))
        else:
            m1_auc = m2_auc = m1_ap = m2_ap = np.nan
        if positives.any():
            m1_sensitivity = float(
                (m1_probability[positives] >= 0.5).mean()
            )
            m2_sensitivity = float(
                (m2_probability[positives] >= 0.5).mean()
            )
        else:
            m1_sensitivity = m2_sensitivity = np.nan
        rows.append({
            "iteration": iteration,
            "sampled_patient_clusters": len(groups),
            "sampled_rows": len(indexes),
            "positive_count": int(positives.sum()),
            "m1_roc_auc": m1_auc,
            "m2_roc_auc": m2_auc,
            "roc_auc_difference_m2_minus_m1": m2_auc - m1_auc,
            "m1_average_precision": m1_ap,
            "m2_average_precision": m2_ap,
            "average_precision_difference_m2_minus_m1": m2_ap - m1_ap,
            "m1_sensitivity": m1_sensitivity,
            "m2_sensitivity": m2_sensitivity,
            "sensitivity_difference_m2_minus_m1": (
                m2_sensitivity - m1_sensitivity
            ),
        })
    return pd.DataFrame(rows)


def summarize_paired_bootstrap(samples, merged_frame, seed=42):
    """Summarize observed and percentile bootstrap differences."""
    observed = {
        "roc_auc_difference_m2_minus_m1": (
            roc_auc_score(merged_frame["target"], merged_frame["m2_probability"])
            - roc_auc_score(
                merged_frame["target"], merged_frame["m1_probability"]
            )
        ),
        "average_precision_difference_m2_minus_m1": (
            average_precision_score(
                merged_frame["target"], merged_frame["m2_probability"]
            )
            - average_precision_score(
                merged_frame["target"], merged_frame["m1_probability"]
            )
        ),
        "sensitivity_difference_m2_minus_m1": (
            (merged_frame.loc[merged_frame["target"] == 1, "m2_probability"] >= 0.5).mean()
            - (merged_frame.loc[merged_frame["target"] == 1, "m1_probability"] >= 0.5).mean()
        ),
    }
    summaries = {}
    for column, observed_value in observed.items():
        valid = samples[column].dropna().to_numpy(dtype=float)
        summaries[column] = {
            "observed_difference": float(observed_value),
            "bootstrap_median_difference": float(np.median(valid)),
            "percentile_2_5": float(np.percentile(valid, 2.5)),
            "percentile_97_5": float(np.percentile(valid, 97.5)),
            "valid_iterations": int(len(valid)),
            "skipped_iterations": int(len(samples) - len(valid)),
        }
    jointly_valid = samples[list(observed)].notna().all(axis=1)
    return {
        "method": "paired_patient_level_cluster_bootstrap",
        "seed": int(seed),
        "requested_iterations": int(len(samples)),
        "valid_iterations": int(jointly_valid.sum()),
        "skipped_iterations": int((~jointly_valid).sum()),
        "metrics": summaries,
    }


def plot_threshold_tradeoffs(m1_grid, m2_grid, output_path):
    """Plot sensitivity, specificity, precision, and F1 by threshold."""
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, metric, title in zip(
        axes.ravel(),
        ["sensitivity", "specificity", "precision", "f1"],
        ["Sensitivity", "Specificity", "Precision", "F1"],
    ):
        axis.plot(m1_grid["threshold"], m1_grid[metric], marker="o", label="M1")
        axis.plot(m2_grid["threshold"], m2_grid[metric], marker="s", label="M2")
        axis.axvline(0.5, color="black", linestyle="--", alpha=0.6)
        axis.set(title=title, xlabel="Threshold", ylabel=title)
        axis.set_xlim(0.08, 0.92)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_failure_montage(cases, project_root, output_path):
    """Plot six deterministic original images per confusion category."""
    project_root = Path(project_root)
    columns = max(int((cases["category"] == category).sum()) for category in CONFUSION_CATEGORIES)
    figure, axes = plt.subplots(4, columns, figsize=(3.0 * columns, 11.5))
    for row, category in enumerate(CONFUSION_CATEGORIES):
        subset = cases.loc[cases["category"] == category].sort_values("rank")
        for column in range(columns):
            axis = axes[row, column]
            axis.axis("off")
            if column >= len(subset):
                continue
            item = subset.iloc[column]
            with Image.open(project_root / item["image_path"]) as image:
                axis.imshow(image.convert("RGB"))
            age = "unknown" if pd.isna(item["age_approx"]) else f"{item['age_approx']:g}"
            sex = "unknown" if pd.isna(item["sex"]) else str(item["sex"])
            site = (
                "unknown"
                if pd.isna(item["anatom_site_general_challenge"])
                else str(item["anatom_site_general_challenge"])
            )
            axis.set_title(
                f"{category} M2={item['m2_probability']:.3f}\n"
                f"age={age} {sex}\n{site}",
                fontsize=8,
            )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_subgroup_performance(m1_metrics, m2_metrics, output_path):
    """Plot selected sensitivity/specificity subgroup estimates."""
    panels = [
        ("sex", "sensitivity", "Sensitivity by sex"),
        ("age", "sensitivity", "Sensitivity by age"),
        ("site", "sensitivity", "Sensitivity by site"),
        ("site", "specificity", "Specificity by site"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(15, 9))
    for axis, (variable, metric, title) in zip(axes.ravel(), panels):
        left = m1_metrics.loc[m1_metrics["group_variable"] == variable]
        right = m2_metrics.loc[m2_metrics["group_variable"] == variable]
        values = left["group_value"].tolist()
        positions = np.arange(len(values))
        width = 0.38
        axis.bar(positions - width / 2, left[metric], width, label="M1")
        axis.bar(positions + width / 2, right[metric], width, label="M2")
        axis.set_xticks(positions, values, rotation=30, ha="right")
        axis.set_ylim(0, 1.08)
        axis.set(title=title, ylabel=metric.replace("_", " ").title())
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        if metric == "sensitivity":
            for position, count in zip(positions, left["positive_count"]):
                axis.text(position, 1.02, f"n+={count}", ha="center", fontsize=7)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def plot_bootstrap_differences(samples, output_path):
    """Plot paired ROC-AUC and AP difference distributions."""
    columns = [
        ("roc_auc_difference_m2_minus_m1", "M2 - M1 ROC-AUC"),
        (
            "average_precision_difference_m2_minus_m1",
            "M2 - M1 Average Precision",
        ),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, (column, title) in zip(axes, columns):
        axis.hist(samples[column].dropna(), bins=35, alpha=0.8)
        axis.axvline(0, color="black", linestyle="--")
        axis.set(title=title, xlabel="Paired difference", ylabel="Iterations")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def _git_head(project_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _assert_official_threshold(grid, expected_counts):
    row = grid.loc[np.isclose(grid["threshold"], 0.5)]
    if len(row) != 1:
        raise RuntimeError("Threshold grid must contain exactly one 0.5 row.")
    actual = {key: int(row.iloc[0][key]) for key in expected_counts}
    if actual != expected_counts:
        raise RuntimeError(
            f"Threshold 0.5 regression mismatch: {actual} != {expected_counts}"
        )


def run_saved_prediction_analyses(
    project_root,
    bootstrap_iterations=1000,
    bootstrap_seed=42,
):
    """Generate authoritative saved-prediction Phase I analyses."""
    project_root = Path(project_root)
    m1_path = project_root / "outputs/predictions/M1_convnext_image.csv"
    m2_path = project_root / "outputs/predictions/M2_convnext_metadata.csv"
    m1 = load_prediction_file(m1_path, expected_rows=6627)
    m2 = load_prediction_file(
        m2_path,
        required_extra_columns=M2_METADATA_COLUMNS,
        expected_rows=6627,
    )
    merged = merge_model_predictions(m1, m2)

    threshold_directory = project_root / "outputs/analysis/threshold"
    m1_grid = calculate_threshold_grid(
        merged["target"], merged["m1_probability"]
    )
    m2_grid = calculate_threshold_grid(
        merged["target"], merged["m2_probability"]
    )
    _assert_official_threshold(
        m1_grid, {"tn": 4086, "fp": 2424, "fn": 3, "tp": 114}
    )
    _assert_official_threshold(
        m2_grid, {"tn": 4791, "fp": 1719, "fn": 12, "tp": 105}
    )
    threshold_directory.mkdir(parents=True, exist_ok=True)
    m1_grid.to_csv(threshold_directory / "M1_threshold_analysis.csv", index=False)
    m2_grid.to_csv(threshold_directory / "M2_threshold_analysis.csv", index=False)
    plot_threshold_tradeoffs(
        m1_grid,
        m2_grid,
        project_root / "outputs/figures/I_threshold_tradeoffs.png",
    )

    failure_directory = project_root / "outputs/analysis/failures"
    failure_directory.mkdir(parents=True, exist_ok=True)
    transitions = calculate_prediction_transitions(merged)
    transitions.to_csv(
        failure_directory / "M1_M2_disagreements.csv", index=False
    )
    disagreement_summary = build_disagreement_summary(transitions)
    _write_json(
        disagreement_summary,
        failure_directory / "M1_M2_disagreement_summary.json",
    )
    image_directory = project_root / "data/train_images"
    content_hash_by_image = {}
    for image_name in merged["image_name"]:
        with (image_directory / f"{image_name}.jpg").open("rb") as handle:
            content_hash_by_image[image_name] = hashlib.file_digest(
                handle, "sha256"
            ).hexdigest()
    failures = select_failure_cases(
        merged,
        per_category=6,
        content_hash_by_image=content_hash_by_image,
    )
    failures["image_path"] = failures["image_name"].map(
        lambda value: f"data/train_images/{value}.jpg"
    )
    failure_columns = [
        "rank",
        "category",
        "image_name",
        "patient_id",
        "target",
        "m2_probability",
        "m2_prediction",
        *M2_METADATA_COLUMNS,
        "image_path",
        "m1_probability",
        "m1_prediction",
        "probability_delta_m2_minus_m1",
        "m1_category",
        "m2_category",
    ]
    failures[failure_columns].to_csv(
        failure_directory / "M2_failure_cases.csv", index=False
    )
    plot_failure_montage(
        failures,
        project_root,
        project_root / "outputs/figures/I_failure_cases.png",
    )

    subgroup_directory = project_root / "outputs/analysis/subgroups"
    subgroup_directory.mkdir(parents=True, exist_ok=True)
    m1_subgroups = calculate_subgroup_metrics(
        merged, "M1", "m1_probability"
    )
    m2_subgroups = calculate_subgroup_metrics(
        merged, "M2", "m2_probability"
    )
    subgroup_comparison = compare_subgroups(m1_subgroups, m2_subgroups)
    m1_subgroups.to_csv(
        subgroup_directory / "M1_subgroup_metrics.csv", index=False
    )
    m2_subgroups.to_csv(
        subgroup_directory / "M2_subgroup_metrics.csv", index=False
    )
    subgroup_comparison.to_csv(
        subgroup_directory / "M1_M2_subgroup_comparison.csv", index=False
    )
    plot_subgroup_performance(
        m1_subgroups,
        m2_subgroups,
        project_root / "outputs/figures/I_subgroup_performance.png",
    )

    bootstrap_summary = None
    if int(bootstrap_iterations) > 0:
        bootstrap_directory = project_root / "outputs/analysis/bootstrap"
        bootstrap_directory.mkdir(parents=True, exist_ok=True)
        bootstrap_samples = paired_patient_bootstrap(
            merged,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        )
        bootstrap_summary = summarize_paired_bootstrap(
            bootstrap_samples,
            merged,
            seed=bootstrap_seed,
        )
        bootstrap_samples.to_csv(
            bootstrap_directory / "M1_M2_bootstrap_samples.csv", index=False
        )
        _write_json(
            bootstrap_summary,
            bootstrap_directory / "M1_M2_bootstrap_summary.json",
        )
        plot_bootstrap_differences(
            bootstrap_samples,
            project_root / "outputs/figures/I_bootstrap_differences.png",
        )

    return {
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_head(project_root),
        "source_prediction_files": {
            "M1": str(m1_path.relative_to(project_root)),
            "M2": str(m2_path.relative_to(project_root)),
        },
        "prediction_audit": {
            "M1_rows": len(m1),
            "M2_rows": len(m2),
            "M1_unique_images": int(m1["image_name"].nunique()),
            "M2_unique_images": int(m2["image_name"].nunique()),
            "same_images": set(m1["image_name"]) == set(m2["image_name"]),
            "same_targets": True,
            "same_patient_ids": True,
            "M1_probability_range": [
                float(m1["probability"].min()),
                float(m1["probability"].max()),
            ],
            "M2_probability_range": [
                float(m2["probability"].min()),
                float(m2["probability"].max()),
            ],
        },
        "threshold_grid": list(THRESHOLD_GRID),
        "failure_selection": {
            "policy": "deterministic_high_confidence_by_confusion_category",
            "selected_per_category": 6,
            "case_count": int(len(failures)),
        },
        "disagreement_summary": disagreement_summary,
        "subgroups": {
            "variables": ["sex", "age", "site"],
            "age_definitions": {
                "<40": "age < 40",
                "40-59": "40 <= age < 60",
                "60+": "age >= 60",
                "unknown": "missing age",
            },
            "small_positive_caution_threshold": SMALL_POSITIVE_THRESHOLD,
        },
        "bootstrap": bootstrap_summary,
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate Phase I analyses from frozen predictions."
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args(args)


def main(args=None):
    arguments = parse_args(args)
    project_root = (
        Path(arguments.project_root)
        if arguments.project_root
        else Path(__file__).resolve().parents[1]
    )
    result = run_saved_prediction_analyses(
        project_root,
        bootstrap_iterations=arguments.bootstrap_iterations,
        bootstrap_seed=arguments.bootstrap_seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
