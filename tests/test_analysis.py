import numpy as np
import pandas as pd
import pytest

import src.analysis as analysis_module
from src.analysis import (
    THRESHOLD_GRID,
    assign_age_band,
    assign_confusion_category,
    calculate_prediction_transitions,
    calculate_subgroup_metrics,
    calculate_threshold_grid,
    compare_subgroups,
    merge_model_predictions,
    paired_patient_bootstrap,
    select_failure_cases,
    summarize_paired_bootstrap,
    validate_prediction_frame,
)


def _prediction_frame(probabilities=None):
    probabilities = probabilities or [0.1, 0.8, 0.6, 0.2]
    return pd.DataFrame({
        "image_name": ["a", "b", "c", "d"],
        "patient_id": ["p1", "p2", "p3", "p4"],
        "target": [0, 1, 0, 1],
        "probability": probabilities,
        "prediction": [int(value >= 0.5) for value in probabilities],
    })


def _merged_frame():
    return pd.DataFrame({
        "image_name": ["a", "b", "c", "d", "e", "f"],
        "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
        "target": [0, 1, 0, 1, 0, 1],
        "m1_probability": [0.7, 0.9, 0.2, 0.1, 0.8, 0.4],
        "m1_prediction": [1, 1, 0, 0, 1, 0],
        "m2_probability": [0.1, 0.4, 0.8, 0.9, 0.7, 0.2],
        "m2_prediction": [0, 0, 1, 1, 1, 0],
        "age_approx": [30, 40, 59.9, 60, np.nan, 70],
        "sex": ["female", "female", "male", "male", None, "female"],
        "anatom_site_general_challenge": [
            "torso",
            "torso",
            "head/neck",
            "head/neck",
            None,
            "torso",
        ],
    })


def test_threshold_metrics_and_half_threshold_are_exact():
    grid = calculate_threshold_grid(
        [0, 0, 1, 1],
        [0.1, 0.5, 0.5, 0.9],
        thresholds=[0.5],
    )

    row = grid.iloc[0]
    assert row[["tn", "fp", "fn", "tp"]].tolist() == [1, 1, 0, 2]
    assert row["predicted_positive_count"] == 3


def test_threshold_grid_is_monotonic_for_counts_and_rates():
    grid = calculate_threshold_grid(
        [0, 0, 1, 1],
        [0.1, 0.4, 0.6, 0.9],
        thresholds=THRESHOLD_GRID,
    )

    assert grid["predicted_positive_count"].is_monotonic_decreasing
    assert grid["sensitivity"].is_monotonic_decreasing
    assert grid["specificity"].is_monotonic_increasing


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_prediction_validation_rejects_invalid_probabilities(value):
    frame = _prediction_frame()
    frame.loc[0, "probability"] = value

    with pytest.raises(ValueError, match="probability"):
        validate_prediction_frame(frame)


def test_merge_uses_image_name_and_ignores_row_order():
    m1 = _prediction_frame()
    m2 = _prediction_frame().iloc[::-1].reset_index(drop=True)
    m2["age_approx"] = [60, 50, 40, 30]
    m2["sex"] = "female"
    m2["anatom_site_general_challenge"] = "torso"

    merged = merge_model_predictions(m1, m2)

    assert merged["image_name"].tolist() == ["a", "b", "c", "d"]
    assert merged.loc[merged["image_name"] == "a", "m1_probability"].item() == 0.1


def test_merge_rejects_target_mismatch():
    m1 = _prediction_frame()
    m2 = _prediction_frame()
    m2["age_approx"] = 50
    m2["sex"] = "female"
    m2["anatom_site_general_challenge"] = "torso"
    m2.loc[0, "target"] = 1

    with pytest.raises(ValueError, match="targets differ"):
        merge_model_predictions(m1, m2)


def test_confusion_category_assignment():
    categories = assign_confusion_category(
        [1, 0, 0, 1],
        [0.9, 0.1, 0.8, 0.2],
    )

    assert categories.tolist() == ["TP", "TN", "FP", "FN"]


def test_failure_selection_uses_confidence_ordering():
    frame = pd.concat([_merged_frame(), _merged_frame()], ignore_index=True)
    frame["image_name"] = [f"image_{index}" for index in range(len(frame))]
    selected = select_failure_cases(frame, per_category=2)

    for category, ascending in {"FN": True, "FP": False, "TP": False, "TN": True}.items():
        values = selected.loc[
            selected["category"] == category, "m2_probability"
        ].tolist()
        assert values == sorted(values, reverse=not ascending)


def test_failure_selection_can_require_unique_image_content():
    frame = pd.DataFrame({
        "image_name": ["fn1", "fn2", "fn3", "tn1"],
        "patient_id": ["p1", "p1", "p2", "p3"],
        "target": [1, 1, 1, 0],
        "m1_probability": [0.4, 0.4, 0.4, 0.1],
        "m1_prediction": [0, 0, 0, 0],
        "m2_probability": [0.1, 0.2, 0.3, 0.1],
        "m2_prediction": [0, 0, 0, 0],
    })
    hashes = {
        "fn1": "same",
        "fn2": "same",
        "fn3": "different",
        "tn1": "tn",
    }

    selected = select_failure_cases(
        frame,
        per_category=2,
        content_hash_by_image=hashes,
    )

    assert list(selected.loc[selected["category"] == "FN", "image_name"]) == [
        "fn1",
        "fn3",
    ]


def test_prediction_transition_labels_are_correct():
    transitions = calculate_prediction_transitions(_merged_frame())

    assert transitions["transition_category"].tolist()[:4] == [
        "M1_POS_TO_M2_NEG",
        "M1_POS_TO_M2_NEG",
        "M1_NEG_TO_M2_POS",
        "M1_NEG_TO_M2_POS",
    ]
    assert transitions.loc[0, "confusion_transition"] == "M1_FP_TO_M2_TN"
    assert transitions.loc[1, "confusion_transition"] == "M1_TP_TO_M2_FN"


@pytest.mark.parametrize(
    ("age", "expected"),
    [(39.9, "<40"), (40, "40-59"), (59.9, "40-59"), (60, "60+"), (np.nan, "unknown")],
)
def test_age_band_boundaries(age, expected):
    assert assign_age_band(age) == expected


def test_subgroup_counts_and_small_positive_flags():
    frame = _merged_frame()
    metrics = calculate_subgroup_metrics(frame, "M1", "m1_probability")

    for variable in ["sex", "age", "site"]:
        subset = metrics.loc[metrics["group_variable"] == variable]
        assert subset["N"].sum() == len(frame)
        assert subset["positive_count"].sum() == int(frame["target"].sum())
    assert metrics["small_positive_count"].all()


def test_single_class_subgroup_keeps_undefined_metrics_missing():
    frame = _merged_frame()
    frame["sex"] = ["negative", "positive", "negative", "positive", "negative", "positive"]
    metrics = calculate_subgroup_metrics(frame, "M2", "m2_probability")
    negative = metrics.loc[
        (metrics["group_variable"] == "sex")
        & (metrics["group_value"] == "negative")
    ].iloc[0]

    assert pd.isna(negative["roc_auc"])
    assert pd.isna(negative["average_precision"])
    assert not bool(negative["metric_defined"])


def test_subgroup_comparison_preserves_counts_and_deltas():
    frame = _merged_frame()
    m1 = calculate_subgroup_metrics(frame, "M1", "m1_probability")
    m2 = calculate_subgroup_metrics(frame, "M2", "m2_probability")

    comparison = compare_subgroups(m1, m2)

    assert comparison["N"].sum() == len(frame) * 3
    assert "delta_sensitivity" in comparison


def test_paired_patient_bootstrap_is_reproducible():
    frame = _merged_frame()
    first = paired_patient_bootstrap(frame, iterations=20, seed=42)
    second = paired_patient_bootstrap(frame, iterations=20, seed=42)

    pd.testing.assert_frame_equal(first, second)
    summary = summarize_paired_bootstrap(first, frame, seed=42)
    assert summary["requested_iterations"] == 20
    assert summary["seed"] == 42


def test_paired_patient_bootstrap_preserves_cluster_multiplicity(monkeypatch):
    frame = pd.DataFrame({
        "image_name": ["a1", "a2", "b1"],
        "patient_id": ["A", "A", "B"],
        "target": [0, 1, 0],
        "m1_probability": [0.1, 0.9, 0.2],
        "m2_probability": [0.2, 0.4, 0.9],
    })

    class FixedGenerator:
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 2)
            return np.array([0, 0, 1])

    monkeypatch.setattr(
        analysis_module.np.random,
        "default_rng",
        lambda seed: FixedGenerator(),
    )
    sample = paired_patient_bootstrap(frame, iterations=1, seed=42).iloc[0]

    assert sample["sampled_patient_clusters"] == 2
    assert sample["sampled_rows"] == 5
    assert sample["positive_count"] == 2
    assert sample["m1_roc_auc"] == 1.0
    assert sample["m2_roc_auc"] == pytest.approx(2 / 3)
