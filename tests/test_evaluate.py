import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    calculate_binary_metrics,
    make_prediction_dataframe,
)


def test_binary_metrics_known_confusion_matrix():
    y_true = np.array([
        0, 0, 1, 1
    ])

    y_prob = np.array([
        0.10,
        0.80,
        0.90,
        0.40,
    ])

    metrics = calculate_binary_metrics(
        y_true,
        y_prob,
        threshold=0.5,
    )

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1

    assert metrics["accuracy"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["f1"] == 0.5


def test_threshold_changes_predictions():
    y_true = np.array([
        0, 0, 1, 1
    ])

    y_prob = np.array([
        0.10,
        0.40,
        0.45,
        0.90,
    ])

    metrics_05 = calculate_binary_metrics(
        y_true,
        y_prob,
        threshold=0.5,
    )

    metrics_03 = calculate_binary_metrics(
        y_true,
        y_prob,
        threshold=0.3,
    )

    assert (
        metrics_03["sensitivity"]
        >= metrics_05["sensitivity"]
    )


def test_prediction_dataframe():
    metadata = pd.DataFrame({
        "image_name": [
            "image_a",
            "image_b",
        ],
        "patient_id": [
            "patient_a",
            "patient_b",
        ],
    })

    output = make_prediction_dataframe(
        metadata,
        y_true=[0, 1],
        y_prob=[0.2, 0.8],
        threshold=0.5,
    )

    assert len(output) == 2

    assert list(output.columns) == [
        "image_name",
        "patient_id",
        "target",
        "probability",
        "prediction",
    ]

    assert output["prediction"].tolist() == [
        0, 1
    ]


def test_non_binary_float_targets_are_rejected():
    with pytest.raises(ValueError, match="only 0 and 1"):
        calculate_binary_metrics(
            [0.2, 1.0],
            [0.1, 0.9],
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [np.nan, np.inf, -np.inf],
)
def test_non_finite_probabilities_are_rejected(
    invalid_probability,
):
    with pytest.raises(ValueError, match="NaN or infinite"):
        calculate_binary_metrics(
            [0, 1],
            [0.1, invalid_probability],
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [-0.1, 1.1],
)
def test_out_of_range_probabilities_are_rejected(
    invalid_probability,
):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calculate_binary_metrics(
            [0, 1],
            [0.1, invalid_probability],
        )


def test_target_probability_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="equal length"):
        calculate_binary_metrics(
            [0, 1],
            [0.1],
        )


@pytest.mark.parametrize(
    ("targets", "probabilities"),
    [
        ([0], [0.1, 0.9]),
        ([0, 1], [0.1]),
    ],
)
def test_prediction_dataframe_length_mismatch_is_rejected(
    targets,
    probabilities,
):
    metadata = pd.DataFrame({
        "image_name": ["image_a", "image_b"],
        "patient_id": ["patient_a", "patient_b"],
    })

    with pytest.raises(ValueError, match="equal length"):
        make_prediction_dataframe(
            metadata,
            y_true=targets,
            y_prob=probabilities,
        )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan])
def test_invalid_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        calculate_binary_metrics(
            [0, 1],
            [0.1, 0.9],
            threshold=threshold,
        )


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_single_class_ranking_metrics_are_none(labels):
    metrics = calculate_binary_metrics(
        labels,
        [0.1, 0.2],
    )

    assert metrics["roc_auc"] is None
    assert metrics["pr_auc_average_precision"] is None
