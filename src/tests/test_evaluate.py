import numpy as np
import pandas as pd

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