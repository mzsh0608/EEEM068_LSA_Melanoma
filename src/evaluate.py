import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    y_true,
    y_prob,
    threshold=0.5,
):
    """
    Calculate binary-classification metrics.

    PR-AUC is represented by scikit-learn's Average Precision
    statistic rather than trapezoidal integration of the
    precision-recall curve. If ``y_true`` contains only one class,
    ROC-AUC and Average Precision are returned as ``None``.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1).

    y_prob:
        Predicted probability for the positive class.

    threshold:
        Probability threshold used to convert probabilities
        into binary predictions.

    Returns
    -------
    dict
        Dictionary containing threshold-independent and
        threshold-dependent metrics.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError(
            "y_true and y_prob must be one-dimensional."
        )

    if len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must have equal length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate an empty dataset."
        )

    try:
        numeric_targets = y_true.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "y_true must contain only numeric binary labels."
        ) from exc

    if not np.all(np.isfinite(numeric_targets)):
        raise ValueError(
            "y_true must not contain NaN or infinite values."
        )

    if not np.all(np.isin(numeric_targets, [0, 1])):
        raise ValueError(
            "y_true must contain only 0 and 1."
        )

    if not np.all(np.isfinite(y_prob)):
        raise ValueError(
            "y_prob must not contain NaN or infinite values."
        )

    if np.any(
        (y_prob < 0) | (y_prob > 1)
    ):
        raise ValueError(
            "y_prob must contain probabilities in [0, 1]."
        )

    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "threshold must be a finite number in [0, 1]."
        ) from exc

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must lie between 0 and 1."
        )

    y_true = numeric_targets.astype(int)

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    has_both_classes = np.unique(y_true).size == 2

    roc_auc = (
        float(roc_auc_score(y_true, y_prob))
        if has_both_classes
        else None
    )
    average_precision = (
        float(average_precision_score(y_true, y_prob))
        if has_both_classes
        else None
    )

    metrics = {
        "threshold": float(threshold),

        # Threshold-independent metrics
        "roc_auc": roc_auc,

        # Average Precision is used as the summary
        # statistic for the precision-recall curve.
        "pr_auc_average_precision": average_precision,

        # Threshold-dependent metrics
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            (sensitivity + specificity) / 2
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "sensitivity": float(
            sensitivity
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),

        "n_samples": int(
            len(y_true)
        ),

        "n_positive": int(
            np.sum(y_true == 1)
        ),

        "n_negative": int(
            np.sum(y_true == 0)
        ),
    }

    return metrics


def make_prediction_dataframe(
    metadata_df,
    y_true,
    y_prob,
    threshold=0.5,
    extra_columns=None,
):
    """
    Create a standard prediction table that can be reused
    for Logistic Regression and later neural networks.
    """

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if not (
        len(metadata_df)
        == len(y_true)
        == len(y_prob)
    ):
        raise ValueError(
            "metadata_df, y_true, and y_prob must have equal length."
        )

    columns = [
        col
        for col in [
            "image_name",
            "patient_id",
        ]
        if col in metadata_df.columns
    ]

    if extra_columns:
        for col in extra_columns:
            if (
                col in metadata_df.columns
                and col not in columns
            ):
                columns.append(col)

    output = (
        metadata_df[columns]
        .reset_index(drop=True)
        .copy()
    )

    output["target"] = y_true
    output["probability"] = y_prob
    output["prediction"] = (
        y_prob >= threshold
    ).astype(int)

    return output


def save_metrics_json(
    metrics,
    output_path,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )
