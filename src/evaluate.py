import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
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

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must have equal length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate an empty dataset."
        )

    if not np.all(
        np.isin(y_true, [0, 1])
    ):
        raise ValueError(
            "y_true must contain only 0 and 1."
        )

    if np.any(
        (y_prob < 0) | (y_prob > 1)
    ):
        raise ValueError(
            "y_prob must contain probabilities in [0, 1]."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must lie between 0 and 1."
        )

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

    metrics = {
        "threshold": float(threshold),

        # Threshold-independent metrics
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),

        # Average Precision is used as the summary
        # statistic for the precision-recall curve.
        "pr_auc_average_precision": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),

        # Threshold-dependent metrics
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
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

    if len(metadata_df) != len(y_true):
        raise ValueError(
            "metadata_df and predictions must have equal length."
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