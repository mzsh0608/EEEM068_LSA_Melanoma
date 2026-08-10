import pandas as pd
import pytest

from src.final_tables import (
    _comparison_table,
    _normalise_history,
    _verify_subgroup_partitions,
)


def test_normalise_history_maps_ap_and_marks_best_epoch():
    history = pd.DataFrame({
        "epoch": [1, 2],
        "train_loss": [1.0, 0.8],
        "val_loss": [0.9, 0.7],
        "val_roc_auc": [0.7, 0.8],
        "val_pr_auc": [0.1, 0.2],
        "val_accuracy": [0.6, 0.7],
        "val_balanced_accuracy": [0.55, 0.65],
        "val_precision": [0.2, 0.3],
        "val_sensitivity": [0.5, 0.6],
        "val_specificity": [0.6, 0.7],
        "val_f1": [0.3, 0.4],
        "learning_rate": [1e-4, 1e-4],
        "seconds": [1.0, 1.1],
    })

    result = _normalise_history(history, "B0")

    assert "val_pr_auc" not in result.columns
    assert result["val_average_precision"].tolist() == [0.1, 0.2]
    assert result["is_best_roc_auc_epoch"].tolist() == [False, True]


def test_comparison_table_uses_exact_right_minus_left_arithmetic():
    rows = []
    for model_id, offset in [("B0", 1), ("M1", 3)]:
        row = {"model_id": model_id}
        row.update({metric: offset + index for index, metric in enumerate([
            "roc_auc", "average_precision", "accuracy", "balanced_accuracy",
            "precision", "sensitivity", "specificity", "f1", "tn", "fp",
            "fn", "tp",
        ])})
        rows.append(row)

    result = _comparison_table(
        pd.DataFrame(rows),
        "B0",
        "M1",
        "M1_minus_B0",
        {"comparison_type": "test"},
    )

    assert len(result) == 12
    assert result["M1_minus_B0"].eq(2).all()
    assert result["comparison_type"].eq("test").all()


def test_subgroup_partition_validation_accepts_complete_and_rejects_partial():
    complete = pd.DataFrame({
        "model": ["M1", "M1", "M2", "M2"],
        "group_variable": ["sex", "sex", "sex", "sex"],
        "n": [4, 6, 5, 5],
    })
    assert _verify_subgroup_partitions(complete, 10)

    partial = complete.copy()
    partial.loc[0, "n"] = 3
    with pytest.raises(ValueError, match="do not reconcile"):
        _verify_subgroup_partitions(partial, 10)
