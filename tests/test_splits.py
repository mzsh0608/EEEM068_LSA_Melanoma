import pandas as pd
import pytest

from src.splits import create_patient_folds


def make_test_dataframe():
    rows = []

    # 15 patients, 2 images each.
    # Enough groups of each class for 3-fold splitting.
    for patient_idx in range(15):
        target = patient_idx % 2

        for image_idx in range(2):
            rows.append({
                "image_name": f"img_{patient_idx}_{image_idx}",
                "patient_id": f"patient_{patient_idx}",
                "target": target,
            })

    return pd.DataFrame(rows)


def test_patient_groups_do_not_cross_folds():
    df = make_test_dataframe()

    result = create_patient_folds(
        df,
        n_splits=3,
        seed=42,
    )

    folds_per_patient = (
        result
        .groupby("patient_id")["fold"]
        .nunique()
    )

    assert (result["fold"] >= 0).all()
    assert folds_per_patient.max() == 1


def test_non_default_index_is_supported():
    df = make_test_dataframe()

    # Deliberately create a non-default DataFrame index.
    df.index = range(100, 100 + len(df))

    result = create_patient_folds(
        df,
        n_splits=3,
        seed=42,
    )

    assert (result["fold"] >= 0).all()

    folds_per_patient = (
        result
        .groupby("patient_id")["fold"]
        .nunique()
    )

    assert folds_per_patient.max() == 1


def test_split_is_deterministic():
    df = make_test_dataframe()

    result_a = create_patient_folds(
        df,
        n_splits=3,
        seed=42,
    )

    result_b = create_patient_folds(
        df,
        n_splits=3,
        seed=42,
    )

    pd.testing.assert_series_equal(
        result_a["fold"],
        result_b["fold"],
    )


def test_missing_patient_ids_are_handled():
    df = make_test_dataframe()

    df.loc[0, "patient_id"] = None
    df.loc[1, "patient_id"] = None

    result = create_patient_folds(
        df,
        n_splits=3,
        seed=42,
    )

    assert (result["fold"] >= 0).all()


def test_duplicate_image_names_are_rejected():
    df = make_test_dataframe()

    df.loc[1, "image_name"] = df.loc[0, "image_name"]

    with pytest.raises(
        ValueError,
        match="Duplicate image_name",
    ):
        create_patient_folds(
            df,
            n_splits=3,
            seed=42,
        )