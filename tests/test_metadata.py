import joblib
import numpy as np
import pandas as pd
import pytest

from src.metadata import (
    METADATA_COLUMNS,
    fit_metadata_preprocessor,
    prepare_metadata,
    save_metadata_artifacts,
    transform_metadata,
)


def _frames():
    train = pd.DataFrame({
        "age_approx": [10.0, 20.0, np.nan, 40.0],
        "sex": ["male", "female", None, "female"],
        "anatom_site_general_challenge": [
            "torso",
            "head/neck",
            "torso",
            None,
        ],
        "target": [0, 1, 0, 1],
        "diagnosis": ["nevus", "melanoma", "nevus", "melanoma"],
        "patient_id": ["p1", "p2", "p3", "p4"],
    })
    validation = pd.DataFrame({
        "age_approx": [1000.0, np.nan],
        "sex": ["validation_only", None],
        "anatom_site_general_challenge": ["validation_site", None],
        "target": [0, 1],
        "diagnosis": ["unknown", "melanoma"],
        "patient_id": ["v1", "v2"],
    })
    return train, validation


def test_numeric_statistics_are_fitted_from_training_only():
    train, validation = _frames()

    prepared = prepare_metadata(train, validation)

    assert prepared.summary["age_imputation_median"] == 20.0
    assert prepared.summary["age_scaler_mean"] == 22.5
    assert np.isclose(
        prepared.summary["age_scaler_scale"],
        np.std([10.0, 20.0, 20.0, 40.0]),
    )


def test_validation_only_categories_do_not_change_dimension():
    train, validation = _frames()

    prepared = prepare_metadata(train, validation)

    assert prepared.train_matrix.shape[1] == prepared.val_matrix.shape[1]
    assert prepared.summary["validation_only_categories"] == {
        "sex": ["validation_only"],
        "anatom_site_general_challenge": ["validation_site"],
    }


def test_missing_values_transform_to_finite_values():
    train, validation = _frames()

    prepared = prepare_metadata(train, validation)

    assert np.isfinite(prepared.train_matrix).all()
    assert np.isfinite(prepared.val_matrix).all()


def test_strict_feature_whitelist_excludes_prohibited_columns():
    train, validation = _frames()
    preprocessor = fit_metadata_preprocessor(train)
    baseline = transform_metadata(preprocessor, validation)
    modified = validation.copy()
    for column in [
        "target",
        "diagnosis",
        "benign_malignant",
        "patient_id",
        "image_name",
        "tfrecord",
        "width",
        "height",
    ]:
        modified[column] = ["changed", "different"]

    transformed = transform_metadata(preprocessor, modified)

    np.testing.assert_allclose(transformed, baseline)
    assert list(METADATA_COLUMNS) == [
        "age_approx",
        "sex",
        "anatom_site_general_challenge",
    ]


def test_transform_is_deterministic():
    train, validation = _frames()
    preprocessor = fit_metadata_preprocessor(train)

    first = transform_metadata(preprocessor, validation)
    second = transform_metadata(preprocessor, validation)

    np.testing.assert_allclose(first, second)


def test_output_is_two_dimensional_float32_and_finite():
    train, validation = _frames()

    prepared = prepare_metadata(train, validation)

    assert prepared.train_matrix.ndim == 2
    assert prepared.val_matrix.ndim == 2
    assert prepared.train_matrix.dtype == np.float32
    assert prepared.val_matrix.dtype == np.float32
    assert np.isfinite(prepared.train_matrix).all()
    assert np.isfinite(prepared.val_matrix).all()


@pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array has been deprecated:"
    "DeprecationWarning"
)
def test_serialization_round_trip_preserves_transform(tmp_path):
    train, validation = _frames()
    prepared = prepare_metadata(train, validation)
    preprocessor_path = tmp_path / "metadata.joblib"
    summary_path = tmp_path / "metadata.json"

    save_metadata_artifacts(
        prepared,
        preprocessor_path,
        summary_path,
    )
    reloaded = joblib.load(preprocessor_path)
    transformed = transform_metadata(reloaded, validation)

    np.testing.assert_allclose(transformed, prepared.val_matrix)
    assert summary_path.is_file()


def test_feature_names_and_dimension_come_from_fitted_encoder():
    train, validation = _frames()

    prepared = prepare_metadata(train, validation)

    assert prepared.summary["metadata_dimension"] == len(
        prepared.feature_names
    )
    assert "age_approx" in prepared.feature_names
    assert not any("target" in name for name in prepared.feature_names)
    assert not any("diagnosis" in name for name in prepared.feature_names)


def test_missing_required_metadata_column_is_rejected():
    train, _ = _frames()

    try:
        fit_metadata_preprocessor(train.drop(columns="sex"))
    except ValueError as error:
        assert "Missing metadata columns" in str(error)
    else:
        raise AssertionError("Missing metadata column was not rejected.")
