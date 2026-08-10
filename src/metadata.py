"""Leakage-safe preprocessing for the whitelisted M2 metadata."""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


METADATA_COLUMNS = [
    "age_approx",
    "sex",
    "anatom_site_general_challenge",
]
NUMERIC_COLUMNS = ["age_approx"]
CATEGORICAL_COLUMNS = ["sex", "anatom_site_general_challenge"]


@dataclass(frozen=True)
class MetadataPreparation:
    """Fitted preprocessor and aligned train/validation evidence."""

    preprocessor: ColumnTransformer
    train_matrix: np.ndarray
    val_matrix: np.ndarray
    feature_names: list[str]
    summary: dict


def _metadata_frame(dataframe):
    missing = set(METADATA_COLUMNS).difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    frame = dataframe.loc[:, METADATA_COLUMNS].copy()
    frame["age_approx"] = pd.to_numeric(
        frame["age_approx"], errors="coerce"
    )
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].astype(object)
        frame[column] = frame[column].where(frame[column].notna(), np.nan)
    return frame


def build_metadata_preprocessor():
    """Create the fixed age/sex/site preprocessing graph."""
    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        ),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="unknown",
                keep_empty_features=True,
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32,
            ),
        ),
    ])
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def fit_metadata_preprocessor(train_df):
    """Fit all learned metadata state using training rows only."""
    preprocessor = build_metadata_preprocessor()
    preprocessor.fit(_metadata_frame(train_df))
    return preprocessor


def transform_metadata(preprocessor, dataframe):
    """Transform rows into a finite dense float32 metadata matrix."""
    transformed = np.asarray(
        preprocessor.transform(_metadata_frame(dataframe)),
        dtype=np.float32,
    )
    if transformed.ndim != 2:
        raise RuntimeError("Transformed metadata must be two-dimensional.")
    if not np.isfinite(transformed).all():
        raise RuntimeError("Transformed metadata contains NaN or infinity.")
    return transformed


def _category_values(frame, column):
    values = frame[column].astype(object).where(
        frame[column].notna(), "unknown"
    )
    return {str(value) for value in values}


def build_metadata_summary(preprocessor, train_df, val_df):
    """Extract learned training statistics and validation audit evidence."""
    train_frame = _metadata_frame(train_df)
    val_frame = _metadata_frame(val_df)
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    encoder = categorical.named_steps["encoder"]
    sex_categories = [str(value) for value in encoder.categories_[0]]
    site_categories = [str(value) for value in encoder.categories_[1]]
    fitted_categories = {
        "sex": set(sex_categories),
        "anatom_site_general_challenge": set(site_categories),
    }
    validation_only = {
        column: sorted(
            _category_values(val_frame, column) - fitted_categories[column]
        )
        for column in CATEGORICAL_COLUMNS
    }

    return {
        "input_columns": list(METADATA_COLUMNS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "fit_partition": "training_only",
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "missing_counts": {
            "training": {
                column: int(train_frame[column].isna().sum())
                for column in METADATA_COLUMNS
            },
            "validation": {
                column: int(val_frame[column].isna().sum())
                for column in METADATA_COLUMNS
            },
        },
        "age_imputation_median": float(imputer.statistics_[0]),
        "age_scaler_mean": float(scaler.mean_[0]),
        "age_scaler_scale": float(scaler.scale_[0]),
        "training_categories": {
            "sex": sex_categories,
            "anatom_site_general_challenge": site_categories,
        },
        "validation_only_categories": validation_only,
        "feature_names": [
            str(name) for name in preprocessor.get_feature_names_out()
        ],
        "metadata_dimension": int(
            len(preprocessor.get_feature_names_out())
        ),
        "policies": {
            "numeric_imputation": "median",
            "numeric_scaling": "standard",
            "categorical_missing_value": "unknown",
            "categorical_encoding": "one_hot",
            "handle_unknown": "ignore",
        },
    }


def prepare_metadata(train_df, val_df):
    """Fit on training only and transform aligned training/validation rows."""
    preprocessor = fit_metadata_preprocessor(train_df)
    train_matrix = transform_metadata(preprocessor, train_df)
    val_matrix = transform_metadata(preprocessor, val_df)
    if train_matrix.shape[1] != val_matrix.shape[1]:
        raise RuntimeError("Train and validation metadata dimensions differ.")
    summary = build_metadata_summary(preprocessor, train_df, val_df)
    return MetadataPreparation(
        preprocessor=preprocessor,
        train_matrix=train_matrix,
        val_matrix=val_matrix,
        feature_names=list(summary["feature_names"]),
        summary=summary,
    )


def save_metadata_artifacts(
    preparation,
    preprocessor_path,
    summary_path,
    validation_df=None,
):
    """Serialize metadata state and optionally verify its validation output."""
    preprocessor_path = Path(preprocessor_path)
    summary_path = Path(summary_path)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preparation.preprocessor, preprocessor_path)
    if validation_df is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*is deprecated.*",
                category=DeprecationWarning,
                module="joblib.*",
            )
            restored = joblib.load(preprocessor_path)
        restored_matrix = transform_metadata(restored, validation_df)
        np.testing.assert_allclose(
            restored_matrix,
            preparation.val_matrix,
            rtol=0.0,
            atol=0.0,
        )
        preparation.summary["serialization_round_trip_verified"] = True
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(preparation.summary, handle, indent=2)
    return preprocessor_path, summary_path
