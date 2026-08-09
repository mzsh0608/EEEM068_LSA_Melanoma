import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def create_patient_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Assign patient-aware stratified folds.

    Images belonging to the same known patient are always
    placed in the same fold.

    Rows without a patient_id are treated as separate
    independent groups using their image_name.
    """

    required = {
        "image_name",
        "patient_id",
        "target"
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["image_name"].duplicated().any():
        raise ValueError(
            "Duplicate image_name values detected."
        )

    result = df.copy()

    # Build grouping ID.
    # Known patient IDs remain grouped.
    result["_group_id"] = result["patient_id"].astype("string")

    missing_patient = result["patient_id"].isna()

    result.loc[
        missing_patient,
        "_group_id"
    ] = (
        "missing_patient_"
        + result.loc[
            missing_patient,
            "image_name"
        ].astype(str)
    )

    result["fold"] = -1

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    fold_column_index = result.columns.get_loc("fold")

    for fold, (_, val_idx) in enumerate(
        splitter.split(
            X=result,
            y=result["target"],
            groups=result["_group_id"]
        )
    ):
        result.iloc[val_idx, fold_column_index] = fold

    if (result["fold"] < 0).any():
        raise RuntimeError(
            "Some rows were not assigned to a fold."
        )

    # Verify each patient/group belongs to exactly one fold.
    group_fold_counts = (
        result.groupby("_group_id")["fold"].nunique()
    )

    if group_fold_counts.max() != 1:
        raise RuntimeError(
            "Patient leakage detected across folds."
        )

    return result.drop(columns=["_group_id"])