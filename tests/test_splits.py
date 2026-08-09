import pandas as pd

from src.splits import create_patient_folds


def test_patient_groups_do_not_cross_folds():
    rows = []

    # 12 synthetic patients, two images each.
    # Alternating class at patient level.
    for patient_idx in range(12):
        target = patient_idx % 2

        for image_idx in range(2):
            rows.append({
                "image_name":
                    f"img_{patient_idx}_{image_idx}",
                "patient_id":
                    f"patient_{patient_idx}",
                "target":
                    target
            })

    df = pd.DataFrame(rows)

    result = create_patient_folds(
        df,
        n_splits=3,
        seed=42
    )

    # Every row must receive a fold.
    assert (result["fold"] >= 0).all()

    # Each patient must occur in exactly one fold.
    folds_per_patient = (
        result
        .groupby("patient_id")["fold"]
        .nunique()
    )

    assert folds_per_patient.max() == 1