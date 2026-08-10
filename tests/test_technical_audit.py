import pandas as pd

from src.technical_audit import (
    classify_exact_duplicate_records,
    sha256_file,
    summarize_exact_duplicates,
)


def test_sha256_file_uses_exact_file_bytes(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    third = tmp_path / "third.jpg"
    first.write_bytes(b"same image bytes")
    second.write_bytes(b"same image bytes")
    third.write_bytes(b"different image bytes")

    assert sha256_file(first) == sha256_file(second)
    assert sha256_file(first) != sha256_file(third)


def test_duplicate_classification_and_critical_summary():
    frame = pd.DataFrame({
        "image_name": ["a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2"],
        "sha256": ["a", "a", "b", "b", "c", "c", "d", "d"],
        "patient_id": ["p1", "p1", "p2", "p3", "p4", "p4", "p5", "p6"],
        "target": [0, 0, 0, 0, 1, 1, 0, 1],
        "fold": [1, 1, 2, 2, 0, 3, 0, 4],
        "split_role": [
            "training",
            "training",
            "training",
            "training",
            "validation",
            "training",
            "validation",
            "training",
        ],
    })

    duplicates = classify_exact_duplicate_records(frame)
    classes = set(duplicates["duplicate_class"])
    assert classes == {
        "A_same_patient_same_split",
        "B_different_patient_same_split",
        "C_same_patient_cross_split",
        "D_different_patient_cross_split",
    }
    summary = summarize_exact_duplicates(frame, duplicates)
    assert summary["duplicate_hash_groups"] == 4
    assert summary["duplicate_image_records"] == 8
    assert summary["same_patient_same_split_groups"] == 1
    assert summary["different_patient_same_split_groups"] == 1
    assert summary["same_patient_cross_split_groups"] == 1
    assert summary["different_patient_cross_split_groups"] == 1
    assert summary["target_conflict_groups"] == 1
    assert summary["critical_leakage_detected"] is True
