"""Reusable integrity and reproducibility audits for the frozen experiments."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

from src.models import build_model
from src.utils import load_config


DUPLICATE_CLASSES = {
    (True, True): "A_same_patient_same_split",
    (False, True): "B_different_patient_same_split",
    (True, False): "C_same_patient_cross_split",
    (False, False): "D_different_patient_cross_split",
}

EXPERIMENTS = {
    "B0": {
        "directory": "B0_resnet18",
        "config": "configs/resnet18.yaml",
        "checkpoint": "outputs/checkpoints/B0_resnet18_best.pt",
        "prediction": "outputs/predictions/B0_resnet18.csv",
    },
    "M1": {
        "directory": "M1_convnext_image",
        "config": "configs/convnext_image.yaml",
        "checkpoint": "outputs/checkpoints/M1_convnext_image_best.pt",
        "prediction": "outputs/predictions/M1_convnext_image.csv",
    },
    "M2": {
        "directory": "M2_convnext_metadata",
        "config": "configs/convnext_metadata.yaml",
        "checkpoint": "outputs/checkpoints/M2_convnext_metadata_best.pt",
        "prediction": "outputs/predictions/M2_convnext_metadata.csv",
    },
}


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_exact_image_frame(train_path, folds_path, image_directory):
    """Hash every JPEG and join authoritative patient, target, and fold data."""
    train = pd.read_csv(train_path)
    folds = pd.read_csv(folds_path)
    required_train = {"image_name", "patient_id", "target"}
    required_folds = {"image_name", "fold"}
    if not required_train.issubset(train.columns):
        raise ValueError("Training metadata lacks required audit columns.")
    if not required_folds.issubset(folds.columns):
        raise ValueError("Fold manifest lacks required audit columns.")
    if train["image_name"].duplicated().any():
        raise ValueError("Training metadata contains duplicate image names.")
    if folds["image_name"].duplicated().any():
        raise ValueError("Fold manifest contains duplicate image names.")

    image_paths = sorted(Path(image_directory).glob("*.jpg"))
    image_names = [path.stem for path in image_paths]
    if len(image_names) != len(set(image_names)):
        raise ValueError("Image directory contains duplicate JPEG stems.")
    expected_names = set(train["image_name"].astype(str))
    if set(image_names) != expected_names:
        missing = expected_names.difference(image_names)
        unexpected = set(image_names).difference(expected_names)
        raise ValueError(
            "Image/metadata mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}."
        )

    hashes = pd.DataFrame({
        "image_name": image_names,
        "sha256": [sha256_file(path) for path in image_paths],
    })
    result = hashes.merge(
        train[["image_name", "patient_id", "target"]],
        on="image_name",
        validate="one_to_one",
    ).merge(
        folds[["image_name", "fold"]],
        on="image_name",
        validate="one_to_one",
    )
    result["split_role"] = result["fold"].eq(0).map({
        True: "validation",
        False: "training",
    })
    return result.sort_values("image_name").reset_index(drop=True)


def classify_exact_duplicate_records(image_frame):
    """Return annotated records for every repeated image-content hash."""
    counts = image_frame.groupby("sha256")["image_name"].size()
    duplicate_hashes = set(counts.loc[counts > 1].index)
    columns = [
        "image_name",
        "sha256",
        "patient_id",
        "target",
        "fold",
        "split_role",
        "group_size",
        "patient_count",
        "split_count",
        "target_count",
        "duplicate_class",
        "target_conflict",
    ]
    if not duplicate_hashes:
        return pd.DataFrame(columns=columns)

    records = []
    duplicates = image_frame.loc[
        image_frame["sha256"].isin(duplicate_hashes)
    ]
    for _, group in duplicates.groupby("sha256", sort=True):
        patient_count = int(group["patient_id"].nunique(dropna=False))
        split_count = int(group["split_role"].nunique(dropna=False))
        target_count = int(group["target"].nunique(dropna=False))
        duplicate_class = DUPLICATE_CLASSES[
            (patient_count == 1, split_count == 1)
        ]
        annotated = group.copy()
        annotated["group_size"] = len(group)
        annotated["patient_count"] = patient_count
        annotated["split_count"] = split_count
        annotated["target_count"] = target_count
        annotated["duplicate_class"] = duplicate_class
        annotated["target_conflict"] = target_count > 1
        records.append(annotated)
    return (
        pd.concat(records, ignore_index=True)[columns]
        .sort_values(["sha256", "image_name"])
        .reset_index(drop=True)
    )


def summarize_exact_duplicates(image_frame, duplicate_records):
    """Create a JSON-safe duplicate-content summary."""
    groups = duplicate_records.drop_duplicates("sha256")
    class_counts = groups["duplicate_class"].value_counts()
    target_conflicts = groups.loc[groups["target_conflict"]]
    cross_split = groups.loc[groups["split_count"] > 1]
    different_patient = groups.loc[groups["patient_count"] > 1]
    return {
        "hash_algorithm": "SHA-256 of JPEG file bytes",
        "total_images_hashed": int(len(image_frame)),
        "unique_sha256_count": int(image_frame["sha256"].nunique()),
        "duplicate_hash_groups": int(len(groups)),
        "duplicate_image_records": int(len(duplicate_records)),
        "same_patient_same_split_groups": int(
            class_counts.get("A_same_patient_same_split", 0)
        ),
        "different_patient_same_split_groups": int(
            class_counts.get("B_different_patient_same_split", 0)
        ),
        "same_patient_cross_split_groups": int(
            class_counts.get("C_same_patient_cross_split", 0)
        ),
        "different_patient_cross_split_groups": int(
            class_counts.get("D_different_patient_cross_split", 0)
        ),
        "target_conflict_groups": int(len(target_conflicts)),
        "cross_split_groups": int(len(cross_split)),
        "different_patient_groups": int(len(different_patient)),
        "maximum_duplicate_group_size": (
            int(groups["group_size"].max()) if not groups.empty else 1
        ),
        "critical_leakage_detected": bool(
            len(cross_split) or len(target_conflicts)
        ),
        "classification": (
            "dataset_quality_and_within_patient_dependence"
            if len(groups) and not len(cross_split) and not len(target_conflicts)
            else "no_exact_duplicates" if groups.empty else "critical"
        ),
    }


def _build_model_for_parameter_count(project_root, config_path):
    config = load_config(project_root / config_path)
    model_config = config["model"]
    metadata_dim = None
    if model_config.get("use_metadata", False):
        summary_path = (
            project_root
            / "logs/M2_convnext_metadata/metadata_summary.json"
        )
        metadata_dim = json.loads(summary_path.read_text(encoding="utf-8"))[
            "metadata_dimension"
        ]
    return build_model(
        architecture=model_config["architecture"],
        pretrained=False,
        weights_name=model_config["weights"],
        num_outputs=model_config["num_outputs"],
        metadata_dim=metadata_dim,
        metadata_embedding_dim=model_config.get("metadata_embedding_dim", 32),
        metadata_activation=model_config.get("metadata_activation", "gelu"),
        metadata_dropout=model_config.get("metadata_dropout", 0.20),
    )


def build_training_timing_audit(project_root):
    """Summarize immutable timing records with explicit timing scopes."""
    project_root = Path(project_root)
    records = []
    for model_name, specification in EXPERIMENTS.items():
        log_directory = project_root / "logs" / specification["directory"]
        history = pd.read_csv(log_directory / "history.csv")
        metrics = json.loads(
            (log_directory / "metrics.json").read_text(encoding="utf-8")
        )
        model = _build_model_for_parameter_count(
            project_root, specification["config"]
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        sum_seconds = float(history["seconds"].sum())
        total_seconds = float(metrics["total_seconds"])
        records.append({
            "model": model_name,
            "epochs": int(len(history)),
            "parameters": int(parameter_count),
            "sum_epoch_seconds": sum_seconds,
            "mean_epoch_seconds": float(history["seconds"].mean()),
            "median_epoch_seconds": float(history["seconds"].median()),
            "reported_total_duration_seconds": total_seconds,
            "non_epoch_overhead_seconds": total_seconds - sum_seconds,
            "epoch_seconds_scope": "training and validation pass per epoch",
            "reported_total_scope": (
                "fit setup, epoch loops, checkpoint reload, and final "
                "validation; excludes final artifact writes"
            ),
        })
        del model
    return pd.DataFrame(records)


def _git_state(project_root, relative_path):
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_path],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    ).returncode == 0
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative_path],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    ).returncode == 0
    return "tracked" if tracked else "ignored" if ignored else "untracked"


def build_checkpoint_manifest(project_root):
    """Identify local checkpoints and the tracked evidence for each model."""
    project_root = Path(project_root)
    records = []
    for model_name, specification in EXPERIMENTS.items():
        relative_path = specification["checkpoint"]
        checkpoint = project_root / relative_path
        log_root = f"logs/{specification['directory']}"
        evidence = {
            "config": specification["config"],
            "environment": f"{log_root}/environment.json",
            "history": f"{log_root}/history.csv",
            "metrics": f"{log_root}/metrics.json",
            "training_log": f"{log_root}/training.log",
            "predictions": specification["prediction"],
        }
        records.append({
            "model": model_name,
            "path": relative_path,
            "exists_locally": checkpoint.is_file(),
            "bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            "git_state": _git_state(project_root, relative_path),
            "tracked_evidence": {
                key: {
                    "path": value,
                    "exists": (project_root / value).is_file(),
                    "git_state": _git_state(project_root, value),
                }
                for key, value in evidence.items()
            },
        })
    return {
        "checkpoint_policy": "local checkpoints remain Git-ignored",
        "models": records,
    }


def _write_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def generate_technical_audit_outputs(project_root):
    """Generate exact-duplicate, timing, and checkpoint audit artifacts."""
    project_root = Path(project_root).resolve()
    analysis_root = project_root / "outputs/analysis"
    integrity_root = analysis_root / "data_integrity"
    technical_root = analysis_root / "technical_audit"
    integrity_root.mkdir(parents=True, exist_ok=True)
    technical_root.mkdir(parents=True, exist_ok=True)

    image_frame = build_exact_image_frame(
        project_root / "data/train.csv",
        project_root / "data/train_folds.csv",
        project_root / "data/train_images",
    )
    duplicate_records = classify_exact_duplicate_records(image_frame)
    duplicate_summary = summarize_exact_duplicates(
        image_frame, duplicate_records
    )
    duplicate_records.to_csv(
        integrity_root / "exact_image_duplicates.csv", index=False
    )
    _write_json(
        duplicate_summary,
        integrity_root / "exact_image_duplicate_summary.json",
    )

    timing = build_training_timing_audit(project_root)
    timing.to_csv(technical_root / "training_timing_audit.csv", index=False)
    checkpoint_manifest = build_checkpoint_manifest(project_root)
    _write_json(
        checkpoint_manifest,
        technical_root / "checkpoint_manifest.json",
    )
    return {
        "duplicate_summary": duplicate_summary,
        "timing_rows": int(len(timing)),
        "checkpoint_models": len(checkpoint_manifest["models"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    result = generate_technical_audit_outputs(arguments.project_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
