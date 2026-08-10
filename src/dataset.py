"""Datasets and DataLoaders for fixed-fold melanoma experiments."""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REQUIRED_COLUMNS = {
    "image_name",
    "patient_id",
    "target",
    "image_path",
}


class MelanomaDataset(Dataset):
    """Map a dataframe row to one RGB image-classification sample."""

    def __init__(
        self,
        dataframe,
        transform=None,
        metadata_columns=None,
        use_metadata=False,
    ):
        self.dataframe = dataframe.copy().reset_index(drop=True)
        self.transform = transform
        self.metadata_columns = list(metadata_columns or [])
        self.use_metadata = use_metadata

        missing = REQUIRED_COLUMNS.difference(self.dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        if self.dataframe["image_name"].duplicated().any():
            raise ValueError("Duplicate image_name values detected.")

        targets = pd.to_numeric(
            self.dataframe["target"],
            errors="coerce",
        )
        if targets.isna().any() or not targets.isin([0, 1]).all():
            raise ValueError("target must contain only 0 and 1.")

        if self.use_metadata:
            if not self.metadata_columns:
                raise ValueError(
                    "metadata_columns are required when use_metadata=True."
                )
            missing_metadata = set(self.metadata_columns).difference(
                self.dataframe.columns
            )
            if missing_metadata:
                raise ValueError(
                    "Missing metadata columns: "
                    f"{sorted(missing_metadata)}"
                )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = Path(row["image_path"])

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        sample = {
            "image": image,
            "target": torch.tensor(
                float(row["target"]),
                dtype=torch.float32,
            ),
            "image_name": str(row["image_name"]),
            "patient_id": str(row["patient_id"]),
        }

        if self.use_metadata:
            values = row[self.metadata_columns].to_numpy(
                dtype=np.float32
            )
            sample["metadata"] = torch.as_tensor(
                values,
                dtype=torch.float32,
            )

        return sample


def load_fold_dataframes(
    train_csv,
    fold_csv,
    image_dir,
    validation_fold=0,
):
    """Load the permanent fold manifest without regenerating folds."""
    metadata = pd.read_csv(train_csv)
    folds = pd.read_csv(fold_csv)

    if "image_name" not in metadata.columns:
        raise ValueError("train.csv is missing image_name.")
    if not {"image_name", "fold"}.issubset(folds.columns):
        raise ValueError("fold CSV must contain image_name and fold.")

    if metadata["image_name"].duplicated().any():
        raise ValueError("Duplicate image_name values in train.csv.")
    if folds["image_name"].duplicated().any():
        raise ValueError("Duplicate image_name values in fold CSV.")

    extra_fold_images = set(folds["image_name"]).difference(
        metadata["image_name"]
    )
    if extra_fold_images:
        raise ValueError("Fold CSV contains images absent from train.csv.")

    merged = metadata.merge(
        folds[["image_name", "fold"]],
        on="image_name",
        how="left",
        validate="one_to_one",
    )

    if merged["fold"].isna().any():
        raise ValueError("Every training row must have a fold assignment.")

    numeric_folds = pd.to_numeric(merged["fold"], errors="coerce")
    if numeric_folds.isna().any() or not np.equal(
        numeric_folds,
        np.floor(numeric_folds),
    ).all():
        raise ValueError("Fold assignments must be integers.")
    merged["fold"] = numeric_folds.astype(int)

    image_dir = Path(image_dir)
    merged["image_path"] = merged["image_name"].map(
        lambda image_name: image_dir / f"{image_name}.jpg"
    )

    train_df = merged[
        merged["fold"] != int(validation_fold)
    ].copy().reset_index(drop=True)
    val_df = merged[
        merged["fold"] == int(validation_fold)
    ].copy().reset_index(drop=True)

    train_patients = set(train_df["patient_id"].dropna().astype(str))
    val_patients = set(val_df["patient_id"].dropna().astype(str))
    overlap = train_patients.intersection(val_patients)
    if overlap:
        raise ValueError(
            f"Patient overlap detected across train/validation: {len(overlap)}"
        )

    return train_df, val_df


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=False,
    seed=42,
):
    """Create shuffled training and deterministic validation loaders."""
    persistent_workers = bool(persistent_workers and num_workers > 0)
    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": drop_last,
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        sampler=None,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=None,
        **common,
    )

    return train_loader, val_loader
