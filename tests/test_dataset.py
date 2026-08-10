from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.dataset import (
    MelanomaDataset,
    create_dataloaders,
    load_fold_dataframes,
)
from src.transforms import get_train_transforms, get_val_transforms


def _write_image(path, colour=(100, 120, 140)):
    Image.new("RGB", (32, 48), colour).save(path, format="JPEG")


def _dataframe(image_paths):
    return pd.DataFrame({
        "image_name": [path.stem for path in image_paths],
        "patient_id": [f"patient_{i}" for i in range(len(image_paths))],
        "target": [i % 2 for i in range(len(image_paths))],
        "image_path": image_paths,
    })


def test_validation_dataset_output(tmp_path):
    image_path = tmp_path / "image_a.jpg"
    _write_image(image_path)
    dataset = MelanomaDataset(
        _dataframe([image_path]),
        transform=get_val_transforms(),
    )

    sample = dataset[0]

    assert sample["image"].shape == (3, 224, 224)
    assert sample["image"].dtype == torch.float32
    assert sample["target"].dtype == torch.float32
    assert sample["target"].ndim == 0
    assert sample["image_name"] == "image_a"
    assert sample["patient_id"] == "patient_0"


def test_training_transform_output_shape(tmp_path):
    image_path = tmp_path / "image_a.jpg"
    _write_image(image_path)
    dataset = MelanomaDataset(
        _dataframe([image_path]),
        transform=get_train_transforms(),
    )

    assert dataset[0]["image"].shape == (3, 224, 224)


def test_validation_transform_is_deterministic(tmp_path):
    image_path = tmp_path / "image_a.jpg"
    _write_image(image_path)
    transform = get_val_transforms()
    dataset = MelanomaDataset(
        _dataframe([image_path]),
        transform=transform,
    )

    assert torch.equal(dataset[0]["image"], dataset[0]["image"])


def test_missing_image_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "missing.jpg"
    dataset = MelanomaDataset(_dataframe([missing_path]))

    with pytest.raises(FileNotFoundError, match="Image not found"):
        dataset[0]


def test_missing_required_column_is_rejected(tmp_path):
    image_path = tmp_path / "image_a.jpg"
    frame = _dataframe([image_path]).drop(columns="patient_id")

    with pytest.raises(ValueError, match="Missing required columns"):
        MelanomaDataset(frame)


def test_duplicate_image_name_is_rejected(tmp_path):
    first = tmp_path / "image_a.jpg"
    second = tmp_path / "image_b.jpg"
    frame = _dataframe([first, second])
    frame.loc[1, "image_name"] = frame.loc[0, "image_name"]

    with pytest.raises(ValueError, match="Duplicate image_name"):
        MelanomaDataset(frame)


def test_load_fold_dataframes_respects_manifest(tmp_path):
    train_csv = tmp_path / "train.csv"
    fold_csv = tmp_path / "folds.csv"
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    metadata = pd.DataFrame({
        "image_name": ["a", "b", "c", "d"],
        "patient_id": ["p1", "p1", "p2", "p3"],
        "target": [0, 0, 1, 0],
    })
    folds = pd.DataFrame({
        "image_name": ["a", "b", "c", "d"],
        "fold": [1, 1, 0, 2],
    })
    metadata.to_csv(train_csv, index=False)
    folds.to_csv(fold_csv, index=False)

    train_df, val_df = load_fold_dataframes(
        train_csv,
        fold_csv,
        image_dir,
        validation_fold=0,
    )

    assert train_df["image_name"].tolist() == ["a", "b", "d"]
    assert val_df["image_name"].tolist() == ["c"]
    assert set(train_df["patient_id"]).isdisjoint(val_df["patient_id"])
    assert val_df.loc[0, "fold"] == 0
    assert val_df.loc[0, "image_path"] == image_dir / "c.jpg"


def test_create_dataloaders_with_zero_workers(tmp_path):
    image_paths = []
    for index in range(4):
        path = tmp_path / f"image_{index}.jpg"
        _write_image(path, colour=(index * 20, 100, 150))
        image_paths.append(path)

    frame = _dataframe(image_paths)
    train_dataset = MelanomaDataset(
        frame,
        transform=get_val_transforms(),
    )
    val_dataset = MelanomaDataset(
        frame,
        transform=get_val_transforms(),
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=2,
        num_workers=0,
        persistent_workers=True,
    )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert train_batch["image"].shape == (2, 3, 224, 224)
    assert val_batch["image"].shape == (2, 3, 224, 224)
    assert train_loader.persistent_workers is False
    assert val_loader.persistent_workers is False


def test_dataset_metadata_mode_returns_encoded_float32_vector(tmp_path):
    image_paths = []
    for index in range(2):
        path = tmp_path / f"image_{index}.jpg"
        _write_image(path)
        image_paths.append(path)
    metadata = np.array(
        [[-0.5, 1.0, 0.0], [0.5, 0.0, 1.0]],
        dtype=np.float32,
    )
    dataset = MelanomaDataset(
        _dataframe(image_paths),
        transform=get_val_transforms(),
        metadata_array=metadata,
        use_metadata=True,
    )

    sample = dataset[0]

    assert sample["metadata"].shape == (3,)
    assert sample["metadata"].dtype == torch.float32
    assert torch.isfinite(sample["metadata"]).all()


def test_dataloader_collates_metadata_as_batch_by_dimension(tmp_path):
    image_paths = []
    for index in range(4):
        path = tmp_path / f"image_{index}.jpg"
        _write_image(path)
        image_paths.append(path)
    metadata = np.arange(20, dtype=np.float32).reshape(4, 5)
    dataset = MelanomaDataset(
        _dataframe(image_paths),
        transform=get_val_transforms(),
        metadata_array=metadata,
        use_metadata=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    batch = next(iter(loader))

    assert batch["metadata"].shape == (2, 5)
    assert batch["metadata"].dtype == torch.float32
    assert torch.isfinite(batch["metadata"]).all()


def test_image_only_dataset_does_not_expose_metadata(tmp_path):
    image_path = tmp_path / "image_a.jpg"
    _write_image(image_path)
    dataset = MelanomaDataset(
        _dataframe([image_path]),
        transform=get_val_transforms(),
        use_metadata=False,
    )

    assert "metadata" not in dataset[0]
