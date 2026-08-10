import json

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.train import (
    fit_model,
    parse_args,
    run_full_experiment,
    train_one_epoch,
    validate_one_epoch,
)


class TinyDataset(Dataset):
    def __init__(self):
        self.targets = [0.0, 1.0, 0.0, 1.0]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "image": torch.full((3, 8, 8), float(index) / 4),
            "target": torch.tensor(self.targets[index]),
            "image_name": f"image_{index}",
            "patient_id": f"patient_{index}",
        }


def _components(batch_size=1):
    loader = DataLoader(TinyDataset(), batch_size=batch_size, shuffle=False)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 1))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    return loader, model, criterion, optimizer


def test_train_one_epoch_preserves_batch_dimension():
    loader, model, criterion, optimizer = _components(batch_size=1)

    loss = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
    )

    assert loss > 0


def test_validate_one_epoch_returns_metrics_and_predictions():
    loader, model, criterion, _ = _components(batch_size=2)

    metrics, predictions = validate_one_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
    )

    assert metrics["n_samples"] == 4
    assert "loss" in metrics
    assert predictions.shape == (4, 5)
    assert list(predictions.columns) == [
        "image_name",
        "patient_id",
        "target",
        "probability",
        "prediction",
    ]


def test_fit_model_writes_reproducible_artifacts(tmp_path):
    loader, model, criterion, optimizer = _components(batch_size=2)
    config = {
        "training": {
            "epochs": 2,
            "amp": False,
            "early_stopping": {"enabled": True, "patience": 1},
        },
        "evaluation": {
            "threshold": 0.5,
            "checkpoint_metric": "roc_auc",
            "checkpoint_mode": "max",
        },
        "outputs": {
            "checkpoint": str(tmp_path / "best.pt"),
            "predictions": str(tmp_path / "predictions.csv"),
            "training_history": str(tmp_path / "history.csv"),
            "metrics": str(tmp_path / "metrics.json"),
            "environment": str(tmp_path / "environment.json"),
            "training_log": str(tmp_path / "training.log"),
        },
    }

    result = fit_model(
        model,
        loader,
        loader,
        criterion,
        optimizer,
        torch.device("cpu"),
        config,
    )

    assert result["best_epoch"] >= 1
    assert isinstance(result["history"], pd.DataFrame)
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "history.csv").is_file()
    assert (tmp_path / "predictions.csv").is_file()
    assert (tmp_path / "metrics.json").is_file()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["best_epoch"] == result["best_epoch"]
    assert metrics["actual_epochs"] == len(result["history"])


def test_cli_requires_explicit_fit():
    setup_arguments = parse_args(["--config", "config.yaml"])
    fit_arguments = parse_args(["--config", "config.yaml", "--fit"])

    assert setup_arguments.fit is False
    assert setup_arguments.smoke_test is False
    assert fit_arguments.fit is True


def test_full_experiment_creates_required_artifacts(tmp_path):
    loader, model, criterion, optimizer = _components(batch_size=2)
    baseline_directory = tmp_path / "outputs" / "predictions"
    baseline_directory.mkdir(parents=True)
    baseline = pd.DataFrame({
        "image_name": [f"image_{index}" for index in range(4)],
        "patient_id": [f"patient_{index}" for index in range(4)],
        "target": [0, 1, 0, 1],
        "probability": [0.1, 0.8, 0.2, 0.7],
        "prediction": [0, 1, 0, 1],
    })
    baseline.to_csv(
        baseline_directory / "H0_logistic_unweighted.csv", index=False
    )
    baseline.to_csv(
        baseline_directory / "H1_logistic_weighted.csv", index=False
    )
    baseline.to_csv(baseline_directory / "B0_resnet18.csv", index=False)

    experiment_directory = tmp_path / "logs" / "M1_convnext_image"
    config = {
        "experiment_id": "M1",
        "experiment_name": "test_convnext_image",
        "seed": 42,
        "validation_fold": 0,
        "model": {
            "architecture": "convnext_tiny",
            "weights": "IMAGENET1K_V1",
            "fine_tune_all": True,
        },
        "data": {"image_size": 224, "batch_size": 2},
        "transforms": {
            "resize_interpolation": "bilinear",
            "horizontal_flip_probability": 0.5,
            "vertical_flip_probability": 0.5,
            "rotation_degrees": 15,
            "brightness": 0.1,
            "contrast": 0.1,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "training": {
            "epochs": 2,
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "loss": "weighted_bce",
            "amp": False,
            "early_stopping": {"enabled": True, "patience": 1},
        },
        "evaluation": {
            "threshold": 0.5,
            "checkpoint_metric": "roc_auc",
            "checkpoint_mode": "max",
        },
        "outputs": {
            "experiment_directory": str(experiment_directory),
            "checkpoint": str(tmp_path / "best.pt"),
            "predictions": str(tmp_path / "predictions.csv"),
            "training_history": str(experiment_directory / "history.csv"),
            "metrics": str(experiment_directory / "metrics.json"),
            "environment": str(experiment_directory / "environment.json"),
            "training_log": str(experiment_directory / "training.log"),
            "training_curves": str(tmp_path / "training_curves.png"),
            "confusion_matrix": str(tmp_path / "confusion_matrix.png"),
            "roc_pr_curve": str(tmp_path / "roc_pr.png"),
        },
    }
    frame = pd.DataFrame({"target": [0, 1, 0, 1]})
    experiment = {
        "device": torch.device("cpu"),
        "train_df": frame.copy(),
        "val_df": frame.copy(),
        "train_loader": loader,
        "val_loader": loader,
        "model": model,
        "criterion": criterion,
        "loss_info": {"loss": "weighted_bce", "pos_weight": 1.0},
        "optimizer": optimizer,
    }

    result = run_full_experiment(experiment, config, tmp_path)

    assert result["predictions"].shape[0] == 4
    resolved = json.loads(
        (experiment_directory / "config.json").read_text()
    )["resolved"]
    expected_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert resolved["total_parameters"] == expected_parameters
    assert resolved["trainable_parameters"] == expected_parameters
    for path in [
        experiment_directory / "config.json",
        experiment_directory / "environment.json",
        experiment_directory / "history.csv",
        experiment_directory / "metrics.json",
        experiment_directory / "training.log",
        tmp_path / "best.pt",
        tmp_path / "predictions.csv",
        tmp_path / "training_curves.png",
        tmp_path / "confusion_matrix.png",
        tmp_path / "roc_pr.png",
    ]:
        assert path.is_file()
        assert path.stat().st_size > 0
