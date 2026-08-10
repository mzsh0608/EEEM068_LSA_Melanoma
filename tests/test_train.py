import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.train import fit_model, train_one_epoch, validate_one_epoch


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
