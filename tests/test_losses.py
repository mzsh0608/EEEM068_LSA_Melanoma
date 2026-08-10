import pytest
import torch

from src.losses import build_loss, calculate_pos_weight


def test_known_pos_weight_calculation():
    assert calculate_pos_weight([0, 0, 0, 1]) == 3.0


@pytest.mark.parametrize("loss_name", ["bce", "weighted_bce"])
def test_bce_losses_are_finite(loss_name):
    criterion, info = build_loss(
        loss_name,
        train_targets=[0, 0, 1],
        device=torch.device("cpu"),
    )
    loss = criterion(
        torch.tensor([0.1, -0.2, 0.3]),
        torch.tensor([0.0, 0.0, 1.0]),
    )

    assert torch.isfinite(loss)
    assert info["loss"] == loss_name


def test_empty_targets_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        calculate_pos_weight([])


def test_non_binary_targets_are_rejected():
    with pytest.raises(ValueError, match="only 0 and 1"):
        calculate_pos_weight([0, 0.5, 1])


def test_targets_without_positives_are_rejected():
    with pytest.raises(ValueError, match="without positives"):
        calculate_pos_weight([0, 0, 0])
