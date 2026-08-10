"""Loss construction for imbalanced binary classification."""

import numpy as np
import torch
from torch import nn


def calculate_pos_weight(targets):
    """Return negative/positive count using binary training targets."""
    if isinstance(targets, torch.Tensor):
        values = targets.detach().cpu().numpy()
    else:
        values = np.asarray(targets)

    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot calculate pos_weight from empty targets.")

    try:
        values = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("targets must be numeric binary values.") from exc

    if not np.all(np.isfinite(values)) or not np.all(
        np.isin(values, [0, 1])
    ):
        raise ValueError("targets must contain only 0 and 1.")

    positives = int(np.sum(values == 1))
    negatives = int(np.sum(values == 0))
    if positives == 0:
        raise ValueError("Cannot calculate pos_weight without positives.")

    return negatives / positives


def build_loss(loss_name, train_targets, device):
    """Build BCE loss and return its reproducibility metadata."""
    normalized_name = str(loss_name).lower()

    if normalized_name == "bce":
        return nn.BCEWithLogitsLoss(), {
            "loss": "bce",
            "pos_weight": None,
        }

    if normalized_name == "weighted_bce":
        weight = calculate_pos_weight(train_targets)
        pos_weight = torch.tensor(
            [weight],
            dtype=torch.float32,
            device=device,
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight), {
            "loss": "weighted_bce",
            "pos_weight": float(weight),
        }

    raise ValueError(f"Unsupported loss: {loss_name}")
