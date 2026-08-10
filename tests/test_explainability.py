import numpy as np
import pytest
import torch
from torch import nn

from src.explainability import (
    GradCAM,
    overlay_cam,
    select_last_convnext_spatial_layer,
)
from src.models import build_model


class TinyFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.metadata = nn.Linear(2, 2)
        self.classifier = nn.Linear(6, 1)

    def forward(self, image, metadata):
        image_features = self.pool(self.features(image)).flatten(1)
        metadata_features = self.metadata(metadata)
        return self.classifier(
            torch.cat([image_features, metadata_features], dim=1)
        )


def test_gradcam_captures_spatial_activation_gradient_and_resizes():
    model = TinyFusionModel().eval()
    with GradCAM(model, model.features) as gradcam:
        result = gradcam.generate(
            torch.randn(2, 3, 8, 8),
            torch.randn(2, 2),
            output_size=(16, 12),
        )

    cam = result["cam"]
    assert result["activation_shape"] == [2, 4, 8, 8]
    assert result["gradient_shape"] == [2, 4, 8, 8]
    assert cam.shape == (2, 16, 12)
    assert torch.isfinite(cam).all()
    assert cam.min() >= 0
    assert cam.max() <= 1


def test_gradcam_repeated_execution_does_not_accumulate_hooks():
    model = TinyFusionModel().eval()
    initial_hooks = len(model.features._forward_hooks)
    gradcam = GradCAM(model, model.features)
    assert len(model.features._forward_hooks) == initial_hooks + 1

    gradcam.generate(torch.randn(1, 3, 8, 8), torch.randn(1, 2))
    gradcam.generate(torch.randn(1, 3, 8, 8), torch.randn(1, 2))

    assert len(model.features._forward_hooks) == initial_hooks + 1
    gradcam.close()
    gradcam.close()
    assert len(model.features._forward_hooks) == initial_hooks


def test_gradcam_rejects_nonspatial_target_layer():
    model = TinyFusionModel().eval()
    with GradCAM(model, model.classifier) as gradcam:
        with pytest.raises(RuntimeError, match=r"shape \[B,C,H,W\]"):
            gradcam.generate(
                torch.randn(1, 3, 8, 8),
                torch.randn(1, 2),
            )


def test_overlay_cam_preserves_shape_and_range():
    image = np.full((8, 6, 3), 128, dtype=np.uint8)
    cam = np.linspace(0, 1, 48, dtype=np.float32).reshape(8, 6)

    overlay = overlay_cam(image, cam)

    assert overlay.shape == image.shape
    assert np.isfinite(overlay).all()
    assert overlay.min() >= 0
    assert overlay.max() <= 1


def test_convnext_metadata_exposes_valid_last_spatial_target():
    model = build_model(
        "convnext_tiny_metadata",
        pretrained=False,
        metadata_dim=5,
    ).eval()
    target_layer, target_name = select_last_convnext_spatial_layer(model)

    with GradCAM(model, target_layer, target_name) as gradcam:
        result = gradcam.generate(
            torch.randn(1, 3, 64, 64),
            torch.randn(1, 5),
            output_size=(64, 64),
        )

    assert target_name == "features.7.2"
    assert result["activation_shape"] == [1, 768, 2, 2]
    assert result["gradient_shape"] == [1, 768, 2, 2]
    assert result["cam"].shape == (1, 64, 64)
