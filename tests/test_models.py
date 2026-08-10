import pytest
import torch

from src.models import build_model


def test_resnet18_binary_output_shape():
    model = build_model("resnet18", pretrained=False)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, 1)


def test_resnet18_final_classifier_has_one_output():
    model = build_model("resnet18", pretrained=False)

    assert model.fc.out_features == 1


def test_unsupported_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model("convnext_tiny", pretrained=False)


def test_unsupported_weights_are_rejected():
    with pytest.raises(ValueError, match="Unsupported ResNet18 weights"):
        build_model(
            "resnet18",
            pretrained=False,
            weights_name="UNKNOWN",
        )
