"""Model factory for image-classification experiments."""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(
    architecture,
    pretrained=True,
    weights_name="IMAGENET1K_V1",
    num_outputs=1,
):
    """Build a binary ResNet18 without an in-model sigmoid."""
    if str(architecture).lower() != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture}")

    if weights_name != "IMAGENET1K_V1":
        raise ValueError(f"Unsupported ResNet18 weights: {weights_name}")

    if int(num_outputs) != 1:
        raise ValueError("ResNet18 binary classification requires one output.")

    weights = (
        ResNet18_Weights.IMAGENET1K_V1
        if pretrained
        else None
    )
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, int(num_outputs))
    return model
