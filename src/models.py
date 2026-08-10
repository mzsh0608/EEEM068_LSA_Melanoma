"""Model factory for image-classification experiments."""

from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    convnext_tiny,
    resnet18,
)


SUPPORTED_WEIGHTS = "IMAGENET1K_V1"


def get_final_classifier(model):
    """Return the validated final linear classifier of a supported model."""
    if isinstance(getattr(model, "fc", None), nn.Linear):
        return model.fc

    classifier = getattr(model, "classifier", None)
    if (
        isinstance(classifier, nn.Sequential)
        and len(classifier) > 0
        and isinstance(classifier[-1], nn.Linear)
    ):
        return classifier[-1]

    raise RuntimeError("Supported model must expose a final linear classifier.")


def build_model(
    architecture,
    pretrained=True,
    weights_name="IMAGENET1K_V1",
    num_outputs=1,
):
    """Build a supported image model with one raw binary logit."""
    architecture = str(architecture).lower()

    if int(num_outputs) != 1:
        raise ValueError("Binary classification requires one output.")

    if architecture == "resnet18":
        if weights_name != SUPPORTED_WEIGHTS:
            raise ValueError(f"Unsupported ResNet18 weights: {weights_name}")

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        classifier = get_final_classifier(model)
        model.fc = nn.Linear(classifier.in_features, int(num_outputs))
        return model

    if architecture == "convnext_tiny":
        if weights_name != SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported ConvNeXt-Tiny weights: {weights_name}"
            )

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        classifier = get_final_classifier(model)
        model.classifier[-1] = nn.Linear(
            classifier.in_features,
            int(num_outputs),
        )
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")
