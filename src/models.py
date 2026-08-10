"""Model factory for image and image-metadata experiments."""

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    convnext_tiny,
    resnet18,
)


SUPPORTED_WEIGHTS = "IMAGENET1K_V1"


class ConvNeXtMetadataFusion(nn.Module):
    """Fuse the standard ConvNeXt image embedding with a small metadata MLP."""

    def __init__(
        self,
        weights,
        metadata_dim,
        metadata_embedding_dim=32,
        metadata_dropout=0.20,
    ):
        super().__init__()
        metadata_dim = int(metadata_dim)
        metadata_embedding_dim = int(metadata_embedding_dim)
        metadata_dropout = float(metadata_dropout)
        if metadata_dim < 1:
            raise ValueError("metadata_dim must be at least one.")
        if metadata_embedding_dim < 1:
            raise ValueError("metadata_embedding_dim must be at least one.")
        if not 0 <= metadata_dropout < 1:
            raise ValueError("metadata_dropout must lie in [0, 1).")

        backbone = convnext_tiny(weights=weights)
        image_classifier = get_final_classifier(backbone)
        self.metadata_dim = metadata_dim
        self.metadata_embedding_dim = metadata_embedding_dim
        self.image_embedding_dim = int(image_classifier.in_features)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.image_pre_classifier = nn.Sequential(
            *list(backbone.classifier.children())[:-1]
        )
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, metadata_embedding_dim),
            nn.GELU(),
            nn.Dropout(metadata_dropout),
        )
        self.fusion_classifier = nn.Linear(
            self.image_embedding_dim + metadata_embedding_dim,
            1,
        )

    def forward(self, image, metadata):
        if metadata.ndim != 2:
            raise ValueError("metadata must have shape [batch, metadata_dim].")
        if metadata.shape[0] != image.shape[0]:
            raise ValueError("Image and metadata batch sizes must match.")
        if metadata.shape[1] != self.metadata_dim:
            raise ValueError(
                f"Expected metadata dimension {self.metadata_dim}; "
                f"received {metadata.shape[1]}."
            )

        image_embedding = self.features(image)
        image_embedding = self.avgpool(image_embedding)
        image_embedding = self.image_pre_classifier(image_embedding)
        metadata_embedding = self.metadata_mlp(metadata)
        fused = torch.cat([image_embedding, metadata_embedding], dim=1)
        return self.fusion_classifier(fused)


def get_final_classifier(model):
    """Return the validated final linear classifier of a supported model."""
    if isinstance(getattr(model, "fusion_classifier", None), nn.Linear):
        return model.fusion_classifier

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
    metadata_dim=None,
    metadata_embedding_dim=32,
    metadata_activation="gelu",
    metadata_dropout=0.20,
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

    if architecture == "convnext_tiny_metadata":
        if weights_name != SUPPORTED_WEIGHTS:
            raise ValueError(
                f"Unsupported ConvNeXt-Tiny weights: {weights_name}"
            )
        if metadata_dim is None:
            raise ValueError("metadata_dim is required for metadata fusion.")
        if str(metadata_activation).lower() != "gelu":
            raise ValueError("Only GELU metadata activation is supported.")

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        return ConvNeXtMetadataFusion(
            weights=weights,
            metadata_dim=metadata_dim,
            metadata_embedding_dim=metadata_embedding_dim,
            metadata_dropout=metadata_dropout,
        )

    raise ValueError(f"Unsupported architecture: {architecture}")
