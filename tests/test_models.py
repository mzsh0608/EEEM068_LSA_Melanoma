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


def test_convnext_tiny_binary_output_shape():
    model = build_model("convnext_tiny", pretrained=False)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, 1)


def test_convnext_tiny_final_classifier_has_one_output():
    model = build_model("convnext_tiny", pretrained=False)

    assert model.classifier[-1].out_features == 1


@pytest.mark.parametrize(
    ("architecture", "metadata_dim"),
    [
        ("resnet18", None),
        ("convnext_tiny", None),
        ("convnext_tiny_metadata", 6),
    ],
)
def test_binary_models_do_not_contain_sigmoid(architecture, metadata_dim):
    model = build_model(
        architecture,
        pretrained=False,
        metadata_dim=metadata_dim,
    )

    assert not any(
        isinstance(module, torch.nn.Sigmoid) for module in model.modules()
    )


def test_unsupported_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model("unsupported", pretrained=False)


def test_unsupported_weights_are_rejected():
    with pytest.raises(ValueError, match="Unsupported ResNet18 weights"):
        build_model(
            "resnet18",
            pretrained=False,
            weights_name="UNKNOWN",
        )


def test_unsupported_convnext_weights_are_rejected():
    with pytest.raises(ValueError, match="Unsupported ConvNeXt-Tiny weights"):
        build_model(
            "convnext_tiny",
            pretrained=False,
            weights_name="UNKNOWN",
        )


def test_convnext_metadata_binary_output_shape():
    model = build_model(
        "convnext_tiny_metadata",
        pretrained=False,
        metadata_dim=7,
    )
    model.eval()

    with torch.no_grad():
        output = model(
            torch.randn(2, 3, 224, 224),
            torch.randn(2, 7),
        )

    assert output.shape == (2, 1)
    assert model.image_embedding_dim == 768
    assert model.metadata_mlp[0].in_features == 7
    assert model.metadata_mlp[0].out_features == 32
    assert model.fusion_classifier.out_features == 1


def test_convnext_metadata_rejects_wrong_metadata_dimension():
    model = build_model(
        "convnext_tiny_metadata",
        pretrained=False,
        metadata_dim=7,
    )

    with pytest.raises(ValueError, match="Expected metadata dimension 7"):
        model(torch.randn(1, 3, 64, 64), torch.randn(1, 6))


def test_convnext_metadata_all_branches_receive_finite_gradients():
    model = build_model(
        "convnext_tiny_metadata",
        pretrained=False,
        metadata_dim=5,
    )
    output = model(torch.randn(1, 3, 64, 64), torch.randn(1, 5))

    output.sum().backward()

    for module in [
        model.features,
        model.metadata_mlp,
        model.fusion_classifier,
    ]:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
