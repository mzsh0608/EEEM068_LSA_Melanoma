"""Shared image transformations for melanoma classifiers."""

from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_interpolation(interpolation):
    if isinstance(interpolation, InterpolationMode):
        return interpolation

    if str(interpolation).lower() == "bilinear":
        return InterpolationMode.BILINEAR

    raise ValueError(
        f"Unsupported resize interpolation: {interpolation}"
    )


def get_train_transforms(
    image_size=224,
    horizontal_flip_probability=0.5,
    vertical_flip_probability=0.5,
    rotation_degrees=15,
    brightness=0.10,
    contrast=0.10,
    normalization_mean=None,
    normalization_std=None,
    resize_interpolation="bilinear",
):
    """Build the stochastic training transformation pipeline."""
    interpolation = _resolve_interpolation(resize_interpolation)
    mean = normalization_mean or IMAGENET_MEAN
    std = normalization_std or IMAGENET_STD

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=interpolation,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(
            p=horizontal_flip_probability
        ),
        transforms.RandomVerticalFlip(
            p=vertical_flip_probability
        ),
        transforms.RandomRotation(
            degrees=rotation_degrees,
            interpolation=interpolation,
        ),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(
    image_size=224,
    normalization_mean=None,
    normalization_std=None,
    resize_interpolation="bilinear",
):
    """Build the deterministic validation transformation pipeline."""
    interpolation = _resolve_interpolation(resize_interpolation)
    mean = normalization_mean or IMAGENET_MEAN
    std = normalization_std or IMAGENET_STD

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=interpolation,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
