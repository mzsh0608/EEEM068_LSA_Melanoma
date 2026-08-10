"""Hook-based Grad-CAM for the frozen M2 image branch."""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis import CONFUSION_CATEGORIES
from src.metadata import transform_metadata
from src.models import build_model
from src.transforms import get_val_transforms
from src.utils import load_config


class GradCAM:
    """Capture one spatial activation and its raw-logit gradient."""

    def __init__(self, model, target_layer, target_layer_name=None):
        self.model = model
        self.target_layer = target_layer
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        self._tensor_hook = None
        self._closed = False
        self._forward_hook = target_layer.register_forward_hook(
            self._capture_activation
        )

    def _capture_activation(self, module, inputs, output):
        del module, inputs
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("Grad-CAM target layer must return a tensor.")
        if self._tensor_hook is not None:
            self._tensor_hook.remove()
        self.activations = output
        self.gradients = None
        self._tensor_hook = output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient):
        self.gradients = gradient

    def generate(self, image, metadata=None, output_size=None):
        """Backpropagate the raw melanoma logit and return normalized CAMs."""
        if self._closed:
            raise RuntimeError("GradCAM hooks have been closed.")
        self.model.zero_grad(set_to_none=True)
        if metadata is None:
            raw_logits = self.model(image)
        else:
            raw_logits = self.model(image, metadata)
        if raw_logits.shape != (image.shape[0], 1):
            raise RuntimeError(
                "Grad-CAM requires one raw logit per image; "
                f"received {list(raw_logits.shape)}."
            )
        raw_logits[:, 0].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture tensors.")
        if self.activations.ndim != 4:
            raise RuntimeError(
                "Grad-CAM target activation must have shape [B,C,H,W]."
            )
        if self.gradients.shape != self.activations.shape:
            raise RuntimeError("Grad-CAM activation and gradient shapes differ.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1))
        flat = cam.flatten(start_dim=1)
        minimum = flat.min(dim=1).values[:, None, None]
        maximum = flat.max(dim=1).values[:, None, None]
        span = maximum - minimum
        cam = torch.where(
            span > torch.finfo(cam.dtype).eps,
            (cam - minimum) / span.clamp_min(torch.finfo(cam.dtype).eps),
            torch.zeros_like(cam),
        )
        if output_size is not None:
            cam = functional.interpolate(
                cam[:, None],
                size=tuple(int(value) for value in output_size),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        if not torch.isfinite(cam).all():
            raise RuntimeError("Grad-CAM produced non-finite values.")
        return {
            "cam": cam,
            "raw_logits": raw_logits.detach(),
            "activation_shape": list(activations.shape),
            "gradient_shape": list(gradients.shape),
        }

    def close(self):
        """Remove all persistent and tensor hooks."""
        if self._closed:
            return
        self._forward_hook.remove()
        if self._tensor_hook is not None:
            self._tensor_hook.remove()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        self.close()


def select_last_convnext_spatial_layer(model):
    """Return the final ConvNeXt block before adaptive pooling by identity."""
    features = getattr(model, "features", None)
    if not isinstance(features, torch.nn.Sequential) or not features:
        raise ValueError("Model does not expose ConvNeXt spatial features.")
    final_stage = features[-1]
    if not isinstance(final_stage, torch.nn.Sequential) or not final_stage:
        raise ValueError("ConvNeXt final feature stage is unavailable.")
    target_layer = final_stage[-1]
    names = [
        name
        for name, module in model.named_modules()
        if module is target_layer
    ]
    if len(names) != 1:
        raise RuntimeError("Could not identify a unique Grad-CAM target path.")
    return target_layer, names[0]


def overlay_cam(image, cam, alpha=0.45, colormap="jet"):
    """Blend a normalized CAM over an RGB image without changing content."""
    image = np.asarray(image, dtype=np.float32)
    cam = np.asarray(cam, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape [H,W,3].")
    if cam.shape != image.shape[:2]:
        raise ValueError("CAM dimensions must match image height and width.")
    if not np.isfinite(cam).all() or cam.min() < 0 or cam.max() > 1:
        raise ValueError("CAM must be finite and normalized to [0,1].")
    if image.max() > 1:
        image = image / 255.0
    heatmap = plt.get_cmap(colormap)(cam)[..., :3].astype(np.float32)
    blended = (1 - float(alpha)) * image + float(alpha) * heatmap
    return np.clip(blended, 0.0, 1.0)


def load_m2_explainability_components(project_root, device=None):
    """Load the frozen M2 checkpoint and saved metadata preprocessor."""
    project_root = Path(project_root)
    config = load_config(project_root / "configs/convnext_metadata.yaml")
    summary = json.loads(
        (
            project_root
            / "logs/M2_convnext_metadata/metadata_summary.json"
        ).read_text(encoding="utf-8")
    )
    model_config = config["model"]
    model = build_model(
        architecture=model_config["architecture"],
        pretrained=False,
        weights_name=model_config["weights"],
        num_outputs=model_config["num_outputs"],
        metadata_dim=summary["metadata_dimension"],
        metadata_embedding_dim=model_config["metadata_embedding_dim"],
        metadata_activation=model_config["metadata_activation"],
        metadata_dropout=model_config["metadata_dropout"],
    )
    checkpoint_path = (
        project_root / "outputs/checkpoints/M2_convnext_metadata_best.pt"
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device).eval()

    preprocessor_path = (
        project_root
        / "logs/M2_convnext_metadata/metadata_preprocessor.joblib"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        preprocessor = joblib.load(preprocessor_path)
    target_layer, target_layer_name = select_last_convnext_spatial_layer(model)
    transform = get_val_transforms(
        image_size=config["data"]["image_size"],
        normalization_mean=config["transforms"]["normalization_mean"],
        normalization_std=config["transforms"]["normalization_std"],
        resize_interpolation=config["transforms"]["resize_interpolation"],
    )
    return {
        "model": model,
        "preprocessor": preprocessor,
        "transform": transform,
        "target_layer": target_layer,
        "target_layer_name": target_layer_name,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "preprocessor_path": preprocessor_path,
    }


def generate_m2_gradcam(project_root, cases_per_category=3):
    """Generate Grad-CAM overlays for deterministic M2 failure cases."""
    project_root = Path(project_root)
    failure_path = (
        project_root / "outputs/analysis/failures/M2_failure_cases.csv"
    )
    cases = pd.read_csv(failure_path)
    selected = pd.concat(
        [
            cases.loc[cases["category"] == category]
            .sort_values("rank")
            .head(int(cases_per_category))
            for category in CONFUSION_CATEGORIES
        ],
        ignore_index=True,
    )
    if set(selected["category"]) != set(CONFUSION_CATEGORIES):
        raise RuntimeError("Every confusion category must have Grad-CAM cases.")

    components = load_m2_explainability_components(project_root)
    model = components["model"]
    device = components["device"]
    metadata_matrix = transform_metadata(
        components["preprocessor"], selected
    )
    figure, axes = plt.subplots(
        4,
        int(cases_per_category) * 2,
        figsize=(4.4 * int(cases_per_category), 12.5),
    )
    case_records = []
    with GradCAM(
        model,
        components["target_layer"],
        components["target_layer_name"],
    ) as gradcam:
        for row_index, category in enumerate(CONFUSION_CATEGORIES):
            subset = selected.loc[selected["category"] == category]
            for case_index, (frame_index, item) in enumerate(subset.iterrows()):
                image_path = project_root / item["image_path"]
                with Image.open(image_path) as source:
                    original = source.convert("RGB")
                    original_array = np.asarray(original)
                    image_tensor = components["transform"](original)
                metadata_tensor = torch.as_tensor(
                    metadata_matrix[frame_index],
                    dtype=torch.float32,
                    device=device,
                )[None]
                result = gradcam.generate(
                    image_tensor[None].to(device),
                    metadata_tensor,
                    output_size=original_array.shape[:2],
                )
                cam = result["cam"][0].detach().cpu().numpy()
                probability = float(
                    torch.sigmoid(result["raw_logits"])[0, 0].item()
                )
                if abs(probability - float(item["m2_probability"])) > 1e-5:
                    raise RuntimeError(
                        "Grad-CAM forward probability differs from saved M2 evidence."
                    )
                overlay = overlay_cam(original_array, cam)
                left = axes[row_index, case_index * 2]
                right = axes[row_index, case_index * 2 + 1]
                left.imshow(original_array)
                right.imshow(overlay)
                left.set_title(
                    f"{category} original\n{item['image_name']}",
                    fontsize=8,
                )
                right.set_title(
                    f"Grad-CAM M2={probability:.3f}",
                    fontsize=8,
                )
                left.axis("off")
                right.axis("off")
                case_records.append({
                    "category": category,
                    "image_name": str(item["image_name"]),
                    "saved_probability": float(item["m2_probability"]),
                    "forward_probability": probability,
                    "activation_shape": result["activation_shape"],
                    "gradient_shape": result["gradient_shape"],
                    "cam_shape": list(cam.shape),
                    "cam_min": float(cam.min()),
                    "cam_max": float(cam.max()),
                    "cam_standard_deviation": float(cam.std()),
                    "cam_finite": bool(np.isfinite(cam).all()),
                    "cam_nontrivial": bool(np.ptp(cam) > 1e-6),
                })
    if not any(record["cam_nontrivial"] for record in case_records):
        raise RuntimeError("All authoritative Grad-CAM outputs are constant.")
    figure.tight_layout()
    figure_path = project_root / "outputs/figures/I_gradcam.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)

    summary = {
        "checkpoint": str(
            components["checkpoint_path"].relative_to(project_root)
        ),
        "checkpoint_epoch": components["checkpoint_epoch"],
        "metadata_preprocessor": str(
            components["preprocessor_path"].relative_to(project_root)
        ),
        "metadata_preprocessor_refitted": False,
        "metadata_included_during_forward": True,
        "target_layer": components["target_layer_name"],
        "gradient_target": "raw_melanoma_logit",
        "cases_per_category": int(cases_per_category),
        "case_count": int(len(case_records)),
        "hooks_closed_after_generation": True,
        "cases": case_records,
        "figure": str(figure_path.relative_to(project_root)),
    }
    summary_path = (
        project_root / "outputs/analysis/gradcam/M2_gradcam_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Generate M2 Grad-CAM from the frozen best checkpoint."
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--cases-per-category", type=int, default=3)
    return parser.parse_args(args)


def main(args=None):
    arguments = parse_args(args)
    project_root = (
        Path(arguments.project_root)
        if arguments.project_root
        else Path(__file__).resolve().parents[1]
    )
    summary = generate_m2_gradcam(
        project_root,
        cases_per_category=arguments.cases_per_category,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
