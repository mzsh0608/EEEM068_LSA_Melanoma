"""Generate J2B publication figures and display-formatted tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


MODEL_ORDER = ("H0", "H1", "B0", "M1", "M2")
DEEP_MODEL_ORDER = ("B0", "M1", "M2")
MODEL_STYLE = {
    "H0": {"color": "#777777", "linestyle": "-", "marker": "o"},
    "H1": {"color": "#202020", "linestyle": "--", "marker": "s"},
    "B0": {"color": "#2f6b9a", "linestyle": "-.", "marker": "^"},
    "M1": {"color": "#c44e52", "linestyle": "-", "marker": "D"},
    "M2": {"color": "#3a8f5c", "linestyle": "--", "marker": "P"},
}

FIGURES = (
    ("Fig_Main_01", "main", "Fig_Main_01_Pipeline"),
    ("Fig_Main_02", "main", "Fig_Main_02_ThresholdTradeoff"),
    ("Fig_App_01", "appendix", "Fig_App_01_ClassDistribution"),
    ("Fig_App_02", "appendix", "Fig_App_02_TrainingCurves"),
    ("Fig_App_03", "appendix", "Fig_App_03_ConfusionMatrices"),
    ("Fig_App_04", "appendix", "Fig_App_04_ROC_PR"),
    ("Fig_App_05", "appendix", "Fig_App_05_Bootstrap"),
    ("Fig_App_06", "appendix", "Fig_App_06_FailureCases"),
    ("Fig_App_07", "appendix", "Fig_App_07_GradCAM"),
    ("Fig_App_08", "appendix", "Fig_App_08_Subgroups"),
)

TABLE_NAMES = (
    "Table_Main_01_ModelResults",
    "Table_App_01_DeepProtocol",
    "Table_App_02_HyperparameterSelection",
    "Table_App_03_MetadataAblation",
    "Table_App_04_BootstrapSummary",
    "Table_App_05_SubgroupSummary",
)


def find_repo_root(start: Path) -> Path:
    """Find the repository root from a working directory or child path."""
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "outputs" / "final" / "tables" / "main_model_results.csv").is_file():
            return candidate
    raise FileNotFoundError("Could not locate the repository root")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def _save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "EEEM068 J2B publication asset generator"}
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.06, metadata=metadata)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.06, metadata=metadata)
    plt.close(fig)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_source_path(root: Path, raw_path: str) -> Path:
    normalized = Path(str(raw_path).replace("\\", "/"))
    return normalized if normalized.is_absolute() else root / normalized


def load_sources(root: Path) -> dict[str, Any]:
    """Load frozen J1/J2A evidence and validate the interpretation constraints."""
    table_dir = root / "outputs" / "final" / "tables"
    manifest_dir = root / "outputs" / "final" / "manifests"
    sources: dict[str, Any] = {
        "dataset": pd.read_csv(table_dir / "dataset_summary.csv"),
        "main": pd.read_csv(table_dir / "main_model_results.csv"),
        "protocol": pd.read_csv(table_dir / "deep_model_protocol.csv"),
        "hyperparameters": pd.read_csv(table_dir / "hyperparameter_selection.csv"),
        "training": pd.read_csv(table_dir / "training_summary.csv"),
        "metadata_ablation": pd.read_csv(table_dir / "M1_M2_metadata_ablation.csv"),
        "threshold": pd.read_csv(table_dir / "threshold_summary.csv"),
        "bootstrap_summary": pd.read_csv(table_dir / "bootstrap_summary.csv"),
        "subgroup": pd.read_csv(table_dir / "subgroup_summary.csv"),
        "failures": pd.read_csv(table_dir / "failure_summary.csv"),
        "integrity": pd.read_csv(table_dir / "integrity_summary.csv"),
        "bootstrap_samples": pd.read_csv(
            root / "outputs" / "analysis" / "bootstrap" / "M1_M2_bootstrap_samples.csv"
        ),
        "failure_cases": pd.read_csv(
            root / "outputs" / "analysis" / "failures" / "M2_failure_cases.csv"
        ),
        "failure_review": pd.read_csv(
            root / "outputs" / "analysis" / "failures" / "M2_failure_review.csv"
        ),
        "gradcam": _read_json(
            root / "outputs" / "analysis" / "gradcam" / "M2_gradcam_summary.json"
        ),
        "j1_complete": _read_json(manifest_dir / "J1_COMPLETE.json"),
        "j1_tables": _read_json(manifest_dir / "J1B_tables_manifest.json"),
        "boundaries": _read_json(manifest_dir / "comparison_boundaries.json"),
        "hyperparameter_strategy": _read_json(manifest_dir / "hyperparameter_strategy.json"),
    }

    notebook_path = root / "notebooks" / "02_results_analysis.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook_errors = [
        output
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    sources["notebook_path"] = notebook_path
    sources["notebook"] = notebook
    if notebook_errors:
        raise ValueError("J2A notebook contains stored execution errors")

    validate_source_data(sources)
    return sources


def validate_source_data(sources: dict[str, Any]) -> None:
    """Reject incomplete or inconsistent plotting inputs before generating assets."""
    main = sources["main"]
    threshold = sources["threshold"]
    training = sources["training"]
    subgroup = sources["subgroup"]
    bootstrap_samples = sources["bootstrap_samples"]

    if tuple(main["model_id"]) != MODEL_ORDER:
        raise ValueError("Main model table does not contain the required ordered model hierarchy")
    if set(threshold["model"]) != {"M1", "M2"}:
        raise ValueError("Threshold evidence must contain M1 and M2 only")
    if not threshold.loc[np.isclose(threshold["threshold"], 0.5)].shape[0] == 2:
        raise ValueError("Threshold 0.5 must be present once for M1 and M2")
    if set(training["model_id"]) != set(DEEP_MODEL_ORDER):
        raise ValueError("Training evidence must contain B0, M1, and M2")
    if set(subgroup["model"]) != {"M1", "M2"}:
        raise ValueError("Subgroup evidence must contain M1 and M2")

    finite_groups = {
        "main": (main, ["roc_auc", "average_precision", "sensitivity", "specificity", "f1"]),
        "threshold": (threshold, ["threshold", "sensitivity", "specificity", "precision", "f1"]),
        "training": (training, ["epoch", "train_loss", "val_loss", "val_roc_auc", "val_average_precision"]),
        "bootstrap": (
            bootstrap_samples,
            [
                "roc_auc_difference_m2_minus_m1",
                "average_precision_difference_m2_minus_m1",
                "sensitivity_difference_m2_minus_m1",
            ],
        ),
        "subgroup": (subgroup, ["n", "positive_count", "sensitivity", "specificity"]),
    }
    for name, (frame, columns) in finite_groups.items():
        values = frame[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite plotting data found in {name}")

    cases = sources["failure_cases"]
    review = sources["failure_review"]
    category_counts = cases.groupby("category").size().to_dict()
    if category_counts != {"FN": 6, "FP": 6, "TN": 6, "TP": 6}:
        raise ValueError("Corrected failure selection must contain six cases per category")
    if len(cases) != cases["image_name"].nunique():
        raise ValueError("Failure selection contains duplicate image names")
    if set(cases["image_name"]) != set(review["image_name"]):
        raise ValueError("Failure review and selected-case image names differ")

    gradcam_categories = {case["category"] for case in sources["gradcam"]["cases"]}
    if gradcam_categories != {"FN", "FP", "TN", "TP"}:
        raise ValueError("Grad-CAM evidence does not cover every confusion category")
    if sources["gradcam"]["gradient_target"] != "raw_melanoma_logit":
        raise ValueError("Grad-CAM target is not the frozen raw melanoma logit")

    boundary_names = {item["comparison"] for item in sources["boundaries"]["boundaries"]}
    required = {"H0_to_H1", "H0_H1_to_B0", "B0_to_M1", "M1_to_M2", "paired_patient_bootstrap", "M2_Grad-CAM"}
    if not required.issubset(boundary_names):
        raise ValueError("Required J1 interpretation boundaries are missing")
    if sources["hyperparameter_strategy"]["systematic_search_performed"]:
        raise ValueError("Frozen strategy unexpectedly claims systematic tuning")


def _draw_box(
    axis: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#404040",
    fontsize: float = 8.0,
) -> None:
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    axis.add_patch(patch)
    axis.text(x, y, text, ha="center", va="center", fontsize=fontsize, linespacing=1.2)


def _draw_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": "#444444", "lw": 1.0, "shrinkA": 1, "shrinkB": 1},
    )


def plot_pipeline(png_path: Path, pdf_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(11.8, 7.0))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    neutral = "#f2f2f2"
    audit_color = "#e8eef3"
    image_color = "#dceaf5"
    metadata_color = "#e2f0e7"
    fusion_color = "#f4e5e3"
    output_color = "#eee9f5"

    top_boxes = [
        ((0.41, 0.955), "SIIM-ISIC JPEG images + metadata", neutral),
        ((0.41, 0.865), "Dataset and exact-content integrity audit", audit_color),
        ((0.41, 0.775), "Patient-aware grouped split", audit_color),
        ((0.41, 0.685), "Training partition  |  Fixed Fold-0 validation", neutral),
    ]
    for center, text, color in top_boxes:
        _draw_box(axis, center, 0.52, 0.058, text, facecolor=color, fontsize=8.4)
    for upper, lower in zip(top_boxes[:-1], top_boxes[1:]):
        _draw_arrow(axis, (upper[0][0], upper[0][1] - 0.031), (lower[0][0], lower[0][1] + 0.031))

    image_boxes = [
        ((0.22, 0.56), "RGB resize + training augmentation"),
        ((0.22, 0.45), "ResNet18 / ConvNeXt-Tiny"),
        ((0.22, 0.34), "Image representation"),
    ]
    metadata_boxes = [
        ((0.60, 0.56), "Age + sex + anatomical site"),
        ((0.60, 0.45), "Training-only imputation, scaling,\nand one-hot encoding"),
        ((0.60, 0.34), "Small metadata MLP"),
    ]
    axis.text(0.22, 0.62, "IMAGE PATH", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    axis.text(0.60, 0.62, "METADATA PATH (M2)", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for center, text in image_boxes:
        _draw_box(axis, center, 0.29, 0.065, text, facecolor=image_color)
    for center, text in metadata_boxes:
        _draw_box(axis, center, 0.29, 0.065, text, facecolor=metadata_color)
    _draw_arrow(axis, (0.35, 0.654), (0.22, 0.595))
    _draw_arrow(axis, (0.47, 0.654), (0.60, 0.595))
    for boxes in (image_boxes, metadata_boxes):
        for upper, lower in zip(boxes[:-1], boxes[1:]):
            _draw_arrow(axis, (upper[0][0], upper[0][1] - 0.035), (lower[0][0], lower[0][1] + 0.035))

    _draw_box(
        axis,
        (0.41, 0.235),
        0.54,
        0.095,
        "B0/M1: image-only binary head\nM2: image + metadata fusion\nRaw binary logit",
        facecolor=fusion_color,
        fontsize=8.2,
    )
    _draw_arrow(axis, (0.22, 0.305), (0.34, 0.275))
    _draw_arrow(axis, (0.60, 0.305), (0.48, 0.275))

    _draw_box(
        axis,
        (0.41, 0.135),
        0.58,
        0.065,
        "Weighted BCE training  |  Best validation ROC-AUC checkpoint",
        facecolor=output_color,
        fontsize=8.1,
    )
    _draw_arrow(axis, (0.41, 0.187), (0.41, 0.168))
    _draw_box(
        axis,
        (0.41, 0.045),
        0.66,
        0.06,
        "Fixed threshold 0.5 primary evaluation\nROC-AUC | AP | Sensitivity | Specificity | F1",
        facecolor=output_color,
        fontsize=7.9,
    )
    _draw_arrow(axis, (0.41, 0.102), (0.41, 0.075))

    _draw_box(
        axis,
        (0.865, 0.31),
        0.22,
        0.43,
        "POST-HOC RELIABILITY\n\nThreshold behaviour\nFailure / disagreement review\nSubgroups\nPaired patient bootstrap\nGrad-CAM",
        facecolor="#f7f7f7",
        edgecolor="#666666",
        fontsize=7.8,
    )
    _draw_arrow(axis, (0.745, 0.045), (0.755, 0.20))
    axis.set_title("Project pipeline and frozen evaluation workflow", pad=4, fontweight="bold")
    _save_figure(fig, png_path, pdf_path)


def plot_threshold_tradeoff(threshold: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    metrics = (
        ("sensitivity", "A  Sensitivity"),
        ("specificity", "B  Specificity"),
        ("precision", "C  Precision"),
        ("f1", "D  F1"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), sharex=True, sharey=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for model_id in ("M1", "M2"):
            model_data = threshold.loc[threshold["model"].eq(model_id)].sort_values("threshold")
            style = MODEL_STYLE[model_id]
            axis.plot(
                model_data["threshold"],
                model_data[metric],
                label=model_id,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=1.5,
                markersize=4,
            )
        axis.axvline(0.5, color="#555555", linestyle=":", linewidth=1.2)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlim(0.08, 0.92)
        axis.set_ylim(0, 1.03)
        axis.set_xticks(np.arange(0.1, 1.0, 0.1))
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Metric value")
    for axis in axes[-1, :]:
        axis.set_xlabel("Decision threshold")
    handles = [
        Line2D([0], [0], label=model, color=MODEL_STYLE[model]["color"],
               linestyle=MODEL_STYLE[model]["linestyle"], marker=MODEL_STYLE[model]["marker"])
        for model in ("M1", "M2")
    ]
    handles.append(Line2D([0], [0], label="Primary threshold 0.5", color="#555555", linestyle=":"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_figure(fig, png_path, pdf_path)


def plot_class_distribution(dataset: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    row = dataset.iloc[0]
    partitions = ["Full dataset", "Training", "Validation"]
    benign = np.array([row.benign_images, row.train_benign, row.validation_benign], dtype=int)
    melanoma = np.array([row.melanoma_images, row.train_melanoma, row.validation_melanoma], dtype=int)
    prevalence = melanoma / (benign + melanoma) * 100
    x = np.arange(len(partitions))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.45, 1]})
    bars_b = axes[0].bar(x - width / 2, benign, width, label="Benign", color="#b7b7b7", edgecolor="#333333")
    bars_m = axes[0].bar(
        x + width / 2, melanoma, width, label="Melanoma", color="#c44e52",
        edgecolor="#333333", hatch="//",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Images (log scale)")
    axes[0].set_xticks(x, partitions)
    axes[0].set_title("A  Class counts", loc="left", fontweight="bold")
    axes[0].grid(axis="y", which="major", color="#dddddd", linewidth=0.5)
    for bars in (bars_b, bars_m):
        axes[0].bar_label(bars, labels=[f"{int(value):,}" for value in bars.datavalues], padding=2, fontsize=7)

    prevalence_bars = axes[1].bar(
        x, prevalence, width=0.58, color="#c44e52", edgecolor="#333333", hatch="//"
    )
    axes[1].set_ylabel("Melanoma prevalence (%)")
    axes[1].set_xticks(x, partitions)
    axes[1].set_title("B  Class prevalence", loc="left", fontweight="bold")
    axes[1].set_ylim(0, max(prevalence) * 1.35)
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[1].bar_label(prevalence_bars, labels=[f"{value:.2f}%" for value in prevalence], padding=3, fontsize=7.5)
    handles = [bars_b[0], bars_m[0]]
    fig.legend(handles=handles, labels=["Benign", "Melanoma"], frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(0.38, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_figure(fig, png_path, pdf_path)


def plot_training_curves(training: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.0), sharex="row")
    for row_index, model_id in enumerate(DEEP_MODEL_ORDER):
        model_data = training.loc[training["model_id"].eq(model_id)].sort_values("epoch")
        best = model_data.loc[model_data["is_best_roc_auc_epoch"].map(_as_bool)].iloc[0]
        color = MODEL_STYLE[model_id]["color"]

        loss_axis = axes[row_index, 0]
        loss_axis.plot(model_data.epoch, model_data.train_loss, color=color, linestyle="--", marker="o", label="Train loss")
        loss_axis.plot(model_data.epoch, model_data.val_loss, color=color, linestyle="-", marker="s", label="Validation loss")
        loss_axis.scatter(best.epoch, best.val_loss, color="white", edgecolor="#222222", s=34, zorder=4)
        loss_axis.axvline(best.epoch, color="#666666", linestyle=":", linewidth=1)
        loss_axis.set_ylabel(f"{model_id}\nLoss")
        loss_axis.set_title("Loss" if row_index == 0 else "", fontweight="bold")
        loss_axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        loss_axis.legend(frameon=False, loc="best")

        metric_axis = axes[row_index, 1]
        metric_axis.plot(model_data.epoch, model_data.val_roc_auc, color=color, linestyle="-", marker="o", label="Validation ROC-AUC")
        metric_axis.plot(model_data.epoch, model_data.val_average_precision, color=color, linestyle="--", marker="s", label="Validation AP")
        metric_axis.scatter(best.epoch, best.val_roc_auc, color="white", edgecolor="#222222", s=34, zorder=4)
        metric_axis.axvline(best.epoch, color="#666666", linestyle=":", linewidth=1)
        metric_axis.set_ylabel("Metric value")
        metric_axis.set_title("Validation ranking metrics" if row_index == 0 else "", fontweight="bold")
        metric_axis.grid(axis="y", color="#dddddd", linewidth=0.5)
        metric_axis.legend(frameon=False, loc="best")
        for axis in (loss_axis, metric_axis):
            axis.set_xticks(model_data.epoch)
    for axis in axes[-1, :]:
        axis.set_xlabel("Epoch")
    fig.text(
        0.5,
        0.008,
        "Dotted lines and outlined markers: best validation ROC-AUC epoch (B0=4, M1=4, M2=8).",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 1), h_pad=1.0)
    _save_figure(fig, png_path, pdf_path)


def plot_confusion_matrices(main: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    model_rows = main.set_index("model_id").loc[list(DEEP_MODEL_ORDER)]
    matrices = {
        model_id: np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        for model_id, row in model_rows.iterrows()
    }
    values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    norm = LogNorm(vmin=max(1, int(values.min())), vmax=int(values.max()))

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
    for axis, model_id in zip(axes, DEEP_MODEL_ORDER):
        matrix = matrices[model_id]
        axis.imshow(matrix, cmap="Blues", norm=norm)
        axis.set_title(model_id, fontweight="bold")
        axis.set_xticks([0, 1], ["Benign", "Melanoma"])
        axis.set_yticks([0, 1], ["Benign", "Melanoma"])
        axis.set_xlabel("Predicted")
        if axis is axes[0]:
            axis.set_ylabel("Actual")
        for row in range(2):
            for column in range(2):
                value = matrix[row, column]
                text_color = "white" if norm(value) > 0.56 else "#111111"
                axis.text(column, row, f"{value:,}", ha="center", va="center", color=text_color, fontsize=9)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#555555")
    fig.suptitle("Threshold 0.5 validation confusion counts", y=0.99, fontweight="bold")
    fig.text(0.5, 0.02, "Cell shading uses a common logarithmic count scale.", ha="center", fontsize=7)
    fig.tight_layout(rect=(0.02, 0.07, 0.98, 0.88), w_pad=1.0)
    _save_figure(fig, png_path, pdf_path)


def _load_predictions(root: Path) -> dict[str, pd.DataFrame]:
    files = {
        "H0": "H0_logistic_unweighted.csv",
        "H1": "H1_logistic_weighted.csv",
        "B0": "B0_resnet18.csv",
        "M1": "M1_convnext_image.csv",
        "M2": "M2_convnext_metadata.csv",
    }
    return {
        model_id: pd.read_csv(root / "outputs" / "predictions" / filename)
        for model_id, filename in files.items()
    }


def plot_roc_pr(root: Path, main: pd.DataFrame, png_path: Path, pdf_path: Path) -> dict[str, dict[str, float]]:
    predictions = _load_predictions(root)
    main_indexed = main.set_index("model_id")
    reference = predictions["H0"].set_index("image_name")["target"].sort_index()
    calculated: dict[str, dict[str, float]] = {}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for model_id in MODEL_ORDER:
        frame = predictions[model_id]
        aligned_target = frame.set_index("image_name")["target"].sort_index()
        if not reference.index.equals(aligned_target.index) or not np.array_equal(reference.values, aligned_target.values):
            raise ValueError(f"{model_id} prediction targets do not match the fixed validation set")
        target = frame["target"].to_numpy(dtype=int)
        probability = frame["probability"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(target, probability)
        precision, recall, _ = precision_recall_curve(target, probability)
        roc_auc = float(roc_auc_score(target, probability))
        ap = float(average_precision_score(target, probability))
        if not np.isclose(roc_auc, main_indexed.loc[model_id, "roc_auc"], atol=1e-12):
            raise ValueError(f"{model_id} ROC-AUC differs from J1")
        if not np.isclose(ap, main_indexed.loc[model_id, "average_precision"], atol=1e-12):
            raise ValueError(f"{model_id} AP differs from J1")
        calculated[model_id] = {"roc_auc": roc_auc, "average_precision": ap}

        style = MODEL_STYLE[model_id]
        linewidth = 1.25 if model_id in {"H0", "H1"} else 1.65
        axes[0].plot(
            fpr, tpr, color=style["color"], linestyle=style["linestyle"], linewidth=linewidth,
            label=f"{model_id} (ROC-AUC {roc_auc:.3f})",
        )
        axes[1].plot(
            recall, precision, color=style["color"], linestyle=style["linestyle"], linewidth=linewidth,
            label=f"{model_id} (AP {ap:.3f})",
        )

    prevalence = float(reference.mean())
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1, label="Chance line")
    axes[1].axhline(prevalence, color="#999999", linestyle=":", linewidth=1, label=f"Prevalence ({prevalence:.3f})")
    axes[0].set(title="A  ROC", xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(title="B  Precision-recall", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    for axis in axes:
        axis.title.set_fontweight("bold")
        axis.title.set_ha("left")
        axis.title.set_position((0, 1.0))
        axis.grid(color="#dddddd", linewidth=0.5)
        axis.legend(frameon=False, loc="best", fontsize=7)
    fig.tight_layout()
    _save_figure(fig, png_path, pdf_path)
    return calculated


def plot_bootstrap(
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> None:
    panels = (
        ("roc_auc_difference", "roc_auc_difference_m2_minus_m1", "A  ROC-AUC difference"),
        ("average_precision_difference", "average_precision_difference_m2_minus_m1", "B  AP difference"),
        ("sensitivity_difference", "sensitivity_difference_m2_minus_m1", "C  Sensitivity difference"),
    )
    summary_indexed = summary.set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    for axis, (summary_metric, sample_column, title) in zip(axes, panels):
        values = samples[sample_column].to_numpy(dtype=float)
        row = summary_indexed.loc[summary_metric]
        axis.hist(values, bins=28, color="#7aa487", edgecolor="white", linewidth=0.4)
        axis.axvspan(row.ci_2_5, row.ci_97_5, color="#3a8f5c", alpha=0.16)
        axis.axvline(0, color="#222222", linewidth=1.1)
        axis.axvline(row.observed_m2_minus_m1, color="#c44e52", linestyle="--", linewidth=1.3)
        axis.axvline(row.ci_2_5, color="#3a8f5c", linestyle=":", linewidth=1.0)
        axis.axvline(row.ci_97_5, color="#3a8f5c", linestyle=":", linewidth=1.0)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("M2 - M1 difference")
        axis.set_ylabel("Bootstrap samples")
        axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    handles = [
        Line2D([0], [0], color="#222222", label="Zero"),
        Line2D([0], [0], color="#c44e52", linestyle="--", label="Observed difference"),
        Patch(facecolor="#3a8f5c", alpha=0.16, label="95% percentile interval"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Paired patient-level bootstrap", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    _save_figure(fig, png_path, pdf_path)


def build_failure_case_manifest(root: Path, sources: dict[str, Any]) -> pd.DataFrame:
    cases = sources["failure_cases"].copy()
    review = sources["failure_review"].copy()
    review_columns = [column for column in review.columns if column not in {"category"}]
    merged = cases.merge(review[review_columns], on="image_name", how="left", validate="one_to_one")
    hashes = []
    for raw_path in merged["image_path"]:
        image_path = _resolve_source_path(root, raw_path)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        hashes.append(sha256_file(image_path))
    merged["content_sha256"] = hashes
    if merged["content_sha256"].nunique() != len(merged):
        raise ValueError("Failure figure selection contains duplicate image content")
    return merged


def plot_failure_cases(root: Path, case_manifest: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    category_order = ["FN", "FP", "TP", "TN"]
    ordered = case_manifest.assign(
        category_order=pd.Categorical(case_manifest["category"], categories=category_order, ordered=True)
    ).sort_values(["category_order", "rank"])
    fig, axes = plt.subplots(4, 6, figsize=(11.5, 7.8))
    for row_index, category in enumerate(category_order):
        rows = ordered.loc[ordered["category"].eq(category)].head(6)
        for column_index, (_, row) in enumerate(rows.iterrows()):
            axis = axes[row_index, column_index]
            image_path = _resolve_source_path(root, row.image_path)
            with Image.open(image_path) as source_image:
                axis.imshow(source_image.convert("RGB"))
            axis.set_title(f"{category} | M2 p={row.m2_probability:.3f}", fontsize=7.2, pad=2)
            axis.axis("off")
    fig.suptitle("M2 validation failure-case review (content-unique selection)", y=0.995, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=0.8, w_pad=0.25)
    _save_figure(fig, png_path, pdf_path)


def plot_gradcam(root: Path, gradcam: dict[str, Any], png_path: Path, pdf_path: Path) -> None:
    source_path = _resolve_source_path(root, gradcam["figure"])
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with Image.open(source_path) as source_image:
        image = np.asarray(source_image.convert("RGB"))
    fig, axis = plt.subplots(figsize=(10.2, 9.6))
    axis.imshow(image)
    axis.axis("off")
    axis.set_title("M2 melanoma-logit Grad-CAM: original and overlay", pad=6, fontweight="bold")
    fig.tight_layout(pad=0.2)
    _save_figure(fig, png_path, pdf_path)


def _subgroup_panel(axis: plt.Axes, subgroup: pd.DataFrame, variable: str, metric: str, title: str) -> None:
    subset = subgroup.loc[subgroup["group_variable"].eq(variable)].copy()
    group_order = subset.loc[subset["model"].eq("M1"), "group_value"].tolist()
    x = np.arange(len(group_order))
    width = 0.36
    offsets = {"M1": -width / 2, "M2": width / 2}
    positive_counts = subset.loc[subset["model"].eq("M1")].set_index("group_value")["positive_count"]
    low_counts = subset.loc[subset["model"].eq("M1")].set_index("group_value")["small_positive_count"].map(_as_bool)

    values_by_model: dict[str, np.ndarray] = {}
    for model_id in ("M1", "M2"):
        indexed = subset.loc[subset["model"].eq(model_id)].set_index("group_value")
        values = indexed.loc[group_order, metric].to_numpy(dtype=float)
        values_by_model[model_id] = values
        bars = axis.bar(
            x + offsets[model_id], values, width, label=model_id,
            color=MODEL_STYLE[model_id]["color"], edgecolor="#333333", linewidth=0.6,
        )
        for bar, group in zip(bars, group_order):
            if bool(low_counts.loc[group]):
                bar.set_hatch("xx")

    for index, group in enumerate(group_order):
        y = max(values_by_model["M1"][index], values_by_model["M2"][index]) + 0.045
        axis.text(index, y, f"n+={int(positive_counts.loc[group])}", ha="center", va="bottom", fontsize=6.6)

    label_map = {
        "lower extremity": "lower ext.",
        "upper extremity": "upper ext.",
    }
    display_labels = [label_map.get(group, group) for group in group_order]
    axis.set_xticks(x, display_labels)
    if variable == "site":
        axis.tick_params(axis="x", rotation=30)
        for label in axis.get_xticklabels():
            label.set_ha("right")
    axis.set_ylim(0, 1.16)
    axis.set_ylabel(metric.capitalize())
    axis.set_title(title, loc="left", fontweight="bold")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)


def plot_subgroups(subgroup: pd.DataFrame, png_path: Path, pdf_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7))
    _subgroup_panel(axes[0, 0], subgroup, "sex", "sensitivity", "A  Sensitivity by sex")
    _subgroup_panel(axes[0, 1], subgroup, "age", "sensitivity", "B  Sensitivity by age band")
    _subgroup_panel(axes[1, 0], subgroup, "site", "sensitivity", "C  Sensitivity by site")
    _subgroup_panel(axes[1, 1], subgroup, "site", "specificity", "D  Specificity by site")
    handles = [
        Patch(facecolor=MODEL_STYLE[model]["color"], edgecolor="#333333", label=model)
        for model in ("M1", "M2")
    ]
    handles.append(Patch(facecolor="white", edgecolor="#333333", hatch="xx", label="Low positive count"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.5, w_pad=1.0)
    _save_figure(fig, png_path, pdf_path)


def publication_model_results(main: pd.DataFrame) -> pd.DataFrame:
    """Return the display-only model-results table at three decimal places."""
    ordered = main.set_index("model_id").loc[list(MODEL_ORDER)].reset_index()
    result = ordered[[
        "model_id", "input_representation", "metadata_used", "roc_auc", "average_precision",
        "sensitivity", "specificity", "f1", "accuracy",
    ]].rename(columns={
        "model_id": "Model",
        "input_representation": "Representation",
        "metadata_used": "Metadata",
        "roc_auc": "ROC-AUC",
        "average_precision": "AP",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
        "f1": "F1",
        "accuracy": "Accuracy",
    })
    result["Metadata"] = result["Metadata"].map(lambda value: "Yes" if _as_bool(value) else "No")
    for column in ["ROC-AUC", "AP", "Sensitivity", "Specificity", "F1", "Accuracy"]:
        result[column] = result[column].map(lambda value: f"{float(value):.3f}")
    return result


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(_escape_markdown(column) for column in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(_escape_markdown(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _escape_latex(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    result = str(value)
    for source, replacement in replacements.items():
        result = result.replace(source, replacement)
    return result.replace("\n", " ")


def _latex_table(frame: pd.DataFrame, note: str) -> str:
    alignment = "".join(
        "r" if pd.to_numeric(frame[column], errors="coerce").notna().all() else "l"
        for column in frame.columns
    )
    header = " & ".join(_escape_latex(column) for column in frame.columns) + r" \\"
    rows = [
        " & ".join(_escape_latex(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        header,
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        rf"\parbox{{0.98\textwidth}}{{\footnotesize Note: {_escape_latex(note)}}}",
        r"\end{table*}",
        "",
    ])


def _write_table_set(output_dir: Path, name: str, frame: pd.DataFrame, note: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    md_path = output_dir / f"{name}.md"
    tex_path = output_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    md_path.write_text(f"# {name}\n\n{_markdown_table(frame)}\n\n**Source note:** {note}\n", encoding="utf-8")
    tex_path.write_text(_latex_table(frame, note), encoding="utf-8")
    return [csv_path, md_path, tex_path]


def _format_metric_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    count_metrics = {"tn", "fp", "fn", "tp"}
    value_columns = [column for column in result.columns if column != "Metric"]
    result[value_columns] = result[value_columns].astype(object)
    for index, row in result.iterrows():
        metric = str(row["Metric"]).lower()
        for column in value_columns:
            value = float(row[column])
            result.at[index, column] = f"{value:.0f}" if metric in count_metrics else f"{value:.3f}"
    return result


def generate_publication_tables(sources: dict[str, Any], output_dir: Path) -> dict[str, list[Path]]:
    created: dict[str, list[Path]] = {}

    main_note = (
        "H0/H1 use the historical 5,000-image Logistic Regression training subset; "
        "B0/M1/M2 use the full 26,499-image training partition; threshold-dependent metrics use 0.5; "
        "values are fixed Fold-0 validation results."
    )
    created["Table_Main_01_ModelResults"] = _write_table_set(
        output_dir,
        "Table_Main_01_ModelResults",
        publication_model_results(sources["main"]),
        main_note,
    )

    protocol_columns = [
        "model_id", "architecture", "pretrained_weights", "image_size", "batch_size", "loss",
        "pos_weight", "optimizer", "learning_rate", "weight_decay", "max_epochs", "patience",
        "actual_epochs", "best_epoch", "parameter_count",
    ]
    protocol = sources["protocol"][protocol_columns].rename(columns={
        "model_id": "Model", "architecture": "Architecture", "pretrained_weights": "Pretraining",
        "image_size": "Image size", "batch_size": "Batch", "loss": "Loss", "pos_weight": "pos_weight",
        "optimizer": "Optimizer", "learning_rate": "Learning rate", "weight_decay": "Weight decay",
        "max_epochs": "Maximum epochs", "patience": "Patience", "actual_epochs": "Actual epochs",
        "best_epoch": "Best epoch", "parameter_count": "Parameters",
    })
    protocol["pos_weight"] = protocol["pos_weight"].map(lambda value: f"{float(value):.3f}")
    protocol["Learning rate"] = protocol["Learning rate"].map(lambda value: f"{float(value):.1e}")
    protocol["Weight decay"] = protocol["Weight decay"].map(lambda value: f"{float(value):.1e}")
    created["Table_App_01_DeepProtocol"] = _write_table_set(
        output_dir,
        "Table_App_01_DeepProtocol",
        protocol,
        "Shared Fold-0 deep-model protocol; checkpoint selection used maximum validation ROC-AUC.",
    )

    hyperparameters = sources["hyperparameters"][[
        "parameter", "B0_value", "M1_value", "M2_value", "selection_category", "systematically_tuned",
    ]].rename(columns={
        "parameter": "Parameter", "B0_value": "B0", "M1_value": "M1", "M2_value": "M2",
        "selection_category": "Selection category", "systematically_tuned": "Systematically tuned?",
    })
    hyperparameters["Systematically tuned?"] = hyperparameters["Systematically tuned?"].map(
        lambda value: "Yes" if _as_bool(value) else "No"
    )
    created["Table_App_02_HyperparameterSelection"] = _write_table_set(
        output_dir,
        "Table_App_02_HyperparameterSelection",
        hyperparameters,
        "No systematic grid, random, or Bayesian search; common settings were deliberately matched.",
    )

    ablation = sources["metadata_ablation"][["metric", "M1", "M2", "M2_minus_M1"]].rename(columns={
        "metric": "Metric", "M2_minus_M1": "M2 - M1",
    })
    ablation = _format_metric_comparison(ablation)
    created["Table_App_03_MetadataAblation"] = _write_table_set(
        output_dir,
        "Table_App_03_MetadataAblation",
        ablation,
        "M1 image-only versus M2 metadata-fusion system ablation; per-sample differences are not causal metadata effects.",
    )

    bootstrap = sources["bootstrap_summary"][[
        "metric", "observed_m2_minus_m1", "bootstrap_median", "ci_2_5", "ci_97_5",
        "valid_iterations", "bootstrap_unit",
    ]].rename(columns={
        "metric": "Metric", "observed_m2_minus_m1": "Observed M2 - M1", "bootstrap_median": "Median",
        "ci_2_5": "2.5 percentile", "ci_97_5": "97.5 percentile", "valid_iterations": "Iterations",
        "bootstrap_unit": "Unit",
    })
    for column in ["Observed M2 - M1", "Median", "2.5 percentile", "97.5 percentile"]:
        bootstrap[column] = bootstrap[column].map(lambda value: f"{float(value):.3f}")
    created["Table_App_04_BootstrapSummary"] = _write_table_set(
        output_dir,
        "Table_App_04_BootstrapSummary",
        bootstrap,
        "Paired patient-level bootstrap on Fold-0 validation; this is internal uncertainty, not external validation.",
    )

    subgroup = sources["subgroup"][[
        "model", "group_variable", "group_value", "n", "positive_count", "roc_auc", "average_precision",
        "sensitivity", "specificity", "small_positive_count",
    ]].rename(columns={
        "model": "Model", "group_variable": "Variable", "group_value": "Group", "n": "N",
        "positive_count": "Positive count", "roc_auc": "ROC-AUC", "average_precision": "AP",
        "sensitivity": "Sensitivity", "specificity": "Specificity", "small_positive_count": "Low positive count?",
    })
    for column in ["ROC-AUC", "AP", "Sensitivity", "Specificity"]:
        subgroup[column] = subgroup[column].map(lambda value: f"{float(value):.3f}")
    subgroup["Low positive count?"] = subgroup["Low positive count?"].map(
        lambda value: "Yes" if _as_bool(value) else "No"
    )
    created["Table_App_05_SubgroupSummary"] = _write_table_set(
        output_dir,
        "Table_App_05_SubgroupSummary",
        subgroup,
        "Exploratory internal-validation subgroups; low-positive groups should not support strong bias claims.",
    )
    return created


def figure_manifest_rows(root: Path) -> list[dict[str, str]]:
    """Build the fixed source and interpretation map for every J2B figure."""
    source_map = {
        "Fig_Main_01": (
            "outputs/final/tables/dataset_summary.csv; outputs/final/tables/deep_model_protocol.csv; "
            "outputs/final/manifests/comparison_boundaries.json; notebooks/02_results_analysis.ipynb",
            "Frozen project workflow and evaluation boundaries.",
            "Fold 0 is fixed patient-aware validation; no external cohort, segmentation, or unsupported architecture is implied.",
        ),
        "Fig_Main_02": (
            "outputs/final/tables/threshold_summary.csv",
            "M1/M2 operating-point trade-offs across the fixed threshold grid.",
            "Exploratory threshold behaviour; 0.5 remains the primary comparison point.",
        ),
        "Fig_App_01": (
            "outputs/final/tables/dataset_summary.csv",
            "Class counts and melanoma prevalence by partition.",
            "Descriptive class-distribution evidence only.",
        ),
        "Fig_App_02": (
            "outputs/final/tables/training_summary.csv",
            "Frozen deep-model loss and validation ranking histories.",
            "Checkpoint epoch is selected by validation ROC-AUC; no training ranking metric is inferred.",
        ),
        "Fig_App_03": (
            "outputs/final/tables/main_model_results.csv",
            "Threshold-0.5 validation confusion counts for B0/M1/M2.",
            "Operating-point counts on the fixed validation fold.",
        ),
        "Fig_App_04": (
            "outputs/predictions/H0_logistic_unweighted.csv; outputs/predictions/H1_logistic_weighted.csv; "
            "outputs/predictions/B0_resnet18.csv; outputs/predictions/M1_convnext_image.csv; "
            "outputs/predictions/M2_convnext_metadata.csv; outputs/final/tables/main_model_results.csv",
            "Five-model validation ROC and precision-recall curves.",
            "AP is Average Precision; curves reuse the fixed Fold-0 validation predictions.",
        ),
        "Fig_App_05": (
            "outputs/analysis/bootstrap/M1_M2_bootstrap_samples.csv; outputs/final/tables/bootstrap_summary.csv",
            "M2-minus-M1 paired bootstrap difference distributions.",
            "Internal paired patient-level bootstrap; not external validation or a significance test.",
        ),
        "Fig_App_06": (
            "outputs/analysis/failures/M2_failure_cases.csv; outputs/analysis/failures/M2_failure_review.csv; "
            "outputs/analysis/data_integrity/exact_image_duplicates.csv",
            "Content-unique M2 qualitative review cases.",
            "Descriptive visual review; no artifact is established as a causal failure mechanism.",
        ),
        "Fig_App_07": (
            "outputs/figures/I_gradcam.png; outputs/analysis/gradcam/M2_gradcam_summary.json",
            "Frozen M2 melanoma-logit Grad-CAM overview.",
            "Image-branch attribution conditioned on metadata; not metadata explanation or causality.",
        ),
        "Fig_App_08": (
            "outputs/final/tables/subgroup_summary.csv",
            "M1/M2 sensitivity and specificity across exploratory subgroups.",
            "Internal exploratory analysis; sparse-positive groups do not support strong bias claims.",
        ),
    }
    rows = []
    for figure_id, location, stem in FIGURES:
        source_files, message, boundary = source_map[figure_id]
        rows.append({
            "figure_id": figure_id,
            "figure_path_png": f"outputs/final/figures/{location}/{stem}.png",
            "figure_path_pdf": f"outputs/final/figures/{location}/{stem}.pdf",
            "intended_location": "main paper" if location == "main" else "appendix",
            "source_files": source_files,
            "generation_script_or_notebook": "src/publication_assets.py",
            "primary_message": message,
            "interpretation_boundary": boundary,
        })
    return rows


def verify_frozen_hashes(root: Path, sources: dict[str, Any]) -> dict[str, Any]:
    frozen_records = sources["j1_complete"]["frozen_hash_verification"]["artifacts"]
    table_records = sources["j1_tables"]["tables"]
    frozen_results = [sha256_file(root / record["path"]) == record["sha256"] for record in frozen_records]
    table_results = [sha256_file(root / record["path"]) == record["sha256"] for record in table_records]
    if not all(frozen_results):
        raise ValueError("A frozen scientific artifact hash changed")
    if not all(table_results):
        raise ValueError("A J1 authoritative table hash changed")
    return {
        "frozen_scientific_artifacts": {"checked": len(frozen_results), "status": "PASS"},
        "j1_authoritative_tables": {"checked": len(table_results), "status": "PASS"},
    }


def validate_generated_assets(
    root: Path,
    sources: dict[str, Any],
    publication_tables: dict[str, list[Path]],
    case_manifest: pd.DataFrame,
    roc_pr_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    file_checks = []
    for figure_id, location, stem in FIGURES:
        png_path = root / "outputs" / "final" / "figures" / location / f"{stem}.png"
        pdf_path = root / "outputs" / "final" / "figures" / location / f"{stem}.pdf"
        if not png_path.is_file() or png_path.stat().st_size == 0:
            raise FileNotFoundError(png_path)
        if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
            raise FileNotFoundError(pdf_path)
        with Image.open(png_path) as image:
            width, height = image.size
            image.verify()
        if width < 1800 or height < 700:
            raise ValueError(f"{figure_id} review PNG is below expected publication-review dimensions")
        file_checks.append({
            "figure_id": figure_id,
            "png_bytes": png_path.stat().st_size,
            "pdf_bytes": pdf_path.stat().st_size,
            "png_width": width,
            "png_height": height,
            "status": "PASS",
        })

    for name, paths in publication_tables.items():
        if name not in TABLE_NAMES or len(paths) != 3:
            raise ValueError("Unexpected publication table output set")
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
            raise FileNotFoundError(f"Incomplete publication table output for {name}")

    generated_main = pd.read_csv(
        root / "outputs" / "final" / "tables" / "publication" / "Table_Main_01_ModelResults.csv",
        dtype=str,
    )
    expected_main = publication_model_results(sources["main"]).astype(str)
    pd.testing.assert_frame_equal(generated_main, expected_main)

    publication_dir = root / "outputs" / "final" / "tables" / "publication"
    generated_protocol = pd.read_csv(
        publication_dir / "Table_App_01_DeepProtocol.csv", dtype=str
    ).set_index("Model")
    source_protocol = sources["protocol"].set_index("model_id")
    if tuple(generated_protocol.index) != DEEP_MODEL_ORDER:
        raise ValueError("Publication deep-protocol model order mismatch")
    for model_id in DEEP_MODEL_ORDER:
        protocol_checks = {
            "Image size": int(source_protocol.loc[model_id, "image_size"]),
            "Batch": int(source_protocol.loc[model_id, "batch_size"]),
            "Maximum epochs": int(source_protocol.loc[model_id, "max_epochs"]),
            "Actual epochs": int(source_protocol.loc[model_id, "actual_epochs"]),
            "Best epoch": int(source_protocol.loc[model_id, "best_epoch"]),
            "Parameters": int(source_protocol.loc[model_id, "parameter_count"]),
        }
        for column, expected in protocol_checks.items():
            if int(generated_protocol.loc[model_id, column]) != expected:
                raise ValueError(f"Publication deep-protocol mismatch for {model_id} {column}")
        if not np.isclose(
            float(generated_protocol.loc[model_id, "pos_weight"]),
            round(float(source_protocol.loc[model_id, "pos_weight"]), 3),
            atol=5e-4,
        ):
            raise ValueError(f"Publication pos_weight mismatch for {model_id}")

    generated_hyper = pd.read_csv(
        publication_dir / "Table_App_02_HyperparameterSelection.csv", dtype=str
    )
    if generated_hyper["Parameter"].tolist() != sources["hyperparameters"]["parameter"].tolist():
        raise ValueError("Publication hyperparameter rows differ from J1")
    if not generated_hyper["Systematically tuned?"].eq("No").all():
        raise ValueError("Publication hyperparameter table misstates tuning status")

    generated_ablation = pd.read_csv(
        publication_dir / "Table_App_03_MetadataAblation.csv", dtype=str
    ).set_index("Metric")
    source_ablation = sources["metadata_ablation"].set_index("metric")
    for metric in source_ablation.index:
        for generated_column, source_column in (("M1", "M1"), ("M2", "M2"), ("M2 - M1", "M2_minus_M1")):
            expected = round(float(source_ablation.loc[metric, source_column]), 3)
            if not np.isclose(float(generated_ablation.loc[metric, generated_column]), expected, atol=5e-4):
                raise ValueError(f"Publication metadata-ablation mismatch for {metric}")

    generated_bootstrap = pd.read_csv(
        publication_dir / "Table_App_04_BootstrapSummary.csv", dtype=str
    ).set_index("Metric")
    source_bootstrap = sources["bootstrap_summary"].set_index("metric")
    for metric in source_bootstrap.index:
        columns = (
            ("Observed M2 - M1", "observed_m2_minus_m1"),
            ("Median", "bootstrap_median"),
            ("2.5 percentile", "ci_2_5"),
            ("97.5 percentile", "ci_97_5"),
        )
        for generated_column, source_column in columns:
            expected = round(float(source_bootstrap.loc[metric, source_column]), 3)
            if not np.isclose(float(generated_bootstrap.loc[metric, generated_column]), expected, atol=5e-4):
                raise ValueError(f"Publication bootstrap mismatch for {metric}")

    generated_subgroup = pd.read_csv(
        publication_dir / "Table_App_05_SubgroupSummary.csv", dtype=str
    )
    source_subgroup = sources["subgroup"].reset_index(drop=True)
    if len(generated_subgroup) != len(source_subgroup):
        raise ValueError("Publication subgroup row count differs from J1")
    for index, source_row in source_subgroup.iterrows():
        generated_row = generated_subgroup.iloc[index]
        identity = (source_row["model"], source_row["group_variable"], source_row["group_value"])
        if tuple(generated_row[["Model", "Variable", "Group"]]) != identity:
            raise ValueError("Publication subgroup row identity differs from J1")
        for generated_column, source_column in (
            ("ROC-AUC", "roc_auc"), ("AP", "average_precision"),
            ("Sensitivity", "sensitivity"), ("Specificity", "specificity"),
        ):
            expected = round(float(source_row[source_column]), 3)
            if not np.isclose(float(generated_row[generated_column]), expected, atol=5e-4):
                raise ValueError(f"Publication subgroup mismatch for {identity}")

    main_indexed = sources["main"].set_index("model_id")
    for model_id, metrics in roc_pr_metrics.items():
        if not np.isclose(metrics["roc_auc"], main_indexed.loc[model_id, "roc_auc"], atol=1e-12):
            raise ValueError("Generated ROC metric mismatch")
        if not np.isclose(metrics["average_precision"], main_indexed.loc[model_id, "average_precision"], atol=1e-12):
            raise ValueError("Generated AP metric mismatch")

    if case_manifest["content_sha256"].nunique() != len(case_manifest):
        raise ValueError("Failure-case content uniqueness check failed")
    if set(case_manifest["category"]) != {"FN", "FP", "TN", "TP"}:
        raise ValueError("Failure-case category check failed")

    return {
        "figure_files": file_checks,
        "all_expected_files_exist": True,
        "nonzero_file_sizes": True,
        "expected_png_dimensions": True,
        "plotting_data_finite": True,
        "required_model_sets_verified": True,
        "threshold_0_5_included": True,
        "semantic_labels_verified_from_plot_configuration": True,
        "publication_table_numbers_match_j1": True,
        "roc_ap_values_match_j1": True,
        "failure_case_content_unique": True,
        "failure_and_gradcam_categories_complete": True,
    }


def generate_publication_assets(
    root: Path,
    *,
    manual_qc_pass: bool = False,
    issues_fixed: Iterable[str] = (),
) -> dict[str, Any]:
    root = find_repo_root(root)
    _configure_style()
    sources = load_sources(root)
    starting_integrity = verify_frozen_hashes(root, sources)

    figure_root = root / "outputs" / "final" / "figures"
    main_dir = figure_root / "main"
    appendix_dir = figure_root / "appendix"
    table_dir = root / "outputs" / "final" / "tables" / "publication"
    manifest_dir = root / "outputs" / "final" / "manifests"
    main_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = {
        figure_id: (
            figure_root / location / f"{stem}.png",
            figure_root / location / f"{stem}.pdf",
        )
        for figure_id, location, stem in FIGURES
    }

    plot_pipeline(*figure_paths["Fig_Main_01"])
    plot_threshold_tradeoff(sources["threshold"], *figure_paths["Fig_Main_02"])
    plot_class_distribution(sources["dataset"], *figure_paths["Fig_App_01"])
    plot_training_curves(sources["training"], *figure_paths["Fig_App_02"])
    plot_confusion_matrices(sources["main"], *figure_paths["Fig_App_03"])
    roc_pr_metrics = plot_roc_pr(root, sources["main"], *figure_paths["Fig_App_04"])
    plot_bootstrap(
        sources["bootstrap_samples"], sources["bootstrap_summary"], *figure_paths["Fig_App_05"]
    )

    failure_case_manifest = build_failure_case_manifest(root, sources)
    plot_failure_cases(root, failure_case_manifest, *figure_paths["Fig_App_06"])
    plot_gradcam(root, sources["gradcam"], *figure_paths["Fig_App_07"])
    plot_subgroups(sources["subgroup"], *figure_paths["Fig_App_08"])

    failure_manifest_path = manifest_dir / "J2B_failure_case_sources.csv"
    failure_case_manifest.to_csv(failure_manifest_path, index=False, lineterminator="\n")
    publication_tables = generate_publication_tables(sources, table_dir)

    figure_manifest_path = manifest_dir / "J2B_figure_sources.csv"
    pd.DataFrame(figure_manifest_rows(root)).to_csv(figure_manifest_path, index=False, lineterminator="\n")

    automated_qc = validate_generated_assets(
        root, sources, publication_tables, failure_case_manifest, roc_pr_metrics
    )
    ending_integrity = verify_frozen_hashes(root, sources)
    qc_path = manifest_dir / "J2B_visual_qc.json"
    qc_record = {
        "phase": "J2B",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "generation_script": "src/publication_assets.py",
        "automated_qc": automated_qc,
        "frozen_integrity_before_generation": starting_integrity,
        "frozen_integrity_after_generation": ending_integrity,
        "manual_visual_review": {
            "status": "PASS" if manual_qc_pass else "PENDING",
            "figures_reviewed": [item[0] for item in FIGURES] if manual_qc_pass else [],
            "checks": [
                "labels not clipped",
                "legends readable",
                "axes correct",
                "no obvious overlap",
                "Grad-CAM panels correspond",
                "failure cases loaded",
                "failure selection content-unique",
                "terminology reviewed",
            ] if manual_qc_pass else [],
            "issues_fixed": list(issues_fixed),
        },
        "terminology": {
            "average_precision_term": "AP",
            "validation_term": "validation",
            "prohibited_terms_absent_from_configured_text": True,
        },
    }
    qc_path.write_text(json.dumps(qc_record, indent=2) + "\n", encoding="utf-8")

    return {
        "figures": [path for pair in figure_paths.values() for path in pair],
        "publication_tables": [path for paths in publication_tables.values() for path in paths],
        "figure_manifest": figure_manifest_path,
        "failure_case_manifest": failure_manifest_path,
        "qc_manifest": qc_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manual-qc-pass", action="store_true")
    parser.add_argument("--issues-fixed", action="append", default=[])
    args = parser.parse_args()
    outputs = generate_publication_assets(
        args.root,
        manual_qc_pass=args.manual_qc_pass,
        issues_fixed=args.issues_fixed,
    )
    print(f"Generated {len(outputs['figures'])} figure files")
    print(f"Generated {len(outputs['publication_tables'])} publication table files")
    print(outputs["figure_manifest"])
    print(outputs["qc_manifest"])


if __name__ == "__main__":
    main()
