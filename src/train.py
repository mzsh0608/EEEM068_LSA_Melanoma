"""Reusable PyTorch training engine for fixed-fold experiments."""

import argparse
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import (
    MelanomaDataset,
    create_dataloaders,
    load_fold_dataframes,
)
from src.evaluate import (
    calculate_binary_metrics,
    make_prediction_dataframe,
    save_metrics_json,
)
from src.losses import build_loss
from src.models import build_model
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    capture_environment,
    ensure_parent,
    load_config,
    resolve_device,
    save_json,
    seed_everything,
)


def _amp_active(amp_enabled, device):
    return bool(amp_enabled and device.type == "cuda")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _resolve_output(project_root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(project_root) / path


def _git_head(project_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _check_binary_output(raw_logits, batch_size):
    if raw_logits.shape != (batch_size, 1):
        raise RuntimeError(
            "Model must return logits with shape [batch_size, 1]; "
            f"received {list(raw_logits.shape)}."
        )
    return raw_logits.squeeze(-1)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler=None,
    amp_enabled=False,
):
    """Train for one epoch and return sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = _amp_active(amp_enabled, device)

    if use_amp and scaler is None:
        raise ValueError("A GradScaler is required when CUDA AMP is active.")

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            raw_logits = model(images)
            logits = _check_binary_output(raw_logits, targets.shape[0])
            loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss detected.")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training loader produced no samples.")

    return total_loss / total_samples


def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
):
    """Evaluate one deterministic validation pass."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    targets_all = []
    probabilities_all = []
    image_names = []
    patient_ids = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            raw_logits = model(images)
            logits = _check_binary_output(raw_logits, targets.shape[0])
            loss = criterion(logits, targets)
            probabilities = torch.sigmoid(logits)

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss detected.")
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("Non-finite validation probability detected.")

            batch_size = targets.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            targets_all.extend(targets.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())
            image_names.extend(list(batch["image_name"]))
            patient_ids.extend(list(batch["patient_id"]))

    if total_samples == 0:
        raise RuntimeError("Validation loader produced no samples.")

    metrics = calculate_binary_metrics(
        targets_all,
        probabilities_all,
        threshold=threshold,
    )
    metrics["loss"] = total_loss / total_samples

    metadata = pd.DataFrame({
        "image_name": image_names,
        "patient_id": patient_ids,
    })
    predictions = make_prediction_dataframe(
        metadata,
        targets_all,
        probabilities_all,
        threshold=threshold,
    )
    return metrics, predictions


def build_resolved_config(config, experiment):
    """Combine the locked YAML with values resolved from the real run."""
    train_frame = experiment["train_df"]
    val_frame = experiment["val_df"]
    train_targets = train_frame["target"]
    val_targets = val_frame["target"]
    training_config = config["training"]
    early_config = training_config["early_stopping"]
    device = experiment["device"]

    patient_overlap = None
    if "patient_id" in train_frame and "patient_id" in val_frame:
        train_patients = set(train_frame["patient_id"].dropna().astype(str))
        val_patients = set(val_frame["patient_id"].dropna().astype(str))
        patient_overlap = len(train_patients & val_patients)

    resolved = {
        "experiment_id": config["experiment_id"],
        "experiment_name": config["experiment_name"],
        "architecture": config["model"]["architecture"],
        "weights": config["model"]["weights"],
        "fine_tune_all": bool(config["model"]["fine_tune_all"]),
        "seed": int(config["seed"]),
        "validation_fold": int(config["validation_fold"]),
        "train_samples": int(len(train_targets)),
        "validation_samples": int(len(val_targets)),
        "train_negative": int((train_targets == 0).sum()),
        "train_positive": int((train_targets == 1).sum()),
        "validation_negative": int((val_targets == 0).sum()),
        "validation_positive": int((val_targets == 1).sum()),
        "patient_overlap": patient_overlap,
        "pos_weight": experiment["loss_info"]["pos_weight"],
        "image_size": int(config["data"]["image_size"]),
        "batch_size": int(config["data"]["batch_size"]),
        "num_workers": int(experiment["train_loader"].num_workers),
        "persistent_workers": bool(
            experiment["train_loader"].persistent_workers
        ),
        "optimizer": training_config["optimizer"],
        "learning_rate": float(training_config["learning_rate"]),
        "weight_decay": float(training_config["weight_decay"]),
        "loss": training_config["loss"],
        "amp_requested": bool(training_config["amp"]),
        "amp_active": _amp_active(training_config["amp"], device),
        "max_epochs": int(training_config["epochs"]),
        "early_stopping_enabled": bool(early_config["enabled"]),
        "early_stopping_patience": int(early_config["patience"]),
        "threshold": float(config["evaluation"]["threshold"]),
        "checkpoint_metric": config["evaluation"]["checkpoint_metric"],
        "checkpoint_mode": config["evaluation"]["checkpoint_mode"],
        "preprocessing": {
            "color_mode": "RGB",
            "resize": [
                int(config["data"]["image_size"]),
                int(config["data"]["image_size"]),
            ],
            "resize_interpolation": config["transforms"][
                "resize_interpolation"
            ],
            "normalization_mean": config["transforms"][
                "normalization_mean"
            ],
            "normalization_std": config["transforms"][
                "normalization_std"
            ],
            "validation_augmentation": None,
        },
        "training_augmentation": {
            "horizontal_flip_probability": config["transforms"][
                "horizontal_flip_probability"
            ],
            "vertical_flip_probability": config["transforms"][
                "vertical_flip_probability"
            ],
            "rotation_degrees": config["transforms"][
                "rotation_degrees"
            ],
            "brightness": config["transforms"]["brightness"],
            "contrast": config["transforms"]["contrast"],
        },
    }
    return {"configured": config, "resolved": resolved}


def plot_training_history(history, output_path, best_epoch):
    """Save loss and validation ranking curves for a completed run."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Weighted BCE")
    axes[0].legend()

    axes[1].plot(
        history["epoch"],
        history["val_roc_auc"],
        label="ROC-AUC",
    )
    axes[1].plot(
        history["epoch"],
        history["val_pr_auc"],
        label="Average Precision",
    )
    axes[1].set(title="Validation ranking metrics", xlabel="Epoch")
    axes[1].legend()

    for axis in axes:
        axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.5)
        axis.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(ensure_parent(output_path), dpi=160)
    plt.close(figure)


def plot_confusion_matrix(metrics, output_path):
    """Save the best-checkpoint threshold-0.5 confusion matrix."""
    matrix = [
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ]
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(matrix, cmap="Blues")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(
                column_index,
                row_index,
                f"{value:,}",
                ha="center",
                va="center",
                color=("white" if value > max(map(max, matrix)) / 2 else "black"),
            )
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Benign", "Melanoma"],
        yticklabels=["Benign", "Melanoma"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"B0 confusion matrix at threshold {metrics['threshold']:.1f}",
    )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(ensure_parent(output_path), dpi=160)
    plt.close(figure)


def plot_roc_pr_comparison(prediction_files, output_path):
    """Save ROC and precision-recall curves from authoritative CSVs."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    prevalence = None

    for label, prediction_path in prediction_files:
        frame = pd.read_csv(prediction_path)
        metrics = calculate_binary_metrics(
            frame["target"],
            frame["probability"],
            threshold=0.5,
        )
        false_positive_rate, true_positive_rate, _ = roc_curve(
            frame["target"], frame["probability"]
        )
        precision, recall, _ = precision_recall_curve(
            frame["target"], frame["probability"]
        )
        axes[0].plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{label} (AUC={metrics['roc_auc']:.3f})",
        )
        axes[1].plot(
            recall,
            precision,
            label=(
                f"{label} "
                f"(AP={metrics['pr_auc_average_precision']:.3f})"
            ),
        )
        prevalence = float(frame["target"].mean())

    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.5)
    axes[0].set(
        title="ROC curves",
        xlabel="False positive rate",
        ylabel="True positive rate",
    )
    if prevalence is not None:
        axes[1].axhline(
            prevalence,
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"Prevalence ({prevalence:.3f})",
        )
    axes[1].set(
        title="Precision-recall curves",
        xlabel="Recall",
        ylabel="Precision",
    )
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(ensure_parent(output_path), dpi=160)
    plt.close(figure)


def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    config,
    run_metadata=None,
):
    """Fit a configured model with checkpointing and early stopping."""
    run_metadata = dict(run_metadata or {})
    fit_started_at = _utc_now()
    fit_started = time.perf_counter()
    training_config = config["training"]
    evaluation_config = config["evaluation"]
    output_config = config["outputs"]
    max_epochs = int(training_config["epochs"])
    amp_enabled = bool(training_config.get("amp", False))
    use_amp = _amp_active(amp_enabled, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    early_config = training_config.get("early_stopping", {})
    early_enabled = bool(early_config.get("enabled", False))
    patience = int(early_config.get("patience", 0))
    checkpoint_metric = evaluation_config["checkpoint_metric"]
    checkpoint_mode = evaluation_config.get("checkpoint_mode", "max")
    threshold = float(evaluation_config.get("threshold", 0.5))

    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be 'min' or 'max'.")
    if early_enabled and patience < 1:
        raise ValueError("Early-stopping patience must be at least one.")

    project_root = Path(__file__).resolve().parents[1]
    checkpoint_path = _resolve_output(
        project_root, output_config["checkpoint"]
    )
    history_path = _resolve_output(
        project_root, output_config["training_history"]
    )
    log_path = _resolve_output(
        project_root, output_config["training_log"]
    )
    predictions_path = _resolve_output(
        project_root, output_config["predictions"]
    )
    metrics_path = _resolve_output(project_root, output_config["metrics"])
    environment_path = _resolve_output(
        project_root, output_config["environment"]
    )
    best_metric = float("-inf") if checkpoint_mode == "max" else float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    early_stopping_triggered = False
    history = []
    log_lines = [
        f"start_timestamp={fit_started_at}",
        f"git_commit={run_metadata.get('git_commit')}",
        f"experiment_id={run_metadata.get('experiment_id')}",
        f"device={device}",
        f"gpu_name={run_metadata.get('gpu_name')}",
        (
            "data_counts="
            f"train:{run_metadata.get('train_samples')} "
            f"validation:{run_metadata.get('validation_samples')} "
            f"train_negative:{run_metadata.get('train_negative')} "
            f"train_positive:{run_metadata.get('train_positive')} "
            f"validation_negative:{run_metadata.get('validation_negative')} "
            f"validation_positive:{run_metadata.get('validation_positive')}"
        ),
        f"pos_weight={run_metadata.get('pos_weight')}",
        (
            "configuration="
            f"architecture:{run_metadata.get('architecture')} "
            f"weights:{run_metadata.get('weights')} "
            f"batch_size:{run_metadata.get('batch_size')} "
            f"num_workers:{run_metadata.get('num_workers')} "
            f"optimizer:{run_metadata.get('optimizer')} "
            f"learning_rate:{run_metadata.get('learning_rate')} "
            f"weight_decay:{run_metadata.get('weight_decay')} "
            f"amp_active:{run_metadata.get('amp_active')} "
            f"max_epochs:{max_epochs} patience:{patience}"
        ),
    ]
    ensure_parent(log_path).write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    for epoch in range(1, max_epochs + 1):
        started = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        val_metrics, _ = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
        )
        seconds = time.perf_counter() - started
        current_metric = val_metrics.get(checkpoint_metric)
        if current_metric is None:
            raise RuntimeError(
                f"Checkpoint metric is unavailable: {checkpoint_metric}"
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc_average_precision"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision": val_metrics["precision"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": seconds,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(
            ensure_parent(history_path),
            index=False,
        )

        improved = (
            current_metric > best_metric
            if checkpoint_mode == "max"
            else current_metric < best_metric
        )
        if improved:
            best_metric = float(current_metric)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "config": config,
                },
                ensure_parent(checkpoint_path),
            )
        else:
            epochs_without_improvement += 1

        with ensure_parent(log_path).open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"epoch={epoch} train_loss={train_loss:.8f} "
                f"val_loss={val_metrics['loss']:.8f} "
                f"{checkpoint_metric}={current_metric:.8f} "
                f"seconds={seconds:.2f}\n"
            )

        print(
            f"Epoch {epoch}/{max_epochs}: "
            f"train_loss={train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_roc_auc={val_metrics['roc_auc']:.6f} "
            f"val_ap={val_metrics['pr_auc_average_precision']:.6f} "
            f"seconds={seconds:.1f}",
            flush=True,
        )

        if early_enabled and epochs_without_improvement >= patience:
            early_stopping_triggered = True
            break

    if best_epoch is None:
        raise RuntimeError("Training completed without a valid checkpoint.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics, final_predictions = validate_one_epoch(
        model,
        val_loader,
        criterion,
        device,
        threshold=threshold,
    )
    total_seconds = time.perf_counter() - fit_started
    final_metrics.update({
        "best_epoch": int(best_epoch),
        "best_validation_roc_auc": float(best_metric),
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_mode": checkpoint_mode,
        "configured_max_epochs": max_epochs,
        "actual_epochs": len(history),
        "early_stopping_enabled": early_enabled,
        "early_stopping_patience": patience,
        "early_stopping_triggered": early_stopping_triggered,
        "total_seconds": total_seconds,
    })
    final_predictions.to_csv(
        ensure_parent(predictions_path),
        index=False,
    )
    save_metrics_json(final_metrics, metrics_path)
    environment = capture_environment()
    environment.update({
        "git_commit": run_metadata.get("git_commit"),
        "experiment_id": run_metadata.get("experiment_id"),
    })
    save_json(environment, environment_path)

    with ensure_parent(log_path).open("a", encoding="utf-8") as handle:
        handle.write(
            f"best_epoch={best_epoch} best_metric={best_metric:.8f}\n"
            f"early_stopping_triggered={early_stopping_triggered}\n"
            f"actual_epochs={len(history)} total_seconds={total_seconds:.2f}\n"
            f"end_timestamp={_utc_now()}\n"
            "artifacts="
            f"checkpoint:{checkpoint_path} predictions:{predictions_path} "
            f"metrics:{metrics_path} history:{history_path}\n"
            "warnings=none\n"
        )

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "metrics": final_metrics,
        "predictions": final_predictions,
        "early_stopping_triggered": early_stopping_triggered,
        "total_seconds": total_seconds,
    }


def build_experiment(
    config,
    project_root,
    num_workers_override=None,
):
    """Assemble all configured components without starting training."""
    project_root = Path(project_root)
    seed_everything(config["seed"])
    device = resolve_device()
    data_config = config["data"]
    transform_config = config["transforms"]
    model_config = config["model"]
    training_config = config["training"]

    if model_config.get("use_metadata", False):
        raise ValueError("Metadata models are not implemented in Phase E.")
    if not data_config.get("train_shuffle", True):
        raise ValueError("Phase E requires shuffled training DataLoaders.")
    if str(training_config.get("scheduler", "none")).lower() != "none":
        raise ValueError("Only scheduler=none is supported in Phase E.")

    train_df, val_df = load_fold_dataframes(
        project_root / "data" / "train.csv",
        project_root / "data" / "train_folds.csv",
        project_root / "data" / "train_images",
        validation_fold=config["validation_fold"],
    )

    common_transforms = {
        "image_size": data_config["image_size"],
        "normalization_mean": transform_config["normalization_mean"],
        "normalization_std": transform_config["normalization_std"],
        "resize_interpolation": transform_config["resize_interpolation"],
    }
    train_transform = get_train_transforms(
        horizontal_flip_probability=transform_config[
            "horizontal_flip_probability"
        ],
        vertical_flip_probability=transform_config[
            "vertical_flip_probability"
        ],
        rotation_degrees=transform_config["rotation_degrees"],
        brightness=transform_config["brightness"],
        contrast=transform_config["contrast"],
        **common_transforms,
    )
    val_transform = get_val_transforms(**common_transforms)

    train_dataset = MelanomaDataset(train_df, transform=train_transform)
    val_dataset = MelanomaDataset(val_df, transform=val_transform)
    num_workers = (
        int(num_workers_override)
        if num_workers_override is not None
        else int(data_config["num_workers"])
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=data_config["batch_size"],
        num_workers=num_workers,
        pin_memory=data_config["pin_memory"],
        persistent_workers=data_config["persistent_workers"],
        drop_last=data_config["drop_last"],
        seed=config["seed"],
    )

    model = build_model(
        architecture=model_config["architecture"],
        pretrained=model_config["pretrained"],
        weights_name=model_config["weights"],
        num_outputs=model_config["num_outputs"],
    )
    if not model_config.get("fine_tune_all", True):
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    model = model.to(device)

    criterion, loss_info = build_loss(
        training_config["loss"],
        train_df["target"],
        device,
    )
    if str(training_config["optimizer"]).lower() != "adamw":
        raise ValueError("Only AdamW is supported in Phase E.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    return {
        "device": device,
        "train_df": train_df,
        "val_df": val_df,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "criterion": criterion,
        "loss_info": loss_info,
        "optimizer": optimizer,
    }


def run_full_experiment(experiment, config, project_root):
    """Execute and persist one authoritative configured experiment."""
    project_root = Path(project_root)
    output_config = config["outputs"]
    resolved_config = build_resolved_config(config, experiment)
    run_metadata = dict(resolved_config["resolved"])
    run_metadata["git_commit"] = _git_head(project_root)
    run_metadata["gpu_name"] = (
        torch.cuda.get_device_name(0)
        if experiment["device"].type == "cuda"
        else None
    )

    experiment_directory = _resolve_output(
        project_root, output_config["experiment_directory"]
    )
    config_path = experiment_directory / "config.json"
    save_json(resolved_config, config_path)

    result = fit_model(
        experiment["model"],
        experiment["train_loader"],
        experiment["val_loader"],
        experiment["criterion"],
        experiment["optimizer"],
        experiment["device"],
        config,
        run_metadata=run_metadata,
    )

    training_curves_path = _resolve_output(
        project_root, output_config["training_curves"]
    )
    confusion_matrix_path = _resolve_output(
        project_root, output_config["confusion_matrix"]
    )
    roc_pr_path = _resolve_output(
        project_root, output_config["roc_pr_curve"]
    )
    predictions_path = _resolve_output(
        project_root, output_config["predictions"]
    )
    plot_training_history(
        result["history"], training_curves_path, result["best_epoch"]
    )
    plot_confusion_matrix(result["metrics"], confusion_matrix_path)
    plot_roc_pr_comparison(
        [
            (
                "H0 Logistic Regression",
                project_root
                / "outputs"
                / "predictions"
                / "H0_logistic_unweighted.csv",
            ),
            (
                "H1 Weighted Logistic Regression",
                project_root
                / "outputs"
                / "predictions"
                / "H1_logistic_weighted.csv",
            ),
            ("B0 ResNet18", predictions_path),
        ],
        roc_pr_path,
    )

    log_path = _resolve_output(
        project_root, output_config["training_log"]
    )
    with ensure_parent(log_path).open("a", encoding="utf-8") as handle:
        handle.write(
            "generated_artifacts="
            f"config:{config_path} "
            f"environment:{_resolve_output(project_root, output_config['environment'])} "
            f"training_curves:{training_curves_path} "
            f"confusion_matrix:{confusion_matrix_path} "
            f"roc_pr_curve:{roc_pr_path}\n"
        )

    return result


def run_one_batch_smoke(experiment, config):
    """Run exactly one train step and one validation forward pass."""
    device = experiment["device"]
    model = experiment["model"]
    criterion = experiment["criterion"]
    optimizer = experiment["optimizer"]
    train_batch = next(iter(experiment["train_loader"]))
    images = train_batch["image"].to(device)
    targets = train_batch["target"].to(device)
    use_amp = _amp_active(config["training"].get("amp", False), device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=use_amp):
        raw_logits = model(images)
        logits = _check_binary_output(raw_logits, targets.shape[0])
        loss = criterion(logits, targets)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("Smoke test produced non-finite values.")
    if use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    model.eval()
    val_batch = next(iter(experiment["val_loader"]))
    val_images = val_batch["image"].to(device)
    with torch.no_grad():
        val_raw_logits = model(val_images)
        val_logits = _check_binary_output(
            val_raw_logits,
            val_images.shape[0],
        )
        probabilities = torch.sigmoid(val_logits)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Smoke validation probabilities are non-finite.")

    return {
        "image_shape": list(images.shape),
        "target_shape": list(targets.shape),
        "image_dtype": str(images.dtype),
        "target_dtype": str(targets.dtype),
        "raw_output_shape": list(raw_logits.shape),
        "logit_shape": list(logits.shape),
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "loss": float(loss.detach().item()),
        "finite_loss": bool(torch.isfinite(loss).item()),
        "backward_successful": True,
        "optimizer_step_successful": True,
        "probability_min": float(probabilities.min().item()),
        "probability_max": float(probabilities.max().item()),
        "probabilities_finite": bool(
            torch.isfinite(probabilities).all().item()
        ),
        "probabilities_in_range": bool(
            ((probabilities >= 0) & (probabilities <= 1)).all().item()
        ),
    }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Build the reusable melanoma training pipeline."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-workers", type=int, default=None)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one train batch and one validation batch only.",
    )
    execution.add_argument(
        "--fit",
        action="store_true",
        help="Run the full configured experiment and save its artifacts.",
    )
    return parser.parse_args(args)


def main(args=None):
    arguments = parse_args(args)
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(arguments.config)
    experiment = build_experiment(
        config,
        project_root,
        num_workers_override=arguments.num_workers,
    )

    print("Experiment:", config["experiment_id"])
    print("Device:", experiment["device"])
    print("Training samples:", len(experiment["train_df"]))
    print("Validation samples:", len(experiment["val_df"]))
    print(
        "Training benign:",
        int((experiment["train_df"]["target"] == 0).sum()),
    )
    print(
        "Training melanoma:",
        int((experiment["train_df"]["target"] == 1).sum()),
    )
    print(
        "Validation benign:",
        int((experiment["val_df"]["target"] == 0).sum()),
    )
    print(
        "Validation melanoma:",
        int((experiment["val_df"]["target"] == 1).sum()),
    )
    train_patients = set(experiment["train_df"]["patient_id"].astype(str))
    val_patients = set(experiment["val_df"]["patient_id"].astype(str))
    print("Patient overlap:", len(train_patients & val_patients))
    print("Loss:", experiment["loss_info"]["loss"])
    print("pos_weight:", experiment["loss_info"]["pos_weight"])
    print("Model:", config["model"]["architecture"])
    print("Weights:", config["model"]["weights"])
    print("Image size:", config["data"]["image_size"])
    print("Batch size:", config["data"]["batch_size"])
    print("Workers:", experiment["train_loader"].num_workers)
    print("Max epochs:", config["training"]["epochs"])
    print(
        "Early stopping patience:",
        config["training"]["early_stopping"]["patience"],
    )
    print(
        "Checkpoint metric:",
        config["evaluation"]["checkpoint_metric"],
    )

    if arguments.smoke_test:
        smoke = run_one_batch_smoke(experiment, config)
        print("Smoke test:", smoke)

    if arguments.fit:
        print("Starting authoritative configured experiment.", flush=True)
        result = run_full_experiment(experiment, config, project_root)
        print(
            "Experiment complete:",
            f"best_epoch={result['best_epoch']}",
            f"best_roc_auc={result['best_metric']:.8f}",
            flush=True,
        )
        return

    print("Pipeline setup successful. Full training requires --fit.")


if __name__ == "__main__":
    main()
