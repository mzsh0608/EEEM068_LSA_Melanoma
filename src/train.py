"""Reusable PyTorch training engine for fixed-fold experiments."""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

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


def fit_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    config,
):
    """Fit a configured model with checkpointing and early stopping."""
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

    def resolve_output(value):
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    checkpoint_path = resolve_output(output_config["checkpoint"])
    history_path = resolve_output(output_config["training_history"])
    log_path = resolve_output(output_config["training_log"])
    predictions_path = resolve_output(output_config["predictions"])
    metrics_path = resolve_output(output_config["metrics"])
    environment_path = resolve_output(output_config["environment"])
    best_metric = float("-inf") if checkpoint_mode == "max" else float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    history = []
    ensure_parent(log_path).write_text("", encoding="utf-8")

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

        if early_enabled and epochs_without_improvement >= patience:
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
    final_predictions.to_csv(
        ensure_parent(predictions_path),
        index=False,
    )
    save_metrics_json(final_metrics, metrics_path)
    save_json(capture_environment(), environment_path)

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "metrics": final_metrics,
        "predictions": final_predictions,
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
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one train batch and one validation batch only.",
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
    print("Loss:", experiment["loss_info"]["loss"])
    print("pos_weight:", experiment["loss_info"]["pos_weight"])
    print("Model:", config["model"]["architecture"])
    print("Image size:", config["data"]["image_size"])
    print("Batch size:", config["data"]["batch_size"])

    if arguments.smoke_test:
        smoke = run_one_batch_smoke(experiment, config)
        print("Smoke test:", smoke)

    print("Pipeline setup successful. Full training belongs to Phase F.")


if __name__ == "__main__":
    main()
