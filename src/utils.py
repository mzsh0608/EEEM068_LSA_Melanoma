"""Configuration, reproducibility, device, and JSON utilities."""

import json
import platform
import random
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
import torchvision
import yaml


def load_config(config_path):
    """Load a non-empty YAML mapping from ``config_path``."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Malformed YAML config: {path}") from exc

    if not isinstance(config, dict) or not config:
        raise ValueError("Config must contain a non-empty YAML mapping.")

    return config


def seed_everything(seed):
    """Seed common RNGs and request deterministic CuDNN behaviour."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device():
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_parent(path):
    """Create the parent directory for a future output path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    """Write a JSON-serializable object with stable indentation."""
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def capture_environment():
    """Capture the main runtime versions used by an experiment."""
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "python_full": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if cuda_available
            else None
        ),
    }
