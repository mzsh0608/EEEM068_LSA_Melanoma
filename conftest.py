"""Repository-level pytest configuration for Windows-safe temp paths."""

import os
import time
from pathlib import Path


def pytest_configure(config):
    """Use a fresh repository-local basetemp for every pytest invocation."""
    if config.option.basetemp is None:
        Path(".pytest_tmp").mkdir(exist_ok=True)
        run_name = f"run-{os.getpid()}-{time.time_ns()}"
        config.option.basetemp = str(Path(".pytest_tmp") / run_name)
