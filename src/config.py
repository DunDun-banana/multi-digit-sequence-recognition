"""Configuration loading, path resolution, device selection, and random seeding."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(config).__name__}")
    return config


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve a path relative to base_dir (defaults to current working directory)."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    return (root / path).resolve()


def get_device(device_pref: str = "auto") -> torch.device:
    """Return a PyTorch device based on preference ('auto', 'cpu', or 'cuda')."""
    pref = device_pref.lower().strip()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested in config, but no CUDA device is available.")
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device preference: {device_pref!r}. Use 'auto', 'cpu', or 'cuda'.")


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
