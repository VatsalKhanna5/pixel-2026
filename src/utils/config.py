"""
src/utils/config.py
PIXEL-2026 — Configuration management utilities.

Loads OmegaConf YAML configs, merges overrides from CLI,
sets all random seeds, configures PyTorch backends.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load a YAML config and apply optional dot-notation overrides.

    Args:
        config_path: Path to the YAML file.
        overrides:   List of "key=value" strings, e.g. ["denoiser.lr=5e-5"].

    Returns:
        Merged OmegaConf DictConfig (struct mode enabled).
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Fix all random seeds for full reproducibility.

    Covers: Python random, NumPy, PyTorch CPU/GPU.
    Sets deterministic CUDA operations when requested.

    Note: deterministic=True may slightly reduce throughput on some ops.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True   # faster non-deterministic training


def configure_from_cfg(cfg: DictConfig) -> None:
    """Apply global settings from a loaded config."""
    set_seed(cfg.seed, cfg.deterministic)


def pretty_print(cfg: DictConfig, title: str = "Config") -> None:
    """Print config as YAML to stdout."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(OmegaConf.to_yaml(cfg))
    print(f"{'='*60}\n")


def save_config(cfg: DictConfig, save_path: str | Path) -> None:
    """Persist the resolved config alongside a checkpoint."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_path)


def get_project_root() -> Path:
    """Return the absolute path to the project root (contains pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")
