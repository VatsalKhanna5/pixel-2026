"""
src/utils/logging_utils.py
PIXEL-2026 — Centralised logging and WandB helpers.

Every training script imports get_logger() and (optionally) init_wandb().
All logs are written to logs/<run_name>_<timestamp>.log and to WandB.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional WandB — imported lazily so the module loads even without wandb
_wandb = None


def get_logger(
    name: str,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Build a logger that writes to stdout AND (optionally) a log file.

    Args:
        name:    Logger name (usually __name__ of the calling module).
        log_dir: If provided, also write to log_dir/<name>_<timestamp>.log
        level:   Logging level.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured — avoid duplicate handlers

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s | %(name)s | %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # file handler
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"{name}_{ts}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def init_wandb(
    cfg_dict: dict[str, Any],
    project: str = "pixel-2026",
    name: str | None = None,
    tags: list[str] | None = None,
    entity: str | None = None,
    resume: str = "allow",
) -> Any:
    """
    Initialise a WandB run.

    Args:
        cfg_dict: Flat or nested dict of hyperparameters to log.
        project:  WandB project name.
        name:     Run name (auto-generated if None).
        tags:     List of tags.
        entity:   WandB entity (username or org).
        resume:   "allow" | "must" | "never"

    Returns:
        The wandb.run object (or None if wandb is unavailable).
    """
    global _wandb
    try:
        import wandb
        _wandb = wandb
        run = wandb.init(
            project=project,
            name=name,
            config=cfg_dict,
            tags=tags or [],
            entity=entity,
            resume=resume,
        )
        return run
    except ImportError:
        print("wandb not installed — skipping WandB logging.")
        return None
    except Exception as exc:
        print(f"WandB init failed: {exc} — continuing without WandB.")
        return None


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a dict of metrics to WandB (if active)."""
    if _wandb is not None and _wandb.run is not None:
        _wandb.log(metrics, step=step)
