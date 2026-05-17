"""
src/utils/visualization.py
PIXEL-2026 — Visualisation utilities.

Provides:
  - plot_layout()        : render a 15×15 binary EM layout
  - plot_sparams()       : plot S11/S21 magnitude and phase vs. frequency
  - plot_layout_grid()   : tile multiple layouts for comparison
  - plot_training_curve(): smoothed loss curve with optional WandB sync
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")   # non-interactive backend safe for all environments


FREQ_UNIT = 1e9   # display frequencies in GHz


def plot_layout(
    layout: np.ndarray,
    title: str = "EM Layout",
    port1: tuple[int, int] = (7, 0),
    port2: tuple[int, int] = (7, 14),
    save_path: str | Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Render a binary 15×15 EM layout.

    Conductors are black, dielectric is white, port pixels are marked in red.

    Args:
        layout:    np.ndarray of shape (H, W), values in {0, 1}.
        title:     Plot title.
        port1/2:   (row, col) of port pixels — highlighted in red.
        save_path: If given, save the figure there.
        ax:        Existing Axes to draw on (creates new figure if None).

    Returns:
        The matplotlib Figure object.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()

    disp = np.where(layout == 1, 0.0, 1.0)   # 1=conductor → black (0); 0=void → white (1)
    ax.imshow(disp, cmap="gray", vmin=0, vmax=1, origin="upper")

    # Mark ports
    for (r, c), label in [(port1, "P1"), (port2, "P2")]:
        ax.scatter(c, r, color="red", s=80, zorder=5)
        ax.text(c + 0.4, r, label, color="red", fontsize=7, va="center")

    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{layout.shape[1]} px", fontsize=7)
    ax.set_ylabel(f"{layout.shape[0]} px", fontsize=7)

    if save_path is not None and standalone:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_sparams(
    freqs_hz: np.ndarray,
    s11_mag: np.ndarray,
    s21_mag: np.ndarray,
    s11_target: np.ndarray | None = None,
    s21_target: np.ndarray | None = None,
    title: str = "S-Parameters",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot S11 and S21 magnitude (dB) vs. frequency.

    Args:
        freqs_hz:   Frequency array in Hz, shape (N_f,).
        s11_mag:    |S11| in linear scale, shape (N_f,).
        s21_mag:    |S21| in linear scale, shape (N_f,).
        s11_target: Optional target |S11|.
        s21_target: Optional target |S21|.
        title:      Plot title.
        save_path:  If given, save the figure.

    Returns:
        The matplotlib Figure.
    """
    freqs_ghz = freqs_hz / FREQ_UNIT

    # Clamp to avoid log(0)
    s11_db = 20 * np.log10(np.clip(s11_mag, 1e-10, None))
    s21_db = 20 * np.log10(np.clip(s21_mag, 1e-10, None))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    fig.suptitle(title, fontsize=10)

    ax1.plot(freqs_ghz, s11_db, "b-", linewidth=1.5, label="|S₁₁| generated")
    if s11_target is not None:
        t_db = 20 * np.log10(np.clip(s11_target, 1e-10, None))
        ax1.plot(freqs_ghz, t_db, "r--", linewidth=1.2, label="|S₁₁| target")
    ax1.set_ylabel("|S₁₁| (dB)"); ax1.set_xlabel("Frequency (GHz)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.4)
    ax1.set_ylim([-50, 5])

    ax2.plot(freqs_ghz, s21_db, "g-", linewidth=1.5, label="|S₂₁| generated")
    if s21_target is not None:
        t_db = 20 * np.log10(np.clip(s21_target, 1e-10, None))
        ax2.plot(freqs_ghz, t_db, "r--", linewidth=1.2, label="|S₂₁| target")
    ax2.set_ylabel("|S₂₁| (dB)"); ax2.set_xlabel("Frequency (GHz)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.4)
    ax2.set_ylim([-50, 5])

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_layout_grid(
    layouts: np.ndarray,
    titles: Sequence[str] | None = None,
    ncols: int = 5,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Tile multiple layouts in a grid for comparison.

    Args:
        layouts:   np.ndarray of shape (N, H, W).
        titles:    Optional list of N title strings.
        ncols:     Number of columns in the grid.
        save_path: If given, save the figure.

    Returns:
        The matplotlib Figure.
    """
    N = len(layouts)
    nrows = math.ceil(N / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = np.array(axes).flatten()

    for i, ax in enumerate(axes):
        if i < N:
            t = titles[i] if titles else f"#{i}"
            plot_layout(layouts[i], title=t, ax=ax)
        else:
            ax.set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Uniform moving average for loss curve smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curve(
    losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    title: str = "Training Loss",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training (and optional validation) loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, alpha=0.3, color="steelblue", linewidth=0.8)
    ax.plot(smooth(np.array(losses)), color="steelblue", linewidth=1.5, label="train (smoothed)")
    if val_losses is not None:
        ax.plot(val_losses, color="tomato", linewidth=1.5, label="val")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
