"""
src/utils/visualization.py
============================
Phase 5 — Paper figure generation for PIXEL-2026.

Generates all figures needed for the AAAI-2027 paper:
  Fig 1: Example generated layouts (PIXEL vs baselines)
  Fig 2: S-parameter comparison (target vs surrogate-predicted)
  Fig 3: Evaluation metrics bar chart (all methods)
  Fig 4: Ablation study bar chart
  Fig 5: Diversity scatter (Hamming vs S21 MSE)

All figures saved to paper/figures/ as PDF + PNG.

Usage:
    python -m src.utils.visualization \\
        --eval-dir experiments/full_eval_v1 \\
        --out-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

COLORS = {
    "pixel_guided":         "#2196F3",
    "cfg_only":             "#9C27B0",
    "det_cnn":              "#F44336",
    "cvae":                 "#FF9800",
    "ablation_no_topo":     "#607D8B",
    "ablation_no_drc":      "#78909C",
    "ablation_no_guidance": "#90A4AE",
}

LAYOUT_CMAP = ListedColormap(["#FFFFFF", "#1565C0"])   # white=dielectric, blue=conductor

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi":    150,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
})


# ---------------------------------------------------------------------------
# Figure 1: Layout gallery
# ---------------------------------------------------------------------------

def fig_layout_gallery(
    layouts_dict: dict,     # {label: (N, 15, 15) ndarray, values 0 or 1}
    n_show: int = 5,
    out_path: Path = None,
) -> None:
    methods = list(layouts_dict.keys())
    fig, axes = plt.subplots(len(methods), n_show,
                             figsize=(2.5 * n_show, 2.2 * len(methods)))
    if len(methods) == 1:
        axes = axes[None]

    rng = np.random.default_rng(0)
    for row, label in enumerate(methods):
        arr = (layouts_dict[label] == 1).astype(float)
        idxs = rng.choice(len(arr), size=n_show, replace=False)
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(arr[i], cmap=LAYOUT_CMAP, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=70, va="center")
            if row == 0:
                ax.set_title(f"Sample {col+1}", fontsize=9)
            ax.plot(0,  7, "r>", markersize=5, clip_on=False)
            ax.plot(14, 7, "g>", markersize=5, clip_on=False)

    fig.suptitle("Generated EM Layouts — PIXEL vs Baselines", y=1.01)
    plt.tight_layout()
    _save(fig, out_path, "layout_gallery")


# ---------------------------------------------------------------------------
# Figure 2: S-parameter comparison
# ---------------------------------------------------------------------------

def fig_sparams(
    y_star:      np.ndarray,    # (N, 4, 100) target (normalised)
    y_pred_dict: dict,          # {label: (N, 4, 100)}
    freq_ghz:    np.ndarray,    # (100,)
    n_show: int = 3,
    out_path: Path = None,
) -> None:
    rng = np.random.default_rng(1)
    idxs = rng.choice(len(y_star), size=n_show, replace=False)
    palette = list(COLORS.values())

    fig, axes = plt.subplots(1, n_show, figsize=(4.2 * n_show, 3.5), sharey=True)
    for col, i in enumerate(idxs):
        ax = axes[col]
        ax.plot(freq_ghz, y_star[i, 1], "k-", lw=2, label="Target", zorder=5)
        for (lbl, yp), color in zip(y_pred_dict.items(), palette):
            ax.plot(freq_ghz, yp[i, 1], "--", color=color, lw=1.2, label=lbl)
        ax.set_xlabel("Frequency (GHz)")
        if col == 0:
            ax.set_ylabel(r"$|S_{21}|$ (normalised)")
        ax.set_title(f"Spec #{i}")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=8)
    fig.suptitle(r"$|S_{21}|$ Target vs Surrogate-Predicted Outputs")
    plt.tight_layout()
    _save(fig, out_path, "sparams_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Metrics bar chart (all methods)
# ---------------------------------------------------------------------------

def fig_metrics_bar(results: dict, out_path: Path = None) -> None:
    method_keys = [k for k in results if not k.startswith("ablation")]
    labels = [results[k]["label"] for k in method_keys]
    colors = [COLORS.get(k, "#607D8B") for k in method_keys]

    metrics_spec = [
        ("Connectivity\nyield",    [results[k]["conn_clean"] for k in method_keys], 0.95),
        ("DRC pass\n(post-proc)",  [results[k]["drc_clean"]  for k in method_keys], 0.90),
        ("S21 MSE (×0.01)",        [results[k]["s21_mse"] * 100 for k in method_keys], 5.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(labels))
    for ax, (name, vals, gate) in zip(axes, metrics_spec):
        bars = ax.bar(x, vals, 0.6, color=colors, edgecolor="white", lw=0.8)
        ax.axhline(gate, color="red", lw=1.5, ls="--", label=f"Gate={gate}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(name)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(max(vals), gate),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.legend(fontsize=8)

    fig.suptitle("PIXEL vs Baselines — Key Metrics", fontsize=13)
    plt.tight_layout()
    _save(fig, out_path, "metrics_bar")


# ---------------------------------------------------------------------------
# Figure 4: Ablation bar chart
# ---------------------------------------------------------------------------

def fig_ablation(results: dict, out_path: Path = None) -> None:
    abl_keys = ["pixel_guided"] + [k for k in results if k.startswith("ablation")]
    if len(abl_keys) < 2:
        print("[fig] No ablation results to plot — skipping ablation figure")
        return
    labels = [results[k]["label"] for k in abl_keys]
    colors = [COLORS["pixel_guided"]] + [COLORS.get(k, "#607D8B") for k in abl_keys[1:]]

    specs = [
        ("Connectivity\n(cleaned)", [results[k]["conn_clean"] for k in abl_keys]),
        ("S21 MSE",                 [results[k]["s21_mse"]    for k in abl_keys]),
        ("Hamming diversity",       [results[k]["hamming"]    for k in abl_keys]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(labels))
    for ax, (name, vals) in zip(axes, specs):
        bars = ax.bar(x, vals, 0.6, color=colors, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_title(name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005 * max(vals),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Ablation Study — Contribution of Each Guidance Component", fontsize=13)
    plt.tight_layout()
    _save(fig, out_path, "ablation_study")


# ---------------------------------------------------------------------------
# Figure 5: Diversity vs accuracy scatter
# ---------------------------------------------------------------------------

def fig_diversity_scatter(results: dict, out_path: Path = None) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for key, r in results.items():
        color = COLORS.get(key, "#607D8B")
        ax.scatter(r["s21_mse"], r["hamming"], s=120, color=color, zorder=5,
                   edgecolors="white", lw=0.8, label=r["label"])
        ax.annotate(r["label"], (r["s21_mse"], r["hamming"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5)

    ax.set_xlabel("Surrogate S21 MSE  (↓ better)")
    ax.set_ylabel("Hamming diversity  (↑ better)")
    ax.set_title("Accuracy vs Diversity Trade-off")
    ax.axhline(30, color="grey", ls=":", lw=1, label="Diversity gate (30 bits)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out_path, "diversity_scatter")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig, out_path: Path, name: str) -> None:
    if out_path is None:
        out_path = Path("paper/figures")
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"{name}.pdf")
    fig.savefig(out_path / f"{name}.png")
    plt.close(fig)
    print(f"[fig] Saved {name}.pdf + .png", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", default="experiments/full_eval_v1")
    parser.add_argument("--out-dir",  default="paper/figures")
    args = parser.parse_args()

    summary_path = Path(args.eval_dir) / "full_eval_summary.json"
    if not summary_path.exists():
        print(f"[warn] {summary_path} not found — run full_eval.py first")
        return

    with open(summary_path) as f:
        results = json.load(f)

    out_dir = Path(args.out_dir)
    print(f"[viz] Generating paper figures …")
    fig_metrics_bar(results, out_dir)
    fig_ablation(results,    out_dir)
    fig_diversity_scatter(results, out_dir)
    print(f"[done] Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
