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
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker


# ---------------------------------------------------------------------------
# Style — AAAI-2027 / NeurIPS compatible
# ---------------------------------------------------------------------------

# Short display labels for each method key
SHORT_LABELS = {
    "pixel_guided":         "PIXEL\n(guided)",
    "cfg_only":             "CFG-only",
    "det_cnn":              "Det-CNN",
    "cvae":                 "cVAE",
    "ablation_no_topo":     "−Topo\nguide",
    "ablation_no_drc":      "−DRC\nguide",
    "ablation_no_guidance": "No\nguide",
}

COLORS = {
    "pixel_guided":         "#1565C0",   # strong blue — hero method
    "cfg_only":             "#7B1FA2",   # purple
    "det_cnn":              "#C62828",   # red
    "cvae":                 "#E65100",   # deep orange
    "ablation_no_topo":     "#546E7A",   # blue-grey
    "ablation_no_drc":      "#78909C",
    "ablation_no_guidance": "#90A4AE",
}

LAYOUT_CMAP = ListedColormap(["#F5F5F5", "#1565C0"])   # light grey / strong blue

# Base rcParams — serif font, crisp rendering
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "axes.titleweight": "bold",
    "axes.linewidth":   0.8,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "legend.framealpha":0.85,
    "legend.edgecolor": "#CCCCCC",
    "grid.linewidth":   0.5,
    "grid.alpha":       0.35,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

# AAAI double-column width ≈ 3.3" per column; full width ≈ 7.0"
COL1 = 3.3
COL2 = 6.8


# ---------------------------------------------------------------------------
# Figure 1: Layout gallery
# ---------------------------------------------------------------------------

def fig_layout_gallery(
    layouts_dict: dict,     # {label: (N, 15, 15) ndarray, values 0 or 1}
    n_show: int = 5,
    out_path: Path = None,
) -> None:
    methods = list(layouts_dict.keys())
    fig, axes = plt.subplots(
        len(methods), n_show,
        figsize=(1.55 * n_show, 1.5 * len(methods)),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
    )
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
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_visible(True)
            if col == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0,
                              labelpad=55, va="center")
            if row == 0:
                ax.set_title(f"#{col+1}", fontsize=8)
            # port markers
            ax.plot(0,  7, ">", color="#E53935", markersize=4, clip_on=False)
            ax.plot(14, 7, ">", color="#43A047", markersize=4, clip_on=False)

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

    fig, axes = plt.subplots(
        1, n_show, figsize=(COL2, 2.4), sharey=True,
        gridspec_kw={"wspace": 0.12},
    )
    for col, i in enumerate(idxs):
        ax = axes[col]
        ax.plot(freq_ghz, y_star[i, 1], "k-", lw=1.8, label="Target", zorder=6)
        for key, yp in y_pred_dict.items():
            ax.plot(freq_ghz, yp[i, 1], "--", color=COLORS.get(key, "#607D8B"),
                    lw=1.0, label=SHORT_LABELS.get(key, key), alpha=0.9)
        ax.set_xlabel("Frequency (GHz)", fontsize=8)
        if col == 0:
            ax.set_ylabel(r"$|S_{21}|$ (norm.)", fontsize=8)
        ax.set_title(f"Spec {i+1}", fontsize=9)
        ax.grid(True)
        if col == 0:
            ax.legend(fontsize=7, loc="best", ncol=1)

    _save(fig, out_path, "sparams_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Metrics bar chart — main results
# ---------------------------------------------------------------------------

def fig_metrics_bar(results: dict, out_path: Path = None) -> None:
    # Only non-ablation methods
    method_keys = [k for k in results if not k.startswith("ablation")]
    labels = [SHORT_LABELS.get(k, results[k]["label"]) for k in method_keys]
    colors = [COLORS.get(k, "#607D8B") for k in method_keys]
    x = np.arange(len(labels))
    w = 0.6

    metrics_spec = [
        # (panel title,  values,                             y_range,         gate,   y_label,          fmt)
        ("Connectivity",
         [results[k]["conn_clean"] for k in method_keys],
         (0.93, 1.005), 0.95, "Yield (post-proc.)", ".3f"),
        ("DRC Pass Rate",
         [results[k]["drc_clean"]  for k in method_keys],
         (0.93, 1.005), 0.90, "Pass rate (post-proc.)", ".3f"),
        ("S21 MSE",
         [results[k]["s21_mse"]    for k in method_keys],
         None,           None, "MSE", ".4f"),
        ("Layout Diversity",
         [results[k]["hamming"]    for k in method_keys],
         None,           None, "Hamming bits", ".1f"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.8),
                             gridspec_kw={"wspace": 0.45})

    for ax, (title, vals, y_range, gate, ylabel, fmt) in zip(axes, metrics_spec):
        bars = ax.bar(x, vals, w, color=colors, edgecolor="white", lw=0.6,
                      zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, ha="center")
        ax.set_title(title, pad=4)
        ax.set_ylabel(ylabel, fontsize=7.5, labelpad=2)
        ax.grid(True, axis="y", zorder=0)

        if y_range is not None:
            ax.set_ylim(y_range)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))
        else:
            # auto range with 20% headroom
            lo = min(vals) * 0.92
            hi = max(vals) * 1.10
            ax.set_ylim(lo, hi)

        if gate is not None:
            ax.axhline(gate, color="#E53935", lw=1.0, ls="--",
                       label=f"Gate {gate:.2f}", zorder=4)
            ax.legend(fontsize=7, loc="lower right")

        # value labels — positioned just above each bar
        pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad,
                    format(v, fmt),
                    ha="center", va="bottom", fontsize=6.5, color="#212121")

    _save(fig, out_path, "metrics_bar")


# ---------------------------------------------------------------------------
# Figure 4: Ablation bar chart
# ---------------------------------------------------------------------------

def fig_ablation(results: dict, out_path: Path = None) -> None:
    abl_keys = ["pixel_guided"] + [k for k in results if k.startswith("ablation")]
    if len(abl_keys) < 2:
        print("[fig] No ablation results — skipping")
        return

    labels = [SHORT_LABELS.get(k, results[k]["label"]) for k in abl_keys]
    colors = [COLORS["pixel_guided"]] + [COLORS.get(k, "#607D8B") for k in abl_keys[1:]]
    x = np.arange(len(labels))
    w = 0.55

    specs = [
        ("Connectivity",    [results[k]["conn_clean"] for k in abl_keys],
         (0.97, 1.005), "Yield"),
        ("DRC Pass Rate",   [results[k]["drc_clean"]  for k in abl_keys],
         (0.97, 1.005), "Pass rate"),
        ("S21 MSE",         [results[k]["s21_mse"]    for k in abl_keys],
         None, "MSE"),
        ("Hamming Diversity",[results[k]["hamming"]   for k in abl_keys],
         None, "Hamming bits"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.8),
                             gridspec_kw={"wspace": 0.48})

    for ax, (title, vals, y_range, ylabel) in zip(axes, specs):
        bars = ax.bar(x, vals, w, color=colors, edgecolor="white", lw=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5, ha="center")
        ax.set_title(title, pad=4)
        ax.set_ylabel(ylabel, fontsize=7.5, labelpad=2)
        ax.grid(True, axis="y", zorder=0)

        if y_range is not None:
            ax.set_ylim(y_range)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
        else:
            lo = min(vals) * 0.97
            hi = max(vals) * 1.06
            ax.set_ylim(lo, hi)

        fmt = ".4f" if title == "S21 MSE" else (".1f" if "Diversity" in title else ".3f")
        pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.012
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad,
                    format(v, fmt),
                    ha="center", va="bottom", fontsize=6.5, color="#212121")

    # shared legend at bottom
    patches = [mpatches.Patch(color=colors[i], label=labels[i])
               for i in range(len(labels))]
    fig.legend(handles=patches, loc="lower center", ncol=len(labels),
               fontsize=7.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.12))

    _save(fig, out_path, "ablation_study")


# ---------------------------------------------------------------------------
# Figure 5: Diversity vs accuracy scatter
# ---------------------------------------------------------------------------

def fig_diversity_scatter(results: dict, out_path: Path = None) -> None:
    # Pre-compute reasonable annotation offsets to avoid overlap
    OFFSETS = {
        "pixel_guided":         ( 6,  3),
        "cfg_only":             ( 6, -8),
        "det_cnn":              (-38, -9),
        "cvae":                 ( 6, -8),
        "ablation_no_topo":     ( 6,  3),
        "ablation_no_drc":      ( 6,  3),
        "ablation_no_guidance": (-60, -9),
    }

    fig, ax = plt.subplots(figsize=(COL1 + 0.4, 2.8))

    # Separate ablations so they can be styled differently
    main_keys = [k for k in results if not k.startswith("ablation")]
    abl_keys  = [k for k in results if k.startswith("ablation")]

    for key in abl_keys:
        r = results[key]
        ax.scatter(r["s21_mse"], r["hamming"], s=55,
                   color=COLORS.get(key, "#90A4AE"),
                   marker="s", edgecolors="#FFFFFF", lw=0.5,
                   alpha=0.75, zorder=4)
        off = OFFSETS.get(key, (5, 3))
        ax.annotate(SHORT_LABELS.get(key, r["label"]),
                    (r["s21_mse"], r["hamming"]),
                    textcoords="offset points", xytext=off,
                    fontsize=6.5, color="#546E7A", style="italic")

    for key in main_keys:
        r = results[key]
        ax.scatter(r["s21_mse"], r["hamming"], s=90,
                   color=COLORS.get(key, "#607D8B"),
                   marker="o", edgecolors="#FFFFFF", lw=0.8,
                   zorder=5)
        off = OFFSETS.get(key, (5, 3))
        ax.annotate(SHORT_LABELS.get(key, r["label"]).replace("\n", " "),
                    (r["s21_mse"], r["hamming"]),
                    textcoords="offset points", xytext=off,
                    fontsize=7.5, fontweight="bold",
                    color=COLORS.get(key, "#212121"))

    ax.set_xlabel("Surrogate $S_{21}$ MSE  ($\\downarrow$ better)", fontsize=9)
    ax.set_ylabel("Hamming Diversity  ($\\uparrow$ better)", fontsize=9)
    ax.grid(True)

    # diversity gate line
    ax.axhline(30, color="#B0BEC5", ls=":", lw=1.0, label="Diversity target (30 bits)")

    # ideal corner annotation
    ax.annotate("Ideal", xy=(ax.get_xlim()[0], 30),
                xytext=(3, 3), textcoords="offset points",
                fontsize=7, color="#B0BEC5")

    # legend for marker shapes
    circ  = mpatches.Patch(color="#607D8B", label="Main methods (●)")
    sq    = mpatches.Patch(color="#90A4AE", label="Ablations (■)")
    ax.legend(handles=[circ, sq], fontsize=7.5, loc="lower left")

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
    print("[viz] Generating paper figures …")
    fig_metrics_bar(results, out_dir)
    fig_ablation(results,    out_dir)
    fig_diversity_scatter(results, out_dir)
    print(f"[done] Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
