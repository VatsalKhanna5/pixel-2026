"""
src/utils/visualization.py
============================
Phase 5 — Paper figure generation for PIXEL-2026.

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
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "DejaVu Serif"],
    "font.size":            13,
    "axes.labelsize":       13,
    "axes.titlesize":       14,
    "axes.titleweight":     "bold",
    "axes.titlepad":        10,
    "axes.linewidth":       0.9,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.labelsize":      12,
    "ytick.labelsize":      12,
    "xtick.major.pad":      6,
    "ytick.major.pad":      4,
    "legend.fontsize":      11,
    "legend.framealpha":    0.92,
    "legend.edgecolor":     "#AAAAAA",
    "legend.handlelength":  1.4,
    "grid.linewidth":       0.5,
    "grid.alpha":           0.4,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.08,
})

# Method colours
COLORS = {
    "pixel_guided":         "#1565C0",
    "cfg_only":             "#7B1FA2",
    "det_cnn":              "#C62828",
    "cvae":                 "#E65100",
    "ablation_no_topo":     "#37474F",
    "ablation_no_drc":      "#607D8B",
    "ablation_no_guidance": "#90A4AE",
}

# Short single-line labels — no \n, used on x-ticks of vertical bar charts
TICK_LABELS = {
    "pixel_guided":         "PIXEL",
    "cfg_only":             "CFG-only",
    "det_cnn":              "Det-CNN",
    "cvae":                 "cVAE",
    "ablation_no_topo":     "w/o Topo",
    "ablation_no_drc":      "w/o DRC",
    "ablation_no_guidance": "w/o Guidance",
}

# Full labels for legends and horizontal bar charts
FULL_LABELS = {
    "pixel_guided":         "PIXEL (guided)",
    "cfg_only":             "CFG-only",
    "det_cnn":              "Det-CNN",
    "cvae":                 "cVAE",
    "ablation_no_topo":     "w/o Topology guidance",
    "ablation_no_drc":      "w/o DRC guidance",
    "ablation_no_guidance": "w/o Any guidance",
}

LAYOUT_CMAP = ListedColormap(["#F5F5F5", "#1565C0"])


# ---------------------------------------------------------------------------
# Figure 1: Layout gallery
# ---------------------------------------------------------------------------

def fig_layout_gallery(
    layouts_dict: dict,
    n_show: int = 5,
    out_path: Path = None,
) -> None:
    methods = list(layouts_dict.keys())
    n_rows  = len(methods)
    cell_w, cell_h = 1.8, 1.8
    fig, axes = plt.subplots(
        n_rows, n_show,
        figsize=(cell_w * n_show + 1.5, cell_h * n_rows + 0.6),
        gridspec_kw={"hspace": 0.12, "wspace": 0.06},
    )
    if n_rows == 1:
        axes = axes[None]

    rng = np.random.default_rng(0)
    for row, label in enumerate(methods):
        arr  = (layouts_dict[label] == 1).astype(float)
        idxs = rng.choice(len(arr), size=n_show, replace=False)
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(arr[i], cmap=LAYOUT_CMAP, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_visible(True)
            if col == 0:
                ax.set_ylabel(FULL_LABELS.get(label, label),
                              fontsize=10, rotation=0,
                              labelpad=80, va="center")
            if row == 0:
                ax.set_title(f"Sample {col+1}", fontsize=10)
            ax.plot(0,  7, ">", color="#E53935", ms=4, clip_on=False)
            ax.plot(14, 7, ">", color="#43A047", ms=4, clip_on=False)

    _save(fig, out_path, "layout_gallery")


# ---------------------------------------------------------------------------
# Figure 2: S-parameter comparison
# ---------------------------------------------------------------------------

def fig_sparams(
    y_star:      np.ndarray,
    y_pred_dict: dict,
    freq_ghz:    np.ndarray,
    n_show: int = 3,
    out_path: Path = None,
) -> None:
    rng  = np.random.default_rng(1)
    idxs = rng.choice(len(y_star), size=n_show, replace=False)

    fig, axes = plt.subplots(1, n_show, figsize=(5.0 * n_show, 3.8), sharey=True,
                             gridspec_kw={"wspace": 0.18})
    for col, i in enumerate(idxs):
        ax = axes[col]
        ax.plot(freq_ghz, y_star[i, 1], "k-", lw=2.0, label="Target", zorder=6)
        for key, yp in y_pred_dict.items():
            ax.plot(freq_ghz, yp[i, 1], "--",
                    color=COLORS.get(key, "#607D8B"), lw=1.3,
                    label=FULL_LABELS.get(key, key), alpha=0.9)
        ax.set_xlabel("Frequency (GHz)")
        if col == 0:
            ax.set_ylabel(r"$|S_{21}|$ (normalised)")
        ax.set_title(f"Test Spec #{i + 1}")
        ax.grid(True)
        if col == 0:
            ax.legend(fontsize=10, loc="best")

    _save(fig, out_path, "sparams_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Metrics bar chart — 2×2 grid, one metric per panel
# ---------------------------------------------------------------------------

def fig_metrics_bar(results: dict, out_path: Path = None) -> None:
    # Non-ablation methods only
    keys   = [k for k in results if not k.startswith("ablation")]
    xlbls  = [TICK_LABELS.get(k, k) for k in keys]
    colors = [COLORS.get(k, "#607D8B") for k in keys]
    x      = np.arange(len(keys))
    w      = 0.55

    # (title, values, y_range, gate, y_label, val_fmt)
    panels = [
        ("Connectivity Yield",
         [results[k]["conn_clean"] for k in keys],
         (0.955, 1.012), 0.95, "Proportion (post-processed)", ".3f"),

        ("DRC Pass Rate",
         [results[k]["drc_clean"] for k in keys],
         (0.940, 1.012), 0.90, "Proportion (post-processed)", ".3f"),

        ("Surrogate S21 MSE  (×10⁻³)",
         [results[k]["s21_mse"] * 1e3 for k in keys],
         None, None, "MSE  (×10⁻³)", ".2f"),

        ("Layout Diversity",
         [results[k]["hamming"] for k in keys],
         None, None, "Mean Hamming distance (bits)", ".1f"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10),
                             layout="constrained")

    for idx, (ax, (title, vals, y_range, gate, ylabel, fmt)) in \
            enumerate(zip(axes.flat, panels)):

        bars = ax.bar(x, vals, w, color=colors,
                      edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, rotation=30, ha="right", fontsize=12)
        ax.set_title(title)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, axis="y", zorder=0)

        if y_range is not None:
            ax.set_ylim(y_range)
            # tick every 0.01 for proportion panels
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
        else:
            lo = min(vals) * 0.94
            hi = max(vals) * 1.08
            ax.set_ylim(lo, hi)

        if gate is not None:
            ax.axhline(gate, color="#D32F2F", lw=1.4, ls="--",
                       zorder=4, label=f"Gate = {gate:.2f}")
            ax.legend(loc="lower right", fontsize=11)

        # value annotations — clear of the bar top
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        pad   = yspan * 0.015
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + pad,
                    format(v, fmt),
                    ha="center", va="bottom", fontsize=10.5,
                    color="#212121", fontweight="bold")

    _save(fig, out_path, "metrics_bar")


# ---------------------------------------------------------------------------
# Figure 4: Ablation — horizontal grouped bars, one panel per metric
# ---------------------------------------------------------------------------

def fig_ablation(results: dict, out_path: Path = None) -> None:
    abl_keys = ["pixel_guided"] + [k for k in results if k.startswith("ablation")]
    if len(abl_keys) < 2:
        print("[fig] No ablation results — skipping")
        return

    # Horizontal bar layout: methods on y-axis (readable), bars extend right
    ylbls  = [FULL_LABELS.get(k, results[k]["label"]) for k in abl_keys]
    colors = [COLORS.get(k, "#607D8B") for k in abl_keys]
    y      = np.arange(len(abl_keys))
    h      = 0.55

    panels = [
        ("Connectivity Yield\n(post-processed)",
         [results[k]["conn_clean"] for k in abl_keys],
         (0.975, 1.007), "Proportion", ".4f"),

        ("DRC Pass Rate\n(post-processed)",
         [results[k]["drc_clean"] for k in abl_keys],
         (0.975, 1.007), "Proportion", ".4f"),

        ("Surrogate S21 MSE  (×10⁻³)",
         [results[k]["s21_mse"] * 1e3 for k in abl_keys],
         None, "MSE  (×10⁻³)", ".3f"),

        ("Layout Diversity",
         [results[k]["hamming"] for k in abl_keys],
         None, "Mean Hamming bits", ".2f"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5),
                             layout="constrained")

    for ax, (title, vals, x_range, xlabel, fmt) in zip(axes, panels):
        bars = ax.barh(y, vals, h, color=colors,
                       edgecolor="white", linewidth=0.8, zorder=3)

        ax.set_yticks(y)
        ax.set_yticklabels(ylbls, fontsize=11.5)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.invert_yaxis()              # PIXEL (guided) at top
        ax.grid(True, axis="x", zorder=0)

        if x_range is not None:
            # Extend x limit to give room for bar-end value labels
            span = x_range[1] - x_range[0]
            ax.set_xlim(x_range[0], x_range[1] + span * 0.22)
            # At most 4 ticks, no crowding
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))
        else:
            span = max(vals) - min(vals)
            ax.set_xlim(min(vals) - span * 0.3, max(vals) + span * 0.55)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune="both"))

        # Rotate x-tick labels to prevent overlap
        ax.tick_params(axis="x", labelrotation=30, labelsize=11)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        # value labels at bar ends
        xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        pad   = xspan * 0.010
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + pad,
                    bar.get_y() + bar.get_height() / 2,
                    format(v, fmt),
                    va="center", ha="left", fontsize=10.5,
                    color="#212121", fontweight="bold")

    # Highlight PIXEL bar with a subtle box in all panels
    for ax in axes:
        for bar in ax.patches[:1]:        # first bar = PIXEL
            bar.set_edgecolor("#1565C0")
            bar.set_linewidth(2.0)

    _save(fig, out_path, "ablation_study")


# ---------------------------------------------------------------------------
# Figure 5: Diversity vs accuracy scatter
# ---------------------------------------------------------------------------

def fig_diversity_scatter(results: dict, out_path: Path = None) -> None:
    """
    Accuracy vs diversity scatter.

    The diffusion-based methods (PIXEL + CFG + 3 ablations) cluster tightly
    at nearly identical (S21 MSE, Hamming) coordinates.  Annotating them
    individually causes collision.  Strategy:
      - Draw a shaded ellipse bounding the diffusion cluster
      - Label the cluster with an arrow + text box
      - Annotate cVAE and Det-CNN individually (they are distinct)
      - Use a legend for per-method colour identification
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, Ellipse

    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    all_keys  = list(results.keys())
    diff_keys = [k for k in all_keys
                 if k not in ("det_cnn", "cvae")]   # diffusion cluster
    solo_keys = ["cvae", "det_cnn"]

    # ── 1. Draw diffusion cluster shading ────────────────────────────────
    clust_x = [results[k]["s21_mse"] for k in diff_keys]
    clust_y = [results[k]["hamming"] for k in diff_keys]
    cx, cy  = np.mean(clust_x), np.mean(clust_y)
    rx = (max(clust_x) - min(clust_x)) / 2 + 0.00012
    ry = (max(clust_y) - min(clust_y)) / 2 + 0.08
    ellipse = Ellipse((cx, cy), width=2 * rx, height=2 * ry,
                      facecolor="#BBDEFB", edgecolor="#1565C0",
                      linewidth=1.2, alpha=0.45, zorder=2)
    ax.add_patch(ellipse)

    # ── 2. Plot diffusion points ──────────────────────────────────────────
    for key in diff_keys:
        r = results[key]
        marker = "o" if key == "pixel_guided" else "s"
        size   = 180 if key == "pixel_guided" else 90
        ax.scatter(r["s21_mse"], r["hamming"],
                   s=size, color=COLORS.get(key, "#607D8B"),
                   marker=marker, edgecolors="#FFFFFF", linewidths=1.2,
                   zorder=4, label=FULL_LABELS.get(key, key))

    # Label the cluster with an arrow from above-left
    ax.annotate(
        "Diffusion-based methods\n(PIXEL + CFG + ablations)",
        xy=(cx, cy + ry),
        xytext=(cx - 0.0018, cy + ry + 0.55),
        fontsize=11, color="#1565C0", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color="#1565C0",
                        lw=1.2, connectionstyle="arc3,rad=-0.1"),
        bbox=dict(boxstyle="round,pad=0.3", fc="#E3F2FD",
                  ec="#1565C0", lw=0.8, alpha=0.9),
        zorder=6,
    )

    # ── 3. Plot & annotate standalone methods ─────────────────────────────
    for key in solo_keys:
        r = results[key]
        ax.scatter(r["s21_mse"], r["hamming"],
                   s=200, color=COLORS.get(key, "#607D8B"),
                   marker="o", edgecolors="#FFFFFF", linewidths=1.5,
                   zorder=5, label=FULL_LABELS.get(key, key))
        x_off = -110 if key == "det_cnn" else 10
        y_off = 8
        ax.annotate(
            FULL_LABELS.get(key, key),
            xy=(r["s21_mse"], r["hamming"]),
            xytext=(x_off, y_off), textcoords="offset points",
            fontsize=12, fontweight="bold",
            color=COLORS.get(key, "#212121"),
            arrowprops=dict(arrowstyle="-", color=COLORS.get(key, "#607D8B"),
                            lw=0.8),
            zorder=6,
        )

    # ── 4. Axes, grid, decorations ────────────────────────────────────────
    ax.set_xlabel("Surrogate $S_{21}$ MSE   ($\\downarrow$ better)", fontsize=13)
    ax.set_ylabel("Mean Hamming Diversity  ($\\uparrow$ better)", fontsize=13)
    ax.grid(True)

    # Diversity goal line
    ax.axhline(30, color="#B0BEC5", ls=":", lw=1.5,
               label="Diversity target (30 bits)")

    # Axis limits with padding for annotations
    all_x = [results[k]["s21_mse"] for k in all_keys]
    all_y = [results[k]["hamming"] for k in all_keys]
    ax.set_xlim(min(all_x) - 0.0004, max(all_x) + 0.0018)
    ax.set_ylim(min(all_y) - 0.4, 30 + 0.8)

    # ── 5. Legend (2-column for space) ───────────────────────────────────
    ax.legend(fontsize=10, loc="lower right", ncol=2, framealpha=0.92,
              handletextpad=0.4, columnspacing=0.8)

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
