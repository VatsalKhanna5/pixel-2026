"""
src/evaluation/stats_tests.py
==============================
Phase 7 — Statistical testing battery for paper-grade significance claims.

Runs on Phase 7 EM results after phase7_em.py completes.
All comparisons are PAIRED (same 100 specs for every method).

Tests applied:
  1. Wilcoxon signed-rank (paired, non-parametric)  H1: PIXEL < other
  2. McNemar's test (binary success/fail at threshold)
  3. Bootstrap CI on coverage (B=2000)
  4. Rank-biserial correlation r (effect size for Wilcoxon)
  5. Bonferroni correction: α_corrected = 0.05 / n_comparisons
  6. Post-hoc power analysis

Also reports Phase 6 best-of-1 results for context.

Usage:
    python -m src.evaluation.stats_tests \\
        --em-dir    experiments/phase7/em \\
        --phase6    experiments/em_verify_v1/em_verify_summary.json \\
        --out-dir   experiments/phase7/stats
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Core test functions
# ---------------------------------------------------------------------------

def wilcoxon_test(x, y, alternative="less"):
    """Wilcoxon signed-rank test. H1: x tends to be less than y."""
    d = x - y
    d = d[d != 0]
    if len(d) < 10:
        warnings.warn("Wilcoxon: fewer than 10 non-zero differences.")
    stat, p = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
    return float(stat), float(p)


def rank_biserial(x, y):
    """Effect size r for Wilcoxon: r = 1 - 2U/(n1*n2), range [-1,1]."""
    n1, n2 = len(x), len(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    r = 1.0 - 2.0 * float(u) / (n1 * n2)
    return float(r)


def mcnemar_test(x, y, threshold):
    """
    McNemar's test on binary success/fail.
    H1: PIXEL (x) has higher success rate than y.
    b = PIXEL success & y fail (PIXEL only wins)
    c = PIXEL fail & y success (y only wins)
    """
    sx = x < threshold
    sy = y < threshold
    b = int(np.sum(sx & ~sy))
    c = int(np.sum(~sx & sy))
    if (b + c) == 0:
        return float("nan"), float("nan"), b, c
    # With continuity correction
    chi2 = (abs(b - c) - 1.0) ** 2 / (b + c)
    p = 1.0 - stats.chi2.cdf(chi2, df=1)
    # One-sided: if b > c, PIXEL is better
    p_one = p / 2.0 if b > c else 1.0 - p / 2.0
    return float(chi2), float(p_one), b, c


def bootstrap_coverage(x, threshold, B=2000, alpha=0.05):
    """Bootstrap CI for P(x < threshold)."""
    n     = len(x)
    boots = [np.mean(x[np.random.choice(n, n, replace=True)] < threshold)
             for _ in range(B)]
    ci_lo = float(np.percentile(boots, 100 * alpha / 2))
    ci_hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(np.mean(x < threshold)), ci_lo, ci_hi


def power_post_hoc(x, y, alpha=0.05):
    """
    Approximate post-hoc power using normal approximation to Wilcoxon.
    Returns observed effect size r and estimated power.
    """
    n = len(x)
    r = rank_biserial(x, y)
    # Normal approx: Z = r * sqrt(n)
    z = abs(r) * np.sqrt(n)
    z_alpha = stats.norm.ppf(1 - alpha)
    power = 1.0 - float(stats.norm.cdf(z_alpha - z))
    return float(r), float(power)


# ---------------------------------------------------------------------------
# Full comparison between two paired arrays
# ---------------------------------------------------------------------------

def full_comparison(px_bk, other_bk, label, alpha_bonf=0.05/3,
                    thresholds=(0.001, 0.005, 0.010)):
    """
    px_bk    : (N,) best-of-K EM MSE for PIXEL
    other_bk : (N,) best-of-K EM MSE for other method
    """
    assert len(px_bk) == len(other_bk), "Must be paired (same N specs)"
    N = len(px_bk)

    result = {"label": label, "N": N, "alpha_bonferroni": float(alpha_bonf)}

    # Wilcoxon
    w_stat, w_p = wilcoxon_test(px_bk, other_bk, alternative="less")
    result["wilcoxon"] = {
        "statistic": w_stat, "p_value": w_p,
        "significant": bool(w_p < alpha_bonf),
        "interpretation": ("PIXEL < " + label) if w_p < alpha_bonf else "no significant difference"
    }

    # Effect size
    r, power = power_post_hoc(px_bk, other_bk, alpha=alpha_bonf)
    result["effect_size"] = {
        "rank_biserial_r": r,
        "magnitude": "large" if abs(r) > 0.5 else ("medium" if abs(r) > 0.3 else "small"),
        "post_hoc_power": power,
    }

    # Coverage at each threshold
    result["coverage"] = {}
    for t in thresholds:
        px_cov,  px_lo,  px_hi  = bootstrap_coverage(px_bk,    t)
        ot_cov,  ot_lo,  ot_hi  = bootstrap_coverage(other_bk, t)
        chi2, mc_p, b, c = mcnemar_test(px_bk, other_bk, t)
        result["coverage"][f"t_{t:.3f}"] = {
            "pixel_cov":   px_cov,  "pixel_ci95":  [px_lo, px_hi],
            "other_cov":   ot_cov,  "other_ci95":  [ot_lo, ot_hi],
            "mcnemar_chi2": chi2,   "mcnemar_p":   mc_p,
            "mcnemar_b":   b,       "mcnemar_c":   c,
            "pixel_better": int(b), "other_better": int(c),
        }

    # Summary stats
    result["means"] = {
        "pixel_mean":  float(np.nanmean(px_bk)),
        "other_mean":  float(np.nanmean(other_bk)),
        "pixel_median": float(np.nanmedian(px_bk)),
        "other_median": float(np.nanmedian(other_bk)),
        "ratio_means":  float(np.nanmean(other_bk) / max(np.nanmean(px_bk), 1e-9)),
    }

    return result


# ---------------------------------------------------------------------------
# Diversity analysis
# ---------------------------------------------------------------------------

def diversity_analysis(pixel_div, cvae_div):
    """
    Compare within-spec diversity between PIXEL and cVAE.
    pixel_div, cvae_div: dict from summary["pixel_div"] / summary["cvae_div"]
    """
    ph = np.array(pixel_div["hamming_within"])
    ch = np.array(cvae_div["hamming_within"])
    pe = np.array(pixel_div["best_em_k"])
    ce = np.array(cvae_div["best_em_k"])

    w_stat, w_p = wilcoxon_test(ch, ph, alternative="less")  # H1: cVAE hamming < PIXEL
    return {
        "pixel_mean_hamming":  float(np.mean(ph)),
        "cvae_mean_hamming":   float(np.mean(ch)),
        "wilcoxon_hamming_p":  w_p,
        "pixel_higher_diversity": bool(np.mean(ph) > np.mean(ch)),
        "pixel_mean_best_em":  float(np.nanmean(pe)),
        "cvae_mean_best_em":   float(np.nanmean(ce)),
        "per_spec_hamming_pixel": ph.tolist(),
        "per_spec_hamming_cvae":  ch.tolist(),
    }


# ---------------------------------------------------------------------------
# Phase 6 context (best-of-1)
# ---------------------------------------------------------------------------

def phase6_context(phase6_path):
    """Extract best-of-1 coverage from Phase 6 em_verify results."""
    with open(phase6_path) as f:
        d = json.load(f)
    ctx = {}
    for key, v in d.items():
        ems = np.array([r["em_s21_mse"] for r in v["per_layout"] if r["error"] is None])
        valid = ems[ems <= 0.05]
        ctx[key] = {
            "N": int(v["n_verified"]),
            "cov_0001": float(np.mean(ems < 0.001)),
            "cov_0010": float(np.mean(ems < 0.010)),
            "cond_mean": float(np.mean(valid)),
            "max_em": float(np.max(ems)),
        }
    return ctx


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(results: dict) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("PHASE 7 STATISTICAL REPORT  (paired, N=100 specs, K=5)")
    lines.append("=" * 72)
    lines.append(f"Bonferroni-corrected α = {results['alpha_bonferroni']:.4f}  "
                 f"(3 comparisons: vs CFG, Det-CNN, cVAE)")
    lines.append("")

    for key in ["vs_cfg", "vs_det_cnn", "vs_cvae"]:
        if key not in results:
            continue
        r = results[key]
        lines.append(f"PIXEL vs {r['label']}  (N={r['N']})")
        w = r["wilcoxon"]
        es = r["effect_size"]
        lines.append(f"  Wilcoxon signed-rank: W={w['statistic']:.0f}  "
                     f"p={w['p_value']:.4e}  {'*' if w['significant'] else 'n.s.'}")
        lines.append(f"  Effect size r={es['rank_biserial_r']:+.3f}  "
                     f"({es['magnitude']})  power={es['post_hoc_power']:.2f}")
        lines.append(f"  Means: PIXEL={r['means']['pixel_mean']:.6f}  "
                     f"{r['label']}={r['means']['other_mean']:.6f}  "
                     f"ratio={r['means']['ratio_means']:.2f}×")
        lines.append(f"  Coverage @ 0.001:")
        c = r["coverage"]["t_0.001"]
        lines.append(f"    PIXEL  {100*c['pixel_cov']:.1f}%  "
                     f"[{100*c['pixel_ci95'][0]:.1f}–{100*c['pixel_ci95'][1]:.1f}%]")
        lines.append(f"    {r['label']:<10} {100*c['other_cov']:.1f}%  "
                     f"[{100*c['other_ci95'][0]:.1f}–{100*c['other_ci95'][1]:.1f}%]")
        mc = f"McNemar b={c['mcnemar_b']} c={c['mcnemar_c']} p={c['mcnemar_p']:.4e}"
        lines.append(f"    {mc}")
        lines.append("")

    if "diversity" in results:
        d = results["diversity"]
        lines.append("WITHIN-SPEC DIVERSITY (K=20 per spec, N=20 hardest specs)")
        lines.append(f"  PIXEL mean intra-spec Hamming: {d['pixel_mean_hamming']:.2f} bits")
        lines.append(f"  cVAE  mean intra-spec Hamming: {d['cvae_mean_hamming']:.2f} bits")
        lines.append(f"  Wilcoxon (cVAE<PIXEL Hamming): p={d['wilcoxon_hamming_p']:.4e}")
        lines.append(f"  PIXEL higher diversity: {d['pixel_higher_diversity']}")
        lines.append("")

    if "phase6_context" in results:
        lines.append("PHASE 6 CONTEXT (best-of-1, for comparison)")
        for key, ctx in results["phase6_context"].items():
            lines.append(f"  {key:<20} cov@0.001={100*ctx['cov_0001']:.1f}%  "
                         f"cov@0.010={100*ctx['cov_0010']:.1f}%  "
                         f"cond_mean={ctx['cond_mean']:.6f}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--em-dir",  default="experiments/phase7/em")
    parser.add_argument("--phase6",  default="experiments/em_verify_v1/em_verify_summary.json")
    parser.add_argument("--out-dir", default="experiments/phase7/stats")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    np.random.seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    em_path = Path(args.em_dir) / "em_results_p7.json"
    if not em_path.exists():
        print(f"[error] {em_path} not found. Run phase7_em.py first.")
        return

    with open(em_path) as f:
        em = json.load(f)

    px_bk  = np.array(em["pixel"]["best_em_k"])
    N      = len(px_bk)
    n_comp = sum(k in em for k in ["cfg", "det_cnn", "cvae"])
    alpha_bonf = 0.05 / max(n_comp, 1)

    all_results = {"N": N, "alpha_bonferroni": alpha_bonf}

    for key, label in [("cfg", "CFG-only"), ("det_cnn", "Det-CNN"), ("cvae", "cVAE")]:
        if key not in em:
            print(f"[skip] {key} not in EM results")
            continue
        other_bk = np.array(em[key]["best_em_k"])
        # Align: both should be length N; if Det-CNN has K=1, it's already best-of-1
        res = full_comparison(px_bk, other_bk, label,
                              alpha_bonf=alpha_bonf)
        all_results[f"vs_{key}"] = res

    # Diversity
    if "pixel_div" in em and "cvae_div" in em:
        all_results["diversity"] = diversity_analysis(em["pixel_div"], em["cvae_div"])

    # Phase 6 context
    if Path(args.phase6).exists():
        all_results["phase6_context"] = phase6_context(args.phase6)

    # Save
    stats_json = out_dir / "stats_summary.json"
    with open(stats_json, "w") as f:
        json.dump(all_results, f, indent=2)

    report = print_report(all_results)
    print(report)
    with open(out_dir / "stats_report.txt", "w") as f:
        f.write(report)

    print(f"\n[done] {stats_json}", flush=True)


if __name__ == "__main__":
    main()
