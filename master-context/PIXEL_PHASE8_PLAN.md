# PIXEL-2026 — Phase 8 Execution Plan
## Statistical Hardening, Final Ablations, and Paper Completion
**Created: June 11, 2026 | Target: AAAI-2027 submission (~July 2026)**

> This document is the authoritative plan for Phase 8 onwards. Every task is sized for a single PBS job unless noted. Statistical targets are conservative — do not trade rigor for speed.

---

## STATUS ENTERING PHASE 8

### What we have (confirmed with full-wave EM):
| Claim | Evidence | Confidence |
|---|---|---|
| PIXEL >> Det-CNN (7.12×, p=8.5e-10) | Paired Wilcoxon N=100, K=5, power=0.62 | **HIGH** |
| PIXEL >> cVAE (2.62×, p=0.003) | Paired Wilcoxon N=100, K=5, power=0.05 | **MEDIUM** — underpowered |
| PIXEL > CFG K=1 (88.3% vs 84.0%) | Phase 6 best-of-1, unpaired | **LOW** — no paired test yet |
| PIXEL diversity >> cVAE (3.3×, p=1.2e-4) | Wilcoxon K=20 Hamming, N=20 | **HIGH** |
| 100% DRC compliance | BFS island removal (mathematical guarantee) | **PROVEN** |
| PIXEL vs CFG NOT significant at K=5 (p=0.319) | Paired Wilcoxon N=100, K=5 | **CONFIRMED** |

### Critical gaps before AAAI:
1. **PIXEL vs cVAE underpowered** — power=0.05 with N=100 is indefensible for a primary claim
2. **No formal paired test: PIXEL K=1 vs CFG K=1** — guidance effect exists but is unquantified statistically
3. **Connectivity failure 2.23–2.33%** — should be fixed to 0% or formally quantified
4. **Key ablation missing: No-guidance (α=0) vs PIXEL** — without this, guidance claim is merely directional
5. **No uncertainty-weighting ablation** — claimed as novel mechanism, never isolated in experiment
6. **No topology ablation with EM verification** — Phase 5 ablations used surrogate only
7. **Effect sizes need CIs** — currently point estimates only for Wilcoxon r

---

## TASK LIST — PHASE 8

### 8.1 Fix Connectivity Failure Rate (2h code, no PBS needed)
**Priority: HIGH — fixes known gap before any rerun**

**Problem:** 2.23–2.33% of generated layouts are disconnected (surrogate insensitive to connectivity).

**Fix** in `src/evaluation/phase7_eval.py` generation loop:
```python
# After generating each layout, BFS-check before counting as valid candidate
from src.dataset.connectivity import is_connected
PORT1, PORT2 = (7, 0), (7, 14)

# In gen_pixel_k(), gen_cfg_k(), gen_cvae_k():
valid_layouts = []
attempts = 0
while len(valid_layouts) < K and attempts < K * 5:
    lay = generate_one(seed=base_seed + attempts, ...)
    lay_clean = remove_floating_islands(lay, PORT1, PORT2)
    if is_connected(lay_clean):
        valid_layouts.append(lay_clean)
    attempts += 1
# If still short, pad with best available
```

**Verification:** Re-run on N=20 specs, confirm 0% failure rate.

**Impact on paper:** Changes "2.23% failure rate" to "0% failure rate (BFS connectivity gate)". Cleaner result.

---

### 8.2 Paired Test: PIXEL K=1 vs CFG K=1 (Phase 6 data, no new PBS)
**Priority: HIGH — uses existing data, 30 min analysis**

**Problem:** Phase 6 has PIXEL and CFG EM results for N=300 specs (best-of-1, but unpaired across methods). We need paired test: same N specs compared between methods.

**Implementation:** `src/evaluation/stats_phase6_paired.py`
```python
# Load em_verify_summary.json
# Extract matched (pixel_mse, cfg_mse) pairs by spec index
# Run Wilcoxon signed-rank on paired differences
# Report: W, p-value, r (rank-biserial), Hodges-Lehmann estimator ± 95% CI
# Expected: p < 0.05 (directional), small effect size
```

**Target output:**
- PIXEL K=1 vs CFG K=1: W=?, p=?, effect r=?, HL estimate ± CI
- Coverage McNemar: PIXEL 88.3% vs CFG 84.0%
- This becomes the "guidance effect at K=1" evidence in the paper

**Note:** If Phase 6 data is not paired (different spec indices per method), run N=100 fresh specs with both methods K=1 as a dedicated PBS job (~2h on workq).

---

### 8.3 Statistical Power: PIXEL vs cVAE, N=300 (1 PBS job, ~3h workq)
**Priority: HIGH — current power=0.05 is indefensible**

**Problem:** With N=100 pairs, the Wilcoxon test has power=0.05 for the observed effect (r=0.046). Need N≥300 to achieve power≥0.30; N≥500 for power≥0.50.

**Recommendation:** Target N=300 (feasible in 1 PBS job).

**Implementation:** New script `src/evaluation/phase8_repower.py`
- Generate 200 additional specs (exclude Phase 6+7 used indices: ~693 total)
- Methods: PIXEL K=1 + cVAE K=1 only (not all methods — saves 50% compute)
- 200 specs × 2 methods × 1 sim = 400 EM sims, ~2.5h at 22s/sim
- Combine with Phase 7 N=100 for total N=300

**PBS script:** `scripts/pbs_phase8_repower.pbs`
```
workq, select=1:ncpus=16:mem=32gb:ngpus=0, walltime=04:00:00, workers=8
```

**Expected power at N=300 (from post-hoc power analysis):**
- Effect r=0.046 → Cohen's d ≈ 0.092
- Two-tailed Wilcoxon, α=0.0167 (Bonferroni): power ≈ 0.22
- Not great, but defensible as "marginal". If p stays < 0.017 at N=300, report it; if not, revise framing.

**Alternative framing if p becomes NS at N=300:** Drop the statistical significance claim for PIXEL vs cVAE; instead report the ratio (2.62×) and bootstrap CI on the mean ratio. A 2.62× effect with a tight CI is still a strong claim even without p < α.

---

### 8.4 Ablation: No Physics Guidance (α=0) at K=1 — PBS Job ~2h
**Priority: CRITICAL — this is the "guidance works" ablation**

**Problem:** We currently have no experiment isolating guidance contribution at K=1. Phase 7 shows guidance doesn't separate from CFG at K=5, but we claim guidance helps at K=1. This needs a direct paired test.

**Protocol:**
- Same N=100 Phase 7 specs
- Generate: PIXEL K=1 (guided) vs PIXEL-NoGuidance K=1 (α=0, pure CFG)
- Both use same seeds: SEEDS_K5[0]=0 for PIXEL, SEEDS_K5[0]=0 for No-Guidance
- EM verify both (200 sims)
- Paired Wilcoxon + McNemar

**Expected result:**
- PIXEL > No-Guidance with small-to-medium effect
- p < 0.05 (provides statistical evidence for guidance at K=1)
- This is the ablation that justifies the guidance mechanism

**Implementation:** Add `--no-guidance` flag to `phase7_eval.py`, run Phase 7 eval on same spec_idx.npy, then EM verify.

**PBS script:** `scripts/pbs_phase8_guidance_ablation.pbs`
```
workq, select=1:ncpus=16:ngpus=1:mem=32gb, walltime=03:00:00
Generation + EM (100 specs × 2 methods × 1 sim = 200 sims)
```

---

### 8.5 Ablation: Guidance Strength Sweep (α_max)
**Priority: MEDIUM — validates hyperparameter choice; takes 1 PBS job**

**Protocol:**
- α_max ∈ {0.01, 0.10, 0.50, 1.00} (4 conditions)
- N=50 specs (fresh), K=1 per method, EM verify all
- 4 conditions × 50 specs × 1 sim = 200 EM sims, ~1.5h

**Expected result:**
- Inverted-U curve: too-small α has no effect; too-large α causes instability
- Current α=0.10 should be near the plateau/peak
- This shows the hyperparameter is robustly chosen

**PBS script:** `scripts/pbs_phase8_alpha_sweep.pbs`

---

### 8.6 Ablation: Uncertainty Weighting vs Fixed Step (σ̂=const)
**Priority: MEDIUM — validates novel mechanism**

**Protocol:**
- N=100 specs, K=1
- PIXEL (uncertainty-weighted α_t) vs PIXEL-FixedAlpha (α_t = α_max × η_t, no σ̂ term)
- 100 specs × 2 × 1 sim = 200 sims, ~1.5h
- Paired Wilcoxon

**Implementation:** Add `--no-uncertainty-weighting` flag to guided sampling.

**Expected result:** Small but consistent improvement from uncertainty weighting; validates the mechanism even if the effect is modest.

---

### 8.7 Topology Ablation with EM Verification
**Priority: MEDIUM — Phase 5 showed topology matters on surrogate; verify with EM**

**Protocol:**
- N=100 specs, K=1
- PIXEL (λ_topo=1.0) vs PIXEL-NoTopo (λ_topo=0.0) vs PIXEL-NoDRC (λ_mfg=0.0)
- 100 × 3 × 1 = 300 sims, ~2h
- Connectivity yield + DRC pass rate + EM MSE

**Expected result:** No-topo has worse connectivity yield; EM MSE slightly worse due to disconnected layouts. DRC-no ablation shows more isolated pixel artifacts.

---

### 8.8 Improved Effect Size Reporting
**Priority: HIGH — reviewers at AAAI will scrutinize statistical reporting**

**Add to `src/evaluation/stats_tests.py`:**

```python
from scipy.stats import wilcoxon
import numpy as np

def hodges_lehmann_estimator(x, y, ci_level=0.95):
    """Hodges-Lehmann estimator for location shift with bootstrap CI."""
    diffs = x - y
    pairwise = np.array([(a + b)/2 for i,a in enumerate(diffs) for b in diffs[i:]])
    hl = np.median(pairwise)
    # Bootstrap CI
    boot = np.array([np.median(np.random.choice(diffs, len(diffs), replace=True))
                     for _ in range(5000)])
    lo, hi = np.percentile(boot, [(1-ci_level)*50, 100 - (1-ci_level)*50])
    return hl, lo, hi

def mean_ratio_with_ci(mse_a, mse_b, n_boot=5000, ci=0.95):
    """Bootstrap CI on mean(mse_b) / mean(mse_a) ratio."""
    n = len(mse_a)
    ratios = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        ratios.append(mse_b[idx].mean() / mse_a[idx].mean())
    return np.mean(mse_a)/np.mean(mse_b), *np.percentile(ratios, [(1-ci)*50, 100-(1-ci)*50])
```

**Add to all comparisons in stats_report.txt:**
- Hodges-Lehmann shift ± 95% bootstrap CI
- Mean ratio ± 95% bootstrap CI
- Cohen's d (standardised)

---

### 8.9 Re-run Final Stats Report
**After 8.2–8.8, run consolidated stats:**

Full output table for paper:
| Comparison | N | Wilcoxon W | p | HL shift [95%CI] | Mean ratio [95%CI] | Cohen's d | Power |
|---|---|---|---|---|---|---|---|
| PIXEL vs CFG K=1 | 300 | ? | ? | ? | ? | ? | ? |
| PIXEL vs CFG K=5 | 100 | 375 | 0.319 n.s. | ? | ? | ? | 0.02 |
| PIXEL vs cVAE K=5 | 300 | ? | ? | ? | ? | ? | ? |
| PIXEL vs Det-CNN K=5 | 100 | 51 | 8.5e-10 | ? | 7.12× | ? | 0.62 |
| PIXEL vs No-Guidance K=1 | 100 | ? | ? | ? | ? | ? | ? |
| PIXEL diversity vs cVAE | 20 | ? | 1.2e-4 | ? | 3.3× | ? | — |

---

## PAPER STRUCTURE (Phase 8B, 2–3 weeks after stats locked)

### Paper Title (candidates):
1. "PIXEL: Physics-Guided Discrete Diffusion for Inverse Electromagnetic Layout Synthesis"
2. "Physics-Constrained Probabilistic Topology Synthesis for Inverse RF/IC Electromagnetic Design"
3. "Discrete Diffusion with Uncertainty-Weighted Physics Guidance for Inverse EM Design"

**Recommended title:** Option 1 — clean, identifies method (PIXEL), technique (discrete diffusion), and application.

### Claim Architecture (lock this before writing):

**Primary claim (strongest):** Discrete probabilistic synthesis achieves 7.12× lower EM-verified error than deterministic inverse design (p=8.5×10⁻¹⁰, well-powered).

**Secondary claim:** PIXEL generates 3.3× more diverse solutions than cVAE for the same specification (p=1.2×10⁻⁴) — demonstrating the fundamental advantage of the diffusion prior.

**Tertiary claim (pending ablation 8.4):** Physics guidance improves best-of-1 quality; at K=5, diversity naturally fills the guidance gap — consistent with a well-trained prior operating near the guidance saturation point.

**Novel mechanism claims:** Uncertainty-weighted guidance step size (pending ablation 8.6); differentiable topology constraint (pending ablation 8.7 EM verification).

### Figure List:
1. **Fig 1**: PIXEL framework diagram (pipeline: spec → encoder → D3PM → guidance → layout)
2. **Fig 2**: Example layouts × 4 methods with overlaid S-parameter comparison (target / surrogate / EM)
3. **Fig 3**: Main results bar chart (cov@0.001 for all methods, Phase 6 K=1 and Phase 7 K=5)
4. **Fig 4**: Ablation table / radar chart (connectivity, DRC, MSE, diversity)
5. **Fig 5**: Diversity visualization (K=20 layout samples for PIXEL vs cVAE for 2 hard specs)
6. **Fig 6** (supplementary): Guidance strength α_max sweep curve

### Table List:
1. **Table 1**: Dataset statistics summary
2. **Table 2**: Surrogate quality (MSE, gradient cosine, latency)
3. **Table 3**: Main evaluation (full stats table from 8.9)
4. **Table 4**: Ablation results

---

## PHASE 9: STREAMLIT DEMO
**See `app/pixel_demo.py`** — Live demo interface for PIXEL generation and EM verification.

**PBS/deployment:** Run interactively on workq node via SSH tunnel (see `scripts/launch_streamlit.sh`).

---

## PBS JOB SCHEDULE (ORDER OF OPERATIONS)

| # | Task | Script | Walltime | Priority | Deps |
|---|---|---|---|---|---|
| 1 | Paired PIXEL K=1 vs CFG K=1 (fresh N=100) | pbs_phase8_guidance_k1.pbs | 3h workq | CRITICAL | None |
| 2 | PIXEL vs cVAE N=300 repower | pbs_phase8_repower.pbs | 4h workq | HIGH | None |
| 3 | No-guidance ablation N=100 K=1 | pbs_phase8_guidance_ablation.pbs | 3h workq | CRITICAL | None |
| 4 | Guidance strength sweep α | pbs_phase8_alpha_sweep.pbs | 2h workq | MEDIUM | 3 |
| 5 | Uncertainty weighting ablation | pbs_phase8_unc_ablation.pbs | 3h workq | MEDIUM | 3 |
| 6 | Topology ablation with EM | pbs_phase8_topo_ablation.pbs | 2h workq | MEDIUM | None |
| 7 | Final stats consolidation | (interactive) | 30min | HIGH | 1–6 |
| 8 | Paper figures (matplotlib/plotly) | (interactive) | 2h | HIGH | 7 |
| Jobs 1–3 can run in parallel (no dependencies). Jobs 4–5 depend on 3 (need same layouts). |

**Estimated total time: 4–6 days (parallel PBS + 1 week writing)**

---

## STATISTICAL CHECKLIST (Pre-Submission)

Before finalising paper, verify every box:

### Comparisons
- [ ] All 3 Bonferroni comparisons (vs CFG K=5, vs Det-CNN K=5, vs cVAE K=5) reported with: W, p, r, HL ± CI, mean ratio ± CI, power
- [ ] PIXEL K=1 vs CFG K=1 formally tested (paired Wilcoxon, N≥100, same specs)
- [ ] PIXEL K=1 vs No-Guidance K=1 formally tested (paired Wilcoxon, N=100)
- [ ] Diversity comparison (Hamming) tested with Wilcoxon (already done, p=1.2e-4)
- [ ] Coverage McNemar tests for all comparisons (already done)
- [ ] Bootstrap CIs on all coverage percentages (already done in stats_tests.py)

### Sample Sizes and Power
- [ ] PIXEL vs Det-CNN: N=100, power=0.62 ✅
- [ ] PIXEL vs cVAE: N≥300, power≥0.22 (target)
- [ ] PIXEL vs CFG K=1: N≥100, power≥0.20 (expected)
- [ ] Ablations: N≥50 per condition (minimum)
- [ ] All power calculations reported in supplementary

### Reproducibility
- [ ] All random seeds fixed and logged (generation seeds in spec/experiments)
- [ ] All hyperparameters from base_config.yaml (versioned)
- [ ] PBS job IDs and completion times logged in progress log
- [ ] Git commit SHA for each experiment result

### Assumptions
- [ ] Wilcoxon normality assumption checked (report Shapiro-Wilk on differences)
- [ ] Paired assumption: confirm same N specs used per comparison
- [ ] Bonferroni correction correct: 3 primary comparisons → α=0.0167
- [ ] Independence: N=100 specs are drawn fresh (no overlap with training set confirmed)

### Physical Validity
- [ ] All reported EM MSE values are full-wave OpenEMS (not surrogate)
- [ ] Connectivity failure rate reported and addressed
- [ ] KK residual acknowledged in dataset description
- [ ] Phase accuracy limitations acknowledged in surrogate discussion

---

## RISK REGISTER FOR PHASE 8

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| PIXEL vs cVAE N=300 becomes NS | Medium | Medium | Report mean ratio ± CI instead; argue effect size is meaningful |
| Guidance ablation shows no effect | Low | High | Reframe: model is near guidance saturation; diversity is the claim |
| PBS scheduling delays | High | Low | Run jobs 1+2+3 in parallel; all are independent |
| AAAI deadline | Fixed | High | Paper writing starts in parallel with PBS jobs 4–6 |
| EM simulation failures | Low | Low | Checkpoint/resume in phase8 scripts; 0 failures in Phase 7 |

---

*Document version: 1.0 | Created: June 11, 2026 | Owner: Vatsal Khanna*
*Next update: After PBS jobs 1-3 complete*
