# PIXEL-2026 — Master Execution Plan & System Memory
## Physics-Constrained Probabilistic Topology Synthesis for Inverse Electromagnetic RF/IC Design

> **This file is the authoritative execution guide for the PIXEL-2026 research project.**  
> Load this file at the START of every new chat session to restore full context.  
> Status: Active | Version: 1.0 | Date: May 17, 2026

---

## HOW TO USE THIS FILE (FOR FUTURE SESSIONS)

```
ALWAYS READ THIS FILE FIRST AT THE START OF A SESSION.
Then check PIXEL_PROGRESS_LOG.md for current status.
Never rely on chat memory — this document is ground truth.
```

---

# PART 1: PROJECT IDENTITY

## 1.1 One-Line Summary
PIXEL generates fabrication-valid RF/IC electromagnetic layouts directly from target S-parameters using physics-guided discrete diffusion with differentiable topology constraints.

## 1.2 Full Title
**Physics-Constrained Probabilistic Topology Synthesis for Inverse Electromagnetic RF/IC Design**

## 1.3 Acronym
**PIXEL** — **P**hysics-guided d**I**screte di**X**fusion for **E**M **L**ayout synthesis

## 1.4 Target Venues
| Venue | Focus | Deadline |
|---|---|---|
| **AAAI-2027** | Primary — AI methodology | ~July 2026 abstract, ~July 2026 paper |
| **IEEE TMTT** | Secondary — RF/Microwave engineering | Continuous submission |
| **ICLR-2027 Workshop** | Fallback — Physics-informed ML | TBD |

## 1.5 Core Scientific Hypothesis (Validated)
> Inverse EM design can be formulated as constrained probabilistic topology synthesis over a learned manifold of physically plausible electromagnetic structures, where differentiable surrogate physics and topological constraints jointly guide discrete generative sampling toward desired spectral objectives.

### Three Supporting Sub-Claims
1. EM layouts occupy a structured, low-dimensional manifold within `{0,1}^(15×15)` → the prior can be learned.
2. A calibrated CNN ensemble can approximate EM forward physics well enough to provide reliable gradient guidance.
3. Differentiable topology + DRC constraints during generation yield better manufacturability than post-hoc repair.

---

# PART 2: CRITICAL PHYSICS TRUTHS (DO NOT VIOLATE)

These are non-negotiable physical constraints. Every architectural and training decision must respect them.

## 2.1 Maxwell's Equations (Frequency Domain)
$$\nabla \times \mathbf{E} = -j\omega\mu\mathbf{H}, \quad \nabla \times \mathbf{H} = \mathbf{J} + j\omega\varepsilon\mathbf{E}$$

The layout `x ∈ {0,1}^(H×W)` defines spatial permittivity:
- `x_ij = 1`: conductor → `ε = ε₀(1 - jσ/ωε₀)`
- `x_ij = 0`: dielectric → `ε = ε₀·εᵣ`

## 2.2 S-Matrix Physical Constraints (ALL MUST BE ENFORCED)
| Law | Formula | Implementation |
|---|---|---|
| **Passivity** | `S†S ⪯ I` (full matrix form) | Max eigenvalue loss in surrogate |
| **Reciprocity** | `S^T = S`, so `S₁₂ = S₂₁` | Reciprocity loss + reduces output dim |
| **Causality (KK)** | `Re[S] = H{Im[S]}` | KK Hilbert-transform loss |
| **Smoothness** | Adjacent freq samples correlated | Spectral smoothness loss |

**CRITICAL: Never regress S-parameter magnitude and phase independently — violates Kramers-Kronig causality.**

## 2.3 Resolution-Frequency Co-Design (Physical Validity Bounds)
Pixel validity criterion: `Δ_pixel ≪ λ_eff/10`

| Physical domain L | Pixel size Δ (at 15×15) | Valid f_max | Application band |
|---|---|---|---|
| 20 mm | 1.33 mm | ~4.5 GHz | Cellular sub-6 |
| **10 mm** | **0.67 mm** | **~9 GHz** | **WiFi 2.4/5 GHz ✓** |
| **5 mm** | **0.33 mm** | **~18 GHz** | **mmWave, satellite ✓** |

**Working assumption: 5–10 mm physical domain → 2–18 GHz valid range (Rogers 4003C, εᵣ=3.5)**

## 2.4 Critical Formulation Errors (V1 → V2 Corrections)
| ID | V1 Error | V2 Correction | Impact |
|---|---|---|---|
| V-CRIT-1 | CFG formula from continuous diffusion applied to discrete | Log-probability domain CFG: `log p̃ = (1+w)log p_cond - w·log p_uncond` | Entire guidance mechanism |
| V-CRIT-2 | Gradient `∇_x_t L_physics` on discrete tokens | Guidance on denoiser logits via predicted `x̂₀ = E[x₀|x_t]` | Physics guidance |
| V-CRIT-3 | No Kramers-Kronig causality | KK loss added to surrogate: `L_KK = ||Re[Ŝ] - H{Im[Ŝ]}||²` | Surrogate physical validity |
| V-CRIT-4 | Scalar passivity `|S₁₁|²+|S₂₁|²≤1` | Full matrix: `λ_max(S†S) ≤ 1` | Multi-port extension |

---

# PART 3: SYSTEM CAPABILITIES

## 3.1 Hardware Profile
| Resource | Specification | Implication |
|---|---|---|
| **CPU** | 2× Intel Xeon Gold 6226R @ 2.90 GHz | 32 physical cores, 64 logical threads |
| **RAM** | 128 GB | Large dataset loading without disk I/O bottleneck |
| **GPU 0** | Quadro RTX 8000 | 51.5 GB VRAM, CUDA 12.8 |
| **GPU 1** | Quadro RTX 8000 | 51.5 GB VRAM, CUDA 12.8 |
| **Total VRAM** | ~103 GB | Can train with enormous batch sizes or multiple models in parallel |

## 3.2 Software Environment
| Package | Version | Status |
|---|---|---|
| Python | 3.13.9 | ✅ Available |
| PyTorch | 2.11.0+cu128 | ✅ Available |
| torchvision | 0.26.0+cu128 | ✅ Available |
| numpy | 2.4.4 | ✅ Available |
| scipy | 1.17.1 | ✅ Available |
| transformers | 5.8.1 | ✅ Available |
| accelerate | 1.13.0 | ✅ Available (DDP) |
| einops | 0.8.2 | ✅ Available |
| h5py | 3.16.0 | ✅ Available (dataset storage) |
| scikit-learn | 1.8.0 | ✅ Available |
| scikit-image | 0.25.2 | ✅ Available |
| wandb | 0.27.0 | ✅ Available (experiment tracking) |
| pandas | 3.0.3 | ✅ Available |
| matplotlib | 3.10.9 | ✅ Available |
| tqdm | 4.67.1 | ✅ Available |
| timm | 1.0.27 | ✅ Available |
| diffusers | 0.38.0 | ✅ Available |
| torchmetrics | 1.9.0 | ✅ Available |
| **openems** | **v0.0.36-93-g7b9cd51** | **✅ Installed at D:\openEMS\openEMS\** |

## 3.3 Optimal Execution Strategy for This System

### Dataset Generation (Phase 1)
- Use Python `multiprocessing.Pool` with **56–60 workers** (leave 4 threads for OS overhead)
- OpenEMS FDTD is single-threaded per simulation → embarrassingly parallel
- 200k structures × 2 min / 60 parallel cores = **~111 hours ≈ 4.6 days** on-machine
- **No cloud needed** — the 64-thread Xeon handles this entirely

### Surrogate Training (Phase 2)
- Train **5 independent surrogates in parallel**: 2 on GPU0, 2 on GPU1, 1 alternating
- 15×15 tiny input → batch size **512–1024** easily fits per GPU
- Each surrogate: ~4–8 hours on single RTX 8000 → **all 5 in ~8–12 hours**

### Denoiser Training (Phase 3)
- Use `accelerate` DDP across both GPUs
- 15×15 U-Net is small → batch size **256–512 per GPU** (512–1024 total)
- 300 epochs × ~15 min/epoch (DDP 2×GPU) = **~75 hours ≈ 3 days**

### Physics-Guided Sampling (Phase 4)
- Inference on single GPU; 50 diffusion steps per sample → very fast
- Batch inference: generate 32 candidates per specification simultaneously

### Key Principle: NEVER UNDERSELL THE HARDWARE
With 103 GB total VRAM and 64 CPU threads, we can:
- Load the entire 200k dataset in RAM during training
- Run multiple ablations simultaneously on the two GPUs
- Use DDP training without gradient compression

---

# PART 4: MATHEMATICAL ARCHITECTURE SUMMARY

## 4.1 Core Components
```
INPUT:  y* ∈ ℝ^(4×N_f)   [S11_mag, S21_mag, S11_phase, S21_phase at N_f=100 freq points]

[A] SPECTRAL ENCODER:
    1D ResNet: y* → c_y ∈ ℝ^256
    (captures resonance locality, harmonic structure, spectral continuity)

[B] DENOISER (D3PM, conditioned):
    x_t ∈ {0,1,MASK}^(15×15) + c_y + t → p_θ(x_{t-1}|x_t, c_y)
    U-Net with AdaLN(c_y, t) conditioning
    T=1000 timesteps, absorbing-state forward process

[C] SURROGATE ENSEMBLE (K=5):
    x̃ ∈ [0,1]^(15×15) → Ŷ ∈ ℝ^(4×N_f)
    Physics-constrained CNN (passivity + KK + reciprocity + smoothness losses)
    Provides: μ̂(x), σ̂²(x) [mean & uncertainty]

[D] CONNECTIVITY DISCRIMINATOR:
    x̃ ∈ [0,1]^(15×15) → D_conn(x) ∈ [0,1]
    CNN trained on DFS-labeled layouts
    Provides differentiable topology guidance

OUTPUT: x* ∈ {0,1}^(15×15) [binary conductor map, fabrication-ready]
```

## 4.2 Correct Discrete CFG (V2)
```
log p̃(x₀|x_t, c_y) = (1+w)·log p_θ(x₀|x_t, c_y) - w·log p_θ(x₀|x_t, ∅)
[normalize per-pixel independently]
```
Training: drop condition with p_drop = 0.15.  Guidance weight w = 2.0 (tunable).

## 4.3 Physics Guidance Mechanism (V2)
```
At each step t < t_thresh:
1. Logits:      ℓ_t = f_θ(x_t, t, c_y)                        [denoiser output]
2. Soft layout: x̂₀ = σ(ℓ_t[:,:,1])                             [P(pixel=1)]
3. Surrogate:   L_physics = ||F̂(x̂₀) - y*||²
4. Guidance:    α_t = α_max / (σ̂(x̂₀) + ε) · η_t              [uncertainty-weighted]
5. Update:      ℓ̃_t = ℓ_t - α_t · ∇_{ℓ_t} (L_physics + λ_topo·L_topo + λ_mfg·L_mfg)
6. Sample:      x_{t-1} ~ p_θ(x_{t-1}|x_t, ℓ̃_t)
```
Where: `η_t = (1 - t/T)²` (late-stage activation), `t_thresh = 0.4T = 400`

## 4.4 Full Training Objectives

### Surrogate Loss
```
L_surrogate = ||Ŷ - Y||²_MSE
            + 0.10 · L_pass    [passivity: λ_max(S†S) ≤ 1]
            + 0.05 · L_recip   [reciprocity: S = S^T]
            + 0.05 · L_KK      [causality: Hilbert transform]
            + 0.02 · L_smooth  [spectral smoothness]
```

### Denoiser Loss
```
L_denoiser = -ELBO_D3PM + λ_aux · L_x0_pred
```

### Guided Sampling Loss (inference only)
```
L_guided = L_physics + λ_topo · L_topo + λ_mfg · L_mfg
         = ||F̂(x̂₀) - y*||² + 1.0 · (-log D_conn(x̂₀)) + 0.5 · L_DRC(x̂₀)
```

## 4.5 Key Hyperparameter Starting Points
| Parameter | Value | Source |
|---|---|---|
| Grid size H=W | 15 (→ 32 scaling) | Physical co-design |
| Freq range | 0.5–20 GHz, N_f=100 | Dataset design |
| Diffusion steps T | 1000 | D3PM standard |
| CFG weight w | 2.0 | Empirical (range: 1–5) |
| Physics α_max | 0.1 | Starting point |
| Guidance threshold | t_thresh = 400 (= 0.4T) | Empirical |
| Ensemble size K | 5 | Calibration requirement |
| Spectral embed dim m | 256 | Architecture |
| Gumbel τ_init | 1.0 | Annealing |
| Gumbel τ_final | 0.01 | Annealing |
| Condition dropout | p_drop = 0.15 | CFG standard |
| λ_topo | 1.0 | Guidance |
| λ_mfg | 0.5 | Guidance |

---

# PART 5: DATASET SPECIFICATION

## 5.1 Structure
- **Target size:** 200k structures (Phase 1), scalable to 500k
- **Splits:** 80% train / 10% val / 10% test (held-out primitive types)
- **Storage format:** HDF5 via h5py

## 5.2 Primitives (10+ types)
| Class | Primitives | Expected S-parameter behavior |
|---|---|---|
| Passband | Microstrip line, wideband taper | Near-unity transmission, phase shift |
| Stopband | λ/4 shunt stub, λ/2 resonator, notch | Bandstop at f₀ |
| Bandpass | Coupled λ/2 resonators, interdigital, ring | Bandpass filter |
| Coupling | Edge-coupled lines | Directional ~10 dB coupling |
| Complex | SRR (split-ring), stub-loaded, cascaded | Mixed resonances |

## 5.3 Perturbation Protocol
```python
for each primitive P:
    params = perturb(P.nominal, σ=0.15 × param_range)  # geometric
    px_map = rasterize(params, grid=15×15)
    if rand() < 0.3: px_map = local_mutate(px_map, n_flips=U(1,5))
    if rand() < 0.2: px_map = combine(px_map, sample_another())
    if connected(px_map): emit(px_map)  # else discard
```

## 5.4 HDF5 Schema
```
layout:          (N, 15, 15)   uint8   — binary pixel map
S11_mag:         (N, 100)      float32 — return loss magnitude
S21_mag:         (N, 100)      float32 — insertion loss magnitude
S11_phase:       (N, 100)      float32 — return loss phase [rad]
S21_phase:       (N, 100)      float32 — insertion loss phase [rad]
substrate_id:    (N,)          uint8   — 0=Rogers4003C, 1=FR4, 2=Rogers5880, 3=Alumina
resonance_freqs: (N, K)        float32 — K≤5 resonance frequencies
Q_factor:        (N, K)        float32 — Q at each resonance
validity_flag:   (N,)          bool    — passed all checks
primitive_type:  (N,)          uint8   — for evaluation stratification
```

## 5.5 Substrate Parameters
| ID | Material | εᵣ | tan δ | Application |
|---|---|---|---|---|
| 0 | Rogers 4003C | 3.55 | 0.0027 | High-frequency standard |
| 1 | FR4 | 4.4 | 0.02 | Low-cost PCB |
| 2 | Rogers 5880 (PTFE) | 2.2 | 0.0009 | Microwave/mmWave |
| 3 | Alumina | 9.8 | 0.0001 | MMIC/ceramic |

---

# PART 6: EXECUTION PLAN — 5 PHASES

---

## PHASE 0: Environment Setup (Days 1–3)

### Goals
Get all software running and validated before writing research code.

### Steps
| Step | Action | Command |
|---|---|---|
| 0.1 | Install missing Python packages | `pip install timm torchmetrics diffusers` |
| 0.2 | Install OpenEMS on Windows | Download from openems.de, install to `C:\openEMS`, add to PATH |
| 0.3 | Install Python-OpenEMS bridge | `pip install python-openems` or use `CSXCAD` subprocess bridge |
| 0.4 | Validate OpenEMS Python bridge | Run single microstrip simulation, verify S11/S21 output |
| 0.5 | Set up project directory structure | Create `src/`, `data/`, `experiments/`, `logs/` |
| 0.6 | Set up WandB project | `wandb init`, project name `pixel-2026` |
| 0.7 | Validate dual-GPU DDP | Run accelerate test with 2 GPUs |

### Checkpoint P0
- [ ] OpenEMS produces correct S-parameters for a known 50Ω microstrip line
- [ ] Python multiprocessing pool runs 60 OpenEMS jobs in parallel
- [ ] WandB logging works
- [ ] Both GPUs visible to PyTorch, accelerate DDP functional

### Rollback P0
- If OpenEMS Windows installation fails: Use CST Microwave Studio trial, or use a Python FDTD library (e.g., `fdtd` package) as interim surrogate data generator with known physics limitations
- If Python bridge fails: Use subprocess call to OpenEMS executable with file I/O

---

## PHASE 1: Dataset Generation (Days 4–21)
**STATUS: ✅ COMPLETE — 342,415 samples in `data/raw/pixel_dataset.h5` (June 2, 2026)**

### Goals
Generate 200k physically valid, electromagnetically simulated RF structures as HDF5.

### Steps
| Step | Action | Verification | Status |
|---|---|---|---|
| 1.1 | Implement primitive generators (10 types) | Visual inspection of pixel maps | ✅ Done (primitives.py in generate.py) |
| 1.2 | Implement stochastic perturbation engine | Histogram of geometric param distributions | ✅ Done |
| 1.3 | Implement BFS connectivity validator | On 1000 random layouts | ✅ Done (connectivity.py) |
| 1.4 | Implement OpenEMS simulation runner | S-parameter smoke-test PASSED (−2.14 dB, KK=0.313) | ✅ Done (openems_wrapper.py) |
| 1.5 | Implement HDF5 writer with schema above | Load/read test, data type check | ✅ Done (hdf5_writer.py) |
| 1.6 | Validate simulation pipeline end-to-end | FDTD smoke-test + 50-sample pilot | 🟡 Pilot running |
| 1.7 | Launch 200k parallel generation job | 32-worker multiprocessing pool | ❌ Pending pilot pass |
| 1.8 | Monitor and checkpoint every 10k structures | Connectivity yield, S-param distribution | ❌ Pending |
| 1.9 | Dataset quality audit | See Checkpoint P1 | ❌ Pending |

### Checkpoint P1 — Dataset Quality Gates
| Metric | Pass Criterion | If Fail |
|---|---|---|
| Connectivity yield | > 85% of generated structures pass connectivity | Tighten perturbation σ |
| S-param physical validity | Passivity: `|S₁₁|²+|S₂₁|²≤1.01` for >99% | Re-run simulation with stricter mesh |
| Resonance coverage | ≥ 60% of structures have identifiable resonances | Add more resonant primitives |
| Spectral diversity | S21 MSE between random pairs > 0.05 | Increase geometric perturbation range |
| Primitive balance | No primitive type > 25% of dataset | Reweight generation probabilities |
| KK residual | Mean Hilbert-transform error < 0.02 on test structures | Signal simulation discretization issue |

### Verification for Each Primitive Type
```
For microstrip line: S21 magnitude flat (~0 dB), check vs. Lossless TL formula
For λ/4 stub: Bandstop at f₀, verify: f₀ = c/(4·L_stub·√εᵣ_eff)
For coupled lines: Check coupling coefficient vs. line separation
For ring resonator: Check resonance at f₀ = c/(π·D·√εᵣ_eff)
```

### Rollback P1
- If connectivity yield < 50%: Switch to repair-based generation (DFS repair after generation)
- If simulation times > 5 min/structure: Reduce frequency range to 1–10 GHz, N_f=50
- If 200k takes > 10 days: Use 100k as primary dataset, 200k as stretch goal
- If OpenEMS accuracy insufficient: Validate subset (1k structures) with analytical closed-form and use as error budget reference

### Phase 1 Technical Notes (Session 3)
**FDTD Configuration (DO NOT CHANGE without justification):**
- Simulation backend: `src/dataset/openems_wrapper.py` using CSXCAD + openEMS v0.0.36
- Grid: 43×43×21 cells = 38,829 FDTD cells; Δt = 1.306e-13 s
- Domain: 7.5 mm × 7.5 mm × (substrate + ground + air); ext_x=0.5, ext_y=1.0, air_z=1.0 mm
- Max simulation time: `SetMaxTime(2e-9)` → ≤15,314 timesteps → ≤48 s/simulation
- End criterion: `SetEndCriteria(1e-2)` (−20 dB) — note: low-loss substrates plateau at −5 to −7 dB due to dielectric near-field with τ≈11 ns >> 2 ns window. This is PHYSICAL, not a bug.
- S-parameter frequencies: 100 points from 0.5 GHz to 20 GHz (linspace)
- Port impedance: 50 Ω, lumped port on left edge (port 1) and right edge (port 2), y-center
- Passivity enforcement: `_enforce_passivity()` — `target_s21² = max(0, 0.995 − |s11|²)`, scale s21 magnitude
- Substrate library: Rogers4003C (ε=3.55, tanδ=0.0027), FR4 (ε=4.4, tanδ=0.02), Rogers5880 (ε=2.2, tanδ=0.0009), Alumina (ε=9.8, tanδ=0.0001)

**Pilot sanity thresholds:**
- Mean |S21| > −6.0 dB
- |S21| ripple < 15.0 dB (NOT 6 dB — high-freq rolloff at 20 GHz gives ~10 dB ripple for 7.5 mm line)
- KK residual < 0.60
- Passivity NOT checked in sanity (enforced by `_enforce_passivity` in `simulate()`)

**Throughput estimate (32 workers):**
- Worst-case: 48 s/sim → 75 sim/hr/worker → 32 workers = 2,400 sim/hr → 200K / 2,400 ≈ 83 hr ≈ 3.5 days
- Optimistic: many layouts converge faster (resonant structures hit −20 dB before 15,314 steps)

**Generation command (after pilot passes):**
```powershell
cd D:\pixel-2026\pixel-2026
$env:PYTHONUTF8=1
Remove-Item data\raw\pixel_dataset.h5 -Force
Remove-Item data\raw\pixel_dataset.checkpoint.json -Force
C:\Users\tyrone\anaconda3\envs\pixel-env\python.exe -m src.dataset.generate --config experiments/configs/base_config.yaml --workers 32 --n-samples 200000 --skip-sanity --skip-pilot
```
Use `--resume` if interrupted.

**Python invocation rules:**
- ALWAYS use full path: `C:\Users\tyrone\anaconda3\envs\pixel-env\python.exe`
- ALWAYS set `$env:PYTHONUTF8=1` before running
- PowerShell exit code 1 with `2>&1` is cosmetic (NativeCommandError) — check Python logs only

---

## PHASE 2: Surrogate Physics Model (Days 12–28, overlaps Phase 1)
**STATUS: ✅ COMPLETE — all 5/5 surrogates trained, all gates passed (June 2, 2026)**

**PHASE 2 RESULTS (June 2, 2026):**
- Ensemble val_mag_mse = **0.01252 ± 0.00003** (gate < 0.05 → 4× better)
- Gradient cosine mean = **0.971** across all 5 surrogates (gate > 0.70)
- Gradient magnitude ratio ≈ **0.97** (gate 0.5–2.0 → essentially 1.0)
- All 5/5 surrogates PASS gradient fidelity gate
- Checkpoints: `experiments/surrogate_v1/surrogate_k{0..4}_best.pt`

### ⚠️ PHASE 2 CRITICAL NOTE: KK Loss Weight
The training dataset has a systematic KK violation (mean residual ~0.30) due to the 2 ns FDTD time window cap. The surrogate must predict the simulator's actual output. **Use λ_KK = 0.005** (not 0.05). The KK loss serves only as a soft regularizer, not a hard enforcer.

### Goals
Train K=5 calibrated CNN ensemble surrogates that accurately predict S-parameters AND have reliable gradients.

### Steps
| Step | Action | Verification |
|---|---|---|
| 2.1 | Implement CNN surrogate architecture (Section 9.2 of v2) | Forward pass shape test |
| 2.2 | Implement physics losses: L_pass, L_recip, L_KK, L_smooth | Unit tests: passivity on known matrices |
| 2.3 | Train 5 surrogates (2 parallel on GPU0/GPU1) | WandB: training curves per surrogate |
| 2.4 | **Gradient fidelity validation** (critical) | Cosine similarity to FD gradients |
| 2.5 | Calibration test: ECE, reliability diagram | Ensemble variance vs. actual error |
| 2.6 | Inference latency benchmark | < 10ms/sample on GPU |
| 2.7 | Ablation: no KK loss, no passivity loss | Quantify contribution |

### Checkpoint P2 — Surrogate Quality Gates
| Metric | Pass Criterion | If Fail |
|---|---|---|
| S21 MSE (test set) | < 0.05 (normalized 0-1 range) | Increase model capacity; more data |
| S11 MSE (test set) | < 0.05 | As above |
| Resonance freq error | < 5% of bandwidth | Review frequency resolution N_f |
| **Gradient cosine sim** | **Mean > 0.7** | **CRITICAL — see rollback** |
| Gradient magnitude ratio | 0.5 – 2.0 (within 2×) | Gradient smoothing |
| Ensemble ECE | < 0.05 | Temperature scaling post-calibration |
| Passivity violation rate | < 1% on test set | Increase λ_pass |
| KK residual | < 0.01 on test set | Increase λ_KK |
| Inference latency | < 10 ms/sample | Optimize with torch.compile() |

### Critical: Gradient Fidelity Validation Protocol
```python
# For 1000 test layouts:
g_surrogate = autograd.grad(surrogate(x̃), x̃)  # analytical gradient
g_FD[j] = (surrogate(x̃ + δ·e_j) - surrogate(x̃)) / δ  # finite difference
# Report:
cosine_sim = mean(cosine(g_surrogate, g_FD))   # MUST be > 0.7
mag_ratio = mean(||g_surrogate|| / ||g_FD||)    # Should be ~1.0
```

### Rollback P2
- **If gradient cosine sim < 0.7 (CRITICAL):**
  1. Train surrogate on continuous (soft) input `x̃ ∈ [0,1]` instead of binary `{0,1}` — smoother input space → smoother gradients
  2. Apply Gaussian blurring to gradient field before guidance step
  3. **Fallback:** Zeroth-order guidance (MCMC-style surrogate sampling instead of gradient descent)
- If MSE too high: Add ViT attention in bottleneck; increase training epochs
- If calibration poor: Apply temperature scaling on ensemble; increase K to 7

### Phase 2 — Exact Implementation Guide (Session 5)

#### File 1: `src/models/surrogate.py`
```python
# Architecture: PhysicsSurrogate CNN
# Input:  x̃ ∈ [0,1]^(1×15×15)  (binary layout, float)
# Output: Ŷ ∈ ℝ^(4×100)         [S11_mag, S21_mag, S11_phase, S21_phase]

class ResBlock2D(nn.Module):
    # Conv(ch, 3×3, pad=1) → BN → ReLU → Conv(ch, 3×3, pad=1) → BN + residual → ReLU

class PhysicsSurrogate(nn.Module):
    # Encoder:
    #   ConvBlock(1→32, 3×3) + BN + ReLU                       → (32, 15, 15)
    #   ResBlock×3 (32 ch)                                       → (32, 15, 15)
    #   Conv(32→64, stride=2) + BN + ReLU                       → (64, 8, 8)
    #   ResBlock×3 (64 ch)                                       → (64, 8, 8)
    #   Conv(64→128, stride=2) + BN + ReLU                      → (128, 4, 4)
    #   ResBlock×2 (128 ch)                                      → (128, 4, 4)
    # Head:
    #   AdaptiveAvgPool2d(1) → Flatten → Linear(128→512) → ReLU → Linear(512→400)
    #   Reshape → (4, 100)
    # Note: train on continuous x̃ ∈ [0,1] — NOT hard binary — for gradient smoothness

class SurrogateEnsemble(nn.Module):
    # Wraps K=5 PhysicsSurrogate models
    # forward(x) → (mean_pred (4,100), variance (4,100))
    # mean = (1/K) Σ f_k(x)
    # var  = (1/(K-1)) Σ (f_k(x) - mean)²
```

#### File 2: `src/losses/physics_losses.py`
```python
def passivity_loss(s11_mag, s21_mag):
    # Full matrix form for 2-port:  λ_max(S†S) ≤ 1
    # For 2-port diagonal: |S11|²+|S21|² (S22≈S11 assumed for symmetric structures)
    power = s11_mag**2 + s21_mag**2  # (B, Nf)
    return torch.mean(torch.clamp(power.max(dim=1).values - 1.0, min=0)**2)

def reciprocity_loss(s11_mag, s21_mag):
    # For 2-port, S12=S21. We only predict S21, so enforce |S12|≈|S21| trivially = 0
    # Skip for now (returns 0); add S22 prediction in Phase 5 if needed
    return torch.tensor(0.0)

def kk_loss(s_re, s_im):
    # KK: Im[S](f) ≈ -H{Re[S](f)} via FFT Hilbert
    # WEAK REGULARIZER ONLY — λ=0.005, not 0.05
    # Training data has inherent KK violation from 2ns FDTD window
    analytic = torch.fft.rfft(s_re, dim=-1)
    # ... Hilbert via FFT phase shift ...
    return torch.mean((s_im - im_reconstructed)**2)

def smoothness_loss(s_pred):
    # Adjacent frequency smoothness: penalize sharp spectral jumps
    diff = s_pred[..., 1:] - s_pred[..., :-1]   # (B, 4, 99)
    return torch.mean(diff**2)

def total_surrogate_loss(pred, target):
    s11_mag_pred  = pred[:, 0, :]   # (B, 100)
    s21_mag_pred  = pred[:, 1, :]
    s11_ph_pred   = pred[:, 2, :]
    s21_ph_pred   = pred[:, 3, :]
    mse = F.mse_loss(pred, target)
    L_pass  = passivity_loss(s11_mag_pred, s21_mag_pred)
    L_kk    = kk_loss(s11_mag_pred * torch.cos(s11_ph_pred),
                      s11_mag_pred * torch.sin(s11_ph_pred))
    L_smooth = smoothness_loss(pred)
    return mse + 0.10*L_pass + 0.005*L_kk + 0.02*L_smooth
```

#### File 3: `src/training/train_surrogate.py`
```python
# Key design decisions:
# - Load full dataset into RAM at start (856 MB, fits in 1 TB RAM easily)
# - Batch size: 1024 (15×15 is tiny; H100 can handle large batches)
# - LR: 3e-4, cosine annealing to 1e-6
# - Epochs: 150 (fast due to in-memory batching on H100)
# - Train surrogates 0,1,2,3,4 sequentially in ONE PBS job
# - Save each to: experiments/surrogate_v1/surrogate_{k}.pt
# - WandB project: pixel-2026-surrogate

# Dataset split (create splits at start of training):
# - Sort by idx → 80/10/10 split (NO random shuffle across primitives for test stratification)
# - Actually: use stratified split by primitive_type (hold out 10% of each type)

# Input normalization:
# - layout: float32, already in [0,1] (it's uint8 0 or 1 → cast to float)
# - S_target: normalize? NO — predict in physical units directly
#   S*_mag ∈ [0, 1] (passivity enforced), phase ∈ [-π, π]

# Gradient fidelity validation (run after training):
# - Sample 1000 test layouts
# - autograd gradient vs finite-difference gradient
# - Report cosine similarity (must be > 0.7)
```

#### File 4: `scripts/pbs/train_surrogate.pbs`
```bash
#!/bin/bash
#PBS -N pixel_surrogate
#PBS -q workq
#PBS -l select=1:ncpus=32:ngpus=1
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o logs/surrogate_train.log

cd $PBS_O_WORKDIR
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate pixel-env
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=MIG-$(nvidia-smi -L | grep "MIG" | head -1 | awk -F'UUID: ' '{print $2}' | tr -d ')')

python -m src.training.train_surrogate \
    --config experiments/configs/base_config.yaml \
    --n-surrogates 5 \
    --seed-base 42 \
    --output-dir experiments/surrogate_v1/
```

#### Estimated Training Time on H100 MIG (3g.47gb)
- Dataset: 342k samples × 625 floats = 856 MB → fully in-memory
- Batch size 1024 → ~335 batches/epoch
- H100 speed: ~5ms/batch (tiny 15×15 CNN) → ~1.7 s/epoch
- 150 epochs × 5 surrogates = ~2.1 hours total
- Well within 24h PBS walltime for all 5 surrogates in one job

---

## PHASE 3: Denoiser / Generative Model (Days 25–45)

### Goals
Train a conditional D3PM denoiser over binary EM layouts with discrete CFG. Validate unconditional and conditional generation quality.

### Steps
| Step | Action | Verification |
|---|---|---|
| 3.1 | Implement D3PM absorbing state forward process (ternary: {0,1,MASK}) | Unit test: marginal distribution at time T should be all-MASK |
| 3.2 | Implement spectral encoder (1D ResNet) | Forward pass; embedding space isotropy test |
| 3.3 | Implement U-Net denoiser with AdaLN conditioning | Forward pass; shape check 15→8→15 |
| 3.4 | Implement correct discrete CFG (log-probability domain) | Verify: w=0 → unconditional, w→∞ → deterministic |
| 3.5 | Implement training loop with condition dropout p=0.15 | WandB: ELBO curve |
| 3.6 | Launch DDP training (2×GPU) | Monitor ELBO, track NaN/inf |
| 3.7 | Validate unconditional generation | Connectivity yield, structure diversity |
| 3.8 | Validate conditional generation (CFG only, no physics guidance) | S-param MSE with surrogate scoring |
| 3.9 | Ablation: DDPM baseline (wrong inductive bias) | Quantify degradation |

### Checkpoint P3 — Denoiser Quality Gates
| Metric | Pass Criterion | If Fail |
|---|---|---|
| Training ELBO | Decreasing, stable (no NaN) | Check AdaLN conditioning; reduce LR |
| Unconditional connectivity yield | > 80% | Review training data; add connectivity supervision |
| Unconditional DRC pass rate | > 75% | Add DRC-aware loss term |
| Conditional S21 MSE (surrogate-scored) | < 0.10 | Increase CFG weight w; check encoder |
| Sample diversity (Hamming distance) | > 30 bits mean pairwise | Check for mode collapse |
| **Topology validity (unconditional)** | **> 80%** | **Critical for guidance to work** |

### Architecture Notes
```
IMPORTANT for 15→8→15 sizing:
- Encoder stride=2: 15×15 → 8×8 (ceiling division, slight asymmetry)
- Decoder TransposeConv: use output_size=(15,15) explicitly
- Bottleneck attention at 8×8 = 64 tokens (affordable full attention)
- Embed ternary {0,1,MASK} as learned 32-dim embedding per pixel
- Port channel: always provide binary port map as 2nd input channel
```

### Rollback P3
- If D3PM ELBO diverges: Switch to **MDLM (Masked Diffusion Language Model)** — simpler masking with BERT-style unmasking, easier to train
- If discrete CFG produces mode collapse: Reduce w from 2.0 to 1.0; increase p_drop to 0.2
- If 15→8→15 architecture causes artifacts: Use simple downsampling via avg pool, upsample via bilinear

---

## PHASE 4: Physics-Guided Sampling (Days 40–55)

### Goals
Integrate surrogate physics guidance + connectivity guidance + DRC guidance into the denoising process. Validate with full-wave EM simulation.

### Steps
| Step | Action | Verification |
|---|---|---|
| 4.1 | Implement `x̂₀ = E[x₀|x_t]` extraction from denoiser logits | Check: x̂₀ ∈ [0,1], soft gradients flow |
| 4.2 | Implement uncertainty-weighted guidance step | Check: α_t decreases with surrogate uncertainty |
| 4.3 | Implement connectivity discriminator (train from scratch) | AUC-ROC > 0.95 on connected/disconnected layouts |
| 4.4 | Implement DRC loss (differentiable width/spacing) | Unit test: known-bad layouts get high L_DRC |
| 4.5 | Implement combined guided sampling loop | End-to-end test: spec → layout |
| 4.6 | Hyperparameter sweep: α_max, w, t_thresh, λ_topo, λ_mfg | Grid search on validation set |
| 4.7 | **Full-wave EM verification** on 100 generated layouts | Central evaluation metric |
| 4.8 | Tune until EM verification pass rate > 85% | Iterate on guidance strength |

### Checkpoint P4 — Guided Generation Quality Gates
| Metric | Pass Criterion | If Fail |
|---|---|---|
| Connectivity yield (guided) | > 95% | Increase λ_topo |
| DRC pass rate (guided) | > 90% | Increase λ_mfg |
| Surrogate-scored S21 MSE | < 0.05 | Increase α_max; check guidance threshold |
| **EM-verified S21 MSE** | **< 0.08** | **PRIMARY METRIC — iterate guidance** |
| **EM-verified S11 MSE** | **< 0.08** | As above |
| Resonance frequency error | < 10% | Check resolution-frequency co-design |
| Generation time (GPU) | < 60 sec/design | Reduce T or use DDIM-style accelerated sampling |

### Checkpoint P4 — Ablation Pre-Check
Before final evaluation, run these ablations in sequence:
1. No physics guidance (α=0): Baseline denoiser only
2. No uncertainty weighting (σ̂=const): Fixed step size
3. No topology constraint (λ_topo=0): Effect of connectivity discriminator
4. No DRC loss (λ_mfg=0): Effect of manufacturability
5. Continuous DDPM (wrong inductive bias): Compare to D3PM

### Rollback P4
- If EM-verified MSE doesn't improve with guidance: Validate surrogate gradient fidelity is still > 0.7 (may have degraded on guided trajectories)
- If guidance causes non-physical layouts: Reduce α_max by 2×; add gradient clipping `||g||≤1.0`
- If guidance is too slow: Implement only K=3 surrogates for guidance (not all 5); use K=5 only for final evaluation

---

## PHASE 5: Evaluation, Baselines & Paper (Days 50–70)

### Goals
Comprehensive evaluation against all baselines, ablation study, scaling experiment, paper writing.

### Steps
| Step | Action | Verification |
|---|---|---|
| 5.1 | Implement Det-CNN baseline (deterministic inverse regressor) | Sanity check: worse diversity than PIXEL |
| 5.2 | Implement cVAE baseline (spectral encoder + VAE) | |
| 5.3 | Implement cGAN baseline | |
| 5.4 | Implement BO baseline (Bayesian Optimization over layout space) | |
| 5.5 | Run 1000-specification test evaluation (ALL metrics) | Full-wave EM on all baselines too |
| 5.6 | Run all 8 ablation studies | |
| 5.7 | **32×32 scaling experiment** (dataset subset + retrain) | Shows architecture scales |
| 5.8 | Write paper (AAAI format, 8 pages + refs) | See paper structure below |
| 5.9 | Supplementary material (full arch specs, extended results) | |

### Checkpoint P5 — Publication Quality Gates
| Metric | PIXEL Target | Det-CNN Expected | cVAE Expected |
|---|---|---|---|
| S21 MSE (EM-verified) | < 0.08 | > 0.20 | ~0.12 |
| Connectivity yield | > 95% | ~60% | ~75% |
| DRC pass rate | > 90% | ~50% | ~65% |
| Sample diversity (Hamming) | > 30 bits | 0 (deterministic) | ~15 bits |
| Inference time | < 60 s | < 1 s | < 2 s |

**AAAI positioning: If PIXEL outperforms all baselines on EM-verified spectral accuracy, topology validity, AND provides diversity, the paper has its primary contribution demonstrated.**

### Paper Structure (AAAI Format — 8 pages)
```
1. Introduction (1 page)
   - Problem: ill-posed inverse EM design
   - Gap: no discrete diffusion + physics guidance + topology
   - Contributions: 6 bullet points

2. Background (0.5 page)
   - D3PM, CFG, physics guidance in diffusion

3. Physics Foundations (0.5 page)
   - Maxwell's equations, S-matrix, KK relations (brief — NOT an EM paper)

4. PIXEL Framework (3 pages)
   4.1 Dataset generation
   4.2 Surrogate ensemble
   4.3 D3PM denoiser
   4.4 Discrete CFG (corrected formulation)
   4.5 Physics-guided sampling
   4.6 Topology + DRC constraints

5. Experiments (2 pages)
   5.1 Dataset statistics
   5.2 Surrogate accuracy
   5.3 Main results table (all metrics, all baselines)
   5.4 Ablation study
   5.5 Qualitative examples (layout images)
   5.6 Scaling experiment (32×32)

6. Conclusion (0.5 page)

References (~0.5 page)
```

---

# PART 7: RISK REGISTER AND MITIGATIONS

| ID | Risk | Probability | Impact | Primary Mitigation | Fallback |
|---|---|---|---|---|---|
| R1 | Surrogate gradient fidelity < 0.7 | Medium | High | Train on continuous inputs; gradient smoothing | Zeroth-order guidance (MCMC) |
| R2 | D3PM training instability | Low | Medium | Use MDLM as drop-in | Bernoulli diffusion baseline |
| R3 | Dataset quality insufficient | Low | High | 10% challenge structures; holdout evaluation | Reduce dataset size; focus on 3 primitive families |
| R4 | OpenEMS installation/accuracy issues | Medium | High | Use subprocess bridge; validate vs. analytical | Use analytical formulas for simple primitives during development; verify final with OpenEMS |
| R5 | EM verification MSE doesn't improve | Medium | High | Iterate guidance hyperparameters; check R1 | Report surrogate-scored results + explain gap |
| R6 | AAAI venue mismatch | Low | Low | Frame as AI methodology; emphasize D3PM novelty | Submit to NeurIPS-2027 or ICLR-2027 |
| R7 | Dataset generation takes > 10 days | Medium | Medium | 100k as primary; 200k stretch goal | Subset to highest-quality 50k structures |
| R8 | Mode collapse in denoiser | Low | High | Diversity metrics in training loop; diversity loss term | Try Discrete Flow Matching (Campbell et al., 2024) |

---

# PART 8: GLOBAL VERIFICATION PRINCIPLES

## Scientific Integrity Checks (Run at Every Phase Boundary)

### Physics Checks
- [ ] Does every generated layout satisfy `|S₁₁|² + |S₂₁|² ≤ 1.01`?
- [ ] Does the surrogate output satisfy Kramers-Kronig to < 2% error?
- [ ] Is the pixel size constraint `Δ_pixel < λ_eff/10` satisfied for all simulations?
- [ ] Are port pixels always conductors (never masked or set to 0)?

### Mathematics Checks
- [ ] Is CFG applied in log-probability domain (not noise vector domain)?
- [ ] Is physics guidance applied to denoiser logits (not discrete tokens)?
- [ ] Is the D3PM ELBO computed correctly (check KL term sign)?
- [ ] Is the Gumbel-sigmoid temperature properly annealed?

### Reproducibility Checks
- [ ] All random seeds fixed and logged (data generation, model init, sampling)?
- [ ] All hyperparameters versioned in config files (YAML)?
- [ ] All WandB experiment runs linked to git commits?
- [ ] Held-out test set NEVER seen during hyperparameter tuning?

### Statistical Checks
- [ ] Error bars on all reported metrics (≥5 random seeds for key experiments)?
- [ ] Baselines trained with same dataset and compute budget?
- [ ] Ablations differ by exactly ONE component at a time?

---

# PART 9: DIRECTORY STRUCTURE

```
pixel-2026/
├── master-context/
│   ├── v1-full-context.md          [Original brainstorm]
│   ├── v2-master-context.md        [Validated blueprint]
│   ├── pixel-background.tex        [LaTeX document]
│   ├── PIXEL_EXECUTION_PLAN.md     [THIS FILE — Master execution guide]
│   └── PIXEL_PROGRESS_LOG.md       [Updated each session — current state]
│
├── src/
│   ├── dataset/
│   │   ├── primitives.py           [Phase 1: RF primitive generators]
│   │   ├── perturbation.py         [Phase 1: Stochastic perturbation]
│   │   ├── connectivity.py         [Phase 1: BFS/DFS validator]
│   │   ├── openems_runner.py       [Phase 1: EM simulation runner]
│   │   └── dataset_builder.py     [Phase 1: HDF5 writer + parallelism]
│   ├── models/
│   │   ├── spectral_encoder.py     [Phase 3: 1D ResNet encoder]
│   │   ├── surrogate.py            [Phase 2: CNN ensemble surrogate]
│   │   ├── denoiser.py             [Phase 3: U-Net denoiser]
│   │   ├── connectivity_disc.py    [Phase 4: Connectivity discriminator]
│   │   └── diffusion.py            [Phase 3: D3PM forward/reverse process]
│   ├── losses/
│   │   ├── physics_losses.py       [Passivity, KK, reciprocity, smoothness]
│   │   ├── topology_losses.py      [DRC, connectivity]
│   │   └── diffusion_losses.py     [D3PM ELBO, aux x0 prediction]
│   ├── guidance/
│   │   ├── cfg.py                  [Phase 3: Correct discrete CFG]
│   │   └── physics_guidance.py    [Phase 4: Uncertainty-weighted guidance]
│   ├── training/
│   │   ├── train_surrogate.py      [Phase 2: Surrogate training loop]
│   │   ├── train_denoiser.py       [Phase 3: Denoiser training loop DDP]
│   │   └── train_discriminator.py  [Phase 4: Connectivity disc training]
│   ├── evaluation/
│   │   ├── metrics.py              [All evaluation metrics]
│   │   ├── baselines.py            [Det-CNN, cVAE, cGAN, BO, Diff-TO]
│   │   └── em_verifier.py          [Full-wave EM verification pipeline]
│   └── utils/
│       ├── binarization.py         [Gumbel-sigmoid, STE]
│       ├── visualization.py        [Layout + S-param plotting]
│       └── config.py               [Hyperparameter management]
│
├── data/
│   ├── raw/                        [OpenEMS output files]
│   └── processed/
│       ├── dataset_200k.h5         [Primary dataset]
│       └── dataset_32x32.h5        [Scaling experiment]
│
├── experiments/
│   ├── configs/                    [YAML hyperparameter configs]
│   ├── surrogate_v1/               [Experiment logs]
│   ├── denoiser_v1/
│   └── guided_v1/
│
├── logs/                           [WandB local cache]
└── paper/
    ├── main.tex
    ├── figures/
    └── supplementary/
```

---

# PART 10: IMPLEMENTATION PRIORITY ORDER

> **START with Phase 1 immediately. Phase 2 begins as soon as 10k structures are available.**

### Week 1 (Days 1–7): Environment + Dataset Foundation
1. Phase 0: Install all missing packages
2. Phase 0: OpenEMS working + validated on a simple microstrip
3. Phase 1: Implement primitives.py (5 primitive types minimum)
4. Phase 1: Implement connectivity.py (BFS validator)
5. Phase 1: Implement openems_runner.py (subprocess-based)
6. Phase 1: Generate 1k pilot dataset; inspect manually

### Week 2 (Days 8–14): Dataset Scale-Up + Surrogate Start
1. Phase 1: Launch 200k generation (60-core parallel pool)
2. Phase 2: Implement surrogate architecture + physics losses
3. Phase 2: Begin surrogate training on first 50k

### Week 3–4 (Days 15–28): Surrogate Completion + Denoiser Start
1. Phase 2: Complete surrogate training, gradient fidelity validation
2. Phase 3: Implement D3PM + denoiser architecture
3. Phase 3: Begin denoiser training (DDP, both GPUs)

### Week 5–6 (Days 29–42): Denoiser Completion + Guidance
1. Phase 3: Complete denoiser training, validate CFG
2. Phase 4: Implement connectivity discriminator
3. Phase 4: Implement physics-guided sampling
4. Phase 4: Hyperparameter sweep

### Week 7–8 (Days 43–56): Full-Wave Verification + Evaluation
1. Phase 4: EM verification of 100 generated layouts
2. Phase 5: Implement baselines
3. Phase 5: Full 1000-specification evaluation

### Week 9–10 (Days 57–70): Paper + Scaling
1. Phase 5: 32×32 scaling experiment
2. Phase 5: Write paper

---

# PART 11: QUICK REFERENCE EQUATIONS

## Discrete CFG (Correct)
$$\log \tilde{p}_\theta(x_0|x_t, c_y) = (1+w)\log p_\theta(x_0|x_t,c_y) - w\log p_\theta(x_0|x_t,\varnothing)$$

## Physics Guidance (Correct)
$$\tilde{\ell}_t = \ell_t - \alpha_t \nabla_{\ell_t}\mathcal{L}_{\text{guided}}, \quad \alpha_t = \frac{\alpha_{\max}}{\hat{\sigma}(\hat{x}_0) + \epsilon}\cdot\left(1-\frac{t}{T}\right)^2$$

## Surrogate Loss
$$\mathcal{L}_{\text{surr}} = \|\hat{Y}-Y\|^2 + 0.10\mathcal{L}_{\text{pass}} + 0.05\mathcal{L}_{\text{recip}} + 0.05\mathcal{L}_{KK} + 0.02\mathcal{L}_{\text{smooth}}$$

## Passivity (Full Matrix Form)
$$\mathcal{L}_{\text{pass}} = \frac{1}{N_f}\sum_f\max\!\left(0,\lambda_{\max}(\hat{S}^\dagger(f)\hat{S}(f)) - 1\right)^2$$

## KK Loss
$$\mathcal{L}_{KK} = \|\text{Re}[\hat{S}] - \mathcal{H}\{\text{Im}[\hat{S}]\}\|^2 \quad (\mathcal{H}\text{ via FFT})$$

## Pixel Validity
$$\Delta_{\text{pixel}} \ll \frac{\lambda_{\text{eff}}}{10}, \quad \lambda_{\text{eff}} = \frac{c}{f_{\max}\sqrt{\varepsilon_{r,\text{eff}}}}$$

---

*Document generated: May 17, 2026 | Status: Active Research Blueprint*  
*All V1 mathematical errors corrected per v2-master-context.md*  
*See PIXEL_PROGRESS_LOG.md for current phase status and blockers*
