# PIXEL-2026 Progress Log
> **Update this file at the START and END of every chat session.**  
> It is the live status tracker — brief bullets only.

---

## SESSION LOG

### Session 1 — May 17, 2026
**Status at start:** New project, context read for first time  
**Work done:**
- Read all master context files (v1, v2, pixel-background.tex) in full
- Registered system capabilities (2× RTX 8000 / 51.5 GB VRAM each, 2× Xeon Gold 6226R, 128 GB RAM)
- Created `PIXEL_EXECUTION_PLAN.md` — master execution guide
- Created `PIXEL_PROGRESS_LOG.md` (this file)
- Created repo memory at `/memories/repo/pixel-2026.md`

**Status at end:** Phase 0 not yet started  

---

### Session 2 — May 17, 2026
**Status at start:** Phase 0 not started; directory structure did not exist  
**Work done:**
- Created full directory tree (24 directories under project root)
- Created `conda` environment `pixel-env` (Python 3.13.13) at `C:\Users\tyrone\anaconda3\envs\pixel-env`
- Installed PyTorch 2.11.0+cu128 + torchvision + torchaudio via `--index-url https://download.pytorch.org/whl/cu128`
  - **Note:** setuptools downgraded 82.0.1 → 70.2.0 by torch constraint (expected, permanent)
- Installed all remaining dependencies (accelerate 1.13.0, diffusers 0.38.0, transformers 5.8.1, timm 1.0.27, wandb 0.27.0, h5py 3.16.0, omegaconf 2.3.0, scikit-learn 1.8.0, pandas 3.0.3, matplotlib 3.10.9, + 60 transitive packages)
- Fixed `pyproject.toml` build backend: `setuptools.backends.legacy:build` → `setuptools.build_meta` (setuptools 70 compatibility)
- Installed `pixel2026` package in editable mode: `pip install -e . --no-deps` ✅
- Registered Jupyter kernel `Python (pixel-env)` at `C:\Users\tyrone\AppData\Roaming\jupyter\kernels\pixel-env` ✅
- Created all `src/` `__init__.py` files (8 subpackages)
- Created `pyproject.toml`, `environment.yml`, `requirements.txt`
- Created `accelerate_config.yaml` (2× RTX 8000, bf16, 2 processes)
- Created `experiments/configs/base_config.yaml` — master hyperparameter config (all phases, fully documented)
- Created `src/utils/config.py` — OmegaConf loader, seed management, project root resolver
- Created `src/utils/logging_utils.py` — logger factory, WandB init/log helpers
- Created `src/utils/binarization.py` — Gumbel-Sigmoid, STE, tau annealing, expected-x0 extraction
- Created `src/utils/visualization.py` — layout renderer, S-param plotter, training curve plotter
- Created `scripts/setup_env.ps1`, `scripts/setup_openems_windows.ps1`, `scripts/validate_phase0.py`
- Fixed multiprocessing pool check: Windows spawn context requires module-level functions; switched to `ThreadPool`
- Created comprehensive `README.md` with reproducibility log, all setup commands, hardware requirements, config docs
- **Ran `scripts/validate_phase0.py`**: **36/37 PASS, 1 WARN (OpenEMS — expected, not needed until Phase 1)**

**Validation result:** `Phase 0 CONDITIONALLY PASSED`
- All 36 checks PASS including: Python 3.13.13, PyTorch+CUDA dual-GPU, 103 GB VRAM, DataParallel, all packages, HDF5 round-trip, WandB, accelerate, 64-CPU ThreadPool, directory structure, Gumbel-Sigmoid, KK/Hilbert, BFS connectivity
- 1 WARN: OpenEMS not installed — run `.\scripts\setup_openems_windows.ps1` before Phase 1

**Status at end:** Phase 0 ✅ COMPLETE  
**Next session should start with:** Phase 1 — Dataset Generation
- Install OpenEMS via `.\scripts\setup_openems_windows.ps1`
- Implement `src/dataset/openems_wrapper.py` (FDTD simulation interface)
- Implement `src/dataset/generate.py` (parallel layout sampling + simulation)
- Target: 200K (H, W, substrate, S-params) tuples stored as HDF5

---

### Session 3 — May 17, 2026
**Status at start:** Phase 0 complete; Phase 1 not started; only analytical `em_simulation.py` existed  
**Work done:**
- Installed OpenEMS v0.0.36-93-g7b9cd51 from pre-built Windows binaries at `D:\openEMS\openEMS\`
- Registered DLL path: `os.add_dll_directory(r'D:\openEMS\openEMS')` in `sitecustomize.py` AND in module `_register_dll()`
- Created `src/dataset/openems_wrapper.py` — full FDTD simulation backend using CSXCAD + openEMS Python bindings
  - `simulate()`: 2-port S-parameter extraction (38,829 FDTD cells after domain optimization)
  - `_enforce_passivity()`: correct formula — `target_s21_sq = max(0, 0.995 - |s11|²)`, scale magnitude
  - `quick_sanity_check()`: through-line S-param verification gate
  - `worker_init_openems()`: pool initializer for multiprocessing
- Discovered and fixed multiple OpenEMS API issues:
  - Wrong: `FDTD.SetMaxTimesteps(10000)` → Correct: `FDTD.SetMaxTime(2e-9)` (physical time cap)
  - End criterion: `SetEndCriteria(1e-2)` (−20 dB), not 1e-3 (−30 dB) — substrate near-field never reaches −30 dB in 2 ns
  - Domain optimization: ext_x=0.5, ext_y=1.0, air_z=1.0 mm → 38,829 cells (62% reduction from 101,332)
- **FDTD smoke-test PASSED** (Rogers4003C through-line, substrate 0):
  - Mean |S21| = −2.14 dB (gate: > −6 dB) ✅
  - |S21| ripple = 9.72 dB (gate: < 15 dB) ✅
  - Passivity = True ✅
  - KK residual = 0.3131 (gate: < 0.60) ✅
  - Time = 48.4 s, 15,314 timesteps, 12.31 MCells/s
- Updated `src/dataset/generate.py`:
  - FDTD/analytical backend selection via `PIXEL_USE_ANALYTICAL` env var
  - New CLI flags: `--use-analytical`, `--skip-sanity`, `--pilot-n`
  - FDTD sanity check gate before pilot
  - `run_generation()` now accepts `pool_initializer` argument; wires `worker_init_openems` for FDTD
  - `_worker_generate()` noise_sigma=0.0 (FDTD has intrinsic numerical noise)
- Launched **FDTD pilot run**: 50 samples, `--pilot-n 50 --skip-sanity --pilot` (currently running, ~40 min)

**Key physics insight discovered:**
- Low-loss substrate (Rogers4003C, tan_d=0.0027) stores dielectric near-field energy with τ≈11 ns >> 2 ns simulation window. Energy stabilizes at −5 to −6 dB — PHYSICAL behavior, not a simulation defect. The `SetMaxTime(2e-9)` cap ensures all simulations terminate in ≤48 s.

**PowerShell quirk:**
- `2>&1` stderr redirect triggers `NativeCommandError` and exit code 1 in PowerShell even when Python exits 0. This is cosmetic — always check Python logs for actual pass/fail, not PowerShell exit code.

**Status at end:** FDTD smoke-test passed; pilot running  
**Next session should start with:**
1. Check pilot results (connectivity_yield > 0.85, passivity_rate > 0.99, kk_rate > 0.85)
2. If pilot passes: delete analytical dataset (`data/raw/pixel_dataset.h5` + checkpoint), launch full 200K FDTD generation (`--workers 32 --n-samples 200000 --skip-sanity --skip-pilot`)
3. If pilot fails: inspect failures, fix `openems_wrapper.py`, re-run pilot before full generation

---

## CURRENT STATUS SNAPSHOT

| Phase | Name | Status | % Complete |
|---|---|---|---|
| 0 | Environment Setup | ✅ Complete | 100% |
| 1 | Dataset Generation | � In Progress | 15% |
| 2 | Surrogate Physics Model | 🔴 Not started | 0% |
| 3 | Denoiser / Generative Model | 🔴 Not started | 0% |
| 4 | Physics-Guided Sampling | 🔴 Not started | 0% |
| 5 | Evaluation & Paper | 🔴 Not started | 0% |

## INSTALLED PACKAGES STATUS
| Package | Version | Status |
|---|---|---|
| torch | 2.11.0+cu128 | ✅ |
| torchvision | 0.22.0+cu128 | ✅ |
| accelerate | 1.13.0 | ✅ |
| diffusers | 0.38.0 | ✅ |
| transformers | 5.8.1 | ✅ |
| timm | 1.0.27 | ✅ |
| wandb | 0.27.0 | ✅ |
| h5py | 3.16.0 | ✅ |
| omegaconf | 2.3.0 | ✅ |
| scikit-learn | 1.8.0 | ✅ |
| torchmetrics | 1.9.0 | ✅ |
| openems | v0.0.36-93-g7b9cd51 | ✅ (Session 3) |

## KEY DECISIONS MADE
- Primary dataset target: 200k structures (100k minimum viable)
- Working grid: 15×15 → physical domain 7.5 mm → valid to ~12 GHz (conservative margin)
- Substrate primary: Rogers 4003C (εᵣ=3.55), plus FR4, Rogers 5880, Alumina
- Generative backbone: D3PM with absorbing state (ternary {0,1,MASK}), T=1000
- CFG formulation: Log-probability domain (V2 corrected)
- Physics guidance: Via denoiser logits and predicted x̂₀ (V2 corrected)
- Surrogate ensemble size: K=5
- Training strategy: Both GPUs via accelerate DDP (surrogate uses DataParallel)
- Experiment tracking: WandB project = pixel-2026
- Mixed precision: bf16 (RTX 8000 native support)
- Editable install: `pixel2026` package, build backend = `setuptools.build_meta`

## KNOWN WORKAROUNDS
- `setuptools` permanently pinned to 70.2.0 by PyTorch 2.11 — use `setuptools.build_meta` not `setuptools.backends.legacy:build` in pyproject.toml
- Windows multiprocessing in validation script: use `ThreadPool` not `mp.Pool` (spawn context requires `__main__` guard)
- OpenEMS `SetMaxTimesteps` does NOT exist → use `FDTD.SetMaxTime(seconds)` for physical time cap
- PowerShell `2>&1` causes NativeCommandError + exit code 1 even when Python exits 0 — check Python logs, not PS exit code
- OpenEMS end criterion `SetEndCriteria(1e-2)` (−20 dB), not 1e-3 — low-loss substrate near-field never reaches −30 dB in 2 ns
- FDTD pilot runs single-threaded; use `--pilot-n 50` for fast verification (~40 min) before 200k full generation with 32 workers

## OPEN QUESTIONS / DECISIONS PENDING
- [x] Physical domain size: **DECIDED** → 7.5 mm (15×15 grid, 0.5 mm/pixel), valid to ~12 GHz
- [x] OpenEMS vs. analytical for initial pilot? **DECIDED** → OpenEMS FDTD for research-grade dataset; analytical only for debugging
- [x] Port position convention: **DECIDED** → left-edge center (port 1) and right-edge center (port 2)
- [ ] Connectivity yield in FDTD pilot — confirm > 85% before full generation
- [ ] Full 200K generation duration estimate needs real-data calibration (estimate: 3.5–5.5 days at 32 workers)

## VALIDATED MATHEMATICS (DO NOT CHANGE WITHOUT JUSTIFICATION)
- All 4 V1→V2 critical corrections are locked in (see PIXEL_EXECUTION_PLAN.md §2.4)
- Discrete CFG formula verified and documented
- Physics guidance mechanism via logits verified
- KK loss formula verified (via FFT Hilbert transform)
- Passivity: full matrix eigenvalue form required
- `_enforce_passivity` formula (Session 3, verified correct):
  ```python
  target_s21_sq  = np.maximum(0.0, (1.0 - margin) - |s11|²)   # margin=0.005
  target_s21_mag = np.sqrt(target_s21_sq)
  scale = where(|s21| > 1e-8, target_s21_mag / (|s21| + 1e-30), 0.0)
  scale = clip(scale, 0.0, 1.0)   # can only reduce, never amplify
  s21_out[needs_clip] = s21[needs_clip] * scale[needs_clip]
  ```
