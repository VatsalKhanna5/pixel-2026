# PIXEL-2026

**Physics-guided dIscrete diXusion for EM Layout synthesis**

Inverse-design framework for RF/IC passive structures using discrete denoising diffusion (D3PM) with physics-guided sampling.  Target venue: **AAAI-2027** (primary), **IEEE TMTT** (concurrent journal).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Research Objectives](#research-objectives)
3. [Repository Structure](#repository-structure)
4. [Environment Setup](#environment-setup)
5. [Reproducibility Log](#reproducibility-log)
6. [Hardware Requirements](#hardware-requirements)
7. [Configuration System](#configuration-system)
8. [Experiment Phases](#experiment-phases)
9. [Running Experiments](#running-experiments)
10. [Evaluation](#evaluation)
11. [Known Constraints & Workarounds](#known-constraints--workarounds)

---

## Project Overview

PIXEL formulates passive RF/IC layout generation as **conditional discrete denoising diffusion** on a 15×15 binary conductor-placement grid.  Given a target S-parameter response (S11, S21 over 100 frequency points from 0.5–20 GHz), PIXEL generates physically-realisable metal layouts that satisfy:

- **Spectral fidelity** — surrogate-predicted S-parameters match target within prescribed tolerance
- **Connectivity** — conductor path must form a connected graph from port 1 to port 2
- **DRC compliance** — minimum line width, spacing, and clearance constraints
- **Physical passivity** — |S11|² + |S21|² ≤ 1 at all frequencies
- **Kramers–Kronig consistency** — real and imaginary parts of impedance satisfy causality

The physics constraints are enforced during diffusion sampling via analytic gradient guidance applied to the D3PM log-probability, without any re-training of the denoiser.

---

## Research Objectives

| Objective | Target Metric |
|---|---|
| Connectivity yield | ≥ 95% of generated layouts |
| DRC pass rate | ≥ 90% of generated layouts |
| Surrogate \|S21\| MSE (test) | ≤ 0.08 |
| Surrogate \|S11\| MSE (test) | ≤ 0.08 |
| Full-wave EM verification (200 layouts) | \|S21\| MSE ≤ 0.15 |
| Gradient fidelity cosine similarity | ≥ 0.70 |

---

## Repository Structure

```
pixel-2026/
├── README.md                       ← this file
├── pyproject.toml                  ← package build config (editable install)
├── requirements.txt                ← pip requirements with version pins
├── environment.yml                 ← full conda environment spec
├── accelerate_config.yaml          ← HuggingFace Accelerate DDP config (2× RTX 8000)
│
├── src/                            ← installable Python package (pixel2026)
│   ├── dataset/                    ← EM layout dataset generation & loading
│   ├── models/                     ← spectral encoder, surrogate, D3PM denoiser
│   ├── losses/                     ← physics-constrained loss functions
│   ├── guidance/                   ← physics-guided sampling operators
│   ├── training/                   ← training loops (surrogate + denoiser)
│   ├── evaluation/                 ← metrics, full-wave EM verification
│   └── utils/
│       ├── config.py               ← OmegaConf config loader + seed utilities
│       ├── logging_utils.py        ← logger factory + WandB helpers
│       ├── binarization.py         ← Gumbel-Sigmoid, STE, tau annealing
│       └── visualization.py        ← layout / S-param / training-curve plots
│
├── experiments/
│   ├── configs/
│   │   └── base_config.yaml        ← master hyperparameter config (all phases)
│   └── notebooks/                  ← exploratory analysis notebooks
│
├── scripts/
│   ├── setup_env.ps1               ← one-command environment bootstrap
│   ├── setup_openems_windows.ps1   ← OpenEMS FDTD solver installation
│   ├── validate_phase0.py          ← 17-check Phase 0 checkpoint script
│   └── install_editable.ps1        ← standalone editable install helper
│
├── data/
│   ├── raw/                        ← OpenEMS simulation outputs (HDF5)
│   └── processed/                  ← normalised train/val/test splits
│
├── checkpoints/                    ← model checkpoints (git-ignored)
├── logs/                           ← training logs, install logs
├── tests/                          ← pytest unit tests
└── master-context/
    ├── PIXEL_EXECUTION_PLAN.md     ← authoritative research execution guide
    ├── PIXEL_PROGRESS_LOG.md       ← live phase status tracker
    ├── v2-master-context.md        ← project specification
    └── pixel-background.tex        ← LaTeX background / theory document
```

---

## Environment Setup

### Prerequisites

- Windows 10/11 or Linux
- [Anaconda or Miniconda](https://docs.anaconda.com/miniconda/) installed
- CUDA 12.8 compatible GPU driver (≥ 560.x on Windows)
- At least 40 GB VRAM total (project targets 2× Quadro RTX 8000)

### Option A — Automated (recommended)

```powershell
# From the project root:
.\scripts\setup_env.ps1
```

This script:
1. Creates `conda` environment `pixel-env` with Python 3.13
2. Installs PyTorch 2.x with CUDA 12.8 wheels
3. Installs all `requirements.txt` dependencies
4. Runs the editable install (`pip install -e . --no-deps`)
5. Registers the Jupyter kernel as `Python (pixel-env)`
6. Runs `scripts/validate_phase0.py` to confirm everything works

### Option B — Manual (step-by-step)

```powershell
# 1. Create environment
conda create -n pixel-env python=3.13 -y
conda activate pixel-env

# 2. Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Editable install of this package
pip install -e . --no-deps

# 5. Register Jupyter kernel
python -m ipykernel install --user --name pixel-env --display-name "Python (pixel-env)"

# 6. Validate
python scripts/validate_phase0.py
```

### Option C — Reproduce exact environment from spec

```powershell
conda env create -f environment.yml
conda activate pixel-env
pip install -e . --no-deps
python scripts/validate_phase0.py
```

---

## Reproducibility Log

All commands executed during environment setup are preserved here for full reproducibility.

### Phase 0 — Environment Setup (Session 2)

**Date:** 2025-07-16
**Host:** workstation (2× Quadro RTX 8000, 2× Xeon Gold 6226R, 128 GB RAM)
**conda version:** 25.11.1
**OS:** Windows 11

```powershell
# Step 1: Create conda environment
conda create -n pixel-env python=3.13 -y
# → python=3.13.13 at C:\Users\tyrone\anaconda3\envs\pixel-env

# Step 2: Install PyTorch 2.x with CUDA 12.8
conda run -n pixel-env pip install torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu128
# → torch==2.11.0+cu128, torchvision==0.22.0+cu128, torchaudio==2.11.0+cu128
# NOTE: setuptools downgraded from 82.0.1 → 70.2.0 (torch 2.11 requirement)

# Step 3: Install all remaining dependencies
conda run -n pixel-env pip install `
    torchmetrics timm diffusers transformers accelerate datasets `
    numpy scipy scikit-learn scikit-image h5py pandas einops `
    matplotlib seaborn tqdm wandb rich pyyaml omegaconf `
    notebook ipykernel ipywidgets `
    pytest pytest-cov black isort flake8 `
    psutil joblib pynvml
# → Exit code 0. Key versions:
#   accelerate==1.13.0, diffusers==0.38.0, transformers==5.8.1
#   timm==1.0.27, wandb==0.27.0, omegaconf==2.3.0, h5py==3.16.0
#   scikit-learn==1.8.0, pandas==3.0.3, matplotlib==3.10.9

# Step 4: Fix pyproject.toml build backend
# setuptools 70.2.0 lacks setuptools.backends.legacy:build (added in setuptools 72)
# → changed build-backend to "setuptools.build_meta"

# Step 5: Editable install
conda run -n pixel-env pip install -e "D:\pixel-2026\pixel-2026" --no-deps
# → Successfully installed pixel2026-0.1.0

# Step 6: Register Jupyter kernel
conda run -n pixel-env python -m ipykernel install `
    --user --name pixel-env --display-name "Python (pixel-env)"
# → C:\Users\tyrone\AppData\Roaming\jupyter\kernels\pixel-env
```

**Environment fingerprint** (run `conda run -n pixel-env pip list` to verify):

| Package | Version |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.22.0+cu128 |
| accelerate | 1.13.0 |
| diffusers | 0.38.0 |
| transformers | 5.8.1 |
| timm | 1.0.27 |
| wandb | 0.27.0 |
| h5py | 3.16.0 |
| omegaconf | 2.3.0 |
| scikit-learn | 1.8.0 |
| Python | 3.13.13 |

---

## Hardware Requirements

| Resource | Minimum | This Project |
|---|---|---|
| GPU VRAM | 40 GB total | 2× 51.5 GB = 103 GB |
| GPU count | 1 | 2 (DDP training) |
| System RAM | 64 GB | 128 GB |
| CPU cores | 16 | 64 threads (2× Xeon Gold 6226R) |
| Disk (data) | 200 GB | — |
| CUDA | 11.8+ | 12.8 |

### GPU Parallelism Strategy

- **Surrogate training:** `torch.nn.DataParallel` across GPU 0 and GPU 1 (batch=512)
- **Denoiser training:** `accelerate` DDP with `accelerate_config.yaml` (per-GPU batch=256, effective=512)
- **Sampling / guidance:** single-GPU (guidance gradients are sequential per step)

---

## Configuration System

All hyperparameters are centralised in `experiments/configs/base_config.yaml` and loaded via OmegaConf:

```python
from src.utils.config import load_config, configure_from_cfg

cfg = load_config("experiments/configs/base_config.yaml")
configure_from_cfg(cfg)   # sets seeds, cudnn flags

# Override individual values from CLI:
cfg = load_config("experiments/configs/base_config.yaml",
                  overrides=["denoiser.lr=5e-5", "compute.mixed_precision=fp16"])
```

Configs are saved alongside every checkpoint for exact reproducibility:

```python
from src.utils.config import save_config
save_config(cfg, "checkpoints/run_001/config.yaml")
```

---

## Experiment Phases

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Environment setup, validation | ✅ Complete |
| **Phase 1** | Dataset generation (200K layouts via OpenEMS) | 🔴 Not started |
| **Phase 2** | Forward surrogate training (physics-constrained CNN) | 🔴 Not started |
| **Phase 3** | D3PM denoiser training (CFG, absorbing-state) | 🔴 Not started |
| **Phase 4** | Physics-guided sampling + evaluation | 🔴 Not started |
| **Phase 5** | Ablations, baselines, paper preparation | 🔴 Not started |

See `master-context/PIXEL_EXECUTION_PLAN.md` for the full per-phase checklist, rollback procedures, and checkpoint criteria.

---

## Running Experiments

### Validate Phase 0

```powershell
conda activate pixel-env
python scripts/validate_phase0.py
```

### Dataset Generation (Phase 1)

```powershell
# Install OpenEMS first (Windows):
.\scripts\setup_openems_windows.ps1

# Then generate layouts (uses 56 CPU workers):
conda activate pixel-env
python src/dataset/generate.py --config experiments/configs/base_config.yaml
```

### Surrogate Training (Phase 2)

```powershell
conda activate pixel-env
python src/training/train_surrogate.py \
    --config experiments/configs/base_config.yaml
```

### Denoiser Training (Phase 3)

```powershell
# DDP across 2 GPUs via accelerate:
accelerate launch --config_file accelerate_config.yaml \
    src/training/train_denoiser.py \
    --config experiments/configs/base_config.yaml
```

### Sampling with Physics Guidance (Phase 4)

```powershell
conda activate pixel-env
python src/guidance/sample.py \
    --config experiments/configs/base_config.yaml \
    --checkpoint checkpoints/denoiser/best.pt \
    --n-samples 1000
```

---

## Evaluation

```powershell
conda activate pixel-env
python src/evaluation/evaluate.py \
    --config experiments/configs/base_config.yaml \
    --samples experiments/results/samples.h5
```

Produces:
- Connectivity yield and DRC pass rate
- Surrogate MSE (S11, S21) vs. target
- Optional full-wave EM verification via OpenEMS (subset of 200 layouts)
- WandB summary table

---

## Known Constraints & Workarounds

| Constraint | Root cause | Workaround |
|---|---|---|
| `setuptools` pinned to 70.2.0 | PyTorch 2.11 requires `setuptools<82` | Use `setuptools.build_meta` backend in `pyproject.toml` (not `setuptools.backends.legacy`) |
| OpenEMS not in conda/pip | Windows binary distribution only | Use `scripts/setup_openems_windows.ps1`; FDTD calls wrapped in `src/dataset/openems_wrapper.py` |
| `torch.compile` disabled | Stability on Python 3.13 + Windows | Enable after Phase 3 stable; toggle via `compute.compile: true` in config |
| `fsspec` downgraded to 2026.2.0 | `datasets` package pin | Expected; no functional impact |

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{pixel2026,
  title  = {PIXEL: Physics-guided Discrete Diffusion for Inverse EM Layout Synthesis},
  author = {[Authors]},
  year   = {2026},
  note   = {Under review, AAAI 2027}
}
```
