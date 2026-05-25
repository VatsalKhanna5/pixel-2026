#!/bin/bash
# ============================================================
# PIXEL-2026 — HPC Environment Setup Script (Linux / NIT Jalandhar H100)
#
# Run this ONCE on the login node to set up the full environment.
# This does NOT submit any jobs — it installs packages interactively.
#
# Usage:
#   cd /Data1/ec_23104075/projects/pixel-2026
#   bash scripts/setup_hpc_env.sh
#
# After this script completes:
#   1. Transfer HDF5 from Windows: see instructions below
#   2. Build OpenEMS (separate step — takes ~45 min)
#   3. Submit generation jobs: qsub scripts/pbs_generate.pbs
# ============================================================

set -e   # abort on first error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_BASE="/apps/compilers/anaconda3"
ENV_NAME="pixel-env"

echo "========================================================"
echo " PIXEL-2026 HPC Environment Setup"
echo " Project root : $PROJECT_ROOT"
echo " Conda base   : $CONDA_BASE"
echo " Env name     : $ENV_NAME"
echo "========================================================"

# ---- 1. Initialise conda for this shell ----
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- 2. Create the conda environment ----
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[INFO] Environment '$ENV_NAME' already exists — skipping creation."
else
    echo "[INFO] Creating conda environment '$ENV_NAME' with Python 3.11 ..."
    conda create -n "$ENV_NAME" python=3.11 pip -y
fi

conda activate "$ENV_NAME"

# ---- 3. Install PyTorch (CUDA 12.x, compatible with H100 CUDA 13.0 driver) ----
echo "[INFO] Installing PyTorch (CUDA 12.4 wheels) ..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --upgrade

# ---- 4. Install remaining requirements ----
echo "[INFO] Installing project requirements ..."
pip install \
    torchmetrics>=1.4.0 \
    timm>=1.0.0 \
    diffusers>=0.30.0 \
    transformers>=4.45.0 \
    accelerate>=0.34.0 \
    datasets>=2.20.0 \
    numpy>=2.0.0 \
    scipy>=1.13.0 \
    scikit-learn>=1.5.0 \
    scikit-image>=0.24.0 \
    h5py>=3.11.0 \
    pandas>=2.2.0 \
    einops>=0.8.0 \
    matplotlib>=3.9.0 \
    seaborn>=0.13.0 \
    tqdm>=4.66.0 \
    wandb>=0.17.0 \
    rich>=13.8.0 \
    pyyaml>=6.0.0 \
    omegaconf>=2.3.0 \
    psutil>=6.0.0 \
    joblib>=1.4.0 \
    pynvml>=11.5.0 \
    pytest>=8.3.0 \
    ipykernel>=6.29.0

# ---- 5. Install pixel2026 package in editable mode ----
echo "[INFO] Installing pixel2026 in editable mode ..."
pip install -e . --no-deps

# ---- 6. Register Jupyter kernel ----
echo "[INFO] Registering Jupyter kernel ..."
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python (pixel-env)"

# ---- 7. Quick validation ----
echo ""
echo "[INFO] Running quick validation ..."
python - <<'EOF'
import sys
print(f"Python : {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available (login node may show False): {torch.cuda.is_available()}")

import numpy, scipy, h5py, omegaconf, wandb, tqdm, accelerate
print("All core packages imported successfully.")

import src.dataset.primitives as p
print(f"Primitives loaded: {len(p.PRIMITIVE_NAMES)} types")

print("\n[PASS] Environment validation complete.")
EOF

echo ""
echo "========================================================"
echo " Setup complete! Next steps:"
echo ""
echo " 1. Transfer HDF5 from Windows:"
echo "    scp data\\raw\\pixel_dataset.h5 \\"
echo "        ec_23104075@10.10.11.201:/Data1/ec_23104075/projects/pixel-2026/data/raw/"
echo ""
echo " 2. Build OpenEMS (run scripts/build_openems_linux.sh)"
echo ""
echo " 3. Submit generation job:"
echo "    qsub scripts/pbs_generate.pbs"
echo ""
echo " 4. Check job status:"
echo "    qstat -u $USER"
echo "========================================================"
