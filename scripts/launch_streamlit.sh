#!/bin/bash
# ============================================================
# scripts/launch_streamlit.sh
# Launch the PIXEL Streamlit demo on the HPC server.
#
# USAGE (on HPC):
#   cd /Data1/ec_23104075/projects/pixel-2026
#   bash scripts/launch_streamlit.sh [port]
#
# REMOTE ACCESS (from your local PC):
#   Step 1 — open SSH tunnel (run this in a new terminal on your PC):
#       ssh -L 8501:localhost:8501 ec_23104075@10.10.11.201
#
#   Step 2 — open your browser and go to:
#       http://localhost:8501
#
#   If port 8501 is busy, use a different port, e.g. 8502:
#       bash scripts/launch_streamlit.sh 8502
#       ssh -L 8502:localhost:8502 ec_23104075@10.10.11.201
#       → http://localhost:8502
#
# NOTE: This script runs on the login node (CPU).
#       For GPU-accelerated generation, request an interactive workq session:
#           qsub -I -q workq -l select=1:ncpus=8:ngpus=1:mem=16gb -l walltime=04:00:00
#       Then run this script inside the interactive session.
# ============================================================

set -e

PORT=${1:-8501}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=============================================="
echo " PIXEL Streamlit Demo"
echo " Project root: $PROJECT_ROOT"
echo " Port: $PORT"
echo "=============================================="

# ── Activate conda environment ──────────────────────────────────────────────
source /apps/compilers/anaconda3/etc/profile.d/conda.sh
conda activate pixel-env

# ── Set environment ──────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# OpenEMS binary path
export PATH="/Data1/ec_23104075/.conda/envs/pixel-env/bin:$PATH"

# Use CPU by default on login node; set CUDA_VISIBLE_DEVICES if on a GPU node
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Try to detect a free MIG GPU (if on a GPU node)
    FREE_MIG=$(python3 - << 'PYEOF' 2>/dev/null
import subprocess, re, sys
try:
    out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
    mig_list = []; cur_gpu = None
    for line in out.splitlines():
        m = re.search(r'GPU \d+.*UUID: (GPU-[a-f0-9-]+)', line)
        if m: cur_gpu = m.group(1)
        mm = re.search(r'(MIG-[a-f0-9-]+)', line)
        if mm and cur_gpu:
            mig_list.append((mm.group(1), cur_gpu))
    proc = subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=gpu_uuid,used_memory','--format=csv,noheader'],
        text=True, stderr=subprocess.DEVNULL)
    busy = {}
    for ln in proc.strip().splitlines():
        if ln.strip():
            parts = ln.split(',')
            busy[parts[0].strip()] = busy.get(parts[0].strip(),0) + int(parts[1].strip().split()[0])
    for mig, gpu in mig_list:
        if busy.get(gpu, 0) < 10000:
            print(mig); sys.exit(0)
except: pass
PYEOF
    )
    if [ -n "$FREE_MIG" ]; then
        export CUDA_VISIBLE_DEVICES="$FREE_MIG"
        echo "Using GPU MIG: $FREE_MIG"
    else
        export CUDA_VISIBLE_DEVICES=""
        echo "No GPU detected — running on CPU (slower generation)"
    fi
fi

# ── Verify key checkpoints exist ─────────────────────────────────────────────
echo ""
echo "Checking checkpoints..."
for f in \
    "experiments/surrogate_v1/surrogate_k0_best.pt" \
    "experiments/denoiser_v1/denoiser_best.pt" \
    "experiments/discriminator_v1/disc_best.pt"; do
    if [ -f "$PROJECT_ROOT/$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
        echo "  ERROR: Required checkpoint not found. Run Phases 2-4 first."
        exit 1
    fi
done

# ── Launch Streamlit ──────────────────────────────────────────────────────────
echo ""
echo "Starting Streamlit on port $PORT..."
echo ""
echo "  ┌─────────────────────────────────────────────────────┐"
echo "  │ Remote access from your local PC:                   │"
echo "  │                                                     │"
echo "  │   Open a NEW terminal on your PC and run:           │"
echo "  │   ssh -L ${PORT}:localhost:${PORT} ec_23104075@10.10.11.201        │"
echo "  │                                                     │"
echo "  │   Then open your browser at:                        │"
echo "  │   http://localhost:${PORT}                              │"
echo "  └─────────────────────────────────────────────────────┘"
echo ""

cd "$PROJECT_ROOT"

exec streamlit run app/pixel_demo.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.serverAddress localhost \
    --browser.gatherUsageStats false \
    --theme.base light \
    --theme.primaryColor "#2980b9"
