#!/bin/bash
# ============================================================
# PIXEL-2026 — Build OpenEMS from Source (Linux / HPC)
#
# Installs openEMS + CSXCAD Python bindings into the pixel-env
# conda environment. Build time: ~30–50 min.
#
# Requirements: cmake, gcc, git (all available on HPC login node)
#
# Usage:
#   conda activate pixel-env
#   bash scripts/build_openems_linux.sh
#
# After this: verify with  python -c "from openEMS.openEMS import openEMS"
# ============================================================

set -e

CONDA_BASE="/apps/compilers/anaconda3"
ENV_NAME="pixel-env"
INSTALL_DIR="/Data1/ec_23104075/openems_build"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

PYTHON_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
echo "Installing into conda env: $PYTHON_PREFIX"
echo "Build directory          : $INSTALL_DIR"

# ---- 1. Install C++ build dependencies via conda-forge ----
echo "[INFO] Installing C++ build dependencies via conda-forge ..."
# NOTE: vtk is intentionally excluded — openEMS compiles without it (ENABLE_GUI=OFF)
# and vtk is extremely large (~700 MB), causing conda solve timeouts.
conda install -c conda-forge \
    hdf5 \
    tinyxml \
    tinyxml2 \
    fftw \
    boost-cpp \
    make \
    -y

# Export cmake/pkg-config hints so cmake finds conda packages
export CMAKE_PREFIX_PATH="$PYTHON_PREFIX"
export PKG_CONFIG_PATH="$PYTHON_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$PYTHON_PREFIX/lib:$LD_LIBRARY_PATH"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# ---- 2. Clone the openEMS-Project (includes CSXCAD, openEMS, python bindings) ----
if [ ! -d "openEMS-Project" ]; then
    echo "[INFO] Cloning openEMS-Project ..."
    git clone --recursive https://github.com/thliebig/openEMS-Project.git
fi
cd openEMS-Project

# Update to latest if already cloned
git pull --recurse-submodules 2>/dev/null || true

# ---- 3. Build CSXCAD (geometry library) ----
echo "[INFO] Building CSXCAD ..."
cd CSXCAD
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$PYTHON_PREFIX" \
    -DCMAKE_PREFIX_PATH="$PYTHON_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DWITH_VTK=OFF
make -j"$(nproc)"
make install
cd ../..

# ---- 4. Build openEMS solver ----
echo "[INFO] Building openEMS ..."
cd openEMS
mkdir -p build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$PYTHON_PREFIX" \
    -DCMAKE_PREFIX_PATH="$PYTHON_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_GUI=OFF \
    -DENABLE_CTBM=OFF \
    -DWITH_VTK=OFF \
    -DCSXCAD_ROOT="$PYTHON_PREFIX"
make -j"$(nproc)"
make install
cd ../..

# ---- 5. Install Python bindings ----
echo "[INFO] Installing Python bindings ..."
cd python
pip install -e .
cd ..

# ---- 5. Verify ----
echo ""
echo "[INFO] Verifying installation ..."
python - <<'EOF'
from CSXCAD import ContinuousStructure
from openEMS.openEMS import openEMS
from openEMS.physical_constants import C0
print(f"[PASS] CSXCAD imported: ContinuousStructure OK")
print(f"[PASS] openEMS imported: C0 = {C0:.4e} m/s")
EOF

echo ""
echo "[PASS] OpenEMS build complete."
echo "Run a sanity check: python -m src.dataset.openems_wrapper 0"
