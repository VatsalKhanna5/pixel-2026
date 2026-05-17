#!/usr/bin/env python3
"""
scripts/validate_phase0.py
PIXEL-2026 — Phase 0 Checkpoint Validation Script

Runs every check defined in PIXEL_EXECUTION_PLAN.md § Phase 0 Checkpoint P0.
Prints a pass/fail table and exits with code 0 (all pass) or 1 (any fail).

Usage:
    conda activate pixel-env
    python scripts/validate_phase0.py

Or via the setup script (called automatically at the end).
"""

import sys
import importlib
import subprocess
import textwrap
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Colour helpers (no external deps needed)
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
WARN = f"{YELLOW}WARN{RESET}"

results: list[tuple[str, str, str]] = []   # (check_name, status, detail)

ROOT = Path(__file__).resolve().parent.parent


def check(name: str, fn):
    """Run fn(); record PASS/FAIL."""
    try:
        detail = fn()
        results.append((name, PASS, detail or ""))
    except Exception as exc:
        results.append((name, FAIL, str(exc)[:120]))


# ===========================================================================
# CHECK SUITE
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
def _python_version():
    v = sys.version_info
    assert v.major == 3 and v.minor >= 13, f"Expected Python 3.13+, got {v.major}.{v.minor}"
    return f"{v.major}.{v.minor}.{v.micro}"

check("Python ≥ 3.13", _python_version)


# ---------------------------------------------------------------------------
# 2. PyTorch + CUDA
# ---------------------------------------------------------------------------
def _torch():
    import torch
    assert torch.cuda.is_available(), "CUDA not available"
    n = torch.cuda.device_count()
    assert n >= 1, f"Expected ≥1 GPU, got {n}"
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    return f"v{torch.__version__}  |  {n} GPU(s): {', '.join(names)}"

check("PyTorch + CUDA", _torch)


# ---------------------------------------------------------------------------
# 3. VRAM check (≥ 40 GB per GPU for research-grade comfort)
# ---------------------------------------------------------------------------
def _vram():
    import torch
    total_vram = 0.0
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gb = props.total_memory / 1e9
        total_vram += gb
        info.append(f"GPU{i}: {gb:.1f} GB")
    assert total_vram >= 40.0, f"Total VRAM {total_vram:.1f} GB < 40 GB"
    return "  ".join(info) + f"  (total {total_vram:.1f} GB)"

check("VRAM ≥ 40 GB total", _vram)


# ---------------------------------------------------------------------------
# 4. GPU tensor op (smoke test)
# ---------------------------------------------------------------------------
def _gpu_tensor():
    import torch
    a = torch.randn(256, 256, device="cuda:0")
    b = torch.randn(256, 256, device="cuda:0")
    c = a @ b
    assert c.shape == (256, 256)
    return "matrix multiply on cuda:0 — OK"

check("GPU tensor smoke test", _gpu_tensor)


# ---------------------------------------------------------------------------
# 5. Dual-GPU data parallel sanity
# ---------------------------------------------------------------------------
def _dual_gpu():
    import torch
    import torch.nn as nn
    if torch.cuda.device_count() < 2:
        return "SKIPPED — only 1 GPU present"
    model = nn.Linear(64, 64)
    model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
    x = torch.randn(16, 64, device="cuda")
    y = model(x)
    assert y.shape == (16, 64)
    return "DataParallel across GPU0 + GPU1 — OK"

check("Dual-GPU DataParallel", _dual_gpu)


# ---------------------------------------------------------------------------
# 6. Core scientific packages
# ---------------------------------------------------------------------------
REQUIRED = [
    ("numpy",        "2.0"),
    ("scipy",        "1.13"),
    ("sklearn",      "1.5"),
    ("skimage",      "0.24"),
    ("h5py",         "3.11"),
    ("pandas",       "2.2"),
    ("einops",       "0.8"),
    ("matplotlib",   "3.9"),
    ("tqdm",         "4.66"),
    ("wandb",        "0.17"),
    ("yaml",         None),       # pyyaml — no __version__
    ("omegaconf",    "2.3"),
    ("rich",         "13.8"),
    ("psutil",       "6.0"),
    ("joblib",       "1.4"),
    ("pynvml",       None),
]

def _make_pkg_check(mod, min_ver):
    def _fn():
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "unknown")
        if min_ver and ver != "unknown":
            from packaging.version import Version
            assert Version(ver) >= Version(min_ver), \
                f"{mod} version {ver} < required {min_ver}"
        return ver
    _fn.__name__ = f"import_{mod}"
    return _fn

for mod, minv in REQUIRED:
    check(f"import {mod}", _make_pkg_check(mod, minv))


# ---------------------------------------------------------------------------
# 7. ML-specific packages (Phase 2–3 deps)
# ---------------------------------------------------------------------------
ML_PKGS = [
    ("torchmetrics", "1.4"),
    ("timm",         "1.0"),
    ("diffusers",    "0.30"),
    ("transformers", "4.45"),
    ("accelerate",   "0.34"),
]

for mod, minv in ML_PKGS:
    check(f"import {mod}", _make_pkg_check(mod, minv))


# ---------------------------------------------------------------------------
# 8. Pixel2026 package importable (editable install)
# ---------------------------------------------------------------------------
def _pixel_pkg():
    # src/ must be importable as a namespace package
    import importlib
    spec = importlib.util.find_spec("src")
    if spec is None:
        # Try direct path injection
        sys.path.insert(0, str(ROOT))
        spec = importlib.util.find_spec("src")
    assert spec is not None, "src package not importable — run: pip install -e ."
    return f"src found at {spec.submodule_search_locations[0]}"

check("src package importable", _pixel_pkg)


# ---------------------------------------------------------------------------
# 9. HDF5 write/read round-trip
# ---------------------------------------------------------------------------
def _hdf5():
    import h5py, numpy as np, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        layout = np.random.randint(0, 2, (15, 15), dtype=np.uint8)
        spec   = np.random.rand(100).astype(np.float32)
        with h5py.File(path, "w") as fh:
            fh.create_dataset("layout", data=layout)
            fh.create_dataset("S21_mag", data=spec)
        with h5py.File(path, "r") as fh:
            l2 = fh["layout"][:]
            s2 = fh["S21_mag"][:]
        assert np.array_equal(layout, l2)
        assert np.allclose(spec, s2)
        return "write+read 15×15 layout and S21_mag — OK"
    finally:
        os.unlink(path)

check("HDF5 round-trip", _hdf5)


# ---------------------------------------------------------------------------
# 10. WandB importable (login not required for Phase 0)
# ---------------------------------------------------------------------------
def _wandb():
    import wandb
    return f"v{wandb.__version__}  (offline mode OK for now)"

check("wandb importable", _wandb)


# ---------------------------------------------------------------------------
# 11. accelerate DDP config present / creatable
# ---------------------------------------------------------------------------
def _accelerate():
    import accelerate
    return f"v{accelerate.__version__}"

check("accelerate importable", _accelerate)


# ---------------------------------------------------------------------------
# 12. multiprocessing pool sanity (needed for parallel EM simulation)
# ---------------------------------------------------------------------------
def _square(x):
    """Module-level worker function."""
    return x * x


def _mp_pool():
    import multiprocessing as mp
    # Use ThreadPool to avoid Windows spawn-context __main__ guard requirement.
    # The actual dataset generation scripts use proper if __name__ == '__main__' guards.
    from multiprocessing.pool import ThreadPool
    n = mp.cpu_count()
    assert n >= 8, f"Only {n} CPUs detected"
    with ThreadPool(min(4, n)) as pool:
        results_inner = pool.map(_square, [1, 2, 3, 4])
    assert results_inner == [1, 4, 9, 16], f"Unexpected results: {results_inner}"
    return f"{n} logical CPUs  |  ThreadPool(4) map test — OK"

check("multiprocessing pool", _mp_pool)


# ---------------------------------------------------------------------------
# 13. Logs and checkpoints directories exist
# ---------------------------------------------------------------------------
def _dirs():
    required = [
        "logs", "data/raw", "data/processed", "data/interim",
        "src/dataset", "src/models", "src/losses",
        "src/guidance", "src/training", "src/evaluation", "src/utils",
        "experiments/configs", "experiments/surrogate",
        "experiments/denoiser", "experiments/guided",
        "checkpoints/surrogate", "checkpoints/denoiser", "checkpoints/discriminator",
        "paper", "notebooks", "scripts",
    ]
    missing = [d for d in required if not (ROOT / d).is_dir()]
    assert not missing, f"Missing directories: {missing}"
    return f"{len(required)} directories present"

check("directory structure", _dirs)


# ---------------------------------------------------------------------------
# 14. OpenEMS binary / Python bridge (WARN only — not blocking)
# ---------------------------------------------------------------------------
def _openems():
    # Try Python import first
    try:
        from openems import openEMS
        return "Python import OK"
    except ImportError:
        pass
    # Try binary on PATH
    try:
        r = subprocess.run(["openEMS", "--version"], capture_output=True, timeout=5)
        if r.returncode == 0:
            return f"binary on PATH: {r.stdout.decode().strip()[:60]}"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Not found — warn but don't fail Phase 0
    results.append((
        "OpenEMS available",
        WARN,
        "Not found — run scripts/setup_openems_windows.ps1 before Phase 1"
    ))
    return None   # Signal that we already appended ourselves

check("OpenEMS available", _openems)


# ---------------------------------------------------------------------------
# 15. PyTorch Gumbel-Sigmoid (binarization building block)
# ---------------------------------------------------------------------------
def _gumbel_sigmoid():
    import torch, torch.nn.functional as F
    p = torch.sigmoid(torch.randn(4, 15, 15))
    g = -torch.log(-torch.log(torch.clamp(torch.rand_like(p), 1e-8, 1 - 1e-8)))
    tau = 1.0
    soft = torch.sigmoid((torch.log(p) - torch.log(1 - p) + g) / tau)
    assert soft.shape == (4, 15, 15)
    assert soft.min() >= 0.0 and soft.max() <= 1.0
    return f"output ∈ [{soft.min():.3f}, {soft.max():.3f}] — OK"

check("Gumbel-Sigmoid (binarization)", _gumbel_sigmoid)


# ---------------------------------------------------------------------------
# 16. Hilbert transform / KK loss building block (via scipy)
# ---------------------------------------------------------------------------
def _hilbert_kk():
    import numpy as np
    from scipy.signal import hilbert
    # For a causal signal, Re[S] = Hilbert{Im[S]}
    f = np.linspace(0.5e9, 20e9, 100)
    # Synthetic Lorentzian resonance (physically causal)
    f0, Q = 10e9, 20
    s21_complex = 1.0 / (1 + 1j * Q * (f / f0 - f0 / f))
    re_part = s21_complex.real
    im_part = s21_complex.imag
    # KK: Re = H{Im} (up to sign convention)
    re_reconstructed = np.imag(hilbert(im_part))
    kk_error = np.mean(np.abs(re_part - re_reconstructed))
    assert kk_error < 0.05, f"KK residual {kk_error:.4f} too large on test signal"
    return f"KK residual on Lorentzian test signal: {kk_error:.5f} — OK"

check("Hilbert / KK building block", _hilbert_kk)


# ---------------------------------------------------------------------------
# 17. BFS connectivity check (dataset generation prerequisite)
# ---------------------------------------------------------------------------
def _bfs_connectivity():
    import numpy as np
    from collections import deque

    def is_connected(layout, port1=(7, 0), port2=(7, 14)):
        """BFS from port1 to port2 on binary layout."""
        H, W = layout.shape
        if layout[port1] == 0 or layout[port2] == 0:
            return False
        visited = set()
        queue = deque([port1])
        visited.add(port1)
        while queue:
            r, c = queue.popleft()
            if (r, c) == port2:
                return True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<H and 0<=nc<W and (nr,nc) not in visited and layout[nr,nc]==1:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        return False

    # Connected: horizontal line across row 7
    l1 = np.zeros((15,15), dtype=np.uint8)
    l1[7, :] = 1
    assert is_connected(l1), "Horizontal line should be connected"
    # Disconnected: broken at midpoint
    l2 = l1.copy(); l2[7, 7] = 0
    assert not is_connected(l2), "Broken line should be disconnected"
    return "Connected + disconnected cases verified — OK"

check("BFS connectivity (dataset prerequisite)", _bfs_connectivity)


# ===========================================================================
# REPORT
# ===========================================================================
def print_report():
    print()
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{CYAN}  PIXEL-2026  Phase 0 Validation  |  {datetime.now():%Y-%m-%d %H:%M:%S}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"  {'Check':<42} {'Status':<10} {'Detail'}")
    print(f"  {'-'*42} {'-'*10} {'-'*40}")

    passes = fails = warns = 0
    for name, status, detail in results:
        if status == PASS:  passes += 1
        elif status == FAIL: fails += 1
        else:                warns += 1
        detail_short = textwrap.shorten(detail, 55)
        print(f"  {name:<42} {status:<18} {detail_short}")

    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    total = passes + fails + warns
    print(f"  Result:  {passes}/{total} passed  |  {warns} warnings  |  {fails} failures")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print()

    if fails > 0:
        print(f"{RED}Phase 0 INCOMPLETE — fix the {fails} failing check(s) above.{RESET}")
        sys.exit(1)
    elif warns > 0:
        print(f"{YELLOW}Phase 0 CONDITIONALLY PASSED — {warns} warning(s) require attention before Phase 1.{RESET}")
        sys.exit(0)
    else:
        print(f"{GREEN}{BOLD}Phase 0 FULLY PASSED — ready to begin Phase 1 (Dataset Generation).{RESET}")
        sys.exit(0)


print_report()
