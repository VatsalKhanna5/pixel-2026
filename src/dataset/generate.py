"""
src/dataset/generate.py
PIXEL-2026 — Phase 1 Parallel Dataset Generation

Usage:
    conda activate pixel-env
    python -m src.dataset.generate --config experiments/configs/base_config.yaml
    python -m src.dataset.generate --pilot          # pilot validation only (200 samples)
    python -m src.dataset.generate --resume         # resume from last checkpoint
    python -m src.dataset.generate --n-samples 200000 --workers 56

Workflow:
  1. [ALWAYS unless --skip-pilot] Run pilot validation (200 samples, full report)
  2. If pilot passes quality gates → proceed with full generation
  3. Workers: multiprocessing.Pool with N workers (default from config: 56)
  4. Checkpoint every 10K records
  5. On SIGTERM/SIGINT: flush and checkpoint cleanly

Records failing physics validation are DISCARDED (not written to HDF5).
The generation loop overproduces by ~20% to account for attrition.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Project root resolution ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent   # d:\pixel-2026\pixel-2026
sys.path.insert(0, str(_ROOT))

from src.dataset.primitives   import sample_layout
from src.dataset.connectivity import is_connected          # module-level: cached on first import
from src.dataset.physics_validator import validate_record, check_dataset_quality_gates
from src.dataset.hdf5_writer  import CheckpointWriter, resume_from_checkpoint
from src.utils.config         import load_config

# ── Simulation backend selection ────────────────────────────────────────────
# Default: OpenEMS FDTD (ground-truth full-wave simulation).
# Set environment variable PIXEL_USE_ANALYTICAL=1 (or pass --use-analytical)
# to fall back to the analytical transmission-line model (fast, for debugging).
if os.environ.get("PIXEL_USE_ANALYTICAL", "0") == "1":
    from src.dataset.em_simulation import simulate, FREQS
    _BACKEND = "analytical"
else:
    from src.dataset.openems_wrapper import simulate, FREQS, worker_init_openems
    _BACKEND = "fdtd"

logger = logging.getLogger("pixel.generate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ── Graceful shutdown ────────────────────────────────────────────────────────
_SHUTDOWN = mp.Event()

def _signal_handler(signum, frame):
    logger.warning(f"[generate] Received signal {signum} — initiating graceful shutdown …")
    _SHUTDOWN.set()

signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)


# ── Worker function (must be module-level for pickle) ────────────────────────

def _worker_generate(args: tuple) -> dict | None:
    """
    Worker function: generate ONE valid record.

    Args:
        args: (worker_seed, substrate_id)

    Returns:
        dict record suitable for CheckpointWriter.write_record, or None if invalid.
    """
    worker_seed, substrate_id = args
    rng = np.random.default_rng(worker_seed)

    MAX_ATTEMPTS = 10   # max layout retries before returning None
    for _ in range(MAX_ATTEMPTS):
        layout, meta = sample_layout(rng, sigma=0.15)
        ptype = int(meta["type"])

        # Quick connectivity check before expensive simulation
        if not is_connected(layout):
            continue

        # EM simulation (FDTD or analytical depending on _BACKEND)
        try:
            sim = simulate(layout, meta, substrate_id=substrate_id, freqs=FREQS, noise_sigma=0.0)
        except Exception as exc:
            logger.debug(f"[worker] Simulation failed: {exc}")
            continue

        # Physics validation
        report = validate_record(
            layout      = layout,
            s11_mag     = sim["s11_mag"],
            s21_mag     = sim["s21_mag"],
            s11_phase   = sim["s11_phase"],
            s21_phase   = sim["s21_phase"],
            s21_complex = sim["s21_complex"],
        )

        if not report.validity_flag:
            logger.debug(f"[worker] Record failed validation: {report.fail_reasons}")
            continue

        # Build record dict
        record = {
            "layout":         layout,
            "s11_mag":        sim["s11_mag"],
            "s21_mag":        sim["s21_mag"],
            "s11_phase":      sim["s11_phase"],
            "s21_phase":      sim["s21_phase"],
            "substrate_id":   substrate_id,
            "resonance_freqs": sim["resonance_freqs"],
            "q_factors":      sim["q_factors"],
            "validity_flag":  True,
            "primitive_type": ptype,
        }
        return record

    return None   # could not generate a valid record in MAX_ATTEMPTS


# ── Pilot validation ─────────────────────────────────────────────────────────

def run_pilot(cfg, n_pilot: int = 200) -> bool:
    """
    Generate n_pilot samples synchronously, run quality gates, print report.

    Returns True if all gates pass (safe to start full generation).
    """
    logger.info(f"[Pilot] Generating {n_pilot} samples for validation …")
    substrates = [0, 1, 2, 3]  # always use all 4 substrates

    from src.dataset.physics_validator import ValidationReport
    reports: list[ValidationReport] = []
    type_counts = [0] * 11

    rng_master = np.random.default_rng(42)

    ok_count    = 0
    total_tries = 0

    for i in tqdm(range(n_pilot), desc="Pilot generation"):
        substrate_id = int(rng_master.integers(0, 4))
        seed = int(rng_master.integers(0, 2**31))

        # Generate a layout + simulate (even if it will fail validation)
        # so we can collect reports on the full attempt pool
        rng_w = np.random.default_rng(seed)
        layout, meta = sample_layout(rng_w, sigma=0.15)
        total_tries += 1

        # Always try to simulate and validate for accurate gate statistics
        if not is_connected(layout):
            # Create a dummy report with port_connected_ok=False
            from src.dataset.physics_validator import ValidationReport
            dummy = ValidationReport()
            dummy.port_connected_ok = False
            dummy.fail_reasons = ["port_disconnected"]
            reports.append(dummy)
            continue

        try:
            sim = simulate(layout, meta, substrate_id=substrate_id, freqs=FREQS, noise_sigma=0.002)
        except Exception:
            continue

        report = validate_record(
            layout      = layout,
            s11_mag     = sim["s11_mag"],
            s21_mag     = sim["s21_mag"],
            s11_phase   = sim["s11_phase"],
            s21_phase   = sim["s21_phase"],
            s21_complex = sim["s21_complex"],
        )
        reports.append(report)
        if report.validity_flag:
            ok_count += 1
            ptype = int(meta.get("type", 0))
            type_counts[ptype] += 1

    if not reports:
        logger.error("[Pilot] Zero valid records generated — aborting.")
        return False

    gates = check_dataset_quality_gates(reports)

    logger.info("=" * 60)
    logger.info("PILOT VALIDATION REPORT")
    logger.info("=" * 60)
    logger.info(f"  Samples attempted     : {total_tries}")
    logger.info(f"  Valid records         : {ok_count} ({100*ok_count/total_tries:.1f}%)")
    logger.info(f"  Connected structures  : {gates.get('n_connected', '?')} ({100*gates['connectivity_yield']:.1f}%)")
    logger.info(f"  Connectivity yield    : {gates['connectivity_yield']:.3f}  {'✓' if gates['connectivity_pass'] else '✗'} (>0.85)")
    logger.info(f"  Passivity rate        : {gates['passivity_rate']:.3f}   {'✓' if gates['passivity_pass'] else '✗'} (>0.99, connected only)")
    logger.info(f"  KK causality rate     : {gates['kk_rate']:.3f}   {'✓' if gates['kk_pass'] else '✗'} (>0.85, connected only)")
    logger.info(f"  Spectral smooth rate  : {gates['smooth_rate']:.3f}   {'✓' if gates['smooth_pass'] else '✗'} (>0.99, connected only)")
    logger.info(f"  Dynamic range rate    : {gates['dynamic_range_rate']:.3f}   {'✓' if gates['dynamic_range_pass'] else '✗'} (>0.60, connected only)")
    logger.info(f"  Overall validity      : {gates['overall_validity']:.3f}")

    # Primitive balance
    from src.dataset.primitives import PRIMITIVE_NAMES
    logger.info("  Primitive distribution:")
    for i, (name, cnt) in enumerate(zip(PRIMITIVE_NAMES, type_counts)):
        frac = cnt / max(ok_count, 1)
        flag = " ⚠ (>25%)" if frac > 0.25 else ""
        logger.info(f"    [{i:2d}] {name:<25s}: {cnt:4d}  ({frac:.1%}){flag}")

    logger.info("=" * 60)
    if gates["pass"]:
        logger.info("[Pilot] ALL QUALITY GATES PASSED — safe to proceed with full generation.")
    else:
        failed = [k for k in gates if k.endswith("_pass") and not gates[k]]
        logger.error(f"[Pilot] FAILED gates: {failed}")
        logger.error("[Pilot] Fix the issues above before running full generation.")
    logger.info("=" * 60)

    # Save pilot report to disk
    pilot_dir = Path("experiments") / "pilot_reports"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    report_path = pilot_dir / f"pilot_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as fp:
        json.dump({k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                   for k, v in gates.items()}, fp, indent=2)
    logger.info(f"[Pilot] Report saved → {report_path}")

    return bool(gates["pass"])


# ── Full generation ──────────────────────────────────────────────────────────

def run_generation(cfg, n_samples: int, n_workers: int, h5_path: Path,
                    resume: bool = False, pool_initializer=None) -> None:
    """Main parallel generation loop with checkpoint/resume."""
    start_id  = resume_from_checkpoint(h5_path) if resume else 0
    remaining = n_samples - start_id
    if remaining <= 0:
        logger.info(f"[generate] Dataset already has {start_id} records ≥ target {n_samples}. Done.")
        return

    logger.info(f"[generate] Generating {remaining} records (start_id={start_id}) "
                f"using {n_workers} workers …")

    substrates = [0, 1, 2, 3]
    rng_master = np.random.default_rng(1337 + start_id)

    CHECKPOINT_EVERY = 10_000
    batch_size_per_checkpoint = CHECKPOINT_EVERY

    written  = 0
    attempts = 0
    t0       = time.monotonic()

    # batch_size=256 matches _CHUNK_SIZE in hdf5_writer so each flush writes
    # exactly one HDF5 chunk — one compress/write cycle per chunk, no thrashing.
    # SUBMIT_BATCH = n_workers * 4 keeps the pool saturated without huge memory
    # spikes from overly large pool.map() calls.
    SUBMIT_BATCH = n_workers * 4

    with CheckpointWriter(h5_path, batch_size=256) as writer:
        pbar = tqdm(total=remaining, initial=0, desc="Generating", unit="records",
                    dynamic_ncols=True)
        try:
            with mp.Pool(processes=n_workers, initializer=pool_initializer) as pool:
                # Generate in submission batches = n_workers * 4
                # (keeps all workers busy without allocating hundreds of tasks at once)
                while written < remaining and not _SHUTDOWN.is_set():
                    # Prepare arguments
                    sub_id_batch  = [int(rng_master.integers(0, 4)) for _ in range(SUBMIT_BATCH)]
                    seed_batch    = [int(rng_master.integers(0, 2**31)) for _ in range(SUBMIT_BATCH)]
                    args_batch    = list(zip(seed_batch, sub_id_batch))

                    results = pool.map(_worker_generate, args_batch)
                    attempts += SUBMIT_BATCH

                    for rec in results:
                        if rec is None:
                            continue
                        writer.write_record(rec)
                        written += 1
                        pbar.update(1)
                        if written >= remaining:
                            break
                        if written % CHECKPOINT_EVERY == 0:
                            writer.flush()
                            elapsed = time.monotonic() - t0
                            rate    = written / elapsed
                            eta_h   = (remaining - written) / max(rate, 1) / 3600
                            logger.info(
                                f"[generate] Checkpoint: {written}/{remaining} written "
                                f"({100*written/remaining:.1f}%) | "
                                f"rate={rate:.0f}/s | ETA={eta_h:.1f}h"
                            )

        except KeyboardInterrupt:
            logger.warning("[generate] Interrupted — flushing checkpoint …")
            writer.flush()
        finally:
            pbar.close()

    elapsed   = time.monotonic() - t0
    yield_pct = 100.0 * written / max(attempts, 1)
    logger.info(
        f"[generate] Done. Wrote {written + start_id} total records "
        f"in {elapsed/3600:.2f}h "
        f"(yield={yield_pct:.1f}%, attempts={attempts})"
    )


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(description="PIXEL-2026 Phase 1 Dataset Generation")
    parser.add_argument("--config",      default="experiments/configs/base_config.yaml",
                        help="Path to OmegaConf config YAML")
    parser.add_argument("--n-samples",   type=int, default=None,
                        help="Override target dataset size")
    parser.add_argument("--workers",     type=int, default=None,
                        help="Override number of worker processes")
    parser.add_argument("--output",      default=None,
                        help="Override output HDF5 path")
    parser.add_argument("--pilot",       action="store_true",
                        help="Run pilot validation only (no full generation)")
    parser.add_argument("--skip-pilot",  action="store_true",
                        help="Skip pilot validation (not recommended)")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--use-analytical", action="store_true",
                        help="Use analytical TL model instead of OpenEMS FDTD (fast, debug only)")
    parser.add_argument("--skip-sanity",  action="store_true",
                        help="Skip the FDTD sanity check before pilot (not recommended)")
    parser.add_argument("--pilot-n",      type=int, default=200,
                        help="Number of samples for pilot validation (default: 200)")
    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Backend selection must happen before any workers are spawned ─────────
    if args.use_analytical:
        os.environ["PIXEL_USE_ANALYTICAL"] = "1"
        # Re-import so main process also uses analytical backend
        global simulate, FREQS, _BACKEND
        from src.dataset.em_simulation import simulate as _sim_a, FREQS as _f_a
        simulate = _sim_a
        FREQS    = _f_a
        _BACKEND = "analytical"
        logger.warning(
            "[generate] Backend = ANALYTICAL (transmission-line theory). "
            "NOT for production dataset — use FDTD for research-grade work."
        )
    else:
        logger.info("[generate] Backend = FDTD (OpenEMS full-wave simulation).")

    # Load config
    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = load_config(cfg_path)
    else:
        logger.warning(f"Config not found at {cfg_path}; using defaults.")
        cfg = None

    # Resolve parameters
    n_samples = args.n_samples or (int(cfg.dataset.target_size) if cfg else 200_000)
    n_workers = args.workers   or (int(cfg.dataset.n_workers_generation) if cfg else 56)
    h5_path   = Path(args.output) if args.output else Path("data") / "raw" / "pixel_dataset.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[generate] backend={_BACKEND}  target={n_samples:,}  "
        f"workers={n_workers}  output={h5_path}"
    )

    # ── FDTD sanity check (one through-line, verifies OpenEMS works) ─────────
    if _BACKEND == "fdtd" and not getattr(args, "skip_sanity", False):
        from src.dataset.openems_wrapper import quick_sanity_check
        logger.info("[generate] Running FDTD sanity check before pilot …")
        if not quick_sanity_check(substrate_id=0):
            logger.error(
                "[generate] FDTD sanity check FAILED. "
                "OpenEMS may not be working correctly. "
                "Use --use-analytical for a fast debug run, "
                "or investigate the OpenEMS installation."
            )
            sys.exit(1)
        logger.info("[generate] FDTD sanity check PASSED.")

    # ── Pilot validation ─────────────────────────────────────────────────
    if not args.skip_pilot:
        pilot_ok = run_pilot(cfg, n_pilot=args.pilot_n)
        if not pilot_ok:
            logger.error("[generate] Pilot FAILED. Refusing to start full generation.")
            logger.error("[generate] Fix the physics/connectivity issues and re-run.")
            sys.exit(1)
        if args.pilot:
            logger.info("[generate] --pilot mode: exiting after pilot validation.")
            sys.exit(0)
    else:
        logger.warning("[generate] --skip-pilot flag set — skipping pilot validation (NOT recommended).")

    # ── Full generation ───────────────────────────────────────────────────
    pool_init   = worker_init_openems if _BACKEND == "fdtd" else None
    run_generation(cfg, n_samples=n_samples, n_workers=n_workers,
                   h5_path=h5_path, resume=args.resume,
                   pool_initializer=pool_init)


if __name__ == "__main__":
    # Required on Windows for multiprocessing
    mp.freeze_support()
    main()
