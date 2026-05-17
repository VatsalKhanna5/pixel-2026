"""
scripts/audit_dataset.py
PIXEL-2026 — Post-Generation Dataset Integrity Audit

Runs a comprehensive scan of the HDF5 dataset and prints a full statistics report.
Optionally removes invalid records (--purge flag).

Usage:
    conda activate pixel-env
    python scripts/audit_dataset.py --h5 data/raw/pixel_dataset.h5
    python scripts/audit_dataset.py --h5 data/raw/pixel_dataset.h5 --purge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.dataset.connectivity      import is_connected
from src.dataset.physics_validator import (
    validate_record, check_dataset_quality_gates, ValidationReport
)
from src.dataset.primitives import PRIMITIVE_NAMES

logger = logging.getLogger("pixel.audit")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def audit(h5_path: Path, purge: bool = False) -> dict:
    if not h5_path.exists():
        logger.error(f"File not found: {h5_path}")
        sys.exit(1)

    with h5py.File(h5_path, "r") as f:
        n = f["layout"].shape[0]
        logger.info(f"[Audit] Scanning {n:,} records in {h5_path} …")

        reports: list[ValidationReport] = []
        invalid_indices: list[int] = []
        type_counts = [0] * len(PRIMITIVE_NAMES)
        substrate_counts = [0] * 4

        BATCH = 500
        t0 = time.monotonic()

        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            batch_n = end - start

            layouts     = f["layout"][start:end]
            s11_mag     = f["S11_mag"][start:end]
            s21_mag     = f["S21_mag"][start:end]
            s11_phase   = f["S11_phase"][start:end]
            s21_phase   = f["S21_phase"][start:end]
            prim_types  = f["primitive_type"][start:end]
            sub_ids     = f["substrate_id"][start:end]

            for i in range(batch_n):
                report = validate_record(
                    layout    = layouts[i],
                    s11_mag   = s11_mag[i],
                    s21_mag   = s21_mag[i],
                    s11_phase = s11_phase[i],
                    s21_phase = s21_phase[i],
                )
                reports.append(report)
                if not report.validity_flag:
                    invalid_indices.append(start + i)

                pt = int(prim_types[i])
                if 0 <= pt < len(type_counts):
                    type_counts[pt] += 1

                si = int(sub_ids[i])
                if 0 <= si < 4:
                    substrate_counts[si] += 1

            elapsed = time.monotonic() - t0
            if (start // BATCH) % 20 == 0:
                logger.info(f"  {end:>8,}/{n:>8,}  ({100*end/n:.1f}%)  elapsed={elapsed:.0f}s")

    # ── Quality gates ─────────────────────────────────────────────────────
    gates = check_dataset_quality_gates(reports)
    n_invalid = len(invalid_indices)

    # ── Primitive balance ─────────────────────────────────────────────────
    max_type_frac = max(type_counts) / max(n, 1)
    balance_ok    = max_type_frac <= 0.25

    # ── Spectral diversity (S21 MSE between random pairs) ─────────────────
    diversity_score = _estimate_spectral_diversity(h5_path, n_pairs=1000)

    # ── Print report ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIXEL-2026 DATASET AUDIT REPORT")
    logger.info("=" * 70)
    logger.info(f"  File              : {h5_path}")
    logger.info(f"  Total records     : {n:,}")
    logger.info(f"  Invalid records   : {n_invalid:,}  ({100*n_invalid/max(n,1):.2f}%)")
    logger.info("")
    logger.info("  QUALITY GATES:")
    logger.info(f"    Connectivity yield   : {gates['connectivity_yield']:.4f}  {'PASS' if gates['connectivity_pass'] else 'FAIL'}")
    logger.info(f"    Passivity rate       : {gates['passivity_rate']:.4f}  {'PASS' if gates['passivity_pass'] else 'FAIL'}")
    logger.info(f"    KK causality rate    : {gates['kk_rate']:.4f}  {'PASS' if gates['kk_pass'] else 'FAIL'}")
    logger.info(f"    Spectral smooth rate : {gates['smooth_rate']:.4f}  {'PASS' if gates['smooth_pass'] else 'FAIL'}")
    logger.info(f"    Dynamic range rate   : {gates['dynamic_range_rate']:.4f}  {'PASS' if gates['dynamic_range_pass'] else 'FAIL'}")
    logger.info(f"    Spectral diversity   : {diversity_score:.4f}  {'PASS' if diversity_score>0.05 else 'FAIL'}  (S21 MSE >0.05)")
    logger.info(f"    Primitive balance    : max_frac={max_type_frac:.3f}  {'PASS' if balance_ok else 'FAIL'}  (<25% per type)")
    logger.info("")
    logger.info("  PRIMITIVE DISTRIBUTION:")
    for i, (name, cnt) in enumerate(zip(PRIMITIVE_NAMES, type_counts)):
        frac = cnt / max(n, 1)
        flag = " *** OVER 25% ***" if frac > 0.25 else ""
        logger.info(f"    [{i:2d}] {name:<28s}: {cnt:8,}  ({frac:.1%}){flag}")
    logger.info("")
    logger.info("  SUBSTRATE DISTRIBUTION:")
    sub_names = ["Rogers4003C", "FR4", "Rogers5880", "Alumina"]
    for i, (sname, cnt) in enumerate(zip(sub_names, substrate_counts)):
        logger.info(f"    [{i}] {sname:<15s}: {cnt:8,}  ({cnt/max(n,1):.1%})")
    logger.info("")
    logger.info(f"  OVERALL: {'PASS' if gates['pass'] and balance_ok and diversity_score>0.05 else 'FAIL'}")
    logger.info("=" * 70)

    # ── Save report JSON ──────────────────────────────────────────────────
    report_path = h5_path.with_suffix(".audit.json")
    report_dict = {
        "n_total": n,
        "n_invalid": n_invalid,
        "quality_gates": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if isinstance(v, float) else int(v))
                          for k, v in gates.items()},
        "spectral_diversity": float(diversity_score),
        "primitive_balance_ok": bool(balance_ok),
        "primitive_distribution": {PRIMITIVE_NAMES[i]: int(type_counts[i]) for i in range(len(PRIMITIVE_NAMES))},
        "substrate_distribution": {sub_names[i]: int(substrate_counts[i]) for i in range(4)},
    }
    with open(report_path, "w") as fp:
        json.dump(report_dict, fp, indent=2)
    logger.info(f"[Audit] Report saved → {report_path}")

    # ── Purge invalid records ─────────────────────────────────────────────
    if purge and n_invalid > 0:
        logger.info(f"[Audit] --purge: removing {n_invalid} invalid records …")
        _purge_invalid(h5_path, invalid_indices)
        logger.info("[Audit] Purge complete.")

    return report_dict


def _estimate_spectral_diversity(h5_path: Path, n_pairs: int = 1000) -> float:
    """
    Estimate spectral diversity: mean S21 MSE between random pairs.
    A score > 0.05 indicates sufficient diversity (per PIXEL_EXECUTION_PLAN.md §P1).
    """
    with h5py.File(h5_path, "r") as f:
        n = f["S21_mag"].shape[0]
        if n < 2:
            return 0.0
        rng = np.random.default_rng(0)
        idx_a = rng.integers(0, n, size=n_pairs)
        idx_b = rng.integers(0, n, size=n_pairs)
        # Avoid self-comparisons
        same = idx_a == idx_b
        idx_b[same] = (idx_b[same] + 1) % n
        mse_sum = 0.0
        BATCH = 100
        for i in range(0, n_pairs, BATCH):
            ia = idx_a[i:i+BATCH]
            ib = idx_b[i:i+BATCH]
            a  = f["S21_mag"][ia]
            b  = f["S21_mag"][ib]
            mse_sum += float(np.mean((a - b)**2)) * len(ia)
        return mse_sum / n_pairs


def _purge_invalid(h5_path: Path, invalid_indices: list[int]) -> None:
    """
    Remove invalid records from HDF5 by rewriting the file (keeping valid only).
    Creates a backup at <stem>.bak.h5 before overwriting.
    """
    import shutil
    from src.dataset.hdf5_writer import _DATASETS

    backup_path = h5_path.with_suffix(".bak.h5")
    shutil.copy2(h5_path, backup_path)
    logger.info(f"[Purge] Backup created: {backup_path}")

    invalid_set = set(invalid_indices)
    with h5py.File(backup_path, "r") as src:
        n = src["layout"].shape[0]
        valid_indices = [i for i in range(n) if i not in invalid_set]
        n_valid = len(valid_indices)
        logger.info(f"[Purge] Keeping {n_valid}/{n} records …")

        with h5py.File(h5_path, "w") as dst:
            # Copy attributes
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["purged_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Create datasets
            for name, dtype, shape_suffix, fill in _DATASETS:
                shape    = (n_valid,) + shape_suffix
                maxshape = (None,)    + shape_suffix
                chunks   = (min(256, n_valid),) + shape_suffix
                dst.create_dataset(name, shape=shape, maxshape=maxshape,
                                   dtype=dtype, chunks=chunks,
                                   compression="gzip", compression_opts=1,
                                   fillvalue=fill)

            # Copy valid records in batches
            BATCH = 1000
            for out_start in range(0, n_valid, BATCH):
                out_end = min(out_start + BATCH, n_valid)
                in_indices = valid_indices[out_start:out_end]
                for name, _, _, _ in _DATASETS:
                    # h5py fancy indexing requires sorted list
                    in_sorted = sorted(in_indices)
                    data = src[name][in_sorted]
                    dst[name][out_start:out_end] = data


def main():
    parser = argparse.ArgumentParser(description="PIXEL-2026 Dataset Audit")
    parser.add_argument("--h5",    required=True, help="Path to HDF5 dataset file")
    parser.add_argument("--purge", action="store_true", help="Remove invalid records")
    args = parser.parse_args()

    result = audit(Path(args.h5), purge=args.purge)
    sys.exit(0 if result["quality_gates"]["pass"] else 1)


if __name__ == "__main__":
    main()
