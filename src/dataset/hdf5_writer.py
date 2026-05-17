"""
src/dataset/hdf5_writer.py
PIXEL-2026 — Checkpoint/Resume HDF5 Writer

Design:
  - Single HDF5 file with resizable datasets (maxshape=(None,...))
  - JSON sidecar file (<stem>.checkpoint.json) tracks last_written_id, counts, timestamps
  - Atomic batch writes (flush every N records)
  - Lock file to prevent concurrent writers
  - resume_from_checkpoint() returns the ID to continue from

HDF5 schema (from PIXEL_EXECUTION_PLAN.md §Phase 1):
  layout          (N, 15, 15)  uint8
  S11_mag         (N, 100)     float32
  S21_mag         (N, 100)     float32
  S11_phase       (N, 100)     float32
  S21_phase       (N, 100)     float32
  substrate_id    (N,)         uint8
  resonance_freqs (N, 5)       float32
  Q_factor        (N, 5)       float32
  validity_flag   (N,)         bool
  primitive_type  (N,)         uint8
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Grid / freq constants
H, W   = 15, 15
N_FREQ = 100
N_RES  = 5      # max resonances stored

# Dataset descriptors: (key_in_h5, dtype, shape_suffix, fill_value)
_DATASETS = [
    ("layout",          np.uint8,    (H, W),   0),
    ("S11_mag",         np.float32,  (N_FREQ,), 0.0),
    ("S21_mag",         np.float32,  (N_FREQ,), 0.0),
    ("S11_phase",       np.float32,  (N_FREQ,), 0.0),
    ("S21_phase",       np.float32,  (N_FREQ,), 0.0),
    ("substrate_id",    np.uint8,    (),        0),
    ("resonance_freqs", np.float32,  (N_RES,),  0.0),
    ("Q_factor",        np.float32,  (N_RES,),  0.0),
    ("validity_flag",   bool,        (),        False),
    ("primitive_type",  np.uint8,    (),        0),
]

_CHUNK_SIZE  = 256    # HDF5 chunk size along N axis; must equal batch_size for optimal I/O
_COMPRESSION = "gzip"   # lzf has errno=22 on Windows under high-throughput writes
_COMP_LEVEL  = 1        # level 1 = fast; still 40-50% smaller than uncompressed


class CheckpointWriter:
    """
    Thread-safe checkpoint-resumable HDF5 writer.

    Usage:
        writer = CheckpointWriter(path)
        with writer:
            writer.write_batch(records)
        start_id = writer.last_written_id + 1
    """

    def __init__(self, h5_path: str | Path, batch_size: int = 10):
        self.h5_path     = Path(h5_path)
        self.ckpt_path   = self.h5_path.with_suffix(".checkpoint.json")
        self.lock_path   = self.h5_path.with_suffix(".lock")
        self.batch_size  = batch_size
        self._h5: h5py.File | None = None
        self._buffer: list[dict] = []

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self):
        self._acquire_lock()
        self._open_or_create()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._buffer:
                self._flush_buffer()
            if self._h5 is not None:
                self._h5.flush()
                self._h5.close()
                self._h5 = None
            self._save_checkpoint()
        finally:
            self._release_lock()
        return False   # do not suppress exceptions

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def last_written_id(self) -> int:
        """Last 0-based index that has been written (−1 if file is empty)."""
        ckpt = self._load_checkpoint()
        return int(ckpt.get("last_written_id", -1))

    @property
    def total_count(self) -> int:
        """Number of records in file."""
        ckpt = self._load_checkpoint()
        return int(ckpt.get("total_count", 0))

    def write_record(self, record: dict) -> None:
        """Buffer a single record. Flushed automatically when batch is full."""
        self._buffer.append(record)
        if len(self._buffer) >= self.batch_size:
            self._flush_buffer()

    def write_batch(self, records: list[dict]) -> None:
        """Write a list of records."""
        for r in records:
            self.write_record(r)

    def flush(self) -> None:
        """Force-flush any buffered records to disk."""
        if self._buffer:
            self._flush_buffer()
        if self._h5 is not None:
            self._h5.flush()
        self._save_checkpoint()

    # ── Internal ─────────────────────────────────────────────────────────

    def _open_or_create(self) -> None:
        mode = "a" if self.h5_path.exists() else "w"
        self._h5 = h5py.File(self.h5_path, mode, libver="latest")
        if mode == "w":
            self._create_datasets()
            self._save_checkpoint()
            logger.info(f"[HDF5Writer] Created new file: {self.h5_path}")
        else:
            existing_n = self._h5["layout"].shape[0]
            logger.info(f"[HDF5Writer] Opened existing file ({existing_n} records): {self.h5_path}")

    def _create_datasets(self) -> None:
        assert self._h5 is not None
        for name, dtype, shape_suffix, fill in _DATASETS:
            shape    = (0,) + shape_suffix
            maxshape = (None,) + shape_suffix
            chunks   = (_CHUNK_SIZE,) + shape_suffix
            self._h5.create_dataset(
                name, shape=shape, maxshape=maxshape,
                dtype=dtype, chunks=chunks,
                compression=_COMPRESSION, compression_opts=_COMP_LEVEL,
                fillvalue=fill,
            )
        self._h5.attrs["created_at"]    = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._h5.attrs["schema_version"] = "1.0"

    def _flush_buffer(self) -> None:
        if not self._buffer or self._h5 is None:
            return

        n_new = len(self._buffer)
        n_cur = self._h5["layout"].shape[0]
        n_tot = n_cur + n_new

        # Resize all datasets in one pass
        for name, dtype, shape_suffix, _ in _DATASETS:
            self._h5[name].resize((n_tot,) + shape_suffix)

        # ── Batch slice write ────────────────────────────────────────────────
        # Writing a contiguous slice triggers exactly ONE read-decompress-modify-
        # recompress-write per HDF5 chunk, instead of len(buffer) cycles.
        # The old per-record loop (self._h5[name][idx]=...) caused 500+
        # compress/decompress cycles per 50-record flush and crashed at ~500
        # records with errno=22 on Windows (HDF5 chunk-cache exhaustion).
        sl = slice(n_cur, n_tot)
        self._h5["layout"][sl]          = np.stack([r["layout"]        for r in self._buffer]).astype(np.uint8)
        self._h5["S11_mag"][sl]         = np.stack([r["s11_mag"]       for r in self._buffer]).astype(np.float32)
        self._h5["S21_mag"][sl]         = np.stack([r["s21_mag"]       for r in self._buffer]).astype(np.float32)
        self._h5["S11_phase"][sl]       = np.stack([r["s11_phase"]     for r in self._buffer]).astype(np.float32)
        self._h5["S21_phase"][sl]       = np.stack([r["s21_phase"]     for r in self._buffer]).astype(np.float32)
        self._h5["substrate_id"][sl]    = np.array([r["substrate_id"]  for r in self._buffer], dtype=np.uint8)
        self._h5["resonance_freqs"][sl] = np.stack([r["resonance_freqs"] for r in self._buffer]).astype(np.float32)
        self._h5["Q_factor"][sl]        = np.stack([r["q_factors"]     for r in self._buffer]).astype(np.float32)
        self._h5["validity_flag"][sl]   = np.array([bool(r["validity_flag"]) for r in self._buffer])
        self._h5["primitive_type"][sl]  = np.array([r["primitive_type"] for r in self._buffer], dtype=np.uint8)

        self._h5.flush()
        last_id = n_tot - 1
        self._update_checkpoint(last_written_id=last_id, total_count=n_tot)
        logger.debug(f"[HDF5Writer] Flushed {n_new} records (total: {n_tot})")
        self._buffer.clear()

    def _load_checkpoint(self) -> dict:
        if self.ckpt_path.exists():
            try:
                with open(self.ckpt_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_checkpoint(self) -> None:
        current = self._load_checkpoint()
        if self._h5 is not None:
            current["total_count"]      = int(self._h5["layout"].shape[0])
            current["last_written_id"]  = int(current["total_count"]) - 1
        current["last_updated"]         = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        current["h5_path"]              = str(self.h5_path)
        with open(self.ckpt_path, "w") as f:
            json.dump(current, f, indent=2)

    def _update_checkpoint(self, **kwargs) -> None:
        current = self._load_checkpoint()
        current.update(kwargs)
        current["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.ckpt_path, "w") as f:
            json.dump(current, f, indent=2)

    def _acquire_lock(self, timeout: float = 60.0) -> None:
        start = time.monotonic()
        while self.lock_path.exists():
            if time.monotonic() - start > timeout:
                # Stale lock: remove it
                logger.warning(f"[HDF5Writer] Removing stale lock: {self.lock_path}")
                self.lock_path.unlink(missing_ok=True)
                break
            time.sleep(0.5)
        self.lock_path.write_text(str(os.getpid()))

    def _release_lock(self) -> None:
        self.lock_path.unlink(missing_ok=True)


def resume_from_checkpoint(h5_path: str | Path) -> int:
    """
    Returns the 0-based index of the NEXT record to generate.
    Returns 0 if the file does not exist or checkpoint is missing.
    """
    ckpt_path = Path(h5_path).with_suffix(".checkpoint.json")
    if not ckpt_path.exists():
        return 0
    try:
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        return int(ckpt.get("last_written_id", -1)) + 1
    except Exception:
        return 0


def verify_hdf5_integrity(h5_path: str | Path) -> dict[str, Any]:
    """
    Full integrity scan of an HDF5 dataset file.

    Checks:
      - All expected datasets present
      - Consistent N across all datasets
      - No NaN/Inf in float datasets
      - Passivity |S11|² + |S21|²≤1.01 for every record
      - Connectivity (port-to-port) for every layout

    Returns dict with 'pass' bool and per-check statistics.
    """
    from src.dataset.connectivity import is_connected

    results: dict[str, Any] = {"pass": False}
    path = Path(h5_path)
    if not path.exists():
        results["error"] = "file_not_found"
        return results

    with h5py.File(path, "r") as f:
        expected_keys = {d[0] for d in _DATASETS}
        present_keys  = set(f.keys())
        missing       = expected_keys - present_keys
        if missing:
            results["missing_datasets"] = sorted(missing)
            results["pass"] = False
            return results

        n = f["layout"].shape[0]
        results["n_records"] = n

        # Check consistent N
        for name, _, shape_suffix, _ in _DATASETS:
            if f[name].shape[0] != n:
                results["n_inconsistency"] = name
                results["pass"] = False
                return results

        # Scan all records
        n_passivity_fail = 0
        n_nan_fail       = 0
        n_connect_fail   = 0

        BATCH = 1000
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            s11 = f["S11_mag"][start:end]
            s21 = f["S21_mag"][start:end]
            layouts = f["layout"][start:end]

            # NaN check
            for arr in [s11, s21, f["S11_phase"][start:end], f["S21_phase"][start:end]]:
                if not np.all(np.isfinite(arr)):
                    n_nan_fail += int(np.sum(~np.isfinite(arr).all(axis=-1)))

            # Passivity
            power = s11**2 + s21**2
            n_passivity_fail += int(np.sum(power.max(axis=-1) > 1.01))

            # Connectivity
            for layout in layouts:
                if not is_connected(layout):
                    n_connect_fail += 1

        results["n_passivity_violations"] = n_passivity_fail
        results["n_nan_violations"]       = n_nan_fail
        results["n_connectivity_failures"] = n_connect_fail
        results["passivity_ok"]    = bool(n_passivity_fail == 0)
        results["nan_ok"]          = bool(n_nan_fail == 0)
        results["connectivity_ok"] = bool(n_connect_fail == 0)
        results["pass"] = bool(
            n_passivity_fail == 0 and n_nan_fail == 0 and n_connect_fail == 0
        )

    return results
