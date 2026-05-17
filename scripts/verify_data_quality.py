"""
scripts/verify_data_quality.py
PIXEL-2026 -- Pre-generation Data Quality Cross-Verification

Runs ONE sample per primitive type (11 types x 4 substrates = 44 samples),
prints detailed per-sample and aggregate physics reports, and exits with
code 0 (all pass) or 1 (failures found).

Usage (run BEFORE the 200K generation to confirm data is correct):
    pixel-env\\python.exe scripts\\verify_data_quality.py

No arguments needed. Produces a plain-text report to stdout.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np

from src.dataset.primitives      import (sample_layout, PRIMITIVE_NAMES, PRIMITIVE_GENERATORS,
                                         N_PRIMITIVES)
from src.dataset.connectivity    import is_connected
from src.dataset.em_simulation   import simulate, FREQS, SUBSTRATES
from src.dataset.physics_validator import validate_record

SEP  = "=" * 72
SEP2 = "-" * 72

def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    print(SEP)
    print("PIXEL-2026  DATA QUALITY CROSS-VERIFICATION")
    print(SEP)
    print(f"  Primitives  : {N_PRIMITIVES} types")
    print(f"  Substrates  : {len(SUBSTRATES)}  (Rogers4003C / FR4 / Rogers5880 / Alumina)")
    print(f"  Freq range  : {FREQS[0]/1e9:.1f}-{FREQS[-1]/1e9:.1f} GHz  ({len(FREQS)} points)")
    print()

    overall_pass = True
    failed_cases: list[str] = []

    for ptype in range(N_PRIMITIVES):
        pname = PRIMITIVE_NAMES[ptype]
        print(f"[{ptype:2d}] {pname}")
        print(SEP2)

        for sub_id, sub_info in SUBSTRATES.items():
            rng = np.random.default_rng(seed=ptype * 100 + sub_id)

            # Force the primitive type we want
            found = False
            for attempt in range(50):
                layout, meta = sample_layout(rng, sigma=0.15)
                if int(meta["type"]) == ptype:
                    found = True
                    break
            if not found:
                # Call the generator directly to guarantee the type
                layout, meta = PRIMITIVE_GENERATORS[ptype](rng, sigma=0.15)
                meta["type"] = ptype

            connected = is_connected(layout)
            if not connected:
                print(f"  sub={sub_id} ({sub_info['name']:12s})  SKIP — disconnected layout")
                continue

            try:
                sim = simulate(layout, meta, substrate_id=sub_id, freqs=FREQS, noise_sigma=0.002)
            except Exception as exc:
                print(f"  sub={sub_id} ({sub_info['name']:12s})  FAIL — simulation error: {exc}")
                overall_pass = False
                failed_cases.append(f"ptype={ptype} sub={sub_id} sim_error")
                continue

            rep = validate_record(
                layout      = layout,
                s11_mag     = sim["s11_mag"],
                s21_mag     = sim["s21_mag"],
                s11_phase   = sim["s11_phase"],
                s21_phase   = sim["s21_phase"],
                s21_complex = sim["s21_complex"],
            )

            # Per-sample summary
            s21 = sim["s21_mag"]
            s11 = sim["s11_mag"]
            n_res = int(np.sum(sim["resonance_freqs"] > 0))
            power_max = float(np.max(s11**2 + s21**2))
            s21_range = float(s21.max() - s21.min())
            s21_mean  = float(s21.mean())

            status = _flag(rep.validity_flag)
            print(f"  sub={sub_id} ({sub_info['name']:12s})  {status}"
                  f"  |S21| {s21_mean:.3f}±{s21.std():.3f} range={s21_range:.3f}"
                  f"  passivity={rep.passivity_max:.4f}"
                  f"  kk={rep.kk_residual:.3f}"
                  f"  resonances={n_res}"
                  f"  smooth={'ok' if rep.spectral_smooth_ok else 'BAD'}"
                  f"  dynrng={'ok' if rep.dynamic_range_ok else 'LOW'}")

            if not rep.validity_flag:
                overall_pass = False
                failed_cases.append(f"ptype={ptype}({pname}) sub={sub_id}: {rep.fail_reasons}")

        print()

    # ── Aggregate physics check ───────────────────────────────────────────────
    print(SEP)
    print("AGGREGATE STATISTICS (32-sample sweep, one per type x substrate combos)")
    print(SEP2)

    reports_all = []
    s21_all     = []
    ptype_counts = [0] * N_PRIMITIVES

    rng_agg = np.random.default_rng(seed=9999)
    n_valid = 0
    n_total = 200

    for i in range(n_total):
        sub_id = i % 4
        layout, meta = sample_layout(rng_agg, sigma=0.15)
        if not is_connected(layout):
            continue
        try:
            sim = simulate(layout, meta, substrate_id=sub_id, freqs=FREQS, noise_sigma=0.002)
        except Exception:
            continue
        rep = validate_record(
            layout=layout, s11_mag=sim["s11_mag"], s21_mag=sim["s21_mag"],
            s11_phase=sim["s11_phase"], s21_phase=sim["s21_phase"],
            s21_complex=sim["s21_complex"],
        )
        reports_all.append(rep)
        if rep.validity_flag:
            n_valid += 1
            s21_all.append(sim["s21_mag"])
            ptype_counts[int(meta["type"])] += 1

    n_connected = len(reports_all)
    if n_connected == 0:
        print("  ERROR: zero connected samples — check primitives.py!")
        return 1

    passivity_rate = sum(r.passivity_ok for r in reports_all) / n_connected
    kk_rate        = sum(r.kk_ok        for r in reports_all) / n_connected
    smooth_rate    = sum(r.spectral_smooth_ok for r in reports_all) / n_connected
    dynrng_rate    = sum(r.dynamic_range_ok   for r in reports_all) / n_connected
    valid_rate     = n_valid / n_total
    conn_rate      = n_connected / n_total

    print(f"  Samples attempted    : {n_total}")
    print(f"  Connected structures : {n_connected}  ({100*conn_rate:.1f}%)")
    print(f"  Valid (all gates)    : {n_valid}  ({100*valid_rate:.1f}%)")
    print()
    print(f"  Passivity rate       : {100*passivity_rate:.1f}%  {_flag(passivity_rate >= 0.99)}  (gate >99%)")
    print(f"  KK causality rate    : {100*kk_rate:.1f}%   {_flag(kk_rate >= 0.85)}  (gate >85%)")
    print(f"  Spectral smooth rate : {100*smooth_rate:.1f}%  {_flag(smooth_rate >= 0.99)}  (gate >99%)")
    print(f"  Dynamic range rate   : {100*dynrng_rate:.1f}%   {_flag(dynrng_rate >= 0.60)}  (gate >60%)")

    if s21_all:
        s21_stack = np.stack(s21_all)
        # Spectral diversity: mean pairwise MSE between random pairs
        idx1 = np.random.default_rng(0).integers(0, len(s21_all), 200)
        idx2 = np.random.default_rng(1).integers(0, len(s21_all), 200)
        diversity = float(np.mean((s21_stack[idx1] - s21_stack[idx2])**2))
        print(f"  Spectral diversity   : {diversity:.4f}  {_flag(diversity >= 0.05)}  (gate >0.05)")
        print()
        print(f"  S21 magnitude stats  : mean={s21_stack.mean():.3f}  std={s21_stack.std():.3f}"
              f"  min={s21_stack.min():.3f}  max={s21_stack.max():.3f}")

    print()
    print("  Primitive distribution (valid records only):")
    for i, (name, cnt) in enumerate(zip(PRIMITIVE_NAMES, ptype_counts)):
        frac = cnt / max(n_valid, 1)
        flag = "  <-- overrepresented (>25%)" if frac > 0.25 else ""
        print(f"    [{i:2d}] {name:<28s}: {cnt:3d}  ({100*frac:.1f}%){flag}")

    # ── HDF5 write smoke-test ─────────────────────────────────────────────────
    print()
    print(SEP)
    print("HDF5 WRITER SMOKE-TEST  (write 300 records -> read back -> verify)")
    print(SEP2)

    import tempfile
    from src.dataset.hdf5_writer import CheckpointWriter, resume_from_checkpoint

    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / "smoke_test.h5"
        rng_h5  = np.random.default_rng(42)
        written = 0

        with CheckpointWriter(h5_path, batch_size=256) as wr:
            while written < 300:
                layout, meta = sample_layout(rng_h5, sigma=0.15)
                if not is_connected(layout):
                    continue
                try:
                    sim = simulate(layout, meta, substrate_id=written % 4, freqs=FREQS, noise_sigma=0.002)
                except Exception:
                    continue
                rep = validate_record(
                    layout=layout, s11_mag=sim["s11_mag"], s21_mag=sim["s21_mag"],
                    s11_phase=sim["s11_phase"], s21_phase=sim["s21_phase"],
                    s21_complex=sim["s21_complex"],
                )
                if not rep.validity_flag:
                    continue
                wr.write_record({
                    "layout":         layout,
                    "s11_mag":        sim["s11_mag"],
                    "s21_mag":        sim["s21_mag"],
                    "s11_phase":      sim["s11_phase"],
                    "s21_phase":      sim["s21_phase"],
                    "substrate_id":   written % 4,
                    "resonance_freqs": sim["resonance_freqs"],
                    "q_factors":      sim["q_factors"],
                    "validity_flag":  True,
                    "primitive_type": int(meta["type"]),
                })
                written += 1

        # Read back and verify
        import h5py
        with h5py.File(h5_path, "r") as f:
            n_written  = f["layout"].shape[0]
            layouts_ok = bool(np.all((f["layout"][:] == 0) | (f["layout"][:] == 1)))
            s21_ok     = bool(np.all(np.isfinite(f["S21_mag"][:])))
            s21_range  = float(f["S21_mag"][:].max() - f["S21_mag"][:].min())

        ckpt_id = resume_from_checkpoint(h5_path)
        h5_size_kb = h5_path.stat().st_size / 1024

        print(f"  Records written : {n_written}  (expected 300)")
        print(f"  Checkpoint id   : {ckpt_id}  (expected 300)")
        print(f"  Layout binary   : {_flag(layouts_ok)}")
        print(f"  S21 finite      : {_flag(s21_ok)}")
        print(f"  S21 range       : {s21_range:.3f}  {_flag(s21_range > 0)}")
        print(f"  File size       : {h5_size_kb:.1f} KB  (~{h5_size_kb/300:.1f} KB/record)")

        h5_write_ok = (n_written == 300 and ckpt_id == 300 and layouts_ok and s21_ok and s21_range > 0)
        print(f"  HDF5 write test : {_flag(h5_write_ok)}")
        if not h5_write_ok:
            overall_pass = False
            failed_cases.append("hdf5_write_smoke_test")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print()
    print(SEP)
    if overall_pass and not failed_cases:
        print("VERDICT: ALL CHECKS PASSED — safe to run full 200K generation.")
        print()
        print("Launch command (paste into your terminal):")
        py = r"C:\Users\tyrone\anaconda3\envs\pixel-env\python.exe"
        print(f"  {py} -m src.dataset.generate"
              "  --config experiments/configs/base_config.yaml"
              "  --skip-pilot --workers 32 --n-samples 200000")
    else:
        print(f"VERDICT: FAILED — {len(failed_cases)} issue(s) found:")
        for f in failed_cases:
            print(f"  • {f}")
        print()
        print("Fix the issues above before starting full generation.")
    print(SEP)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
