"""
src/evaluation/phase7_em.py
============================
Phase 7 — EM verification of K layouts per spec.

Runs OpenEMS FDTD on all K×N layouts from phase7_eval.py outputs.
Results are saved per-method as (N, K) EM MSE arrays plus a summary JSON.

Uses spawn multiprocessing to avoid fork+PyTorch VSZ blowup.

Usage:
    python -m src.evaluation.phase7_em \\
        --gen-dir experiments/phase7/gen \\
        --out-dir experiments/phase7/em  \\
        --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Top-level worker function (must be picklable for spawn)
# ---------------------------------------------------------------------------

def _simulate_one(args):
    """
    args = (flat_id, spec_i, k_i, layout_np, substrate_id, y_star_np, sim_tmp_dir)
    Returns dict with flat_id, spec_i, k_i, em_s21_mse, em_s11_mse, elapsed, error.
    """
    flat_id, spec_i, k_i, layout_np, substrate_id, y_star_np, sim_tmp_dir = args

    from src.dataset.openems_wrapper import simulate, FREQS
    import time as _time

    t0 = _time.monotonic()
    try:
        result = simulate(
            layout       = layout_np,
            meta         = {"type": 0},
            substrate_id = int(substrate_id),
            freqs        = FREQS,
            sim_base_dir = sim_tmp_dir,
        )
        elapsed = _time.monotonic() - t0

        s21_em  = result["s21_mag"].astype(np.float32)
        s11_em  = result["s11_mag"].astype(np.float32)
        s21_tgt = y_star_np[1].astype(np.float32)
        s11_tgt = y_star_np[0].astype(np.float32)

        return {
            "flat_id":    int(flat_id),
            "spec_i":     int(spec_i),
            "k_i":        int(k_i),
            "em_s21_mse": float(np.mean((s21_em - s21_tgt) ** 2)),
            "em_s11_mse": float(np.mean((s11_em - s11_tgt) ** 2)),
            "elapsed":    float(elapsed),
            "error":      None,
        }
    except Exception as exc:
        return {
            "flat_id":    int(flat_id),
            "spec_i":     int(spec_i),
            "k_i":        int(k_i),
            "em_s21_mse": float("nan"),
            "em_s11_mse": float("nan"),
            "elapsed":    float(_time.monotonic() - t0),
            "error":      str(exc),
        }


# ---------------------------------------------------------------------------
# Build sim_args list from (N, K, 15, 15) layouts
# ---------------------------------------------------------------------------

def _build_args(layouts_nk, y_star_np, substrate_ids, sim_tmp_dir, offset=0):
    """
    layouts_nk    : (N, K, 15, 15) np.ndarray  int8/int64
    y_star_np     : (N, 4, 100)    np.ndarray  float32
    substrate_ids : (N,)           np.ndarray  int32
    offset        : flat_id offset to keep IDs unique across calls
    Returns list of args tuples.
    """
    N, K = layouts_nk.shape[:2]
    args_list = []
    for i in range(N):
        for k in range(K):
            flat_id = offset + i * K + k
            args_list.append((
                flat_id,
                i, k,
                layouts_nk[i, k].astype(np.uint8),
                int(substrate_ids[i]),
                y_star_np[i],
                str(sim_tmp_dir),
            ))
    return args_list


# ---------------------------------------------------------------------------
# Run EM and aggregate to (N, K) arrays
# ---------------------------------------------------------------------------

def run_em_nk(sim_args, n_workers, N, K):
    """
    Run EM on all sim_args, return (N, K) em_s21_mse and metadata.
    """
    import gc
    import multiprocessing as mp
    gc.collect()
    mp_ctx = mp.get_context('spawn')
    t0 = time.time()
    with mp_ctx.Pool(processes=n_workers) as pool:
        raw = pool.map(_simulate_one, sim_args)
    elapsed = time.time() - t0

    em_mse  = np.full((N, K), np.nan, dtype=np.float32)
    sim_t   = np.full((N, K), np.nan, dtype=np.float32)
    errors  = []
    for r in raw:
        i, k = r["spec_i"], r["k_i"]
        if r["error"] is None:
            em_mse[i, k] = r["em_s21_mse"]
            sim_t[i, k]  = r["elapsed"]
        else:
            errors.append(r)

    n_ok  = int(np.sum(~np.isnan(em_mse)))
    n_err = len(errors)
    print(f"  Finished {n_ok}/{N*K} sims  ({n_err} errors)  "
          f"wall={elapsed:.0f}s  mean_t={np.nanmean(sim_t):.1f}s", flush=True)
    return em_mse, sim_t, errors, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", default="experiments/phase7/gen")
    parser.add_argument("--out-dir", default="experiments/phase7/em")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    gen_dir = Path(args.gen_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_tmp = Path("data/tmp/phase7_em")
    sim_tmp.mkdir(parents=True, exist_ok=True)

    print(f"[phase7_em] gen_dir={gen_dir}  workers={args.workers}", flush=True)

    y_star_bk  = np.load(gen_dir / "y_star_bk.npy").astype(np.float32)   # (N, 4, 100)
    y_star_div = np.load(gen_dir / "y_star_div.npy").astype(np.float32)   # (n_div, 4, 100)
    sub_bk     = np.load(gen_dir / "substrate_ids_bk.npy")
    sub_div    = np.load(gen_dir / "substrate_ids_div.npy")
    N, n_div   = len(y_star_bk), len(y_star_div)
    print(f"[data] N_bk={N}  N_div={n_div}", flush=True)

    summary = {}
    summary_path = out_dir / "em_results_p7.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"[resume] partial results: {list(summary.keys())}", flush=True)

    # ── Best-of-K methods ─────────────────────────────────────────────────
    method_files = {
        "pixel":   gen_dir / f"layouts_pixel_k5.npy",
        "cfg":     gen_dir / f"layouts_cfg_k5.npy",
        "cvae":    gen_dir / f"layouts_cvae_k5.npy",
        "det_cnn": gen_dir / f"layouts_det_k1.npy",
    }

    flat_offset = 0
    for key, lpath in method_files.items():
        if key in summary:
            print(f"[skip] {key} already in results", flush=True)
            # advance flat_offset
            if lpath.exists():
                lshape = np.load(lpath, mmap_mode='r').shape
                flat_offset += lshape[0] * lshape[1]
            continue
        if not lpath.exists():
            print(f"[skip] {lpath} not found", flush=True)
            continue

        layouts = np.load(lpath).astype(np.int64)   # (N, K, 15, 15)
        Nl, K   = layouts.shape[:2]
        print(f"\n[{key}] N={Nl}  K={K}  total_sims={Nl*K} …", flush=True)

        sim_args = _build_args(layouts, y_star_bk, sub_bk, sim_tmp, offset=flat_offset)
        em_mse, sim_t, errors, wall = run_em_nk(sim_args, args.workers, Nl, K)
        flat_offset += Nl * K

        # Best-of-K
        best_em = np.nanmin(em_mse, axis=1)   # (N,)

        summary[key] = {
            "N": int(Nl), "K": int(K),
            "em_mse_nk":    em_mse.tolist(),
            "best_em_k":    best_em.tolist(),
            "mean_sim_t":   float(np.nanmean(sim_t)),
            "wall_s":       float(wall),
            "n_errors":     len(errors),
            "error_msgs":   [e["error"] for e in errors[:5]],
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] {summary_path}", flush=True)

    # ── Within-spec diversity ─────────────────────────────────────────────
    div_files = {
        "pixel_div": gen_dir / f"layouts_pixel_k20_div.npy",
        "cvae_div":  gen_dir / f"layouts_cvae_k20_div.npy",
    }

    for key, lpath in div_files.items():
        if key in summary:
            print(f"[skip] {key} already in results", flush=True)
            continue
        if not lpath.exists():
            print(f"[skip] {lpath} not found", flush=True)
            continue

        layouts = np.load(lpath).astype(np.int64)   # (n_div, K, 15, 15)
        Nd, K   = layouts.shape[:2]
        print(f"\n[{key}] N_div={Nd}  K={K}  total_sims={Nd*K} …", flush=True)

        sim_args = _build_args(layouts, y_star_div, sub_div, sim_tmp,
                               offset=flat_offset)
        em_mse, sim_t, errors, wall = run_em_nk(sim_args, args.workers, Nd, K)
        flat_offset += Nd * K

        # Pairwise Hamming within each spec's K layouts
        layouts_long = layouts.astype(np.int64)   # (Nd, K, 15, 15)
        hamming_within = []
        for i in range(Nd):
            dists = []
            for ka in range(K):
                for kb in range(ka + 1, K):
                    d = int(np.sum(layouts_long[i, ka] != layouts_long[i, kb]))
                    dists.append(d)
            hamming_within.append(float(np.mean(dists)) if dists else 0.0)

        summary[key] = {
            "N_div": int(Nd), "K": int(K),
            "em_mse_nk":       em_mse.tolist(),
            "best_em_k":       np.nanmin(em_mse, axis=1).tolist(),
            "hamming_within":  hamming_within,
            "mean_hamming":    float(np.mean(hamming_within)),
            "mean_sim_t":      float(np.nanmean(sim_t)),
            "wall_s":          float(wall),
            "n_errors":        len(errors),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[saved] {summary_path}", flush=True)

    # ── Final summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 7 EM SUMMARY")
    print("=" * 70)
    for key, v in summary.items():
        if "em_mse_nk" not in v:
            continue
        bk   = np.array(v["best_em_k"])
        ok   = np.sum(~np.isnan(bk))
        print(f"  {key:<15} N={v.get('N', v.get('N_div',0))} K={v['K']}")
        print(f"    Best-of-K mean: {np.nanmean(bk):.6f}  "
              f"< 0.001: {100*np.mean(bk<0.001):.1f}%  "
              f"< 0.010: {100*np.mean(bk<0.010):.1f}%")
        if "hamming_within" in v:
            print(f"    Mean intra-spec Hamming: {v['mean_hamming']:.2f} bits")
    print("=" * 70)
    print(f"\n[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
