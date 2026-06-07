"""
src/evaluation/em_verify.py
============================
Phase 6 — OpenEMS EM verification of generated layouts.

Loads layouts saved by full_eval.py, runs true FDTD simulation on a
strategic subset, and reports:
  - EM-verified S21 MSE  (true physical accuracy)
  - Surrogate S21 MSE   (proxy used during training/eval)
  - Surrogate-EM gap    (how faithful the proxy is)

The subset is chosen to be representative:
  - N_pixel  layouts from PIXEL (guided)
  - N_other  layouts from each other method
  - Layouts are sorted by surrogate S21 MSE so the selection
    covers the full performance range (not just random).

Runs OpenEMS simulations in parallel with multiprocessing.Pool.
Each worker opens its own OpenEMS context — no shared state needed.

Usage:
    python -m src.evaluation.em_verify \\
        --eval-dir experiments/full_eval_v2 \\
        --n-pixel 300 \\
        --n-other 100 \\
        --workers 8

Pilot mode (calibrate timing first):
    python -m src.evaluation.em_verify \\
        --eval-dir experiments/full_eval_v2 \\
        --n-pixel 10 --n-other 5 --workers 4 --pilot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Per-worker simulation function (top-level so picklable by multiprocessing)
# ---------------------------------------------------------------------------

def _simulate_one(args):
    """
    Simulate one layout.  Called in a worker process.

    args = (idx, layout_np, substrate_id, y_star_np, sim_tmp_dir)

    Returns dict with:
        idx, em_s21_mse, em_s11_mse, surr_s21_mse, elapsed, error
    """
    idx, layout_np, substrate_id, y_star_np, sim_tmp_dir = args

    # Import here (worker process) — avoids pickling issues
    from src.dataset.openems_wrapper import simulate, FREQS
    import math as _math

    t0 = time.monotonic()
    try:
        result = simulate(
            layout       = layout_np,
            meta         = {"type": 0},
            substrate_id = int(substrate_id),
            freqs        = FREQS,
            sim_base_dir = sim_tmp_dir,
        )
        elapsed = time.monotonic() - t0

        # S21 MSE against target (using magnitude only, normalised to [0,1])
        s21_em  = result["s21_mag"].astype(np.float32)       # (100,)
        s11_em  = result["s11_mag"].astype(np.float32)

        # y_star_np shape: (4, 100) → channels: [s11_mag, s21_mag, s11_phase/π, s21_phase/π]
        s21_tgt = y_star_np[1].astype(np.float32)            # channel 1
        s11_tgt = y_star_np[0].astype(np.float32)

        em_s21_mse = float(np.mean((s21_em - s21_tgt) ** 2))
        em_s11_mse = float(np.mean((s11_em - s11_tgt) ** 2))

        return {
            "idx":         int(idx),
            "em_s21_mse":  em_s21_mse,
            "em_s11_mse":  em_s11_mse,
            "elapsed":     float(elapsed),
            "error":       None,
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        return {
            "idx":         int(idx),
            "em_s21_mse":  float("nan"),
            "em_s11_mse":  float("nan"),
            "elapsed":     float(elapsed),
            "error":       str(exc),
        }


# ---------------------------------------------------------------------------
# Surrogate scoring (single-process, GPU)
# ---------------------------------------------------------------------------

def surrogate_s21_mse_batch(
    layouts_np: np.ndarray,     # (N, 15, 15)  int8/int64
    y_star_np:  np.ndarray,     # (N, 4, 100)
    eval_dir:   Path,
) -> np.ndarray:
    """
    Score a batch of layouts with the surrogate ensemble.
    Returns per-layout surrogate S21 MSE shape (N,).
    """
    from src.utils.config import load_config
    from src.models.surrogate import PhysicsSurrogate, SurrogateEnsemble
    from src.training.train_surrogate import _build_port_map
    import torch

    cfg = load_config("experiments/configs/base_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scfg = cfg.surrogate

    surr_models = []
    for k in range(scfg.ensemble_size):
        m = PhysicsSurrogate(in_ch=scfg.in_channels, base_ch=scfg.base_channels,
                             n_freq=cfg.dataset.n_freq)
        ck = torch.load(f"experiments/surrogate_v1/surrogate_k{k}_best.pt",
                        map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
        surr_models.append(m.to(device).eval())
    surr_ens = SurrogateEnsemble(surr_models)
    port_map = _build_port_map(cfg).to(device)

    layouts_t = torch.from_numpy(layouts_np.astype(np.int64))
    y_star_t  = torch.from_numpy(y_star_np.astype(np.float32))

    per_layout = []
    BATCH = 256
    with torch.no_grad():
        for i in range(0, len(layouts_t), BATCH):
            lb = layouts_t[i:i+BATCH].to(device)
            yb = y_star_t[i:i+BATCH].to(device)
            # Surrogate expects (B, 2, 15, 15): Ch0=binary layout float, Ch1=port_map
            # Layout values: 0=empty, 1=conductor (no MASK tokens at inference)
            layout_bin = (lb == 1).float().unsqueeze(1)   # (B, 1, 15, 15)
            inp = torch.cat([layout_bin, port_map.unsqueeze(0).expand(len(lb), -1, -1, -1)], dim=1)
            mean_pred, _ = surr_ens(inp)     # (B, 4, 100)
            s21_pred = mean_pred[:, 1, :]    # magnitude, channel 1
            s21_tgt  = yb[:, 1, :]
            mse = ((s21_pred - s21_tgt) ** 2).mean(dim=1).cpu().numpy()
            per_layout.append(mse)

    return np.concatenate(per_layout)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir",  default="experiments/full_eval_v2",
                        help="Directory containing full_eval outputs")
    parser.add_argument("--out-dir",   default="experiments/em_verify_v1")
    parser.add_argument("--n-pixel",   type=int, default=300,
                        help="Layouts to verify from PIXEL (guided)")
    parser.add_argument("--n-other",   type=int, default=100,
                        help="Layouts to verify from each other method")
    parser.add_argument("--workers",   type=int, default=8,
                        help="Parallel OpenEMS workers")
    parser.add_argument("--pilot",     action="store_true",
                        help="Pilot mode: use n-pixel=10, n-other=5, 4 workers")
    args = parser.parse_args()

    if args.pilot:
        args.n_pixel  = 10
        args.n_other  = 5
        args.workers  = 4

    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_tmp  = Path("data/tmp/em_verify")
    sim_tmp.mkdir(parents=True, exist_ok=True)

    print(f"[em_verify] eval_dir={eval_dir}  n_pixel={args.n_pixel}  "
          f"n_other={args.n_other}  workers={args.workers}", flush=True)

    # ── Load shared data ────────────────────────────────────────────────────
    y_star_np = np.load(eval_dir / "y_star.npy").astype(np.float32)   # (N, 4, 100)
    test_idx  = np.load(eval_dir / "test_idx.npy")                    # (N,)
    N_total   = len(y_star_np)

    # Load substrate IDs from HDF5 if available (default 0 = Rogers4003C)
    from src.utils.config import load_config
    cfg = load_config("experiments/configs/base_config.yaml")
    h5_path = cfg.paths.raw_data + "/pixel_dataset.h5"
    import h5py
    with h5py.File(h5_path, "r") as f:
        if "substrate_id" in f:
            substrate_ids = f["substrate_id"][test_idx].astype(np.int32)
        else:
            substrate_ids = np.zeros(N_total, dtype=np.int32)

    # Methods to verify (key → n_layouts)
    verify_plan = {"pixel_guided": args.n_pixel}
    for key in ["cfg_only", "det_cnn", "cvae"]:
        lpath = eval_dir / f"layouts_{key}.npy"
        if lpath.exists():
            verify_plan[key] = args.n_other

    # ── Per-method surrogate scoring + EM verification ─────────────────────
    all_results = {}

    for key, n_verify in verify_plan.items():
        lpath = eval_dir / f"layouts_{key}.npy"
        if not lpath.exists():
            print(f"[skip] {key} — layouts file not found")
            continue

        layouts_all = np.load(lpath).astype(np.int64)   # (N_total, 15, 15)
        print(f"\n[{key}] Scoring surrogate MSE for all {N_total} layouts …", flush=True)

        # Compute surrogate MSE for all layouts → stratified selection
        surr_mse_all = surrogate_s21_mse_batch(layouts_all, y_star_np, eval_dir)

        # Select n_verify indices: spread across the MSE distribution (percentiles)
        pct = np.linspace(0, 100, n_verify, endpoint=True)
        sorted_order = np.argsort(surr_mse_all)
        sel_positions = np.round(np.percentile(np.arange(N_total), pct)).astype(int)
        sel_positions = np.clip(sel_positions, 0, N_total - 1)
        sel_idx = sorted_order[sel_positions]
        sel_idx = np.unique(sel_idx)[:n_verify]   # deduplicate, cap at n_verify

        print(f"[{key}] Running OpenEMS on {len(sel_idx)} layouts "
              f"with {args.workers} workers …", flush=True)
        print(f"[{key}] Surrogate MSE range: "
              f"{surr_mse_all[sel_idx].min():.5f} – {surr_mse_all[sel_idx].max():.5f}",
              flush=True)

        # Build worker args
        sim_args = [
            (i, layouts_all[i].astype(np.uint8), substrate_ids[i],
             y_star_np[i], str(sim_tmp))
            for i in sel_idx
        ]

        # Run in parallel
        import multiprocessing as mp
        t0 = time.time()
        with mp.Pool(processes=args.workers) as pool:
            sim_results = pool.map(_simulate_one, sim_args)
        elapsed_total = time.time() - t0

        # Aggregate
        em_mses    = [r["em_s21_mse"] for r in sim_results if r["error"] is None]
        surr_mses  = [float(surr_mse_all[r["idx"]]) for r in sim_results if r["error"] is None]
        times      = [r["elapsed"] for r in sim_results if r["error"] is None]
        errors     = [r for r in sim_results if r["error"] is not None]

        n_ok       = len(em_mses)
        n_err      = len(errors)
        mean_em    = float(np.mean(em_mses))   if em_mses   else float("nan")
        mean_surr  = float(np.mean(surr_mses)) if surr_mses else float("nan")
        gap        = float(mean_em - mean_surr) if (em_mses and surr_mses) else float("nan")
        mean_t     = float(np.mean(times))     if times     else float("nan")

        print(f"\n[{key}] Results ({n_ok}/{len(sel_idx)} successful, {n_err} failed):")
        print(f"  EM-verified S21 MSE:  {mean_em:.6f}")
        print(f"  Surrogate  S21 MSE:   {mean_surr:.6f}")
        print(f"  Surrogate-EM gap:     {gap:+.6f}")
        print(f"  Mean sim time/layout: {mean_t:.1f}s")
        print(f"  Total wall time:      {elapsed_total:.1f}s", flush=True)

        all_results[key] = {
            "n_total":          N_total,
            "n_verified":       n_ok,
            "n_failed":         n_err,
            "em_s21_mse_mean":  mean_em,
            "em_s21_mse_std":   float(np.std(em_mses)) if em_mses else float("nan"),
            "surr_s21_mse_mean":mean_surr,
            "surr_s21_mse_std": float(np.std(surr_mses)) if surr_mses else float("nan"),
            "surrogate_em_gap": gap,
            "mean_sim_time_s":  mean_t,
            "per_layout":       [
                {"idx": r["idx"], "em_s21_mse": r["em_s21_mse"],
                 "surr_s21_mse": float(surr_mse_all[r["idx"]]),
                 "elapsed": r["elapsed"], "error": r["error"]}
                for r in sim_results
            ],
            "errors":           [r["error"] for r in errors],
        }

        # Save after each method so partial results survive a kill
        out_path = out_dir / "em_verify_summary.json"
        with open(out_path, "w") as fp:
            json.dump(all_results, fp, indent=2)
        print(f"[saved] {out_path}", flush=True)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EM VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'N_ok':>6} {'EM S21 MSE':>12} {'Surr MSE':>12} {'Gap':>10} {'t/sim':>8}")
    print("-" * 70)
    for k, r in all_results.items():
        print(f"  {k:<23} {r['n_verified']:>6} "
              f"{r['em_s21_mse_mean']:>12.6f} "
              f"{r['surr_s21_mse_mean']:>12.6f} "
              f"{r['surrogate_em_gap']:>+10.6f} "
              f"{r['mean_sim_time_s']:>7.1f}s")
    print("=" * 70)
    print(f"\n[done] Full results: {out_dir/'em_verify_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
