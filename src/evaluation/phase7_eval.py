"""
src/evaluation/phase7_eval.py
==============================
Phase 7 — Best-of-K and Within-Spec Diversity Evaluation.

Generates K independent layouts per spec from each model using
different random seeds. Outputs are saved to experiments/phase7/gen/
for subsequent EM verification by phase7_em.py.

Best-of-K (N=100 specs, K=5):
  PIXEL, cVAE, CFG-only → 5 layouts per spec
  Det-CNN               → 1 layout per spec (deterministic)

Within-spec diversity (N=20 hardest specs, K=20):
  PIXEL, cVAE → 20 layouts per spec

Usage:
    python -m src.evaluation.phase7_eval \\
        --config experiments/configs/base_config.yaml \\
        --out-dir experiments/phase7/gen \\
        --n-specs 100 --k-best 5 --n-div 20 --k-div 20
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from src.evaluation.baselines import DetCNN, cVAE
from src.evaluation.full_eval import (
    generate_in_chunks, load_baselines, load_pixel_models, _guided_chunk,
)
from src.guidance.physics_guidance import generate_guided
from src.training.train_denoiser import generate_samples
from src.training.train_surrogate import _build_port_map
from src.utils.config import load_config

# Per-method seed offsets prevent identical RNG state when same seed is used.
# PIXEL and CFG-only share the same denoiser + same multinomial call site —
# without an offset, torch.manual_seed(s) gives byte-identical layouts because
# the guidance gradient (~1e-5 logit shift) is too small to flip any multinomial
# sample. Offset of 200 000 keeps seeds reproducible and well-separated.
SEEDS_K5          = [0, 42, 123, 456, 789]
SEEDS_K5_CFG      = [s + 200_000 for s in SEEDS_K5]   # CFG-only offset
SEEDS_K20         = [0, 42, 123, 456, 789, 1000, 1001, 1002, 1003, 1004,
                     2000, 2001, 2002, 2003, 2004, 3000, 3001, 3002, 3003, 3004]
SEEDS_K20_CFG     = [s + 200_000 for s in SEEDS_K20]


# ---------------------------------------------------------------------------
# Spec selection
# ---------------------------------------------------------------------------

def select_fresh_specs(
    y_star_all: np.ndarray,          # (N_test, 4, 100)
    test_idx_all: np.ndarray,         # (N_test,) dataset indices
    phase6_used: set,                 # set of 0-based y_star indices already EM-verified
    n_specs: int,
    n_div: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        spec_idx_bk  : (n_specs,)  0-based indices into y_star_all for best-of-K study
        spec_idx_div : (n_div,)    0-based indices into y_star_all for diversity study
                                   (subset of spec_idx_bk — the n_div most complex targets)
    """
    N_test = len(y_star_all)
    all_idx = np.arange(N_test)
    available = np.array([i for i in all_idx if i not in phase6_used])

    # Uniform random selection for unbiased coverage
    chosen = rng.choice(available, size=min(n_specs, len(available)), replace=False)
    chosen = np.sort(chosen)

    # Diversity study: pick n_div specs with the most complex S21 targets
    # Complexity = std across frequency bins of the S21 magnitude
    s21_std = y_star_all[chosen, 1, :].std(axis=1)   # (n_specs,)
    div_local = np.argsort(s21_std)[-n_div:]           # top-n_div most complex
    div_idx = np.sort(chosen[div_local])

    print(f"[spec_select] N_available={len(available)}  chosen={len(chosen)}  "
          f"diversity={len(div_idx)}", flush=True)
    print(f"  S21 std range (selected): "
          f"[{s21_std.min():.4f}, {s21_std.max():.4f}]  "
          f"diversity specs: [{s21_std[div_local].min():.4f}, {s21_std[div_local].max():.4f}]",
          flush=True)
    return chosen, div_idx


# ---------------------------------------------------------------------------
# K-layout generation helpers
# ---------------------------------------------------------------------------

def gen_pixel_k(y_star, seeds, den, enc, diff, surr_ens, disc, port_map,
                device, dcfg, gcfg, cfg_w, chunk=256) -> torch.Tensor:
    """Generate K PIXEL layouts per spec. Returns (N, K, 15, 15)."""
    all_k = []
    for k, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"  [PIXEL] seed {seed} (k={k+1}/{len(seeds)}) …", flush=True)
        lays = generate_in_chunks(
            lambda yb, s=seed: _guided_chunk(
                yb, den, enc, diff, surr_ens, disc, port_map, device,
                dcfg, gcfg, cfg_w,
                lambda_topo=gcfg.lambda_topo, lambda_mfg=gcfg.lambda_mfg,
                alpha_max=gcfg.alpha_max),
            y_star, chunk=chunk, label=f"PIXEL_k{k+1}")
        all_k.append(lays)
    return torch.stack(all_k, dim=1)   # (N, K, 15, 15)


def gen_cfg_k(y_star, seeds, den, enc, diff, port_map,
              device, dcfg, cfg_w, chunk=256) -> torch.Tensor:
    """Generate K CFG-only layouts per spec. Returns (N, K, 15, 15).

    Seeds should differ from PIXEL's seeds (use SEEDS_K5_CFG) so the two
    methods explore independent random trajectories.
    """
    all_k = []
    for k, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"  [CFG-only] seed {seed} (k={k+1}/{len(seeds)}) …", flush=True)
        lays = generate_in_chunks(
            lambda yb: generate_samples(
                den, enc, diff, port_map,
                n_samples=len(yb), device=device,
                cfg_w=cfg_w, s_params=yb, T_steps=dcfg.T),
            y_star, chunk=chunk, label=f"CFG_k{k+1}")
        all_k.append(lays)
    return torch.stack(all_k, dim=1)


def gen_cvae_k(y_star, K, cvae_model, device, chunk=256) -> torch.Tensor:
    """
    Generate K cVAE layouts per spec in one forward pass (n_samples=K).
    Returns (N, K, 15, 15).
    """
    print(f"  [cVAE] K={K} samples per spec …", flush=True)
    N = len(y_star)
    all_parts = []
    for i in range(0, N, chunk):
        yb = y_star[i:i+chunk].to(device)
        with torch.no_grad():
            samp = cvae_model.sample(yb, n_samples=K)   # (B*K, 15, 15)
        B = len(yb)
        samp = samp.view(B, K, 15, 15).cpu()
        all_parts.append(samp)
    return torch.cat(all_parts, dim=0)   # (N, K, 15, 15)


def gen_det_k1(y_star, det_model, device, chunk=256) -> torch.Tensor:
    """Det-CNN deterministic: K=1 layout per spec. Returns (N, 1, 15, 15)."""
    print(f"  [Det-CNN] deterministic …", flush=True)
    parts = []
    for i in range(0, len(y_star), chunk):
        with torch.no_grad():
            lays = det_model.predict(y_star[i:i+chunk].to(device))
        parts.append(lays.cpu())
    return torch.cat(parts, dim=0).unsqueeze(1)   # (N, 1, 15, 15)


# ---------------------------------------------------------------------------
# Save / resume helpers
# ---------------------------------------------------------------------------

def _save(arr, path: Path, label="") -> None:
    np.save(path, arr.numpy().astype(np.int8) if isinstance(arr, torch.Tensor)
            else arr)
    print(f"  [saved] {label or path.name}  shape={arr.shape}", flush=True)


def _load(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.int64)
    t = torch.from_numpy(arr)
    print(f"  [resume] {path.name}  shape={t.shape}", flush=True)
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="experiments/configs/base_config.yaml")
    parser.add_argument("--out-dir",  default="experiments/phase7/gen")
    parser.add_argument("--eval-dir", default="experiments/full_eval_v2")
    parser.add_argument("--em-dir",   default="experiments/em_verify_v1")
    parser.add_argument("--n-specs",  type=int, default=100)
    parser.add_argument("--k-best",   type=int, default=5)
    parser.add_argument("--n-div",    type=int, default=20)
    parser.add_argument("--k-div",    type=int, default=20)
    parser.add_argument("--seed",     type=int, default=999)
    parser.add_argument("--chunk",    type=int, default=256)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    dcfg   = cfg.denoiser; gcfg = cfg.guidance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_w = dcfg.cfg_guidance_weight
    rng   = np.random.default_rng(args.seed)

    print(f"[phase7_eval] device={device}  n_specs={args.n_specs}  "
          f"k_best={args.k_best}  n_div={args.n_div}  k_div={args.k_div}", flush=True)

    # ── Load all test specs ────────────────────────────────────────────────
    eval_dir = Path(args.eval_dir)
    test_idx_all = np.load(eval_dir / "test_idx.npy")   # (N_test,)
    y_star_all   = np.load(eval_dir / "y_star.npy").astype(np.float32)
    N_test = len(y_star_all)
    print(f"[data] test_idx={N_test}  y_star={y_star_all.shape}", flush=True)

    # ── Find phase6 used indices ──────────────────────────────────────────
    em_path = Path(args.em_dir) / "em_verify_summary.json"
    phase6_used = set()
    if em_path.exists():
        with open(em_path) as f:
            em_data = json.load(f)
        for v in em_data.values():
            for r in v["per_layout"]:
                phase6_used.add(int(r["idx"]))
    print(f"[data] phase6 used indices: {len(phase6_used)}", flush=True)

    # ── Select fresh specs ─────────────────────────────────────────────────
    spec_idx_path = out_dir / "spec_idx.npy"
    div_idx_path  = out_dir / "diversity_spec_idx.npy"
    if spec_idx_path.exists() and div_idx_path.exists():
        spec_idx = np.load(spec_idx_path)
        div_idx  = np.load(div_idx_path)
        print(f"[resume] spec_idx={spec_idx.shape}  div_idx={div_idx.shape}", flush=True)
    else:
        spec_idx, div_idx = select_fresh_specs(
            y_star_all, test_idx_all, phase6_used,
            n_specs=args.n_specs, n_div=args.n_div, rng=rng)
        np.save(spec_idx_path, spec_idx)
        np.save(div_idx_path, div_idx)
        print(f"[saved] spec_idx={spec_idx.shape}  div_idx={div_idx.shape}", flush=True)

    # Extract y_star for selected specs
    y_star_bk  = torch.from_numpy(y_star_all[spec_idx])      # (N, 4, 100)
    y_star_div = torch.from_numpy(y_star_all[div_idx])        # (n_div, 4, 100)

    # Map div_idx to positions in spec_idx for later reference
    spec_to_pos = {int(s): i for i, s in enumerate(spec_idx)}
    div_in_bk   = np.array([spec_to_pos[int(d)] for d in div_idx
                             if int(d) in spec_to_pos])

    np.save(out_dir / "y_star_bk.npy",  y_star_bk.numpy())
    np.save(out_dir / "y_star_div.npy", y_star_div.numpy())
    np.save(out_dir / "div_in_bk.npy",  div_in_bk)

    # Save substrate_ids for EM verification
    h5_path = cfg.paths.raw_data + "/pixel_dataset.h5"
    with h5py.File(h5_path, "r") as f:
        if "substrate_id" in f:
            sub_bk  = f["substrate_id"][test_idx_all[spec_idx]].astype(np.int32)
            sub_div = f["substrate_id"][test_idx_all[div_idx]].astype(np.int32)
        else:
            sub_bk  = np.zeros(len(spec_idx), dtype=np.int32)
            sub_div = np.zeros(len(div_idx),  dtype=np.int32)
    np.save(out_dir / "substrate_ids_bk.npy",  sub_bk)
    np.save(out_dir / "substrate_ids_div.npy", sub_div)

    # ── Load models ────────────────────────────────────────────────────────
    enc, den, diff, surr_ens, disc = load_pixel_models(cfg, device)
    baselines = load_baselines(cfg, device)
    port_map  = _build_port_map(cfg)

    # ── Best-of-K generation ──────────────────────────────────────────────
    seeds_bk      = SEEDS_K5[:args.k_best]
    seeds_bk_cfg  = SEEDS_K5_CFG[:args.k_best]     # offset set for CFG-only
    seeds_div     = SEEDS_K20[:args.k_div]
    seeds_div_cfg = SEEDS_K20_CFG[:args.k_div]

    for key, path_name, gen_fn in [
        ("pixel",   f"layouts_pixel_k{args.k_best}.npy",   None),
        ("cfg",     f"layouts_cfg_k{args.k_best}.npy",     None),
        ("cvae",    f"layouts_cvae_k{args.k_best}.npy",    None),
        ("det_cnn", f"layouts_det_k1.npy",                 None),
    ]:
        p = out_dir / path_name
        if p.exists():
            print(f"[resume] {path_name} already exists, skipping.", flush=True)
            continue

        t0 = time.time()
        if key == "pixel":
            lays = gen_pixel_k(y_star_bk, seeds_bk, den, enc, diff,
                               surr_ens, disc, port_map, device, dcfg, gcfg, cfg_w,
                               chunk=args.chunk)
        elif key == "cfg":
            lays = gen_cfg_k(y_star_bk, seeds_bk_cfg, den, enc, diff,
                             port_map, device, dcfg, cfg_w, chunk=args.chunk)
        elif key == "cvae" and "cvae" in baselines:
            lays = gen_cvae_k(y_star_bk, args.k_best,
                              baselines["cvae"], device, chunk=args.chunk)
        elif key == "det_cnn" and "det_cnn" in baselines:
            lays = gen_det_k1(y_star_bk, baselines["det_cnn"], device, chunk=args.chunk)
        else:
            print(f"  [skip] {key} not available", flush=True)
            continue
        elapsed = time.time() - t0
        _save(lays, p, f"{key} K={lays.shape[1]} ({elapsed:.1f}s)")

    # ── Within-spec diversity (K=20 on 20 hardest specs) ──────────────────
    for key, path_name in [
        ("pixel", f"layouts_pixel_k{args.k_div}_div.npy"),
        ("cvae",  f"layouts_cvae_k{args.k_div}_div.npy"),
    ]:
        p = out_dir / path_name
        if p.exists():
            print(f"[resume] {path_name} already exists, skipping.", flush=True)
            continue

        t0 = time.time()
        if key == "pixel":
            lays = gen_pixel_k(y_star_div, seeds_div, den, enc, diff,
                               surr_ens, disc, port_map, device, dcfg, gcfg, cfg_w,
                               chunk=args.chunk)
        elif key == "cvae" and "cvae" in baselines:
            lays = gen_cvae_k(y_star_div, args.k_div,
                              baselines["cvae"], device, chunk=args.chunk)
        else:
            continue
        elapsed = time.time() - t0
        _save(lays, p, f"diversity {key} K={lays.shape[1]} ({elapsed:.1f}s)")

    print(f"\n[done] Phase 7 generation complete. Outputs in {out_dir}", flush=True)
    print("Next: submit pbs_phase7_em.pbs for EM verification.", flush=True)


if __name__ == "__main__":
    main()
