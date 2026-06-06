"""
src/evaluation/full_eval.py
============================
Phase 5 — Full comparative evaluation across all methods on N_test specs.

Methods compared:
  1. PIXEL (guided + post-processed)   — our method
  2. PIXEL (CFG-only + post-processed) — ablation: no physics guidance
  3. Det-CNN                           — deterministic baseline
  4. cVAE                              — probabilistic baseline (K=5 samples/spec)

Ablations (subsets of PIXEL):
  A. No connectivity guidance  (lambda_topo=0)
  B. No DRC guidance           (lambda_mfg=0)
  C. No physics guidance       (alpha_max=0)  — same as CFG-only

Metrics:
  - Surrogate S21 MSE          (proxy for EM accuracy)
  - Surrogate S11 MSE
  - Connectivity yield         (BFS check)
  - DRC pass rate              (raw + post-processed)
  - Hamming diversity          (intra-method)
  - Fill fraction
  - Inference time

EM verification: logged as PENDING — full-wave simulation requires Phase 5
OpenEMS setup.

Usage:
    python -m src.evaluation.full_eval \\
        --config experiments/configs/base_config.yaml \\
        --n-test 200
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
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.dataset.connectivity import is_connected
from src.evaluation.baselines import DetCNN, cVAE
from src.evaluation.guided_eval import (
    clean_batch, compute_connectivity, compute_drc_pass,
    compute_hamming, compute_surrogate_mse,
)
from src.guidance.physics_guidance import generate_guided
from src.models.connectivity_disc import ConnectivityDiscriminator
from src.models.denoiser import EMA, PixelDenoiser
from src.models.diffusion import D3PMAbsorbing
from src.models.spectral_encoder import SpectralEncoder
from src.models.surrogate import PhysicsSurrogate, SurrogateEnsemble
from src.training.train_denoiser import generate_samples
from src.training.train_surrogate import _build_port_map
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def score_layouts(
    layouts:      torch.Tensor,       # (N, 15, 15) long
    y_star:       torch.Tensor,       # (N, 4, 100) float
    surr_ens:     SurrogateEnsemble,
    port_map:     torch.Tensor,
    port1:        tuple,
    port2:        tuple,
    device:       torch.device,
    label:        str,
    t_elapsed:    float = 0.0,
    post_process: bool  = True,
) -> dict:
    """Compute all metrics for a batch of generated layouts."""
    N = len(layouts)
    layouts_clean = clean_batch(layouts, port1, port2) if post_process else layouts

    conn       = compute_connectivity(layouts)
    conn_clean = compute_connectivity(layouts_clean)
    drc_raw    = compute_drc_pass(layouts)
    drc_clean  = compute_drc_pass(layouts_clean)
    mse        = compute_surrogate_mse(layouts_clean, y_star, surr_ens, port_map, device)
    hamm       = compute_hamming(layouts_clean)
    fill       = (layouts_clean == 1).float().mean().item()

    print(f"\n=== {label} ===")
    print(f"  Connectivity (raw/clean): {conn:.3f} / {conn_clean:.3f}")
    print(f"  DRC pass (raw/clean):     {drc_raw:.3f} / {drc_clean:.3f}")
    print(f"  S21 MSE (cleaned):        {mse['s21_mse']:.5f}")
    print(f"  S11 MSE (cleaned):        {mse['s11_mse']:.5f}")
    print(f"  Hamming diversity:        {hamm:.1f} bits")
    print(f"  Fill fraction:            {fill:.3f}")
    print(f"  Time/sample:              {t_elapsed/max(N,1):.3f}s")

    return {
        "label": label, "n": N,
        "conn_raw": conn, "conn_clean": conn_clean,
        "drc_raw": drc_raw, "drc_clean": drc_clean,
        "s21_mse": mse["s21_mse"], "s11_mse": mse["s11_mse"],
        "hamming": hamm, "fill": fill,
        "time_per_sample": t_elapsed / max(N, 1),
    }


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

def load_pixel_models(cfg, device):
    """Load all PIXEL components from checkpoints."""
    ecfg = cfg.encoder; dcfg = cfg.denoiser; scfg = cfg.surrogate

    enc = SpectralEncoder(in_channels=ecfg.in_channels, embed_dim=ecfg.embed_dim).to(device)
    den = PixelDenoiser(token_embed_dim=dcfg.token_embed_dim, base_ch=dcfg.base_channels,
                        cond_embed_dim=ecfg.embed_dim, t_embed_dim=dcfg.timestep_embed_dim,
                        n_res_blocks=dcfg.n_res_blocks).to(device)
    ema = EMA(den)
    ck = torch.load("experiments/denoiser_v1/denoiser_best.pt",
                    map_location=device, weights_only=False)
    den.load_state_dict(ck["denoiser_state"])
    enc.load_state_dict(ck["encoder_state"])
    ema.load_state_dict(ck["ema_state"], device)
    ema.apply_shadow()
    den.eval(); enc.eval()

    diff = D3PMAbsorbing(T=dcfg.T).to(device)

    surr_models = []
    for k in range(scfg.ensemble_size):
        m = PhysicsSurrogate(in_ch=scfg.in_channels, base_ch=scfg.base_channels,
                             n_freq=cfg.dataset.n_freq)
        ck2 = torch.load(f"experiments/surrogate_v1/surrogate_k{k}_best.pt",
                         map_location=device, weights_only=False)
        m.load_state_dict(ck2["model_state_dict"])
        surr_models.append(m.to(device).eval())
    surr_ens = SurrogateEnsemble(surr_models)

    disc = ConnectivityDiscriminator(in_ch=2, base_ch=scfg.base_channels).to(device)
    dk = torch.load("experiments/discriminator_v1/disc_best.pt",
                    map_location=device, weights_only=False)
    disc.load_state_dict(dk["model_state"])
    disc.eval()

    return enc, den, diff, surr_ens, disc


def load_baselines(cfg, device):
    """Load Det-CNN and cVAE from checkpoints (if available)."""
    ecfg = cfg.encoder
    baselines = {}

    det_path = Path("experiments/baselines_v1/det_cnn_best.pt")
    if det_path.exists():
        m = DetCNN(in_channels=ecfg.in_channels, embed_dim=ecfg.embed_dim).to(device)
        ck = torch.load(det_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"]); m.eval()
        baselines["det_cnn"] = m
        print(f"[load] Det-CNN loaded (val_loss={ck['val_loss']:.4f})")
    else:
        print(f"[load] Det-CNN checkpoint not found at {det_path} — skipping")

    cvae_path = Path("experiments/baselines_v1/cvae_best.pt")
    if cvae_path.exists():
        bcfg = cfg.get("baselines", None)
        latent_dim = int(bcfg.cvae.latent_dim) if bcfg else 64
        m = cVAE(in_channels=ecfg.in_channels, embed_dim=ecfg.embed_dim,
                 latent_dim=latent_dim).to(device)
        ck = torch.load(cvae_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"]); m.eval()
        baselines["cvae"] = m
        print(f"[load] cVAE loaded (val_loss={ck['val_loss']:.4f})")
    else:
        print(f"[load] cVAE checkpoint not found at {cvae_path} — skipping")

    return baselines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="experiments/configs/base_config.yaml")
    parser.add_argument("--n-test",   type=int, default=200)
    parser.add_argument("--cfg-w",    type=float, default=None)
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    dcfg   = cfg.denoiser; ecfg = cfg.encoder; gcfg = cfg.guidance
    N      = args.n_test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}  N_test={N}", flush=True)

    out_dir = Path("experiments/full_eval_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_w = args.cfg_w if args.cfg_w is not None else dcfg.cfg_guidance_weight

    # ── Load test specs from HDF5 ──────────────────────────────────────────
    h5 = cfg.paths.raw_data + "/pixel_dataset.h5"
    with h5py.File(h5, "r") as f:
        total = f["layout"].shape[0]
    rng = np.random.default_rng(cfg.seed)
    idx_all = rng.permutation(total)
    n_tr = int(total * cfg.dataset.train_frac)
    n_va = int(total * cfg.dataset.val_frac)
    test_idx = np.sort(idx_all[n_tr + n_va: n_tr + n_va + N])
    with h5py.File(h5, "r") as f:
        s11m = f["S11_mag"][test_idx].astype(np.float32)
        s21m = f["S21_mag"][test_idx].astype(np.float32)
        s11p = f["S11_phase"][test_idx].astype(np.float32)
        s21p = f["S21_phase"][test_idx].astype(np.float32)
    y_star = torch.from_numpy(
        np.stack([s11m, s21m, s11p / math.pi, s21p / math.pi], axis=1))
    port_map = _build_port_map(cfg)
    p1 = tuple(cfg.dataset.port1); p2 = tuple(cfg.dataset.port2)
    print(f"[data] {N} test specs loaded", flush=True)

    # ── Load PIXEL models ──────────────────────────────────────────────────
    enc, den, diff, surr_ens, disc = load_pixel_models(cfg, device)

    results = {}

    # ── 1. PIXEL (guided, full) ────────────────────────────────────────────
    print(f"\n[gen] PIXEL guided (α={gcfg.alpha_max}, t_thresh={gcfg.t_threshold}) …")
    t0 = time.time()
    layouts_pixel = generate_guided(
        y_star, den, enc, diff, surr_ens, disc, port_map, device,
        T=dcfg.T, t_thresh=gcfg.t_threshold,
        alpha_max=gcfg.alpha_max, epsilon=gcfg.epsilon_uncertainty,
        lambda_topo=gcfg.lambda_topo, lambda_mfg=gcfg.lambda_mfg,
        g_max=gcfg.gradient_clip_norm, cfg_w=cfg_w,
    )
    results["pixel_guided"] = score_layouts(
        layouts_pixel, y_star, surr_ens, port_map, p1, p2, device,
        "PIXEL (guided)", t_elapsed=time.time() - t0)

    # ── 2. CFG-only (no physics guidance) ─────────────────────────────────
    print(f"\n[gen] CFG-only (w={cfg_w}) …")
    t0 = time.time()
    layouts_cfg = generate_samples(
        den, enc, diff, port_map, n_samples=N, device=device,
        cfg_w=cfg_w, s_params=y_star, T_steps=dcfg.T)
    results["cfg_only"] = score_layouts(
        layouts_cfg, y_star, surr_ens, port_map, p1, p2, device,
        "CFG-only", t_elapsed=time.time() - t0)

    # ── 3. Ablations ───────────────────────────────────────────────────────
    if not args.no_ablations:
        # A: No topology guidance
        print(f"\n[gen] Ablation: no connectivity guidance (lambda_topo=0) …")
        t0 = time.time()
        abl_no_topo = generate_guided(
            y_star, den, enc, diff, surr_ens, disc, port_map, device,
            T=dcfg.T, t_thresh=gcfg.t_threshold,
            alpha_max=gcfg.alpha_max, epsilon=gcfg.epsilon_uncertainty,
            lambda_topo=0.0, lambda_mfg=gcfg.lambda_mfg,
            g_max=gcfg.gradient_clip_norm, cfg_w=cfg_w,
        )
        results["ablation_no_topo"] = score_layouts(
            abl_no_topo, y_star, surr_ens, port_map, p1, p2, device,
            "Ablation: no topo guidance", t_elapsed=time.time() - t0)

        # B: No DRC guidance
        print(f"\n[gen] Ablation: no DRC guidance (lambda_mfg=0) …")
        t0 = time.time()
        abl_no_drc = generate_guided(
            y_star, den, enc, diff, surr_ens, disc, port_map, device,
            T=dcfg.T, t_thresh=gcfg.t_threshold,
            alpha_max=gcfg.alpha_max, epsilon=gcfg.epsilon_uncertainty,
            lambda_topo=gcfg.lambda_topo, lambda_mfg=0.0,
            g_max=gcfg.gradient_clip_norm, cfg_w=cfg_w,
        )
        results["ablation_no_drc"] = score_layouts(
            abl_no_drc, y_star, surr_ens, port_map, p1, p2, device,
            "Ablation: no DRC guidance", t_elapsed=time.time() - t0)

        # C: No physics guidance (alpha=0, same as CFG-only but via guided_step code)
        print(f"\n[gen] Ablation: no physics guidance (alpha_max=0) …")
        t0 = time.time()
        abl_no_phys = generate_guided(
            y_star, den, enc, diff, surr_ens, disc, port_map, device,
            T=dcfg.T, t_thresh=gcfg.t_threshold,
            alpha_max=0.0, epsilon=gcfg.epsilon_uncertainty,
            lambda_topo=0.0, lambda_mfg=0.0,
            g_max=gcfg.gradient_clip_norm, cfg_w=cfg_w,
        )
        results["ablation_no_guidance"] = score_layouts(
            abl_no_phys, y_star, surr_ens, port_map, p1, p2, device,
            "Ablation: no guidance (CFG only via guided loop)", t_elapsed=time.time() - t0)

    # ── 4. Baselines (Det-CNN, cVAE) ───────────────────────────────────────
    baselines = load_baselines(cfg, device)

    if "det_cnn" in baselines:
        print(f"\n[gen] Det-CNN …")
        t0 = time.time()
        with torch.no_grad():
            det_layouts = baselines["det_cnn"].predict(y_star.to(device))
        results["det_cnn"] = score_layouts(
            det_layouts.cpu(), y_star, surr_ens, port_map, p1, p2, device,
            "Det-CNN", t_elapsed=time.time() - t0)

    if "cvae" in baselines:
        print(f"\n[gen] cVAE (5 samples/spec) …")
        t0 = time.time()
        with torch.no_grad():
            cvae_layouts = baselines["cvae"].sample(y_star.to(device), n_samples=5)
        # Take first sample per spec for primary metric; report all for diversity
        cvae_primary = cvae_layouts[::5].cpu()
        results["cvae"] = score_layouts(
            cvae_primary, y_star, surr_ens, port_map, p1, p2, device,
            "cVAE (1 sample)", t_elapsed=time.time() - t0)
        # Diversity across samples
        results["cvae"]["hamming_all_samples"] = compute_hamming(cvae_layouts.cpu())

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FULL EVALUATION SUMMARY")
    print("="*70)
    print(f"{'Method':<35} {'Conn':>6} {'DRC':>6} {'S21MSE':>8} {'Hamming':>8} {'Time':>7}")
    print("-"*70)
    for key, r in results.items():
        print(f"  {r['label']:<33} {r['conn_clean']:>6.3f} {r['drc_clean']:>6.3f} "
              f"{r['s21_mse']:>8.5f} {r['hamming']:>8.1f} {r['time_per_sample']:>6.3f}s")
    print("="*70)
    print(f"\n  EM-verified S21 MSE: PENDING (Phase 5 OpenEMS — gate <0.08)")

    # Save
    out_path = out_dir / "full_eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
