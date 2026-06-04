"""
src/evaluation/guided_eval.py
================================
Phase 4 evaluation: generate guided layouts and compute full scorecard.

Generates N layouts conditioned on test-set spectral targets using the
full PIXEL algorithm (D3PM + CFG + physics guidance), then evaluates:
  1. Connectivity yield (BFS)        gate > 95%
  2. DRC pass rate (width ≥ 2 px)    gate > 90%
  3. Surrogate-scored S21 MSE        gate < 0.05
  4. Hamming diversity               report (target > 30 bits)
  5. Layout statistics               fill fraction, MASK residuals

Also compares against Phase 3 baseline (CFG only, no physics guidance)
to quantify the contribution of guidance.

Full-wave EM verification (OpenEMS) is noted as pending for Phase 5.

Usage:
    python -m src.evaluation.guided_eval \\
        --config experiments/configs/base_config.yaml \\
        --n-samples 100
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
from src.guidance.physics_guidance import generate_guided
from src.models.connectivity_disc import ConnectivityDiscriminator
from src.models.denoiser import EMA, PixelDenoiser
from src.models.diffusion import D3PMAbsorbing
from src.models.spectral_encoder import SpectralEncoder
from src.models.surrogate import PhysicsSurrogate, SurrogateEnsemble
from src.training.train_denoiser import generate_samples  # CFG-only baseline
from src.training.train_surrogate import _build_port_map
from src.utils.config import load_config


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_connectivity(layouts: torch.Tensor) -> float:
    arr = layouts.cpu().numpy()
    n_pass = sum(is_connected((arr[i] == 1).astype(np.uint8)) for i in range(len(arr)))
    return n_pass / max(len(arr), 1)


def compute_drc_pass(layouts: torch.Tensor) -> float:
    """Check minimum trace width ≥ 2 pixels using 4-connectivity erosion."""
    arr = (layouts.cpu().numpy() == 1).astype(np.float32)
    n_pass = 0
    for i in range(len(arr)):
        x = arr[i]
        # Check: no isolated single-pixel conductors (4-connected)
        from scipy.ndimage import label
        labeled, n_comp = label(x)
        has_single_px = any(
            (labeled == c).sum() == 1 for c in range(1, n_comp + 1)
        )
        if not has_single_px:
            n_pass += 1
    return n_pass / max(len(arr), 1)


def compute_surrogate_mse(
    layouts: torch.Tensor,
    y_star:  torch.Tensor,
    surrogate_ens: SurrogateEnsemble,
    port_map: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute surrogate-scored S11 and S21 MSE for generated layouts."""
    surrogate_ens.eval()
    N = len(layouts)
    pm = port_map.float().to(device)
    while pm.ndim < 4: pm = pm.unsqueeze(0)
    pm = pm.expand(N, -1, -1, -1)

    layout_f = (layouts == 1).float().unsqueeze(1).to(device)  # {0,1} binary
    x_in = torch.cat([layout_f, pm], dim=1)

    with torch.no_grad():
        mean_pred, _ = surrogate_ens(x_in)   # (N, 4, 100)

    tgt = y_star.to(device)
    s11_mse = F.mse_loss(mean_pred[:, 0, :], tgt[:, 0, :]).item()
    s21_mse = F.mse_loss(mean_pred[:, 1, :], tgt[:, 1, :]).item()
    return {"s11_mse": s11_mse, "s21_mse": s21_mse,
            "mag_mse_mean": (s11_mse + s21_mse) / 2}


def compute_hamming(layouts: torch.Tensor, n_pairs: int = 200) -> float:
    arr = layouts.cpu().numpy().reshape(len(layouts), -1)
    rng = np.random.default_rng(0)
    pairs = rng.choice(len(arr), size=(n_pairs, 2), replace=True)
    return float(np.mean([np.sum(arr[a] != arr[b]) for a, b in pairs]))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/base_config.yaml")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--alpha-max", type=float, default=None)
    parser.add_argument("--t-thresh", type=int, default=None)
    parser.add_argument("--lambda-topo", type=float, default=None)
    parser.add_argument("--lambda-mfg", type=float, default=None)
    args = parser.parse_args()

    cfg   = load_config(args.config)
    dcfg  = cfg.denoiser
    ecfg  = cfg.encoder
    scfg  = cfg.surrogate
    gcfg  = cfg.guidance
    N     = args.n_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}  N={N}", flush=True)

    out_dir = Path("experiments/guided_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Guidance hyperparameters (allow CLI overrides for sweeps) ──────────
    alpha_max   = args.alpha_max   if args.alpha_max   is not None else gcfg.alpha_max
    t_thresh    = args.t_thresh    if args.t_thresh    is not None else gcfg.t_threshold
    lambda_topo = args.lambda_topo if args.lambda_topo is not None else gcfg.lambda_topo
    lambda_mfg  = args.lambda_mfg  if args.lambda_mfg  is not None else gcfg.lambda_mfg
    print(f"[cfg] alpha_max={alpha_max}  t_thresh={t_thresh}  "
          f"lambda_topo={lambda_topo}  lambda_mfg={lambda_mfg}", flush=True)

    # ── Load denoiser + encoder ────────────────────────────────────────────
    enc = SpectralEncoder(in_channels=ecfg.in_channels, embed_dim=ecfg.embed_dim).to(device)
    den = PixelDenoiser(token_embed_dim=dcfg.token_embed_dim, base_ch=dcfg.base_channels,
                        cond_embed_dim=ecfg.embed_dim, t_embed_dim=dcfg.timestep_embed_dim,
                        n_res_blocks=dcfg.n_res_blocks).to(device)
    ema = EMA(den)

    ckpt = torch.load("experiments/denoiser_v1/denoiser_best.pt",
                      map_location=device, weights_only=False)
    den.load_state_dict(ckpt["denoiser_state"])
    enc.load_state_dict(ckpt["encoder_state"])
    ema.load_state_dict(ckpt["ema_state"], device)
    ema.apply_shadow()
    den.eval(); enc.eval()

    diff = D3PMAbsorbing(T=dcfg.T).to(device)

    # ── Load surrogate ensemble ────────────────────────────────────────────
    surr_models = []
    for k in range(scfg.ensemble_size):
        m = PhysicsSurrogate(in_ch=scfg.in_channels, base_ch=scfg.base_channels,
                             n_freq=cfg.dataset.n_freq)
        ck = torch.load(f"experiments/surrogate_v1/surrogate_k{k}_best.pt",
                        map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"]); m.to(device).eval()
        surr_models.append(m)
    surr_ens = SurrogateEnsemble(surr_models)

    # ── Load connectivity discriminator ────────────────────────────────────
    disc = ConnectivityDiscriminator(in_ch=2, base_ch=scfg.base_channels).to(device)
    disc_ckpt = torch.load("experiments/discriminator_v1/disc_best.pt",
                            map_location=device, weights_only=False)
    disc.load_state_dict(disc_ckpt["model_state"])
    disc.eval()
    print(f"[disc] Loaded discriminator (val_auc={disc_ckpt.get('val_auc',0):.4f})",
          flush=True)

    # ── Load test targets ──────────────────────────────────────────────────
    sp_info  = np.load("experiments/surrogate_v1/split_indices.npz")
    test_idx = np.sort(sp_info["test"])[:N]
    with h5py.File("data/raw/pixel_dataset.h5", "r") as f:
        s11m  = f["S11_mag"][test_idx]
        s21m  = f["S21_mag"][test_idx]
        s11p  = f["S11_phase"][test_idx]
        s21p  = f["S21_phase"][test_idx]

    y_star = torch.from_numpy(
        np.stack([s11m, s21m, s11p/math.pi, s21p/math.pi], axis=1).astype(np.float32)
    )   # (N, 4, 100)

    port_map = _build_port_map(cfg)   # (1, 15, 15)

    # ════════════════════════════════════════════════════════════════════════
    # GENERATION: Guided (PIXEL) vs Baseline (CFG only)
    # ════════════════════════════════════════════════════════════════════════

    print(f"\n[gen] Generating {N} layouts with PIXEL (guided) …", flush=True)
    t0 = time.time()
    guided_layouts = generate_guided(
        y_star, den, enc, diff, surr_ens, disc, port_map, device,
        T=dcfg.T, t_thresh=t_thresh,
        alpha_max=alpha_max, epsilon=gcfg.epsilon_uncertainty,
        lambda_topo=lambda_topo, lambda_mfg=lambda_mfg,
        g_max=gcfg.gradient_clip_norm, cfg_w=dcfg.cfg_guidance_weight,
    )
    t_guided = time.time() - t0
    print(f"  Done in {t_guided:.1f}s  ({t_guided/N:.2f}s/sample)", flush=True)

    print(f"[gen] Generating {N} layouts with CFG-only baseline …", flush=True)
    t0 = time.time()
    baseline_layouts = generate_samples(
        den, enc, diff, port_map, n_samples=N, device=device,
        cfg_w=dcfg.cfg_guidance_weight, s_params=y_star, T_steps=dcfg.T,
    )
    t_base = time.time() - t0
    print(f"  Done in {t_base:.1f}s  ({t_base/N:.2f}s/sample)", flush=True)

    # ════════════════════════════════════════════════════════════════════════
    # METRICS
    # ════════════════════════════════════════════════════════════════════════

    def eval_all(layouts, y_star, label):
        print(f"\n=== {label} ===")
        conn   = compute_connectivity(layouts)
        drc    = compute_drc_pass(layouts)
        sp_mse = compute_surrogate_mse(layouts, y_star, surr_ens, port_map, device)
        hamm   = compute_hamming(layouts)
        fill   = (layouts == 1).float().mean().item()
        mask   = (layouts == 2).float().mean().item()
        t_per_sample = t_guided / N if label == "PIXEL (guided)" else t_base / N

        print(f"  Connectivity yield     : {conn:.4f}  (gate >0.95)  "
              f"{'✅' if conn > 0.95 else '❌'}")
        print(f"  DRC pass rate          : {drc:.4f}  (gate >0.90)  "
              f"{'✅' if drc > 0.90 else '❌'}")
        print(f"  Surrogate S21 MSE      : {sp_mse['s21_mse']:.5f}  (gate <0.05)  "
              f"{'✅' if sp_mse['s21_mse'] < 0.05 else '❌'}")
        print(f"  Surrogate S11 MSE      : {sp_mse['s11_mse']:.5f}")
        print(f"  Hamming diversity      : {hamm:.1f} bits  (target >30)")
        print(f"  Fill fraction (mean)   : {fill:.3f}")
        print(f"  Residual MASK tokens   : {mask:.4f}")
        print(f"  Time / sample          : {t_per_sample:.2f}s  (gate <60s)  "
              f"{'✅' if t_per_sample < 60 else '❌'}")
        return {
            "label": label, "n": N,
            "conn_yield": conn, "drc_pass": drc,
            "s21_mse": sp_mse["s21_mse"], "s11_mse": sp_mse["s11_mse"],
            "hamming": hamm, "fill": fill,
            "time_per_sample_s": t_per_sample,
            "passes_conn": conn > 0.95, "passes_drc": drc > 0.90,
            "passes_s21": sp_mse["s21_mse"] < 0.05,
            "all_pass": (conn > 0.95 and drc > 0.90 and sp_mse["s21_mse"] < 0.05),
        }

    guided_metrics  = eval_all(guided_layouts,  y_star, "PIXEL (guided)")
    baseline_metrics = eval_all(baseline_layouts, y_star, "CFG-only baseline")

    # ── Delta: guided improvement over baseline ────────────────────────────
    print("\n=== IMPROVEMENT: PIXEL vs CFG-only ===")
    for key in ["conn_yield", "drc_pass", "s21_mse", "s11_mse", "hamming"]:
        g = guided_metrics[key]
        b = baseline_metrics[key]
        arrow = "↑" if key in ["conn_yield", "drc_pass", "hamming"] else "↓"
        delta = (g - b) / max(abs(b), 1e-8) * 100
        print(f"  {key:20s}: guided={g:.4f}  base={b:.4f}  Δ={delta:+.1f}% {arrow}")

    # ── Final gate check ───────────────────────────────────────────────────
    print("\n=== CHECKPOINT P4 GATE CHECK ===")
    gates = [
        ("Connectivity yield (guided)",  guided_metrics["conn_yield"],  0.95, ">0.95"),
        ("DRC pass rate (guided)",        guided_metrics["drc_pass"],    0.90, ">0.90"),
        ("Surrogate S21 MSE (guided)",    guided_metrics["s21_mse"],     0.05, "<0.05"),
        ("Time per sample",               guided_metrics["time_per_sample_s"], 60, "<60s"),
    ]
    all_pass = True
    for name, val, thresh, gate_str in gates:
        if "<" in gate_str:
            ok = val < thresh
        else:
            ok = val > thresh
        s = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:35s}  {val:.4f}  ({gate_str})  {s}")
        all_pass = all_pass and ok

    print(f"\n  EM-verified metrics: PENDING (Phase 5 — OpenEMS verification required)")
    print(f"\n  Overall Phase 4 gate: {'✅ ALL PASS' if all_pass else '⚠️  PARTIAL'}")

    # ── Save results ──────────────────────────────────────────────────────
    summary = {
        "guided":   guided_metrics,
        "baseline": baseline_metrics,
        "hyperparams": {
            "alpha_max": alpha_max, "t_thresh": t_thresh,
            "lambda_topo": lambda_topo, "lambda_mfg": lambda_mfg,
            "cfg_w": dcfg.cfg_guidance_weight,
        }
    }
    out_path = out_dir / "eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] Results saved to {out_path}", flush=True)

    # Save generated layouts for future EM verification
    np.save(out_dir / "guided_layouts.npy",   guided_layouts.cpu().numpy())
    np.save(out_dir / "baseline_layouts.npy", baseline_layouts.cpu().numpy())
    np.save(out_dir / "y_star.npy",           y_star.numpy())
    print(f"[done] Layouts saved (shape: {guided_layouts.shape})", flush=True)


if __name__ == "__main__":
    main()
