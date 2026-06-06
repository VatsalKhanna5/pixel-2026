"""
src/evaluation/full_eval.py
============================
Phase 5/6 — Full comparative evaluation across all methods.

N_test=-1  uses the entire held-out test split (~34 k samples).
Layouts are saved per-method to disk so em_verify.py can run OpenEMS
on a strategic subset without rerunning generation.

Checkpoint/resume: if layouts_{key}.npy already exists in out_dir,
generation is skipped for that method and the file is reloaded.

Usage:
    python -m src.evaluation.full_eval \\
        --config experiments/configs/base_config.yaml \\
        --n-test -1          # -1 = all test samples (~34 k)
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

GEN_CHUNK = 512      # samples per GPU forward pass — keeps VRAM under control


# ---------------------------------------------------------------------------
# Chunked generation helpers
# ---------------------------------------------------------------------------

def _guided_chunk(y_batch, den, enc, diff, surr_ens, disc, port_map, device,
                  dcfg, gcfg, cfg_w, lambda_topo, lambda_mfg, alpha_max):
    return generate_guided(
        y_batch, den, enc, diff, surr_ens, disc, port_map, device,
        T=dcfg.T, t_thresh=gcfg.t_threshold,
        alpha_max=alpha_max, epsilon=gcfg.epsilon_uncertainty,
        lambda_topo=lambda_topo, lambda_mfg=lambda_mfg,
        g_max=gcfg.gradient_clip_norm, cfg_w=cfg_w,
    )


def generate_in_chunks(gen_fn, y_star: torch.Tensor,
                       chunk: int = GEN_CHUNK, label: str = "") -> torch.Tensor:
    """Run gen_fn(y_batch) in chunks; cat results on CPU."""
    n_chunks = math.ceil(len(y_star) / chunk)
    parts = []
    for i in range(0, len(y_star), chunk):
        c = i // chunk + 1
        y_b = y_star[i: i + chunk]
        print(f"  [{label}] chunk {c}/{n_chunks} ({len(y_b)} samples) …", flush=True)
        lays = gen_fn(y_b)
        parts.append(lays.cpu())
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def score_layouts(
    layouts:      torch.Tensor,
    y_star:       torch.Tensor,
    surr_ens:     SurrogateEnsemble,
    port_map:     torch.Tensor,
    port1:        tuple,
    port2:        tuple,
    device:       torch.device,
    label:        str,
    t_elapsed:    float = 0.0,
    post_process: bool  = True,
) -> dict:
    N = len(layouts)
    layouts_clean = clean_batch(layouts, port1, port2) if post_process else layouts

    conn       = compute_connectivity(layouts)
    conn_clean = compute_connectivity(layouts_clean)
    drc_raw    = compute_drc_pass(layouts)
    drc_clean  = compute_drc_pass(layouts_clean)
    mse        = compute_surrogate_mse(layouts_clean, y_star, surr_ens, port_map, device)
    hamm       = compute_hamming(layouts_clean)
    fill       = (layouts_clean == 1).float().mean().item()

    print(f"\n=== {label} (N={N}) ===")
    print(f"  Connectivity (raw/clean): {conn:.4f} / {conn_clean:.4f}")
    print(f"  DRC pass (raw/clean):     {drc_raw:.4f} / {drc_clean:.4f}")
    print(f"  S21 MSE (surrogate):      {mse['s21_mse']:.6f}")
    print(f"  S11 MSE (surrogate):      {mse['s11_mse']:.6f}")
    print(f"  Hamming diversity:        {hamm:.2f} bits")
    print(f"  Fill fraction:            {fill:.4f}")
    print(f"  Time/sample:              {t_elapsed/max(N,1):.4f}s", flush=True)

    return {
        "label": label, "n": N,
        "conn_raw": conn, "conn_clean": conn_clean,
        "drc_raw": drc_raw, "drc_clean": drc_clean,
        "s21_mse": mse["s21_mse"], "s11_mse": mse["s11_mse"],
        "hamming": hamm, "fill": fill,
        "time_per_sample": t_elapsed / max(N, 1),
    }


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_pixel_models(cfg, device):
    ecfg = cfg.encoder; dcfg = cfg.denoiser; scfg = cfg.surrogate
    enc = SpectralEncoder(in_channels=ecfg.in_channels,
                          embed_dim=ecfg.embed_dim).to(device)
    den = PixelDenoiser(token_embed_dim=dcfg.token_embed_dim,
                        base_ch=dcfg.base_channels,
                        cond_embed_dim=ecfg.embed_dim,
                        t_embed_dim=dcfg.timestep_embed_dim,
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
    ecfg = cfg.encoder
    baselines = {}
    det_path = Path("experiments/baselines_v1/det_cnn_best.pt")
    if det_path.exists():
        m = DetCNN(in_channels=ecfg.in_channels,
                   embed_dim=ecfg.embed_dim).to(device)
        ck = torch.load(det_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"]); m.eval()
        baselines["det_cnn"] = m
        print(f"[load] Det-CNN  val_loss={ck['val_loss']:.4f}")
    cvae_path = Path("experiments/baselines_v1/cvae_best.pt")
    if cvae_path.exists():
        bcfg = cfg.get("baselines", None)
        latent_dim = int(bcfg.cvae.latent_dim) if bcfg else 64
        m = cVAE(in_channels=ecfg.in_channels, embed_dim=ecfg.embed_dim,
                 latent_dim=latent_dim).to(device)
        ck = torch.load(cvae_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state"]); m.eval()
        baselines["cvae"] = m
        print(f"[load] cVAE     val_loss={ck['val_loss']:.4f}")
    return baselines


# ---------------------------------------------------------------------------
# Layout save / load helpers
# ---------------------------------------------------------------------------

def _save_layouts(layouts: torch.Tensor, path: Path) -> None:
    np.save(path, layouts.cpu().numpy().astype(np.int8))
    print(f"  [saved] {path}  shape={tuple(layouts.shape)}", flush=True)


def _load_layouts(path: Path) -> torch.Tensor:
    arr = np.load(path).astype(np.int64)
    print(f"  [resume] loaded {path}  shape={arr.shape}", flush=True)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="experiments/configs/base_config.yaml")
    parser.add_argument("--n-test",       type=int, default=-1,
                        help="-1 = use entire held-out test split")
    parser.add_argument("--cfg-w",        type=float, default=None)
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--out-dir",      default="experiments/full_eval_v2")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    dcfg   = cfg.denoiser; ecfg = cfg.encoder; gcfg = cfg.guidance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_w = args.cfg_w if args.cfg_w is not None else dcfg.cfg_guidance_weight

    # ── Determine test indices ─────────────────────────────────────────────
    h5 = cfg.paths.raw_data + "/pixel_dataset.h5"
    with h5py.File(h5, "r") as f:
        total = f["layout"].shape[0]
    rng = np.random.default_rng(cfg.seed)
    idx_all = rng.permutation(total)
    n_tr = int(total * cfg.dataset.train_frac)
    n_va = int(total * cfg.dataset.val_frac)
    N = args.n_test if args.n_test > 0 else (total - n_tr - n_va)
    test_idx = np.sort(idx_all[n_tr + n_va: n_tr + n_va + N])

    print(f"[init] device={device}  N_test={N}  out={out_dir}", flush=True)
    np.save(out_dir / "test_idx.npy", test_idx)   # for em_verify

    # ── Load test specs ────────────────────────────────────────────────────
    ystar_path = out_dir / "y_star.npy"
    if ystar_path.exists():
        y_star_np = np.load(ystar_path)
        y_star = torch.from_numpy(y_star_np)
        print(f"[resume] y_star loaded from {ystar_path}", flush=True)
    else:
        with h5py.File(h5, "r") as f:
            s11m = f["S11_mag"][test_idx].astype(np.float32)
            s21m = f["S21_mag"][test_idx].astype(np.float32)
            s11p = f["S11_phase"][test_idx].astype(np.float32)
            s21p = f["S21_phase"][test_idx].astype(np.float32)
        y_star = torch.from_numpy(
            np.stack([s11m, s21m, s11p / math.pi, s21p / math.pi], axis=1))
        np.save(ystar_path, y_star.numpy())
        print(f"[data] {N} test specs loaded and saved.", flush=True)

    port_map = _build_port_map(cfg)
    p1 = tuple(cfg.dataset.port1); p2 = tuple(cfg.dataset.port2)

    # ── Load models ────────────────────────────────────────────────────────
    enc, den, diff, surr_ens, disc = load_pixel_models(cfg, device)
    baselines = load_baselines(cfg, device)

    results = {}
    summary_path = out_dir / "full_eval_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            results = json.load(f)
        print(f"[resume] partial results loaded: {list(results.keys())}", flush=True)

    # ── Helper: generate or load from checkpoint ───────────────────────────
    def _get_layouts(key: str, gen_fn) -> tuple[torch.Tensor, float]:
        lpath = out_dir / f"layouts_{key}.npy"
        if lpath.exists():
            return _load_layouts(lpath), 0.0
        t0 = time.time()
        layouts = gen_fn()
        elapsed = time.time() - t0
        _save_layouts(layouts, lpath)
        return layouts, elapsed

    # ── 1. PIXEL (guided) ─────────────────────────────────────────────────
    if "pixel_guided" not in results:
        print(f"\n[gen] PIXEL guided …", flush=True)
        def _gen_pixel():
            return generate_in_chunks(
                lambda yb: _guided_chunk(
                    yb, den, enc, diff, surr_ens, disc, port_map, device,
                    dcfg, gcfg, cfg_w,
                    lambda_topo=gcfg.lambda_topo, lambda_mfg=gcfg.lambda_mfg,
                    alpha_max=gcfg.alpha_max),
                y_star, label="PIXEL")
        lays, elapsed = _get_layouts("pixel_guided", _gen_pixel)
        results["pixel_guided"] = score_layouts(
            lays, y_star, surr_ens, port_map, p1, p2, device,
            "PIXEL (guided)", t_elapsed=elapsed)
        with open(summary_path, "w") as f: json.dump(results, f, indent=2)

    # ── 2. CFG-only ────────────────────────────────────────────────────────
    if "cfg_only" not in results:
        print(f"\n[gen] CFG-only …", flush=True)
        def _gen_cfg():
            return generate_in_chunks(
                lambda yb: generate_samples(
                    den, enc, diff, port_map,
                    n_samples=len(yb), device=device,
                    cfg_w=cfg_w, s_params=yb, T_steps=dcfg.T),
                y_star, label="CFG")
        lays, elapsed = _get_layouts("cfg_only", _gen_cfg)
        results["cfg_only"] = score_layouts(
            lays, y_star, surr_ens, port_map, p1, p2, device,
            "CFG-only", t_elapsed=elapsed)
        with open(summary_path, "w") as f: json.dump(results, f, indent=2)

    # ── 3. Ablations ───────────────────────────────────────────────────────
    if not args.no_ablations:
        for abl_key, abl_label, lt, lm, am in [
            ("ablation_no_topo",     "w/o Topo guidance",    0.0,             gcfg.lambda_mfg, gcfg.alpha_max),
            ("ablation_no_drc",      "w/o DRC guidance",     gcfg.lambda_topo,0.0,             gcfg.alpha_max),
            ("ablation_no_guidance", "w/o Any guidance",     0.0,             0.0,             0.0),
        ]:
            if abl_key in results:
                continue
            print(f"\n[gen] Ablation: {abl_label} …", flush=True)
            def _gen_abl(lt=lt, lm=lm, am=am):
                return generate_in_chunks(
                    lambda yb, lt=lt, lm=lm, am=am: _guided_chunk(
                        yb, den, enc, diff, surr_ens, disc, port_map, device,
                        dcfg, gcfg, cfg_w,
                        lambda_topo=lt, lambda_mfg=lm, alpha_max=am),
                    y_star, label=abl_key)
            lays, elapsed = _get_layouts(abl_key, _gen_abl)
            results[abl_key] = score_layouts(
                lays, y_star, surr_ens, port_map, p1, p2, device,
                abl_label, t_elapsed=elapsed)
            with open(summary_path, "w") as f: json.dump(results, f, indent=2)

    # ── 4. Det-CNN ─────────────────────────────────────────────────────────
    if "det_cnn" not in results and "det_cnn" in baselines:
        print(f"\n[gen] Det-CNN …", flush=True)
        def _gen_det():
            parts = []
            for i in range(0, len(y_star), GEN_CHUNK):
                with torch.no_grad():
                    parts.append(
                        baselines["det_cnn"].predict(
                            y_star[i:i+GEN_CHUNK].to(device)).cpu())
            return torch.cat(parts, dim=0)
        lays, elapsed = _get_layouts("det_cnn", _gen_det)
        results["det_cnn"] = score_layouts(
            lays, y_star, surr_ens, port_map, p1, p2, device,
            "Det-CNN", t_elapsed=elapsed)
        with open(summary_path, "w") as f: json.dump(results, f, indent=2)

    # ── 5. cVAE ────────────────────────────────────────────────────────────
    if "cvae" not in results and "cvae" in baselines:
        print(f"\n[gen] cVAE …", flush=True)
        def _gen_cvae():
            parts = []
            for i in range(0, len(y_star), GEN_CHUNK):
                with torch.no_grad():
                    samp = baselines["cvae"].sample(
                        y_star[i:i+GEN_CHUNK].to(device), n_samples=1)
                    parts.append(samp.cpu())
            return torch.cat(parts, dim=0)
        lays, elapsed = _get_layouts("cvae", _gen_cvae)
        results["cvae"] = score_layouts(
            lays, y_star, surr_ens, port_map, p1, p2, device,
            "cVAE", t_elapsed=elapsed)
        with open(summary_path, "w") as f: json.dump(results, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("FULL EVALUATION SUMMARY  (surrogate-validated)")
    print("=" * 72)
    print(f"{'Method':<35} {'N':>6} {'Conn':>6} {'DRC':>6} {'S21MSE':>9} {'Hamming':>8}")
    print("-" * 72)
    for key, r in results.items():
        print(f"  {r['label']:<33} {r['n']:>6} {r['conn_clean']:>6.4f} "
              f"{r['drc_clean']:>6.4f} {r['s21_mse']:>9.6f} {r['hamming']:>8.2f}")
    print("=" * 72)
    print(f"\n  EM-verified S21 MSE: run em_verify.py after this job completes.")

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
