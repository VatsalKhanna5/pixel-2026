"""
src/training/train_surrogate.py
================================
Phase 2: Train the K=5 PhysicsSurrogate CNN ensemble for PIXEL-2026.

All 5 surrogates are trained sequentially in a single PBS job.
Estimated total time: ~30 min on H100 MIG 3g.47gb (15×15 input is tiny).

Usage:
    python -m src.training.train_surrogate \\
        --config experiments/configs/base_config.yaml

Outputs (in cfg.surrogate.output_dir = experiments/surrogate_v1/):
    surrogate_k{i}_best.pt       best val-MSE checkpoint per surrogate
    surrogate_k{i}_final.pt      final-epoch checkpoint
    split_indices.npz            train/val/test index arrays (reproducibility)
    training_summary.json        all metrics + gradient-fidelity results
    wandb/                       offline WandB run directories
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

import h5py

# PIXEL modules
from src.models.surrogate import PhysicsSurrogate, SurrogateEnsemble
from src.losses.physics_losses import surrogate_loss, channel_mse
from src.utils.config import load_config, set_seed

# WandB in offline mode (logs saved locally; sync later with `wandb sync`)
os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402  (must be after env var)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PixelDataset(Dataset):
    """
    In-memory dataset loaded from pixel_dataset.h5.

    All data is loaded into RAM at construction time (~925 MB for 343k samples —
    trivial on the 1 TB HPC node).  num_workers=0 in the DataLoader is therefore
    optimal (no pickling overhead, no file-lock contention).

    Target normalisation:
        S*_mag  ∈ [0,1]   — already normalised by simulator passivity enforcement
        S*_phase ÷ π      → ∈ [-1,1]  — balances MSE scale with magnitude channels
    """

    def __init__(
        self,
        layouts: np.ndarray,    # (N, 15, 15)  uint8
        s_params: np.ndarray,   # (N, 4, 100)  float32  (phases already /π)
        port_map: torch.Tensor, # (1, 15, 15)  float32  — broadcast constant
        augment: bool = False,
        noise_sigma: float = 0.05,
    ) -> None:
        self.layouts   = torch.from_numpy(layouts.astype(np.float32))   # (N,15,15)
        self.s_params  = torch.from_numpy(s_params)                      # (N,4,100)
        self.port_map  = port_map                                         # (1,15,15)
        self.augment   = augment
        self.noise_sigma = noise_sigma

    def __len__(self) -> int:
        return len(self.layouts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        layout = self.layouts[idx].unsqueeze(0)   # (1,15,15)

        if self.augment and self.noise_sigma > 0:
            noise  = torch.randn_like(layout) * self.noise_sigma
            layout = (layout + noise).clamp_(0.0, 1.0)

        x = torch.cat([layout, self.port_map], dim=0)  # (2,15,15)
        y = self.s_params[idx]                          # (4,100)
        return x, y


def _build_port_map(cfg) -> torch.Tensor:
    r, c = cfg.dataset.port1
    port_map = torch.zeros(1, cfg.dataset.grid_h, cfg.dataset.grid_w)
    port_map[0, r, c] = 1.0
    r2, c2 = cfg.dataset.port2
    port_map[0, r2, c2] = 1.0
    return port_map


def load_dataset(
    h5_path: str,
    cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load full HDF5 dataset into RAM and return numpy arrays.

    Returns:
        layouts:    (N, 15, 15)  uint8
        s_params:   (N, 4, 100)  float32  phases normalised by /π
        prim_types: (N,)         uint8    used for stratified split
    """
    print(f"[data] Loading {h5_path} into RAM …", flush=True)
    t0 = time.time()

    with h5py.File(h5_path, "r") as f:
        layouts    = f["layout"][:]          # (N,15,15)  uint8
        s11_mag    = f["S11_mag"][:]         # (N,100)    float32
        s21_mag    = f["S21_mag"][:]
        s11_phase  = f["S11_phase"][:]       # radians
        s21_phase  = f["S21_phase"][:]
        prim_types = f["primitive_type"][:]  # uint8

    # Normalise phase to [-1, 1]
    s11_phase_n = s11_phase / math.pi
    s21_phase_n = s21_phase / math.pi

    # Stack into (N, 4, 100)
    s_params = np.stack([s11_mag, s21_mag, s11_phase_n, s21_phase_n], axis=1).astype(np.float32)

    print(f"[data] Loaded {len(layouts):,} samples in {time.time()-t0:.1f}s  "
          f"(shapes: layouts={layouts.shape}, s_params={s_params.shape})", flush=True)
    return layouts, s_params, prim_types


def make_splits(
    prim_types: np.ndarray,
    train_frac: float = 0.80,
    val_frac: float   = 0.10,
    seed: int         = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified 80/10/10 split by primitive_type.

    Stratification ensures every primitive family (including the under-represented
    'notch' at 1.5%) is present in val and test sets — critical for unbiased
    evaluation on held-out structure types.
    """
    N = len(prim_types)
    test_frac = 1.0 - train_frac - val_frac

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_frac + test_frac), random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(N), prim_types))

    # Split temp into val / test (equal halves)
    temp_types = prim_types[temp_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_sub, test_sub = next(sss2.split(np.zeros(len(temp_idx)), temp_types))
    val_idx  = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]

    print(f"[split] train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}", flush=True)
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Gradient fidelity validation
# ---------------------------------------------------------------------------

def validate_gradient_fidelity(
    model: PhysicsSurrogate,
    test_dataset: PixelDataset,
    device: torch.device,
    n_samples: int = 500,
    delta: float   = 0.01,
) -> Dict[str, float]:
    """
    Validate that autograd gradients of the surrogate w.r.t. the layout
    channel agree with finite-difference estimates.

    Gate (from PIXEL_EXECUTION_PLAN Phase 2): mean cosine similarity > 0.70.
    Below this, gradient-based physics guidance in Phase 4 is unreliable.

    Protocol:
      For each test sample (x, y):
        g_auto = ∂ MSE(f(x), y) / ∂ x  [autograd, layout channel only, (225,)]
        g_fd   = [f(x + δ·eⱼ) - f(x)] / δ  [FD over layout channel, (225,)]
        cos_sim = dot(g_auto, g_fd) / (|g_auto|·|g_fd|)

    Total FD calls: n_samples × 225 ≈ 112k forward passes (~20 s on H100).
    """
    model.eval()
    rng = np.random.default_rng(999)
    indices = rng.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)

    cosine_sims: List[float] = []
    mag_ratios:  List[float] = []

    for idx in indices:
        x, y = test_dataset[idx]
        x = x.unsqueeze(0).to(device)   # (1, 2, 15, 15)
        y = y.unsqueeze(0).to(device)   # (1, 4, 100)

        # --- Analytical gradient via autograd ---
        x_ag = x.clone().detach().requires_grad_(True)
        pred = model(x_ag)
        loss = F.mse_loss(pred, y)
        loss.backward()
        g_auto = x_ag.grad[0, 0].cpu().numpy().ravel()   # layout channel (225,)

        # --- Finite-difference gradient (layout channel only) ---
        g_fd = np.zeros(225, dtype=np.float32)
        with torch.no_grad():
            loss0 = F.mse_loss(model(x), y).item()
            for j in range(225):
                x_p = x.clone()
                x_p[0, 0, j // 15, j % 15] += delta
                g_fd[j] = (F.mse_loss(model(x_p), y).item() - loss0) / delta

        norm_auto = float(np.linalg.norm(g_auto)) + 1e-12
        norm_fd   = float(np.linalg.norm(g_fd))   + 1e-12
        cos_sim   = float(np.dot(g_auto, g_fd) / (norm_auto * norm_fd))
        cosine_sims.append(cos_sim)
        mag_ratios.append(norm_auto / norm_fd)

    result = {
        "grad/cosine_mean": float(np.mean(cosine_sims)),
        "grad/cosine_std":  float(np.std(cosine_sims)),
        "grad/cosine_min":  float(np.min(cosine_sims)),
        "grad/mag_ratio":   float(np.mean(mag_ratios)),
        "grad/pass":        bool(np.mean(cosine_sims) > 0.70),
    }
    return result


# ---------------------------------------------------------------------------
# Training loop for a single surrogate
# ---------------------------------------------------------------------------

def train_one_surrogate(
    k: int,
    train_ds: PixelDataset,
    val_ds:   PixelDataset,
    test_ds:  PixelDataset,
    cfg,
    device: torch.device,
    out_dir: Path,
) -> Dict:
    """Train surrogate index k and return its final metrics dict."""
    seed = cfg.surrogate.seed_base + k
    set_seed(seed, deterministic=False)   # non-deterministic for speed; seed ensures weight diversity

    scfg = cfg.surrogate

    # ---- Model ----
    model = PhysicsSurrogate(
        in_ch    = scfg.in_channels,
        base_ch  = scfg.base_channels,
        n_freq   = cfg.dataset.n_freq,
    ).to(device)

    # ---- Data loaders ----
    # num_workers=0: data is already in RAM; avoids pickling torch tensors
    train_loader = DataLoader(train_ds, batch_size=scfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=scfg.batch_size * 2, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ---- Optimiser + scheduler ----
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr           = scfg.lr,
        weight_decay = scfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser,
        T_max  = scfg.epochs,
        eta_min = 1e-6,
    )

    # ---- WandB (offline) ----
    run = wandb.init(
        project = cfg.wandb.project + "-surrogate",
        name    = f"surrogate_k{k}",
        config  = OmegaConf.to_container(scfg, resolve=True),
        mode    = cfg.wandb.get("mode", "offline"),
        dir     = str(out_dir),
        reinit  = True,
    )

    best_val_mse  = float("inf")
    best_ckpt     = out_dir / f"surrogate_k{k}_best.pt"
    final_ckpt    = out_dir / f"surrogate_k{k}_final.pt"

    print(f"\n{'='*60}", flush=True)
    print(f"  Surrogate k={k}  |  seed={seed}  |  params={model.n_params():,}", flush=True)
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}  batch={scfg.batch_size}", flush=True)
    print(f"{'='*60}", flush=True)

    t_start = time.time()

    for epoch in range(1, scfg.epochs + 1):
        # ---- Training ----
        model.train()
        train_loss_sum = 0.0
        train_comps_sum: Dict[str, float] = {}

        for x_b, y_b in train_loader:
            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            pred = model(x_b)
            loss, comps = surrogate_loss(
                pred, y_b,
                lambda_pass   = scfg.lambda_passivity,
                lambda_kk     = scfg.lambda_kk,
                lambda_smooth = scfg.lambda_smooth,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), scfg.gradient_clip)
            optimiser.step()

            train_loss_sum += comps["loss/total"]
            for kk, vv in comps.items():
                train_comps_sum[kk] = train_comps_sum.get(kk, 0.0) + vv

        scheduler.step()
        n_batches = len(train_loader)
        train_metrics = {k: v / n_batches for k, v in train_comps_sum.items()}

        # ---- Validation ----
        model.eval()
        val_preds_list: List[torch.Tensor] = []
        val_tgts_list:  List[torch.Tensor] = []
        val_loss_sum = 0.0

        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                pred = model(x_b)
                vloss, _ = surrogate_loss(pred, y_b,
                                          lambda_pass=scfg.lambda_passivity,
                                          lambda_kk=scfg.lambda_kk,
                                          lambda_smooth=scfg.lambda_smooth)
                val_loss_sum += vloss.item()
                val_preds_list.append(pred.cpu())
                val_tgts_list.append(y_b.cpu())

        val_preds = torch.cat(val_preds_list, dim=0)
        val_tgts  = torch.cat(val_tgts_list,  dim=0)
        val_chan   = channel_mse(val_preds, val_tgts)
        val_mag_mse = val_chan["val/mag_mse_mean"]

        # ---- Checkpoint ----
        if val_mag_mse < best_val_mse:
            best_val_mse = val_mag_mse
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_mag_mse":      best_val_mse,
                "cfg":              OmegaConf.to_container(scfg),
            }, best_ckpt)

        # ---- Logging ----
        log_dict = {
            **train_metrics,
            **val_chan,
            "val/loss":    val_loss_sum / len(val_loader),
            "lr":          scheduler.get_last_lr()[0],
        }
        wandb.log(log_dict, step=epoch)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(
                f"  [k={k}] epoch {epoch:3d}/{scfg.epochs}  "
                f"train_loss={train_metrics['loss/total']:.5f}  "
                f"val_mag_mse={val_mag_mse:.5f}  "
                f"best={best_val_mse:.5f}  "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    # ---- Save final checkpoint ----
    torch.save({
        "epoch":            scfg.epochs,
        "model_state_dict": model.state_dict(),
        "val_mag_mse":      val_mag_mse,
        "best_val_mag_mse": best_val_mse,
        "cfg":              OmegaConf.to_container(scfg),
    }, final_ckpt)

    total_time = time.time() - t_start
    print(f"  [k={k}] Training done in {total_time:.0f}s  best_val_mag_mse={best_val_mse:.5f}", flush=True)

    # ---- Gradient fidelity validation ----
    print(f"  [k={k}] Running gradient fidelity validation "
          f"({scfg.grad_fidelity_n_samples} samples × 225 FD calls) …", flush=True)
    # Load best checkpoint for gradient validation
    best_state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    t_grad = time.time()
    grad_metrics = validate_gradient_fidelity(
        model, test_ds, device,
        n_samples = scfg.grad_fidelity_n_samples,
        delta     = scfg.grad_fidelity_delta,
    )
    print(
        f"  [k={k}] Gradient fidelity: cosine_mean={grad_metrics['grad/cosine_mean']:.4f}  "
        f"mag_ratio={grad_metrics['grad/mag_ratio']:.4f}  "
        f"PASS={grad_metrics['grad/pass']}  ({time.time()-t_grad:.0f}s)",
        flush=True,
    )
    if not grad_metrics["grad/pass"]:
        print(f"  ⚠️  [k={k}] Gradient cosine < 0.70 — Phase 4 guidance may be unreliable.", flush=True)

    wandb.log(grad_metrics, step=scfg.epochs)
    run.finish()

    return {
        "k":                k,
        "seed":             seed,
        "best_val_mag_mse": best_val_mse,
        "train_time_s":     total_time,
        "best_ckpt":        str(best_ckpt),
        **grad_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PIXEL-2026 surrogate ensemble (Phase 2)")
    parser.add_argument("--config", default="experiments/configs/base_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scfg = cfg.surrogate

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[init] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

    # Output directory
    out_dir = Path(scfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] output dir: {out_dir}", flush=True)

    # ---- Load full dataset ----
    h5_path = Path(cfg.paths.raw_data) / "pixel_dataset.h5"
    layouts, s_params, prim_types = load_dataset(str(h5_path), cfg)

    # ---- Stratified split (done once, same for all K surrogates) ----
    split_cache = out_dir / "split_indices.npz"
    if split_cache.exists():
        print(f"[split] Loading cached split from {split_cache}", flush=True)
        sp = np.load(split_cache)
        train_idx, val_idx, test_idx = sp["train"], sp["val"], sp["test"]
        print(f"[split] train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}", flush=True)
    else:
        train_idx, val_idx, test_idx = make_splits(
            prim_types,
            train_frac = cfg.dataset.train_frac,
            val_frac   = cfg.dataset.val_frac,
            seed       = scfg.seed_base,
        )
        np.savez(split_cache, train=train_idx, val=val_idx, test=test_idx)
        print(f"[split] Saved split indices to {split_cache}", flush=True)

    # ---- Port map (constant for all samples) ----
    port_map = _build_port_map(cfg)  # (1,15,15)

    # ---- Build datasets ----
    train_ds = PixelDataset(
        layouts[train_idx], s_params[train_idx],
        port_map, augment=True, noise_sigma=scfg.noise_sigma,
    )
    val_ds = PixelDataset(
        layouts[val_idx], s_params[val_idx],
        port_map, augment=False,
    )
    test_ds = PixelDataset(
        layouts[test_idx], s_params[test_idx],
        port_map, augment=False,
    )

    # ---- Train each surrogate ----
    all_results: List[Dict] = []

    for k in range(scfg.ensemble_size):
        best_ckpt = out_dir / f"surrogate_k{k}_best.pt"
        if best_ckpt.exists():
            print(f"[skip] surrogate k={k} already trained ({best_ckpt})", flush=True)
            # Still load metrics for summary
            ckpt = torch.load(best_ckpt, map_location="cpu")
            all_results.append({"k": k, "best_val_mag_mse": ckpt["val_mag_mse"], "skipped": True})
            continue

        result = train_one_surrogate(k, train_ds, val_ds, test_ds, cfg, device, out_dir)
        all_results.append(result)

    # ---- Ensemble summary ----
    print(f"\n{'='*60}", flush=True)
    print("  ENSEMBLE SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for r in all_results:
        skipped = r.get("skipped", False)
        tag     = " (skipped)" if skipped else ""
        mse     = r["best_val_mag_mse"]
        cs      = r.get("grad/cosine_mean", float("nan"))
        gp      = r.get("grad/pass", "?")
        print(f"  k={r['k']}  val_mag_mse={mse:.5f}  grad_cosine={cs:.4f}  pass={gp}{tag}", flush=True)

    # Mean ensemble MSE (over surrogates that were freshly trained)
    mse_vals = [r["best_val_mag_mse"] for r in all_results]
    print(f"\n  Mean val_mag_mse (ensemble): {np.mean(mse_vals):.5f} ± {np.std(mse_vals):.5f}", flush=True)

    # ---- Save summary JSON ----
    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[done] Summary saved to {summary_path}", flush=True)

    # Gate check
    grad_passes = [r.get("grad/pass", True) for r in all_results if not r.get("skipped")]
    n_pass = sum(grad_passes)
    n_total = len(grad_passes)
    print(f"[gate] Gradient fidelity: {n_pass}/{n_total} surrogates PASS (threshold=0.70)", flush=True)
    if n_pass < n_total:
        print("[gate] ⚠️  Some surrogates failed gradient gate — check training_summary.json", flush=True)
        print("[gate]     See PIXEL_EXECUTION_PLAN.md Phase 2 rollback for mitigations.", flush=True)
    else:
        print("[gate] ✅  All surrogates pass — ready for Phase 3 / Phase 4.", flush=True)


if __name__ == "__main__":
    main()
