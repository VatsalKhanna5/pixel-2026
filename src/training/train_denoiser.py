"""
src/training/train_denoiser.py
================================
Phase 3: Train the D3PM conditional denoiser for PIXEL-2026.

Usage:
    python -m src.training.train_denoiser \\
        --config experiments/configs/base_config.yaml

What happens:
  - 343k layouts loaded into RAM alongside their S-parameter targets.
  - Same stratified split as Phase 2 (loaded from split_indices.npz).
  - Each batch: sample t ~ U(1,T), corrupt x_0 → x_t, train denoiser.
  - Condition dropout p=0.15 (enables CFG at inference).
  - EMA of model weights maintained throughout (used for validation/sampling).
  - Every `val_every` epochs: generate 256 uncond + 256 cond samples,
    check connectivity yield and surrogate-scored S21 MSE.
  - Checkpoint: best (val connectivity yield) + every 50 epochs.
  - WandB offline. Summary JSON saved at end.

Estimated wall time on H100 MIG 3g.47gb:
  ~300 batches/epoch × ~25 ms/step × 300 epochs ≈ 37 min total.
  Well within 24 h PBS walltime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

# PIXEL modules
from src.dataset.connectivity import is_connected
from src.guidance.cfg import apply_cfg_to_logits
from src.losses.diffusion_losses import diffusion_loss, compute_accuracy
from src.models.denoiser import EMA, PixelDenoiser
from src.models.diffusion import D3PMAbsorbing
from src.models.spectral_encoder import SpectralEncoder
from src.models.surrogate import PhysicsSurrogate, SurrogateEnsemble
from src.training.train_surrogate import PixelDataset, _build_port_map
from src.utils.config import load_config, set_seed

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset — reuse PixelDataset; denoiser needs layouts as long tensors
# ---------------------------------------------------------------------------

class DenoisingDataset(Dataset):
    """
    Returns (layout_long, s_params_norm, port_map) where:
        layout_long: (H, W)   long  in {0,1}    — the x_0 token indices
        s_params:    (4, 100) float normalised   — spectral condition y
    """

    def __init__(
        self,
        layouts:   np.ndarray,    # (N, 15, 15) uint8
        s_params:  np.ndarray,    # (N, 4, 100) float32 (phases /π)
        port_map:  torch.Tensor,  # (1, 15, 15) float32
        dropout_p: float = 0.0,   # condition dropout prob (train only)
    ) -> None:
        self.layouts   = torch.from_numpy(layouts.astype(np.int64))   # (N,H,W) long
        self.s_params  = torch.from_numpy(s_params)                    # (N,4,100)
        self.port_map  = port_map
        self.dropout_p = dropout_p

    def __len__(self) -> int:
        return len(self.layouts)

    def __getitem__(self, idx: int):
        x0 = self.layouts[idx]      # (H, W) long {0,1}
        y  = self.s_params[idx]     # (4,100)
        # Condition dropout: zero out the spectral target
        if self.dropout_p > 0 and torch.rand(1).item() < self.dropout_p:
            y = torch.zeros_like(y)
        return x0, y


# ---------------------------------------------------------------------------
# Generation + validation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    denoiser:  PixelDenoiser,
    encoder:   SpectralEncoder,
    diffusion: D3PMAbsorbing,
    port_map:  torch.Tensor,
    n_samples: int,
    device:    torch.device,
    cfg_w:     float = 0.0,          # 0 = unconditional
    s_params:  torch.Tensor | None = None,  # (n_samples, 4, 100) for conditional
    T_steps:   int = 1000,
) -> torch.Tensor:
    """
    Run the full reverse diffusion chain to generate layouts.

    Returns: (n_samples, 15, 15) long in {0, 1, MASK=2}
    """
    denoiser.eval()
    encoder.eval()

    # Start from all-MASK
    H, W  = 15, 15
    x_t   = torch.full((n_samples, H, W), diffusion.MASK, dtype=torch.long, device=device)
    # Standardise port_map to (n_samples, 1, H, W) regardless of input shape
    pm = port_map.float().to(device)
    while pm.ndim < 4:
        pm = pm.unsqueeze(0)
    pm = pm.expand(n_samples, -1, -1, -1).contiguous()

    c_y = None
    if s_params is not None and cfg_w > 0:
        c_y = encoder(s_params.to(device))   # (B, 256)

    for t_val in range(T_steps, 0, -1):
        t = torch.full((n_samples,), t_val, dtype=torch.long, device=device)

        if cfg_w > 0 and c_y is not None:
            log_probs = apply_cfg_to_logits(
                lambda xt, tt, cy, pm_: denoiser(xt, tt, cy, pm_),
                x_t, t, c_y, pm, w=cfg_w,
            )
            x0_logits_for_posterior = log_probs   # CFG log-probs drive posterior sampling
        else:
            x0_logits_for_posterior = denoiser(x_t, t, c_y, pm)
            log_probs = None

        x_t = diffusion.posterior_sample(
            x_t, x0_logits_for_posterior, t,
            cfg_log_probs=log_probs,
        )

    return x_t   # (N, H, W) long


def connectivity_yield(layouts: torch.Tensor) -> float:
    """Fraction of layouts where port1 and port2 are connected (BFS)."""
    n_pass = 0
    arr = layouts.cpu().numpy()
    for i in range(len(arr)):
        # Binary: treat 0 = dielectric, 1 = conductor
        binary = (arr[i] == 1).astype(np.uint8)
        if is_connected(binary):
            n_pass += 1
    return n_pass / max(len(arr), 1)


def hamming_diversity(layouts: torch.Tensor, n_pairs: int = 200) -> float:
    """Mean Hamming distance between random pairs of layouts."""
    arr = layouts.cpu().numpy().reshape(len(layouts), -1)   # (N, 225)
    rng = np.random.default_rng(0)
    pairs = rng.choice(len(arr), size=(n_pairs, 2), replace=True)
    dists = [np.sum(arr[a] != arr[b]) for a, b in pairs]
    return float(np.mean(dists))


@torch.no_grad()
def surrogate_s21_mse(
    ensemble:  SurrogateEnsemble,
    layouts:   torch.Tensor,    # (N, 15, 15) long {0,1}
    s_targets: torch.Tensor,    # (N, 4, 100) normalised
    port_map:  torch.Tensor,
    device:    torch.device,
) -> float:
    """Score generated layouts with the surrogate ensemble."""
    ensemble.eval()
    N = len(layouts)
    # Standardise port_map to (N, 1, H, W) regardless of input shape
    pm = port_map.float().to(device)
    while pm.ndim < 4:
        pm = pm.unsqueeze(0)
    pm = pm.expand(N, -1, -1, -1)
    layout_f = layouts.float().unsqueeze(1).to(device).clamp(0, 1)  # {0,1} → float
    x_in = torch.cat([layout_f, pm], dim=1)   # (N, 2, 15, 15)

    mean_pred, _ = ensemble(x_in)   # (N, 4, 100)
    # S21 magnitude MSE (channels 0=S11, 1=S21)
    mse = F.mse_loss(mean_pred[:, 1, :], s_targets[:, 1, :].to(device)).item()
    return mse


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PIXEL-2026 Phase 3: D3PM Denoiser")
    parser.add_argument("--config", default="experiments/configs/base_config.yaml")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    dcfg = cfg.denoiser
    ecfg = cfg.encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)
    if device.type == "cuda":
        print(f"[init] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[init] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB",
              flush=True)

    set_seed(cfg.seed, deterministic=False)

    out_dir = Path("experiments/denoiser_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] output dir: {out_dir}", flush=True)

    # ── Load dataset ─────────────────────────────────────────────────────────
    h5_path = Path(cfg.paths.raw_data) / "pixel_dataset.h5"
    print(f"[data] Loading {h5_path} …", flush=True)
    t0 = time.time()
    sp = np.load("experiments/surrogate_v1/split_indices.npz")
    train_idx = np.sort(sp["train"])
    val_idx   = np.sort(sp["val"])
    test_idx  = np.sort(sp["test"])

    with h5py.File(h5_path, "r") as f:
        train_layouts   = f["layout"][train_idx]
        train_s11_mag   = f["S11_mag"][train_idx]
        train_s21_mag   = f["S21_mag"][train_idx]
        train_s11_ph    = f["S11_phase"][train_idx]
        train_s21_ph    = f["S21_phase"][train_idx]

        val_layouts     = f["layout"][val_idx]
        val_s11_mag     = f["S11_mag"][val_idx]
        val_s21_mag     = f["S21_mag"][val_idx]
        val_s11_ph      = f["S11_phase"][val_idx]
        val_s21_ph      = f["S21_phase"][val_idx]

    def make_sparams(s11m, s21m, s11p, s21p):
        return np.stack([s11m, s21m, s11p / math.pi, s21p / math.pi], axis=1).astype(np.float32)

    train_sp = make_sparams(train_s11_mag, train_s21_mag, train_s11_ph, train_s21_ph)
    val_sp   = make_sparams(val_s11_mag,   val_s21_mag,   val_s11_ph,   val_s21_ph)

    print(f"[data] Loaded {len(train_layouts):,} train + {len(val_layouts):,} val "
          f"in {time.time()-t0:.1f}s", flush=True)

    port_map = _build_port_map(cfg).unsqueeze(0)   # (1, 1, 15, 15)

    train_ds = DenoisingDataset(train_layouts, train_sp, port_map[0],
                                dropout_p=dcfg.cfg_dropout_prob)
    val_ds   = DenoisingDataset(val_layouts,   val_sp,   port_map[0])

    train_loader = DataLoader(train_ds, batch_size=dcfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=dcfg.batch_size * 2,
                              shuffle=False, num_workers=0)

    # ── Build models ─────────────────────────────────────────────────────────
    encoder  = SpectralEncoder(in_channels=ecfg.in_channels,
                               embed_dim=ecfg.embed_dim).to(device)
    denoiser = PixelDenoiser(
        token_embed_dim = dcfg.token_embed_dim,
        base_ch         = dcfg.base_channels,
        cond_embed_dim  = ecfg.embed_dim,
        t_embed_dim     = dcfg.timestep_embed_dim,
        n_res_blocks    = dcfg.n_res_blocks,
    ).to(device)

    diffusion = D3PMAbsorbing(T=dcfg.T).to(device)

    print(f"[model] Encoder params: {encoder.n_params():,}", flush=True)
    print(f"[model] Denoiser params: {denoiser.n_params():,}", flush=True)

    # EMA
    ema = EMA(denoiser, decay=dcfg.ema_decay)

    # ── Load surrogate for validation scoring ─────────────────────────────────
    print("[surrogate] Loading K=5 ensemble for validation …", flush=True)
    surr_models = []
    scfg = cfg.surrogate
    for k in range(scfg.ensemble_size):
        m = PhysicsSurrogate(in_ch=scfg.in_channels, base_ch=scfg.base_channels,
                             n_freq=cfg.dataset.n_freq)
        ckpt = torch.load(f"experiments/surrogate_v1/surrogate_k{k}_best.pt",
                          map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.to(device).eval()
        surr_models.append(m)
    surrogate_ens = SurrogateEnsemble(surr_models)

    # ── Optimiser + LR schedule ───────────────────────────────────────────────
    params = list(denoiser.parameters()) + list(encoder.parameters())
    optimiser = torch.optim.AdamW(params, lr=dcfg.lr, weight_decay=dcfg.weight_decay)

    total_steps   = dcfg.epochs * len(train_loader)
    warmup_steps  = min(dcfg.warmup_steps, total_steps // 10)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb.init(
        project = cfg.wandb.project + "-denoiser",
        name    = "denoiser_v1",
        config  = OmegaConf.to_container(dcfg, resolve=True),
        mode    = cfg.wandb.get("mode", "offline"),
        dir     = str(out_dir),
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_conn_yield  = 0.0
    global_step      = 0
    val_every        = 25    # epochs between validation runs
    summary: Dict    = {}

    print(f"\n[train] Starting: {dcfg.epochs} epochs, {len(train_loader)} batches/epoch",
          flush=True)
    t_start = time.time()

    for epoch in range(1, dcfg.epochs + 1):
        denoiser.train()
        encoder.train()

        epoch_loss  = 0.0
        epoch_comps: Dict[str, float] = {}
        n_batches   = 0

        for x0, y in train_loader:
            x0 = x0.to(device, non_blocking=True)   # (B, H, W) long {0,1}
            y  = y.to(device,  non_blocking=True)   # (B, 4, 100)

            # Sample random timestep for each sample
            t = torch.randint(1, diffusion.T + 1, (x0.shape[0],), device=device)

            # Forward corruption: x_0 → x_t
            x_t = diffusion.q_sample(x0, t)   # (B, H, W) long {0,1,MASK}

            # Encode spectral condition (y may be zeroed for dropped samples)
            # Separate cond and uncond within the batch based on dropout flag
            null_mask = (y.abs().sum(dim=(1, 2)) == 0)   # (B,) True where dropped
            c_y       = encoder(y)                        # (B, 256)
            c_y[null_mask] = 0.0                          # zero out dropped

            # Pass c_y=None for null samples by passing zeros (network trained on both)
            logits = denoiser(x_t, t, c_y, port_map[0].to(device))   # (B,2,H,W)

            loss, comps = diffusion_loss(logits, x0, x_t, lambda_aux=dcfg.lambda_aux_x0)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, dcfg.gradient_clip_norm)
            optimiser.step()
            scheduler.step()
            ema.update()

            epoch_loss += comps["loss/total"]
            for k, v in comps.items():
                epoch_comps[k] = epoch_comps.get(k, 0.0) + v
            n_batches  += 1
            global_step += 1

        # Average over batches
        avg_comps = {k: v / n_batches for k, v in epoch_comps.items()}
        avg_comps["lr"] = scheduler.get_last_lr()[0]

        # ── Periodic validation ────────────────────────────────────────────
        val_metrics: Dict[str, float] = {}
        if epoch % val_every == 0 or epoch == dcfg.epochs:
            ema.apply_shadow()   # use EMA weights for eval
            denoiser.eval(); encoder.eval()

            with torch.no_grad():
                # Unconditional generation: 256 layouts
                N_gen = 256
                gen_uncond = generate_samples(
                    denoiser, encoder, diffusion, port_map[0],
                    n_samples=N_gen, device=device, cfg_w=0.0, T_steps=dcfg.T,
                )
                # Connectivity yield (gate > 0.80)
                conn = connectivity_yield(gen_uncond)
                # Hamming diversity (gate > 30 bits)
                hamm = hamming_diversity(gen_uncond)

                # Conditional generation with CFG: 64 samples from val set
                N_cond = min(64, len(val_ds))
                val_indices = torch.randperm(len(val_ds))[:N_cond]
                val_y_batch = torch.stack([val_ds[i][1] for i in val_indices])   # (N_cond,4,100)
                gen_cond = generate_samples(
                    denoiser, encoder, diffusion, port_map[0],
                    n_samples=N_cond, device=device,
                    cfg_w=dcfg.cfg_guidance_weight,
                    s_params=val_y_batch, T_steps=dcfg.T,
                )
                # Score with surrogate (gate < 0.10)
                cond_s21_mse = surrogate_s21_mse(
                    surrogate_ens, gen_cond, val_y_batch, port_map[0].unsqueeze(0), device
                )

            ema.restore()   # restore training weights

            val_metrics = {
                "val/conn_yield":    conn,
                "val/hamming_mean":  hamm,
                "val/cond_s21_mse":  cond_s21_mse,
                "val/pass_conn":     float(conn > 0.80),
                "val/pass_s21":      float(cond_s21_mse < 0.10),
            }

            # Save best checkpoint (connectivity yield)
            if conn > best_conn_yield:
                best_conn_yield = conn
                ema.apply_shadow()
                torch.save({
                    "epoch": epoch,
                    "denoiser_state": denoiser.state_dict(),
                    "encoder_state":  encoder.state_dict(),
                    "ema_state":      ema.state_dict(),
                    "val_conn":       conn,
                    "val_s21_mse":    cond_s21_mse,
                }, out_dir / "denoiser_best.pt")
                ema.restore()

        # ── Log + print ────────────────────────────────────────────────────
        log_dict = {**avg_comps, **val_metrics}
        wandb.log(log_dict, step=epoch)

        if epoch % 10 == 0 or epoch == 1 or epoch == dcfg.epochs:
            elapsed = time.time() - t_start
            vstr = ""
            if val_metrics:
                vstr = (f"  conn={val_metrics['val/conn_yield']:.3f}"
                        f"  hamm={val_metrics['val/hamming_mean']:.1f}"
                        f"  s21={val_metrics['val/cond_s21_mse']:.4f}")
            print(
                f"  [ep {epoch:3d}/{dcfg.epochs}] "
                f"loss={avg_comps['loss/total']:.4f} "
                f"main={avg_comps['loss/main']:.4f} "
                f"masked={avg_comps['stat/masked_frac']:.3f}"
                f"{vstr}  ({elapsed:.0f}s)",
                flush=True,
            )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    ema.apply_shadow()
    torch.save({
        "epoch":          dcfg.epochs,
        "denoiser_state": denoiser.state_dict(),
        "encoder_state":  encoder.state_dict(),
        "ema_state":      ema.state_dict(),
    }, out_dir / "denoiser_final.pt")
    ema.restore()

    total_time = time.time() - t_start
    print(f"\n[done] Training complete in {total_time/3600:.2f}h", flush=True)
    print(f"[done] Best val connectivity yield: {best_conn_yield:.4f}", flush=True)

    # ── Gate check ────────────────────────────────────────────────────────────
    gates = {
        "conn_yield_pass":     best_conn_yield > 0.80,
        "best_conn_yield":     best_conn_yield,
        "total_train_time_h":  total_time / 3600,
    }
    print(f"\n{'='*50}", flush=True)
    print(f"  PHASE 3 GATE CHECK", flush=True)
    print(f"{'='*50}", flush=True)
    for k, v in gates.items():
        print(f"  {k}: {v}", flush=True)

    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(gates, f, indent=2)
    print(f"[done] Summary saved to {summary_path}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
