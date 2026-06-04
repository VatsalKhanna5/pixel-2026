"""
src/training/train_baselines.py
================================
Phase 5 — Train Det-CNN and cVAE baselines for PIXEL comparison.

Both models are trained on the same 80% train split as PIXEL components,
using the same HDF5 dataset and data pipeline.

Det-CNN:  BCE loss between predicted soft layout and ground-truth binary layout.
cVAE:     BCE reconstruction + β·KL divergence (β=1.0).

Usage:
    python -m src.training.train_baselines \\
        --model det-cnn \\
        --config experiments/configs/base_config.yaml

    python -m src.training.train_baselines \\
        --model cvae \\
        --config experiments/configs/base_config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from src.evaluation.baselines import DetCNN, cVAE
from src.utils.config import load_config, set_seed

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PixelDataset(Dataset):
    """Layout + S-param dataset for baseline training."""

    def __init__(self, h5_path: str, indices: np.ndarray, phase_norm: bool = True) -> None:
        self.h5_path    = h5_path
        self.indices    = indices
        self.phase_norm = phase_norm
        with h5py.File(h5_path, "r") as f:
            self.s11m = f["S11_mag"][indices].astype(np.float32)
            self.s21m = f["S21_mag"][indices].astype(np.float32)
            self.s11p = f["S11_phase"][indices].astype(np.float32)
            self.s21p = f["S21_phase"][indices].astype(np.float32)
            self.layouts = f["layout"][indices].astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        pi = math.pi if self.phase_norm else 1.0
        y = np.stack([self.s11m[i], self.s21m[i],
                      self.s11p[i] / pi, self.s21p[i] / pi], axis=0)  # (4, N_f)
        x = self.layouts[i][None]   # (1, 15, 15)
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------------------------------------------------------------------------
# Det-CNN training
# ---------------------------------------------------------------------------

def train_det_cnn(cfg, device: torch.device, out_dir: Path) -> None:
    bcfg = cfg.get("baselines", None)
    lr         = float(bcfg.det_cnn.lr)       if bcfg else 3e-4
    epochs     = int(bcfg.det_cnn.epochs)     if bcfg else 100
    batch_size = int(bcfg.det_cnn.batch_size) if bcfg else 512
    embed_dim  = cfg.encoder.embed_dim

    model = DetCNN(in_channels=cfg.encoder.in_channels,
                   embed_dim=embed_dim, base_ch=64).to(device)
    print(f"[Det-CNN] params: {model.n_params():,}", flush=True)

    h5 = cfg.paths.raw_data + "/pixel_dataset.h5"
    with h5py.File(h5, "r") as f:
        N = f["layout"].shape[0]
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(N)
    n_train = int(N * cfg.dataset.train_frac)
    n_val   = int(N * cfg.dataset.val_frac)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]

    train_ds = PixelDataset(h5, train_idx)
    val_ds   = PixelDataset(h5, val_idx)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    wandb.init(project=cfg.wandb.project + "-baselines", name="det_cnn_v1",
               config={"model": "det_cnn", "lr": lr, "epochs": epochs,
                       "embed_dim": embed_dim},
               mode=cfg.wandb.get("mode", "offline"), dir=str(out_dir))

    best_val = float("inf")
    ckpt_path = out_dir / "det_cnn_best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            pred = model(y)                              # (B, 1, 15, 15)
            loss = F.binary_cross_entropy(pred, x)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(y)
                val_loss += F.binary_cross_entropy(pred, x).item()
        val_loss /= len(val_dl)
        sched.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "epoch": ep, "val_loss": val_loss}, ckpt_path)

        if ep % 10 == 0 or ep == 1:
            print(f"  [ep {ep:3d}/{epochs}] train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}", flush=True)
            wandb.log({"det_cnn/train_loss": tr_loss, "det_cnn/val_loss": val_loss, "epoch": ep})

    print(f"[Det-CNN] done. best_val_loss={best_val:.4f}  ckpt={ckpt_path}", flush=True)
    with open(out_dir / "det_cnn_summary.json", "w") as f:
        json.dump({"best_val_loss": best_val, "epochs": epochs}, f, indent=2)
    wandb.finish()


# ---------------------------------------------------------------------------
# cVAE training
# ---------------------------------------------------------------------------

def train_cvae(cfg, device: torch.device, out_dir: Path) -> None:
    bcfg = cfg.get("baselines", None)
    lr         = float(bcfg.cvae.lr)       if bcfg else 3e-4
    epochs     = int(bcfg.cvae.epochs)     if bcfg else 100
    batch_size = int(bcfg.cvae.batch_size) if bcfg else 512
    latent_dim = int(bcfg.cvae.latent_dim) if bcfg else 64
    beta       = float(bcfg.cvae.beta)     if bcfg else 1.0
    embed_dim  = cfg.encoder.embed_dim

    model = cVAE(in_channels=cfg.encoder.in_channels, embed_dim=embed_dim,
                 latent_dim=latent_dim, base_ch=64, beta=beta).to(device)
    print(f"[cVAE] params: {model.n_params():,}", flush=True)

    h5 = cfg.paths.raw_data + "/pixel_dataset.h5"
    with h5py.File(h5, "r") as f:
        N = f["layout"].shape[0]
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(N)
    n_train = int(N * cfg.dataset.train_frac)
    n_val   = int(N * cfg.dataset.val_frac)
    train_ds = PixelDataset(h5, idx[:n_train])
    val_ds   = PixelDataset(h5, idx[n_train:n_train + n_val])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    wandb.init(project=cfg.wandb.project + "-baselines", name="cvae_v1",
               config={"model": "cvae", "lr": lr, "epochs": epochs,
                       "latent_dim": latent_dim, "beta": beta},
               mode=cfg.wandb.get("mode", "offline"), dir=str(out_dir))

    best_val = float("inf")
    ckpt_path = out_dir / "cvae_best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        tr = {"loss": 0.0, "bce": 0.0, "kl": 0.0}
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            losses = model.loss(x, y)
            opt.zero_grad(); losses["loss"].backward(); opt.step()
            for k in tr: tr[k] += losses[k].item()
        for k in tr: tr[k] /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                val_loss += model.loss(x, y)["loss"].item()
        val_loss /= len(val_dl)
        sched.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state": model.state_dict(),
                        "epoch": ep, "val_loss": val_loss}, ckpt_path)

        if ep % 10 == 0 or ep == 1:
            print(f"  [ep {ep:3d}/{epochs}] loss={tr['loss']:.4f}  bce={tr['bce']:.4f}  "
                  f"kl={tr['kl']:.4f}  val={val_loss:.4f}", flush=True)
            wandb.log({"cvae/train_loss": tr["loss"], "cvae/bce": tr["bce"],
                       "cvae/kl": tr["kl"], "cvae/val_loss": val_loss, "epoch": ep})

    print(f"[cVAE] done. best_val_loss={best_val:.4f}  ckpt={ckpt_path}", flush=True)
    with open(out_dir / "cvae_summary.json", "w") as f:
        json.dump({"best_val_loss": best_val, "epochs": epochs,
                   "latent_dim": latent_dim, "beta": beta}, f, indent=2)
    wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["det-cnn", "cvae", "both"], default="both")
    parser.add_argument("--config", default="experiments/configs/base_config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/baselines_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[init] device={device}  model={args.model}", flush=True)

    if args.model in ("det-cnn", "both"):
        print("\n=== Training Det-CNN ===", flush=True)
        train_det_cnn(cfg, device, out_dir)

    if args.model in ("cvae", "both"):
        print("\n=== Training cVAE ===", flush=True)
        train_cvae(cfg, device, out_dir)


if __name__ == "__main__":
    main()
