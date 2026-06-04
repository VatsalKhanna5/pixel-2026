"""
src/training/train_discriminator.py
=====================================
Phase 4 Step 4.3 — Train the connectivity discriminator.

The discriminator D_conn(x) ∈ [0,1] predicts whether a soft binary layout
has a conducting path from port1 → port2.  It is trained on a balanced
binary classification dataset:

  Positives (connected=1):  the 343k procedural layouts — all connected
  Negatives (connected=0):  fabricated disconnected layouts:
      Type A — randomly-zeroed: take a connected layout, remove the central
               column (col 6-8) + random pixels until BFS fails
      Type B — pure random: uniform binary (fill ~0.05-0.30) — most disconnected

Balance: 50/50 positive/negative.  Oversample negatives since dataset is
all-positive.

Training: BCE with logit stabilisation (label smoothing 0.05), AdamW, 50 epochs.
Gate: AUC-ROC > 0.95 on held-out test set.

Usage:
    python -m src.training.train_discriminator \\
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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.dataset.connectivity import is_connected
from src.models.connectivity_disc import ConnectivityDiscriminator
from src.training.train_surrogate import _build_port_map
from src.utils.config import load_config, set_seed

os.environ.setdefault("WANDB_MODE", "offline")
import wandb  # noqa: E402


# ---------------------------------------------------------------------------
# Negative sample generators
# ---------------------------------------------------------------------------

def _make_random_layout(rng: np.random.Generator, h: int = 15, w: int = 15) -> np.ndarray:
    """Random binary layout; fill fraction uniform in [0.03, 0.35]."""
    fill = rng.uniform(0.03, 0.35)
    layout = (rng.random((h, w)) < fill).astype(np.uint8)
    return layout


def _make_broken_layout(
    connected_layout: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 30,
) -> np.ndarray | None:
    """
    Attempt to break a connected layout by zeroing pixels until disconnected.
    Returns None if disconnected layout couldn't be produced in max_attempts.
    """
    h, w = connected_layout.shape
    for _ in range(max_attempts):
        broken = connected_layout.copy()
        # Zero out a random vertical strip (most likely to break connectivity)
        col = rng.integers(2, w - 2)
        width = rng.integers(1, 4)
        broken[:, max(0, col-width//2): col+width//2+1] = 0
        # Random additional flips for variety
        n_flips = rng.integers(0, 8)
        for _ in range(n_flips):
            r, c = rng.integers(0, h), rng.integers(0, w)
            broken[r, c] = 0
        if not is_connected(broken):
            return broken
    return None


def build_disc_dataset(
    pos_layouts: np.ndarray,   # (N, H, W) uint8  all connected
    n_samples: int,            # total samples per class
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build balanced (pos, neg) arrays for discriminator training.

    Returns:
        layouts: (2*n_samples, H, W)  float32
        labels:  (2*n_samples,)       float32  0 or 1
    """
    rng = np.random.default_rng(seed)
    H, W = pos_layouts.shape[1], pos_layouts.shape[2]
    pos_idx = rng.choice(len(pos_layouts), n_samples, replace=(len(pos_layouts) < n_samples))
    pos_data = pos_layouts[pos_idx].astype(np.float32)

    # Generate negatives
    neg_data = []
    attempts = 0
    max_total = n_samples * 10

    while len(neg_data) < n_samples and attempts < max_total:
        # Alternate between random and broken layouts
        if attempts % 2 == 0:
            layout = _make_random_layout(rng, H, W)
        else:
            src = pos_layouts[rng.integers(0, len(pos_layouts))]
            layout = _make_broken_layout(src, rng)
            if layout is None:
                attempts += 1
                continue
        if not is_connected(layout):
            neg_data.append(layout.astype(np.float32))
        attempts += 1

    # Pad with randoms if needed
    while len(neg_data) < n_samples:
        layout = _make_random_layout(rng, H, W)
        if not is_connected(layout):
            neg_data.append(layout.astype(np.float32))

    neg_data = np.array(neg_data[:n_samples])

    layouts = np.concatenate([pos_data, neg_data], axis=0)
    labels  = np.concatenate([
        np.ones(n_samples, dtype=np.float32),
        np.zeros(n_samples, dtype=np.float32),
    ])
    shuffle = rng.permutation(len(layouts))
    return layouts[shuffle], labels[shuffle]


class DiscDataset(Dataset):
    def __init__(
        self,
        layouts: np.ndarray,   # (N, H, W)
        labels:  np.ndarray,   # (N,)
        port_map: torch.Tensor,
        augment: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.layouts  = torch.from_numpy(layouts)     # (N, H, W) float32
        self.labels   = torch.from_numpy(labels)      # (N,) float32
        self.port_map = port_map                       # (1, H, W)
        self.augment  = augment
        self.rng      = rng or np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.layouts)

    def __getitem__(self, idx: int):
        layout = self.layouts[idx]               # (H, W)
        if self.augment:
            # Horizontal and vertical flips (preserve port positions for ports
            # at [7,0] and [7,14], flip is NOT valid; skip flips; use only noise)
            noise = torch.randn_like(layout) * 0.03
            layout = (layout + noise).clamp(0, 1)
        x = torch.cat([layout.unsqueeze(0), self.port_map], dim=0)  # (2, H, W)
        return x, self.labels[idx]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PIXEL-2026 Phase 4: Connectivity Discriminator")
    parser.add_argument("--config", default="experiments/configs/base_config.yaml")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    dcfg = cfg.discriminator
    set_seed(cfg.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path("experiments/discriminator_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] device={device}  output={out_dir}", flush=True)

    # ── Load connected layouts from dataset ──────────────────────────────────
    h5_path = Path(cfg.paths.raw_data) / "pixel_dataset.h5"
    print(f"[data] Loading layouts from {h5_path} …", flush=True)
    split_info = np.load("experiments/surrogate_v1/split_indices.npz")
    train_idx  = np.sort(split_info["train"])
    val_idx    = np.sort(split_info["val"])
    test_idx   = np.sort(split_info["test"])

    with h5py.File(h5_path, "r") as f:
        train_layouts = f["layout"][train_idx]   # (N_train, 15, 15)
        val_layouts   = f["layout"][val_idx]
        test_layouts  = f["layout"][test_idx]

    n_per_class = 30_000   # 30k pos + 30k neg = 60k total per split
    port_map = _build_port_map(cfg)   # (1, 15, 15)

    print(f"[data] Building balanced discriminator datasets …", flush=True)
    tr_lays, tr_labs = build_disc_dataset(train_layouts, n_per_class, seed=cfg.seed)
    va_lays, va_labs = build_disc_dataset(val_layouts,   n_per_class // 5, seed=cfg.seed+1)
    te_lays, te_labs = build_disc_dataset(test_layouts,  n_per_class // 5, seed=cfg.seed+2)

    pos_rate = tr_labs.mean()
    print(f"[data] Train: {len(tr_lays):,}  pos_rate={pos_rate:.3f}", flush=True)

    train_ds = DiscDataset(tr_lays, tr_labs, port_map, augment=True)
    val_ds   = DiscDataset(va_lays, va_labs, port_map)
    test_ds  = DiscDataset(te_lays, te_labs, port_map)

    train_loader = DataLoader(train_ds, batch_size=dcfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=dcfg.batch_size * 2,
                              shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConnectivityDiscriminator(in_ch=2, base_ch=cfg.surrogate.base_channels).to(device)
    print(f"[model] Discriminator params: {model.n_params():,}", flush=True)

    # Pos weight for BCE: handle class imbalance (50/50 → weight=1.0)
    pos_weight = torch.tensor([dcfg.pos_weight], device=device)
    criterion  = nn.BCELoss()

    optimiser = torch.optim.AdamW(model.parameters(), lr=dcfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=dcfg.epochs, eta_min=1e-6)

    wandb.init(project=cfg.wandb.project + "-disc",
               name="disc_v1",
               config={**OmegaConf.to_container(dcfg, resolve=True), "n_per_class": n_per_class},
               mode=cfg.wandb.get("mode", "offline"),
               dir=str(out_dir))

    best_auc   = 0.0
    t_start    = time.time()
    best_ckpt  = out_dir / "disc_best.pt"

    print(f"\n[train] {dcfg.epochs} epochs, {len(train_loader)} batches/epoch", flush=True)

    for epoch in range(1, dcfg.epochs + 1):
        model.train()
        train_loss = 0.0

        for x_b, y_b in train_loader:
            x_b = x_b.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            pred = model(x_b).squeeze(1)   # (B,)
            loss = criterion(pred, y_b)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        scheduler.step()

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                val_preds.append(model(x_b).squeeze(1).cpu())
                val_labels.append(y_b)

        val_preds  = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_auc    = roc_auc_score(val_labels, val_preds)
        val_acc    = ((val_preds > 0.5) == val_labels.astype(bool)).mean()

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_auc": val_auc},
                       best_ckpt)

        log = {"disc/train_loss": train_loss / len(train_loader),
               "disc/val_auc": val_auc, "disc/val_acc": val_acc}
        wandb.log(log, step=epoch)

        if epoch % 10 == 0 or epoch == 1 or epoch == dcfg.epochs:
            elapsed = time.time() - t_start
            gate_ok = "✅" if val_auc > 0.95 else "..."
            print(f"  [ep {epoch:3d}/{dcfg.epochs}] "
                  f"loss={train_loss/len(train_loader):.4f}  "
                  f"val_auc={val_auc:.4f} {gate_ok}  "
                  f"val_acc={val_acc:.4f}  ({elapsed:.0f}s)", flush=True)

    # ── Test set evaluation ──────────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=dcfg.batch_size * 2, shuffle=False, num_workers=0)
    test_preds, test_labels = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            test_preds.append(model(x_b.to(device)).squeeze(1).cpu())
            test_labels.append(y_b)

    test_preds  = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_auc    = roc_auc_score(test_labels, test_preds)
    test_acc    = ((test_preds > 0.5) == test_labels.astype(bool)).mean()

    print(f"\n[test] AUC={test_auc:.4f}  ACC={test_acc:.4f}  "
          f"(gate AUC > 0.95: {'✅ PASS' if test_auc > 0.95 else '❌ FAIL'})", flush=True)

    summary = {"best_val_auc": best_auc, "test_auc": float(test_auc),
               "test_acc": float(test_acc), "test_pass": bool(test_auc > 0.95)}
    with open(out_dir / "disc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    wandb.finish()


if __name__ == "__main__":
    main()
