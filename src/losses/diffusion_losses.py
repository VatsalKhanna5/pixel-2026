"""
src/losses/diffusion_losses.py
================================
Training objectives for the D3PM absorbing-state denoiser.

Two complementary losses:

1. L_main (masked CE):
   Cross-entropy on predicting x_0 only at MASKED positions.
   This is the "direct" D3PM loss for absorbing diffusion — equivalent to
   the ELBO since at unmasked positions the model trivially sees x_0.

2. L_aux (full CE):
   Cross-entropy on ALL pixel positions, regardless of masking.
   Forces the spectral encoder embedding to explain every pixel, not just
   masked ones — prevents the encoder from being ignored when the layout
   is mostly visible at low t.

   Combined: L = L_main + λ_aux * L_aux

   Note on timestep weighting: at low t, very few pixels are masked so L_main
   has tiny magnitude; at high t, almost all pixels are masked so L_main
   dominates.  The sum across a uniformly-sampled batch of timesteps gives
   an unbiased estimate of the ELBO without explicit per-timestep weighting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple

MASK_TOKEN = 2


def diffusion_loss(
    logits:      torch.Tensor,   # (B, 2, H, W)  raw logits for {0,1}
    x_0:         torch.Tensor,   # (B, H, W)     binary ground truth {0,1}
    x_t:         torch.Tensor,   # (B, H, W)     noisy layout {0,1,MASK}
    lambda_aux:  float = 0.5,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined masked + auxiliary cross-entropy loss.

    Args:
        logits:     (B, 2, H, W)  denoiser output
        x_0:        (B, H, W)     clean binary layout
        x_t:        (B, H, W)     noisy layout (has MASK tokens)
        lambda_aux: weight for auxiliary full-image loss

    Returns:
        (total_loss, component_dict)
    """
    B, _, H, W = logits.shape

    # ── 1. Main loss: only on masked pixels ─────────────────────────────────
    mask = (x_t == MASK_TOKEN)   # (B, H, W)  bool

    n_masked = mask.sum().clamp(min=1)

    if mask.any():
        logits_m = logits.permute(0, 2, 3, 1)[mask]  # (n_masked, 2)
        x0_m     = x_0[mask].long()                  # (n_masked,)
        L_main   = F.cross_entropy(logits_m, x0_m)
    else:
        # No masked tokens in this batch (extremely rare at very small t)
        L_main = logits.sum() * 0.0   # zero but keeps the graph

    # ── 2. Auxiliary loss: all pixels ────────────────────────────────────────
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, 2)  # (B*H*W, 2)
    x0_flat     = x_0.reshape(-1).long()                      # (B*H*W,)
    L_aux       = F.cross_entropy(logits_flat, x0_flat)

    total = L_main + lambda_aux * L_aux

    return total, {
        "loss/total":  total.item(),
        "loss/main":   L_main.item(),
        "loss/aux":    L_aux.item(),
        "stat/masked_frac": (mask.float().mean()).item(),
    }


def compute_accuracy(
    logits: torch.Tensor,
    x_0:    torch.Tensor,
    x_t:    torch.Tensor,
) -> dict:
    """Compute prediction accuracy for monitoring (no gradient)."""
    with torch.no_grad():
        pred = logits.argmax(dim=1)          # (B, H, W)
        mask = (x_t == MASK_TOKEN)

        # Accuracy on masked positions only
        if mask.any():
            acc_masked = (pred[mask] == x_0[mask]).float().mean().item()
        else:
            acc_masked = float("nan")

        # Overall accuracy
        acc_all = (pred == x_0).float().mean().item()

    return {
        "acc/masked": acc_masked,
        "acc/all":    acc_all,
    }
