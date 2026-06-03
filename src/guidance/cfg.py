"""
src/guidance/cfg.py
====================
Discrete Classifier-Free Guidance (CFG) for D3PM denoiser.

CORRECTED FORMULATION (V2 — not the continuous Gaussian CFG):

    log p̃(x₀|x_t, c_y) = (1+w)·log p_θ(x₀|x_t, c_y)
                         - w·log p_θ(x₀|x_t, ∅)

Since each pixel's distribution is predicted independently, this
factorises per-pixel and per-class:

    log p̃(x₀=v | x_t, c_y) ∝ (1+w)·log p_cond(v) - w·log p_uncond(v)

This modifies the categorical distribution over {0,1} per pixel, sharpening
it toward classes predicted more under the condition than the null condition.

Training requirement: condition dropout with p_drop = 0.15 so that the model
learns both p_θ(x₀|x_t, c_y) and p_θ(x₀|x_t, ∅) in a single network.

Guidance weight w:
    w = 0    → unconditional (pure prior)
    w = 2.0  → balanced (recommended starting point)
    w >> 0   → deterministic (high spectral fidelity, low diversity risk)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def discrete_cfg(
    logits_cond:   torch.Tensor,   # (B, 2, H, W)  conditional logits
    logits_uncond: torch.Tensor,   # (B, 2, H, W)  unconditional logits
    w: float = 2.0,
) -> torch.Tensor:
    """
    Apply discrete CFG in log-probability space.

    Returns log-normalised CFG probabilities (B, 2, H, W).
    These are log-probabilities, NOT logits — use .exp() to get probs.
    """
    log_p_cond   = F.log_softmax(logits_cond,   dim=1)   # (B, 2, H, W)
    log_p_uncond = F.log_softmax(logits_uncond, dim=1)

    # Guided log-probabilities
    log_p_guided = (1.0 + w) * log_p_cond - w * log_p_uncond   # (B, 2, H, W)

    # Re-normalise to valid log-probabilities
    log_p_guided = log_p_guided - torch.logsumexp(log_p_guided, dim=1, keepdim=True)

    return log_p_guided   # (B, 2, H, W) normalised log-probs


def apply_cfg_to_logits(
    denoiser_fn,           # callable: (x_t, t, c_y, port_map) → logits (B,2,H,W)
    x_t:      torch.Tensor,
    t:        torch.Tensor,
    c_y:      torch.Tensor,
    port_map: torch.Tensor,
    w: float = 2.0,
) -> torch.Tensor:
    """
    Run both conditional and unconditional forward passes and return
    the CFG log-probabilities.

    Args:
        denoiser_fn: the denoiser forward function
        x_t:         (B, H, W) noisy layout
        t:           (B,) timestep
        c_y:         (B, 256) spectral condition
        port_map:    port binary map
        w:           CFG guidance weight

    Returns:
        log_p_guided: (B, 2, H, W) CFG log-probabilities
    """
    logits_cond   = denoiser_fn(x_t, t, c_y,  port_map)
    logits_uncond = denoiser_fn(x_t, t, None,  port_map)
    return discrete_cfg(logits_cond, logits_uncond, w=w)
