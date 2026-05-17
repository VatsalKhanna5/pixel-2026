"""
src/utils/binarization.py
PIXEL-2026 — Differentiable binarization utilities.

Implements:
  - Gumbel-Sigmoid with temperature annealing (Maddison et al. 2016)
  - Straight-Through Estimator (Bengio et al. 2013)
  - Temperature annealing schedule

Used during surrogate training and physics-guided inference.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Gumbel-Sigmoid
# ---------------------------------------------------------------------------

def gumbel_sigmoid(
    logits: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """
    Gumbel-Sigmoid reparameterised sample.

    For binary variables, this is the Bernoulli equivalent of Gumbel-Softmax.

    Args:
        logits: Raw logits (before sigmoid), shape (*,)
        tau:    Temperature (→ 0 gives hard samples, = 1 gives soft samples)
        hard:   If True, return hard {0,1} in forward, soft gradients in backward
                (straight-through mode).
        eps:    Numerical stability floor for log.

    Returns:
        Soft (or hard-via-STE) sample in [0, 1], same shape as logits.
    """
    # Gumbel noise: two independent Gumbel(0,1) samples for Bernoulli
    u = torch.clamp(torch.rand_like(logits), eps, 1.0 - eps)
    gumbel_noise = -torch.log(-torch.log(u))
    # Soft sample
    soft = torch.sigmoid((logits + gumbel_noise) / tau)
    if hard:
        hard_sample = (soft > 0.5).float()
        # STE: use soft gradients
        soft = hard_sample - soft.detach() + soft
    return soft


def straight_through_binarize(x: Tensor) -> Tensor:
    """
    Hard binarization in forward pass, identity gradient in backward pass
    (Straight-Through Estimator, Bengio et al. 2013).

    Args:
        x: Values in [0, 1] (e.g. sigmoid output or soft probabilities).

    Returns:
        Binary {0.0, 1.0} tensor with straight-through gradients.
    """
    binary = (x > 0.5).float()
    return binary - x.detach() + x     # STE


# ---------------------------------------------------------------------------
# Temperature annealing
# ---------------------------------------------------------------------------

def cosine_anneal_tau(
    step: int,
    total_steps: int,
    tau_init: float = 1.0,
    tau_final: float = 0.01,
) -> float:
    """
    Cosine annealing schedule for Gumbel-Sigmoid temperature.

    Returns tau ∈ [tau_final, tau_init] decreasing from tau_init to tau_final.
    """
    progress = min(step / max(total_steps, 1), 1.0)
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tau_final + (tau_init - tau_final) * cosine_val


def exponential_anneal_tau(
    step: int,
    total_steps: int,
    tau_init: float = 1.0,
    tau_final: float = 0.01,
) -> float:
    """
    Exponential annealing schedule.  More aggressive initial cooling.
    """
    if total_steps <= 0:
        return tau_final
    ratio = math.log(tau_final / tau_init)
    return tau_init * math.exp(ratio * step / total_steps)


# ---------------------------------------------------------------------------
# Expected-x0 extraction from D3PM logits
# ---------------------------------------------------------------------------

def expected_x0_from_logits(logits: Tensor) -> Tensor:
    """
    Compute E[x_0 | x_t] from the denoiser's per-pixel logits.

    In D3PM with absorbing state, the denoiser outputs logits over
    {0, 1, MASK} per pixel.  The "soft layout" used for surrogate
    guidance is P(pixel = 1).

    Args:
        logits: Shape (B, H, W, 3), where dim=-1 indexes {0=dielectric,
                1=conductor, 2=MASK}.

    Returns:
        Soft layout x̂₀ ∈ [0, 1]^(B, H, W) = P(pixel = conductor).
    """
    probs = F.softmax(logits, dim=-1)   # (B, H, W, 3)
    return probs[..., 1]                 # probability of state=1 (conductor)
