"""
src/models/diffusion.py
========================
D3PM absorbing-state forward process for PIXEL-2026 Phase 3.

State space: ternary {0=dielectric, 1=conductor, MASK=2}

Forward process:
    Each non-MASK token at x_0 can be absorbed (become MASK) with cumulative
    probability β̄_t = 1 - ᾱ_t, where ᾱ_t = ∏_{s=1}^{t} (1-β_s).
    At t=T: ᾱ_T ≈ 0 → essentially every token is masked.

    q(x_t=MASK | x_0∈{0,1}) = 1 - ᾱ_t
    q(x_t=x_0  | x_0∈{0,1}) = ᾱ_t

Posterior (for sampling):
    q(x_{t-1}=x_0 | x_t=MASK, x_0) = β_t·ᾱ_{t-1} / (1-ᾱ_t)
    q(x_{t-1}=MASK | x_t=MASK, x_0) = (1-ᾱ_{t-1}) / (1-ᾱ_t)
    If x_t ≠ MASK: x_{t-1} = x_t (no change, token was not yet masked)

Schedule: linear beta from β_min=1e-4 to β_max=0.02 over T=1000 steps.
    ᾱ_T = exp(-∑β_t) ≈ exp(-10) ≈ 4.5e-5  → 99.995% masked at T. ✓
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Tuple

MASK = 2   # integer token index for the absorbing MASK state


class D3PMAbsorbing:
    """
    D3PM forward process with absorbing MASK state.

    All schedules are pre-computed and stored as buffers on the target device.
    Call `.to(device)` after construction.
    """

    def __init__(
        self,
        T: int          = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
    ) -> None:
        self.T    = T
        self.MASK = MASK

        # Beta schedule: linearly spaced, shape (T,) indexed [0..T-1] = [β_1..β_T]
        betas = torch.linspace(beta_min, beta_max, T)   # (T,)

        alphas          = 1.0 - betas                   # (T,)  1-β_t
        alphas_cumprod  = torch.cumprod(alphas, dim=0)  # (T,)  ᾱ_t

        # ᾱ_{t-1}: prepend 1.0 so index [t] gives ᾱ_{t-1} when t≥1
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register = {
            "betas":                betas,
            "alphas":               alphas,
            "alphas_cumprod":       alphas_cumprod,
            "alphas_cumprod_prev":  alphas_cumprod_prev,
        }

    def to(self, device: torch.device) -> "D3PMAbsorbing":
        self.register = {k: v.to(device) for k, v in self.register.items()}
        return self

    def _get(self, key: str, t: torch.Tensor) -> torch.Tensor:
        """Gather schedule value at timestep indices t (1-indexed)."""
        # t: (B,)  integers in [1, T]
        vals = self.register[key]          # (T,)
        return vals[t - 1]                 # (B,)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_t ~ q(x_t | x_0) for a batch of layouts.

        Args:
            x_0: (B, H, W)  binary layout, values in {0, 1}
            t:   (B,)       timestep indices, integers in [1, T]
        Returns:
            x_t: (B, H, W)  noisy layout, values in {0, 1, MASK=2}
        """
        alpha_bar = self._get("alphas_cumprod", t)   # (B,)
        alpha_bar = alpha_bar[:, None, None]          # (B, 1, 1) for broadcast

        # For each pixel: stay at x_0 with prob ᾱ_t, become MASK with prob 1-ᾱ_t
        noise     = torch.rand_like(x_0.float())     # (B, H, W) uniform [0,1)
        mask_flag = (noise > alpha_bar).long()       # 1 where should be masked

        x_t = x_0.clone()
        x_t[mask_flag == 1] = MASK
        return x_t   # (B, H, W) long in {0, 1, 2}

    # ------------------------------------------------------------------
    # Posterior (used during sampling)
    # ------------------------------------------------------------------

    def posterior_sample(
        self,
        x_t: torch.Tensor,
        x_0_logits: torch.Tensor,
        t: torch.Tensor,
        cfg_log_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} ~ p_θ(x_{t-1} | x_t) using the posterior formula
        and the network's predicted x_0 distribution.

        For masked tokens:
            P(x_{t-1}=x_0) = β_t · ᾱ_{t-1} / (1-ᾱ_t)  × p_θ(x_0|x_t)
            P(x_{t-1}=MASK) = (1-ᾱ_{t-1}) / (1-ᾱ_t)

        For unmasked tokens: x_{t-1} = x_t (deterministic, no change).

        Args:
            x_t:           (B, H, W) long  current noisy layout {0,1,MASK}
            x_0_logits:    (B, 2, H, W) float  raw logits for {0,1}
            t:             (B,) long  current timestep [1..T]
            cfg_log_probs: (B, 2, H, W) optional pre-computed CFG log-probs
                           (if provided, overrides x_0_logits for sampling)
        Returns:
            x_{t-1}: (B, H, W) long
        """
        B, H, W = x_t.shape
        device  = x_t.device

        # x_0 probabilities
        if cfg_log_probs is not None:
            p_x0 = cfg_log_probs.exp()    # (B, 2, H, W)  normalised CFG probs
        else:
            p_x0 = F.softmax(x_0_logits, dim=1)   # (B, 2, H, W)

        alpha_bar      = self._get("alphas_cumprod",      t).to(device)  # (B,)
        alpha_bar_prev = self._get("alphas_cumprod_prev", t).to(device)
        beta           = self._get("betas",               t).to(device)

        # Posterior weights for masked tokens — need (B,1,1,1) to broadcast with (B,C,H,W)
        beta_v = beta[:, None, None, None]           # (B, 1, 1, 1)
        ab_v   = alpha_bar_prev[:, None, None, None]
        denom  = (1.0 - alpha_bar)[:, None, None, None]  # 1-ᾱ_t

        # P(x_{t-1}=v) for each binary value v — conditional on x_t=MASK
        p_unmask = (beta_v * ab_v / denom.clamp(1e-8))   # (B, 1, 1) scalar factor
        # Probability of x_{t-1} being 0: p_x0[:,0,:,:] * p_unmask
        # Probability of x_{t-1} being 1: p_x0[:,1,:,:] * p_unmask
        # Probability of x_{t-1} = MASK: (1-ᾱ_{t-1})/(1-ᾱ_t)
        p_stay_masked = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar).clamp(1e-8))
        p_stay_masked = p_stay_masked[:, None, None, None]  # (B, 1, 1, 1)

        # Build categorical probs over {0, 1, MASK} for masked positions
        # shape (B, 3, H, W)
        p0 = p_x0[:, 0:1, :, :] * p_unmask              # (B, 1, H, W)
        p1 = p_x0[:, 1:2, :, :] * p_unmask              # (B, 1, H, W)
        pm = p_stay_masked.expand(B, 1, H, W)            # (B, 1, H, W)
        probs_3 = torch.cat([p0, p1, pm], dim=1)  # (B, 3, H, W)
        probs_3 = probs_3 / probs_3.sum(dim=1, keepdim=True).clamp(1e-8)  # normalise

        # Sample from categorical
        probs_flat = probs_3.permute(0, 2, 3, 1).reshape(-1, 3)   # (B*H*W, 3)
        x_prev_masked = torch.multinomial(probs_flat, 1).squeeze(1)  # (B*H*W,)
        x_prev_masked = x_prev_masked.reshape(B, H, W)

        # For unmasked tokens: copy x_t unchanged
        is_masked = (x_t == MASK)
        x_prev = x_t.clone()
        x_prev[is_masked] = x_prev_masked[is_masked]
        return x_prev

    # ------------------------------------------------------------------
    # Helper: expected x_0 (soft, for Phase 4 guidance)
    # ------------------------------------------------------------------

    def expected_x0(self, x_0_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute soft x̂_0 = E[x_0 | x_t] = P(x_0=1 | x_t) ∈ [0,1].
        Used in Phase 4 to pass continuous input to the surrogate.

        Args:
            x_0_logits: (B, 2, H, W)  raw denoiser output logits
        Returns:
            x_hat_0: (B, H, W)  float in [0, 1]
        """
        return F.softmax(x_0_logits, dim=1)[:, 1, :, :]   # P(pixel=conductor)
