"""
src/guidance/physics_guidance.py
==================================
Phase 4: Physics-guided reverse diffusion sampling for PIXEL-2026.

Implements the full PIXEL inference algorithm:

  For t = T → 1:
    1.  ℓ_cond  = denoiser(x_t, t, c_y)           [conditional logits]
    2.  ℓ_uncond = denoiser(x_t, t, None)          [unconditional logits]
    3.  If t < t_thresh:
          ℓ_leaf = ℓ_cond.detach().requires_grad_(True)   [treat as leaf]
          x̂₀    = softmax(ℓ_leaf)[:,1]                    [P(pixel=1)]
          x̂₀_aug = [x̂₀, port_map]                        [surrogate input]
          ŷ, σ̂² = surrogate_ensemble(x̂₀_aug)             [mean + variance]
          L_phys = ||ŷ_mag − y*_mag||² + 0.1·||ŷ_ph − y*_ph||²
          L_topo = −log D_conn([x̂₀, port_map])
          L_DRC  = L_width(x̂₀) + L_spacing(x̂₀)
          L_guided = L_phys + λ_topo·L_topo + λ_mfg·L_DRC
          g      = ∇_{ℓ_leaf} L_guided             [autograd]
          α_t    = α_max / (σ̂ + ε) · η_t           [per-sample, (B,1,1,1)]
          ℓ̃_cond = ℓ_cond − α_t · clip(g, g_max)   [guided conditional]
        Else:
          ℓ̃_cond = ℓ_cond
    4.  log_p = CFG(ℓ̃_cond, ℓ_uncond, w)            [combine via log-prob]
    5.  x_{t-1} = posterior_sample(x_t, log_p, t)

Key theoretical points:
  - Gradient is w.r.t. LOGITS (leaf), not denoiser parameters
  - CFG acts on GUIDED conditional logits (guidance refines conditional,
    CFG sharpens against unconditional)
  - α_t is per-sample: uncertain layouts get smaller steps
  - Guidance activates at t < t_thresh = 0.4T to avoid noisy early steps
  - Phase update from Phase 2 analysis: magnitude channels weighted 10×
    over phase channels in L_physics (phase prediction is noisier)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from src.guidance.cfg import discrete_cfg
from src.losses.topology_losses import connectivity_loss, drc_loss


# ---------------------------------------------------------------------------
# Core guided sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_reverse_step(
    x_t:          torch.Tensor,       # (B, H, W) long  current noisy layout
    t_val:        int,                 # current timestep integer (T → 1)
    t_tensor:     torch.Tensor,        # (B,) long  same value broadcast
    denoiser,                          # PixelDenoiser
    encoder,                           # SpectralEncoder
    diffusion,                         # D3PMAbsorbing
    surrogate_ens,                     # SurrogateEnsemble
    discriminator,                     # ConnectivityDiscriminator
    port_map:     torch.Tensor,        # (1, 1, H, W) or broadcastable
    c_y:          Optional[torch.Tensor],  # (B, 256) spectral conditioning
    y_star:       torch.Tensor,        # (B, 4, 100) normalised target S-params
    T:            int,
    t_thresh:     int,
    alpha_max:    float,
    epsilon:      float,
    lambda_topo:  float,
    lambda_mfg:   float,
    g_max:        float,
    cfg_w:        float,
) -> torch.Tensor:
    """
    Execute one guided reverse diffusion step and return x_{t-1}.

    Guidance is applied in-graph (torch.enable_grad()) for the logit
    gradient, but the outer context is no_grad (sampling is not trained).
    """
    B, H, W = x_t.shape
    device   = x_t.device

    # ── Standard denoiser forward (no grad context here) ─────────────────
    ell_cond  = denoiser(x_t, t_tensor, c_y, port_map)    # (B, 2, H, W)
    ell_uncond = denoiser(x_t, t_tensor, None, port_map)   # (B, 2, H, W)

    # ── Physics guidance (only when t < t_thresh) ────────────────────────
    if t_val < t_thresh:
        with torch.enable_grad():
            # Treat conditional logits as leaf variable
            ell_leaf = ell_cond.detach().requires_grad_(True)

            # Soft x̂₀: P(pixel=1), differentiable through softmax
            x_hat_0 = F.softmax(ell_leaf, dim=1)[:, 1, :, :]   # (B, H, W)

            # Prepare surrogate input: layout channel + port map
            pm = port_map.float()
            while pm.ndim < 4: pm = pm.unsqueeze(0)
            pm = pm.expand(B, -1, -1, -1)
            x_aug = torch.cat([x_hat_0.unsqueeze(1), pm], dim=1)  # (B, 2, H, W)

            # Surrogate: mean prediction + uncertainty
            mean_pred, var_pred = surrogate_ens(x_aug)            # (B, 4, 100) each
            sigma = var_pred.mean(dim=(1, 2, 3)).sqrt()            # (B,) uncertainty

            # Physics loss: magnitude-weighted (phase noisier from Phase 2 analysis)
            L_phys = (
                F.mse_loss(mean_pred[:, :2, :], y_star[:, :2, :])
                + 0.1 * F.mse_loss(mean_pred[:, 2:, :], y_star[:, 2:, :])
            )

            # Topology loss: connectivity discriminator
            L_topo = connectivity_loss(x_hat_0, discriminator, port_map)

            # DRC loss: width + spacing
            L_drc = drc_loss(x_hat_0)

            L_guided = L_phys + lambda_topo * L_topo + lambda_mfg * L_drc
            L_guided.backward()

        # Gradient w.r.t. logits
        g = ell_leaf.grad.detach()          # (B, 2, H, W)

        # Per-sample timestep-decayed step size
        eta  = (1.0 - t_val / T) ** 2      # scalar, large at low t
        # α_t: (B,) → (B, 1, 1, 1)
        alpha = (alpha_max / (sigma.detach() + epsilon) * eta
                 ).clamp(max=alpha_max * 10).view(B, 1, 1, 1)

        # Clip gradient and apply
        g_clipped  = g.clamp(-g_max, g_max)
        ell_guided = ell_cond - alpha * g_clipped
    else:
        ell_guided = ell_cond

    # ── CFG over guided conditional + unconditional ───────────────────────
    if cfg_w > 0.0 and c_y is not None:
        log_p = discrete_cfg(ell_guided, ell_uncond, w=cfg_w)
        x_prev = diffusion.posterior_sample(x_t, ell_guided, t_tensor,
                                            cfg_log_probs=log_p)
    else:
        x_prev = diffusion.posterior_sample(x_t, ell_guided, t_tensor)

    return x_prev


# ---------------------------------------------------------------------------
# Full guided generation (T → 0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_guided(
    y_star:        torch.Tensor,       # (B, 4, 100) normalised target S-params
    denoiser,
    encoder,
    diffusion,
    surrogate_ens,
    discriminator,
    port_map:      torch.Tensor,       # (1, H, W) or (1, 1, H, W)
    device:        torch.device,
    *,
    T:             int   = 1000,
    t_thresh:      int   = 400,
    alpha_max:     float = 0.10,
    epsilon:       float = 0.01,
    lambda_topo:   float = 1.00,
    lambda_mfg:    float = 0.50,
    g_max:         float = 1.00,
    cfg_w:         float = 2.00,
) -> torch.Tensor:
    """
    Generate layouts conditioned on y_star using the full PIXEL algorithm.

    Args:
        y_star: (B, 4, 100)  target S-parameters (normalised, phases÷π)
        ...
    Returns:
        x_final: (B, H, W)   long  final binary layouts {0=dielectric, 1=conductor}
    """
    denoiser.eval()
    encoder.eval()
    surrogate_ens.eval()
    discriminator.eval()

    B    = y_star.shape[0]
    H, W = 15, 15

    # Spectral conditioning
    c_y = encoder(y_star.to(device))                # (B, 256)

    # Port map standardised to (1, 1, H, W)
    pm = port_map.float().to(device)
    while pm.ndim < 4: pm = pm.unsqueeze(0)

    # Start from all-MASK
    x_t = torch.full((B, H, W), diffusion.MASK, dtype=torch.long, device=device)

    for t_val in range(T, 0, -1):
        t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)
        x_t = guided_reverse_step(
            x_t, t_val, t_tensor,
            denoiser, encoder, diffusion, surrogate_ens, discriminator,
            pm, c_y, y_star.to(device),
            T=T, t_thresh=t_thresh,
            alpha_max=alpha_max, epsilon=epsilon,
            lambda_topo=lambda_topo, lambda_mfg=lambda_mfg,
            g_max=g_max, cfg_w=cfg_w,
        )

    return x_t   # (B, H, W) long {0, 1}
