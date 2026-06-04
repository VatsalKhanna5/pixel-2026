"""
src/losses/topology_losses.py
==============================
Differentiable topology and manufacturability losses for Phase 4 guided sampling.

All losses operate on x̂₀ ∈ [0,1]^(B,15,15) — the soft expected-x₀ from the
denoiser.  They are NOT training losses; they are inference-time guidance
signals that modify the denoiser's output logits.

Loss components (combined in physics_guidance.py):
    L_guided = L_physics + λ_topo·L_topo + λ_mfg·L_DRC

    L_topo: connectivity discriminator guidance  — encourages port-to-port path
    L_DRC:  manufacturability                    — discourages thin isolated traces

DRC detail (v2-master-context §15.1):
    Minimum trace width:  ≥ 2 pixels (penalise isolated single-pixel conductors)
    Minimum spacing:      ≥ 2 pixels (penalise thin gaps / thin dielectric regions)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Connectivity loss (using trained discriminator)
# ---------------------------------------------------------------------------

def connectivity_loss(
    x_hat_0: torch.Tensor,          # (B, H, W) soft layout ∈ [0,1]
    discriminator,                   # ConnectivityDiscriminator
    port_map: torch.Tensor,          # (1, H, W) or (B, 1, H, W)
) -> torch.Tensor:
    """
    L_topo = -log D_conn(x̂₀)   ∈ [0, ∞)

    Maximising D_conn ≡ minimising -log D_conn.
    Gradient ∇_{x̂₀} L_topo pushes the soft layout towards configurations
    the discriminator considers connected.
    """
    B = x_hat_0.shape[0]
    pm = port_map.float()
    while pm.ndim < 4: pm = pm.unsqueeze(0)
    pm = pm.expand(B, -1, -1, -1)

    x_in = torch.cat([x_hat_0.unsqueeze(1), pm], dim=1)  # (B, 2, H, W)
    d = discriminator(x_in)                                # (B, 1)
    return -torch.log(d.clamp(min=1e-8)).mean()


# ---------------------------------------------------------------------------
# DRC losses (pure convolution — no extra modules needed)
# ---------------------------------------------------------------------------

# 4-connected kernel (cross shaped, excluding centre)
_CROSS_KERNEL = torch.tensor(
    [[0., 1., 0.],
     [1., 0., 1.],
     [0., 1., 0.]], dtype=torch.float32
).view(1, 1, 3, 3)


def _get_cross(device: torch.device) -> torch.Tensor:
    return _CROSS_KERNEL.to(device)


def width_loss(x_hat_0: torch.Tensor) -> torch.Tensor:
    """
    Penalise isolated single-pixel conductors (minimum trace width < 2 px).

    Formula (v2 §15.2):
        L_width = Σ_{i,j} x̃_{ij} · ReLU(1 − Σ_{neighbours} x̃_{n})

    A conductor pixel with zero conductor neighbours is "isolated" and
    receives maximum penalty 1.  The penalty scales smoothly with the
    neighbour sum, so gradients always exist.
    """
    x = x_hat_0.unsqueeze(1)                           # (B, 1, H, W)
    kernel = _get_cross(x.device)
    neighbour_sum = F.conv2d(x, kernel, padding=1)     # (B, 1, H, W)
    penalty = x * F.relu(1.0 - neighbour_sum)          # isolated pixel penalty
    return penalty.mean()


def spacing_loss(x_hat_0: torch.Tensor) -> torch.Tensor:
    """
    Penalise traces narrower than 2 pixels via differentiable morphological erosion.

    Erosion via min-pooling: eroded_x = min-pool(x, 3×3).
    A conductor pixel that "erodes away" (eroded=0) is part of a thin feature.
    Penalise such pixels proportional to their softness.
    """
    x = x_hat_0.unsqueeze(1)                           # (B, 1, H, W)
    # Differentiable erosion: negate, max-pool, negate back
    eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
    thin_feature = F.relu(x - eroded)                  # on but eroded away → thin
    return thin_feature.mean()


def drc_loss(x_hat_0: torch.Tensor) -> torch.Tensor:
    """
    Combined DRC loss: L_DRC = L_width + L_spacing
    Both normalised to similar scale by construction.
    """
    return width_loss(x_hat_0) + spacing_loss(x_hat_0)
