"""
src/models/connectivity_disc.py
================================
Connectivity Discriminator for PIXEL-2026 Phase 4.

Predicts P(layout is port-to-port connected) ∈ [0,1] from a soft
binary layout map.  Used as a differentiable topology loss during
physics-guided sampling:

    L_topo = -log D_conn(x̂₀)

Architecture (from v2-master-context §13.1):
    Input: (B, 2, 15, 15)  Ch0=soft layout ∈[0,1], Ch1=port map ∈{0,1}
    Conv(2→32, 3×3) → GELU
    ResBlock × 3 (32 ch)
    GlobalAveragePool → Linear(32→16) → GELU → Linear(16→1) → Sigmoid
    D_conn(x) ∈ [0,1]

Training: binary cross-entropy on a balanced dataset of connected
(from our procedural 343k) and disconnected (fabricated negatives)
layouts.  See src/training/train_discriminator.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscResBlock(nn.Module):
    """Pre-activation ResBlock with GELU — consistent with surrogate/denoiser."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.bn1(x)))
        h = self.conv2(F.gelu(self.bn2(h)))
        return x + h


class ConnectivityDiscriminator(nn.Module):
    """
    Differentiable connectivity predictor.

    Inputs a 2-channel (layout + port_map) float tensor and predicts
    the probability that the layout contains a conducting path from
    port1=[7,0] to port2=[7,14].

    Used in two roles:
      1. During discriminator training: classify real connected/disconnected
      2. During guided sampling: gradient ∇_{x̂₀} (-log D_conn(x̂₀)) nudges
         the predicted x_0 towards topologically connected layouts.
    """

    def __init__(self, in_ch: int = 2, base_ch: int = 32) -> None:
        super().__init__()
        self.stem   = nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False)
        self.stage  = nn.Sequential(*[DiscResBlock(base_ch) for _ in range(3)])
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(base_ch, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 15, 15)  Ch0=layout, Ch1=port_map
        Returns:
            (B, 1)  connectivity probability ∈ [0, 1]
        """
        h = F.gelu(self.stem(x))
        h = self.stage(h)
        return self.head(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
