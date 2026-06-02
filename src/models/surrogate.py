"""
src/models/surrogate.py
=======================
Physics Surrogate CNN ensemble for PIXEL-2026 Phase 2.

Architecture:  2-channel input → Conv stem → 3 ResBlock stages (stride-2 between
               stages) → AdaptiveAvgPool → MLP head → (4, 100) output.
Activation:    GELU throughout — smooth, no dead-neuron saturation, good gradient
               flow for the soft-input regime needed by Phase 4 guidance.
Input:         (B, 2, 15, 15)   Ch0=layout ∈[0,1], Ch1=port_map ∈{0,1}
Output:        (B, 4, 100)      [S11_mag, S21_mag, S11_phase/π, S21_phase/π]

Ensemble:      SurrogateEnsemble wraps K=5 independent models.
               forward() returns (mean, variance) per-element — variance used as
               uncertainty weight α_t in Phase 4 physics-guided sampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """
    Standard pre-activation ResBlock with GELU.

    Pre-activation (BN→act→Conv) gives slightly better gradient flow than
    post-activation for deep networks, and GELU avoids the hard-zero dead
    neuron problem of ReLU while remaining computationally efficient.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.bn1(x)))
        h = self.conv2(F.gelu(self.bn2(h)))
        return x + h


class DownBlock(nn.Module):
    """Stride-2 conv for spatial downsampling between stages."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.bn   = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.gelu(self.bn(x)))


# ---------------------------------------------------------------------------
# Single surrogate model
# ---------------------------------------------------------------------------

class PhysicsSurrogate(nn.Module):
    """
    Single CNN surrogate.  Instantiate 5 with different seeds for the ensemble.

    Spatial flow:
        (B,2,15,15) → stem → stage0(32ch,15×15) → down1 → stage1(64ch,8×8)
                    → down2 → stage2(128ch,4×4) → GAP → MLP → (B,4,100)

    Training note:
        The model is trained on continuous x̃ ∈ [0,1] (not hard binary) with
        σ=0.05 Gaussian noise on the layout channel to ensure smooth gradients
        across the entire [0,1] input manifold — a prerequisite for Phase 4
        gradient-based physics guidance.
    """

    def __init__(
        self,
        in_ch: int = 2,
        base_ch: int = 32,
        n_freq: int = 100,
    ) -> None:
        super().__init__()
        self.n_freq = n_freq

        # Stem: raw pixels → feature maps, no spatial change
        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False)

        # Stage 0: 3 ResBlocks at (base_ch, 15, 15)
        self.stage0 = nn.Sequential(*[ResBlock2D(base_ch) for _ in range(3)])

        # Downsample 15→8
        self.down1  = DownBlock(base_ch, base_ch * 2)
        # Stage 1: 3 ResBlocks at (base_ch*2, 8, 8)
        self.stage1 = nn.Sequential(*[ResBlock2D(base_ch * 2) for _ in range(3)])

        # Downsample 8→4
        self.down2  = DownBlock(base_ch * 2, base_ch * 4)
        # Stage 2: 2 ResBlocks at (base_ch*4, 4, 4)
        self.stage2 = nn.Sequential(*[ResBlock2D(base_ch * 4) for _ in range(2)])

        # Global average pooling → flatten → MLP head
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim  = base_ch * 4   # 128
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, 4 * n_freq),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He init with relu gain ≈ correct for GELU too
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, 15, 15)  float32  Ch0=layout, Ch1=port_map
        Returns:
            (B, 4, 100)  [S11_mag, S21_mag, S11_ph/π, S21_ph/π]
        """
        h = F.gelu(self.stem(x))    # (B, 32, 15, 15)
        h = self.stage0(h)           # (B, 32, 15, 15)
        h = self.down1(h)            # (B, 64,  8,  8)
        h = self.stage1(h)           # (B, 64,  8,  8)
        h = self.down2(h)            # (B, 128, 4,  4)
        h = self.stage2(h)           # (B, 128, 4,  4)
        h = self.pool(h).flatten(1)  # (B, 128)
        h = self.head(h)             # (B, 400)
        return h.reshape(-1, 4, self.n_freq)  # (B, 4, 100)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------

class SurrogateEnsemble(nn.Module):
    """
    Ensemble of K PhysicsSurrogate models.

    Used at Phase 4 inference to provide:
      - Prediction mean  μ̂(x) = (1/K) Σ f_k(x)       → spectral guidance signal
      - Prediction var   σ̂²(x) = (1/(K-1)) Σ (f_k-μ̂)² → uncertainty weighting α_t
    """

    def __init__(self, models: List[PhysicsSurrogate]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    @property
    def K(self) -> int:
        return len(self.models)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mean: (B, 4, 100)  ensemble mean prediction
            var:  (B, 4, 100)  ensemble variance (uncertainty)
        """
        preds = torch.stack([m(x) for m in self.models], dim=0)  # (K, B, 4, 100)
        mean  = preds.mean(dim=0)
        var   = preds.var(dim=0, unbiased=True) if self.K > 1 else torch.zeros_like(mean)
        return mean, var

    def scalar_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-sample scalar uncertainty used for guidance step-size weighting.
        α_t ∝ 1 / (σ̂(x̂₀) + ε)   [see PIXEL_EXECUTION_PLAN Phase 4]

        Returns: (B,)  — mean variance over all frequency points and channels.
        """
        _, var = self.forward(x)
        return var.mean(dim=(-2, -1))  # (B,)

    @classmethod
    def load(
        cls,
        paths: List[str],
        device: torch.device | str = "cpu",
        **model_kwargs,
    ) -> "SurrogateEnsemble":
        """Load a saved ensemble from a list of checkpoint paths."""
        models = []
        for p in paths:
            m = PhysicsSurrogate(**model_kwargs)
            ckpt = torch.load(p, map_location=device)
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(device).eval()
            models.append(m)
        return cls(models)
