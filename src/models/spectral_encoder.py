"""
src/models/spectral_encoder.py
================================
1D ResNet spectral encoder for PIXEL-2026 Phase 3.

Maps a normalised 4-channel S-parameter spectrum (B, 4, 100) to a
conditioning vector c_y ∈ ℝ^256 that guides the D3PM denoiser.

Architecture:
    Stem 1D-conv → 3×ResBlock1D(32) → stride-2 down → 3×ResBlock1D(64)
    → stride-2 down → 3×ResBlock1D(128) → GlobalAvgPool → Linear(256)

Why 1D ResNet over MLP:
    S-parameter curves have strong frequency-locality structure (resonances
    are local in frequency, phase wraps continuously).  1D convolutions
    capture these patterns without the full quadratic cost of attention.
    The two stride-2 downsampling stages (100→50→25) act as frequency-
    scale pyramids, giving the embedding multi-scale spectral sensitivity.

Input normalisation expected:
    Ch0: S11_mag  ∈ [0,1]     (linear magnitude, passivity-enforced)
    Ch1: S21_mag  ∈ [0,1]
    Ch2: S11_phase/π  ∈ [-1,1]
    Ch3: S21_phase/π  ∈ [-1,1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """Pre-activation residual block for 1D feature sequences (B, C, L)."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.bn1   = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.bn2   = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.bn1(x)))
        h = self.conv2(F.gelu(self.bn2(h)))
        return x + h


class DownBlock1D(nn.Module):
    """Stride-2 1D downsampling with channel expansion."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.bn   = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv = nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.gelu(self.bn(x)))


class SpectralEncoder(nn.Module):
    """
    Encodes a 4-channel S-parameter spectrum into a fixed conditioning vector.

    Args:
        in_channels:  Number of S-param channels (default 4).
        base_ch:      Width at the first stage (default 32).
        embed_dim:    Output embedding dimension c_y (default 256).
        n_freq:       Number of frequency points (default 100).
    """

    def __init__(
        self,
        in_channels: int = 4,
        base_ch: int     = 32,
        embed_dim: int   = 256,
        n_freq: int      = 100,
    ) -> None:
        super().__init__()

        self.stem = nn.Conv1d(in_channels, base_ch, kernel_size=7, padding=3, bias=False)

        # Stage 0: base_ch, freq_len=100
        self.stage0 = nn.Sequential(*[ResBlock1D(base_ch) for _ in range(3)])

        # Downsample: 100 → 50
        self.down1  = DownBlock1D(base_ch, base_ch * 2)
        self.stage1 = nn.Sequential(*[ResBlock1D(base_ch * 2) for _ in range(3)])

        # Downsample: 50 → 25
        self.down2  = DownBlock1D(base_ch * 2, base_ch * 4)
        self.stage2 = nn.Sequential(*[ResBlock1D(base_ch * 4) for _ in range(3)])

        # Global average pool + projection
        self.head = nn.Sequential(
            nn.GroupNorm(min(8, base_ch * 4), base_ch * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),   # → (B, base_ch*4, 1)
            nn.Flatten(1),             # → (B, base_ch*4)
            nn.Linear(base_ch * 4, embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (B, 4, 100)  normalised S-parameter spectrum
        Returns:
            c_y: (B, 256)  conditioning embedding
        """
        h = F.gelu(self.stem(y))   # (B, 32, 100)
        h = self.stage0(h)          # (B, 32, 100)
        h = self.down1(h)           # (B, 64,  50)
        h = self.stage1(h)          # (B, 64,  50)
        h = self.down2(h)           # (B, 128, 25)
        h = self.stage2(h)          # (B, 128, 25)
        return self.head(h)         # (B, 256)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
