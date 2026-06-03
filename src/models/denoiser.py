"""
src/models/denoiser.py
=======================
U-Net denoiser for PIXEL-2026 Phase 3.

Maps noisy layout x_t (ternary) + spectral conditioning c_y + timestep t
to a predicted distribution p_θ(x_0 | x_t, c_y) over binary {0,1} per pixel.

Architecture:
    Token embed (3→32) + port projection
    → Stem conv (32→64)
    → 2×ResBlock(64)   [skip to decoder]
    → Stride-2 down → (128, 8×8)
    → 2×ResBlock(128)
    → Self-attention bottleneck at 8×8 = 64 tokens
    → 2×ResBlock(128)
    → TransposeConv(128→64, kernel=3, stride=2, pad=1)  [→ 15×15]
    → concat skip (→ 128) → conv (→ 64)
    → 2×ResBlock(64)
    → Output head: Linear(64→2) per pixel → logits for {0,1}

Conditioning via AdaLN:
    cond = proj_cy(c_y) + proj_t(t_emb)   ∈ ℝ^256
    AdaLN(h, cond) = (1+γ(cond)) · GroupNorm(h) + β(cond)

Discrete CFG during training: 15% condition dropout (c_y → 0 vector).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

MASK_TOKEN = 2


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer sinusoidal embedding for integer timesteps.

    Args:
        t:   (B,) long  timesteps in [1, T]
        dim: output dimension (must be even)
    Returns:
        (B, dim) float
    """
    half = dim // 2
    device = t.device
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device, dtype=torch.float32) / (half - 1)
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)   # (B, dim)


# ---------------------------------------------------------------------------
# Adaptive Layer Normalisation (AdaLN)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """
    Adaptive layer normalisation: scale and shift from conditioning.

    AdaLN(h, cond) = (1 + γ) · GroupNorm(h) + β
    where γ, β = Linear(cond).chunk(2)

    Using GroupNorm (not LayerNorm) for 2D feature maps — avoids permute
    operations and is more stable with small spatial dimensions like 15×15.
    """

    def __init__(self, n_channels: int, cond_dim: int, n_groups: int = 8) -> None:
        super().__init__()
        self.norm   = nn.GroupNorm(min(n_groups, n_channels), n_channels, affine=False)
        self.linear = nn.Linear(cond_dim, 2 * n_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:    (B, C, H, W)  feature map
            cond: (B, cond_dim)  conditioning vector
        Returns:
            (B, C, H, W)
        """
        gamma_beta = self.linear(cond)                # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=1)      # each (B, C)
        gamma = gamma[:, :, None, None]               # (B, C, 1, 1)
        beta  = beta[:, :, None, None]
        return (1.0 + gamma) * self.norm(h) + beta


# ---------------------------------------------------------------------------
# Conditioned ResBlock (2D, with AdaLN)
# ---------------------------------------------------------------------------

class CondResBlock(nn.Module):
    """2D ResBlock conditioned via AdaLN."""

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.adaLN1 = AdaLN(channels, cond_dim)
        self.conv1  = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.adaLN2 = AdaLN(channels, cond_dim)
        self.conv2  = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.adaLN1(x, cond)))
        h = self.conv2(F.gelu(self.adaLN2(h, cond)))
        return x + h


# ---------------------------------------------------------------------------
# Self-attention block at bottleneck
# ---------------------------------------------------------------------------

class SelfAttentionBlock(nn.Module):
    """Full self-attention over spatial tokens.  Affordable at 8×8 = 64 tokens."""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, N, C)
        h = self.norm(h)
        h, _ = self.attn(h, h, h)
        h = h.reshape(B, H, W, C).permute(0, 3, 1, 2)    # (B, C, H, W)
        return x + h


# ---------------------------------------------------------------------------
# Full U-Net denoiser
# ---------------------------------------------------------------------------

class PixelDenoiser(nn.Module):
    """
    D3PM denoiser for 15×15 binary EM layouts.

    Input channels:
        Ch0: noisy layout token embedding (from nn.Embedding(3, 32))
        + port map projected to same dimension

    Conditioning:
        c_y: (B, 256)  spectral embedding (from SpectralEncoder)
        t:   (B,)      timestep integers [1..T] → sinusoidal embedding

    Output:
        (B, 2, 15, 15)  logits over {0=dielectric, 1=conductor} per pixel
                        Represents p_θ(x_0 | x_t, c_y).
    """

    # Spatial size at each U-Net level
    _ENC_SIZE    = (15, 15)
    _BOTTLE_SIZE = (8, 8)

    def __init__(
        self,
        token_embed_dim: int = 32,
        base_ch: int         = 64,
        cond_embed_dim: int  = 256,   # output of SpectralEncoder
        t_embed_dim: int     = 128,
        n_res_blocks: int    = 2,
    ) -> None:
        super().__init__()

        # Combined conditioning dimension after projection
        cond_dim = cond_embed_dim

        # ── Token + port embedding ──────────────────────────────────────────
        self.token_embed = nn.Embedding(3, token_embed_dim)  # {0,1,MASK}
        self.port_proj   = nn.Conv2d(1, token_embed_dim, 1, bias=False)

        # ── Timestep embedding ──────────────────────────────────────────────
        self.t_mlp = nn.Sequential(
            nn.Linear(t_embed_dim, cond_embed_dim),
            nn.GELU(),
            nn.Linear(cond_embed_dim, cond_embed_dim),
        )
        self._t_embed_dim = t_embed_dim

        # ── Spectral conditioning projection ────────────────────────────────
        # Sums with timestep embedding to form a single cond vector
        self.c_proj = nn.Linear(cond_embed_dim, cond_embed_dim)

        # ── Encoder ─────────────────────────────────────────────────────────
        ch0 = base_ch            # 64
        ch1 = base_ch * 2        # 128

        self.stem = nn.Sequential(
            nn.Conv2d(token_embed_dim, ch0, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, ch0), ch0),
        )
        self.enc_blocks = nn.ModuleList(
            [CondResBlock(ch0, cond_dim) for _ in range(n_res_blocks)]
        )

        # Stride-2 down: 15→8
        self.down = nn.Sequential(
            nn.GroupNorm(min(8, ch0), ch0, affine=False),
            nn.GELU(),
            nn.Conv2d(ch0, ch1, 3, stride=2, padding=1, bias=False),
        )

        self.bot_blocks = nn.ModuleList(
            [CondResBlock(ch1, cond_dim) for _ in range(n_res_blocks)]
        )
        self.attn = SelfAttentionBlock(ch1, n_heads=4)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.dec_blocks_bot = nn.ModuleList(
            [CondResBlock(ch1, cond_dim) for _ in range(n_res_blocks)]
        )

        # TransposeConv 8→15: kernel=3, stride=2, pad=1 → (input-1)*2-2+3 = 15 ✓
        self.up = nn.ConvTranspose2d(ch1, ch0, kernel_size=3, stride=2, padding=1)

        # After upsample + skip concat: ch0 + ch0 = 2*ch0 → ch0
        self.skip_conv = nn.Sequential(
            nn.GroupNorm(min(8, ch0 * 2), ch0 * 2, affine=False),
            nn.GELU(),
            nn.Conv2d(ch0 * 2, ch0, 1, bias=False),
        )
        self.dec_blocks = nn.ModuleList(
            [CondResBlock(ch0, cond_dim) for _ in range(n_res_blocks)]
        )

        # ── Output head ─────────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(min(8, ch0), ch0, affine=False)
        self.out_conv = nn.Conv2d(ch0, 2, 1)   # 2 logits: {0=diel, 1=cond}

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Zero-init output projection for stable early training
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x_t:      torch.Tensor,                   # (B, H, W)  long {0,1,2}
        t:        torch.Tensor,                   # (B,)       long
        c_y:      Optional[torch.Tensor],         # (B, 256)   or None (uncond)
        port_map: torch.Tensor,                   # (1, H, W) or (B, H, W) float
    ) -> torch.Tensor:
        """
        Returns:
            logits: (B, 2, H, W)  raw logits for {0,1} = p_θ(x_0|x_t,c_y)
        """
        B, H, W = x_t.shape
        device  = x_t.device

        # ── Build conditioning vector ────────────────────────────────────────
        t_emb = sinusoidal_embedding(t, self._t_embed_dim)  # (B, 128)
        t_emb = self.t_mlp(t_emb)                           # (B, 256)

        if c_y is None:
            c_y = torch.zeros(B, t_emb.shape[-1], device=device, dtype=t_emb.dtype)
        cond = self.c_proj(c_y) + t_emb   # (B, 256)

        # ── Token + port embedding ──────────────────────────────────────────
        tok = self.token_embed(x_t)                          # (B, H, W, 32)
        tok = tok.permute(0, 3, 1, 2)                       # (B, 32, H, W)
        if port_map.ndim == 2:
            port_map = port_map.unsqueeze(0)
        if port_map.ndim == 3:
            port_map = port_map.unsqueeze(0) if port_map.shape[0] != B else port_map.unsqueeze(1)
        port = self.port_proj(port_map.float().to(device))  # (B, 32, H, W)
        h    = F.gelu(self.stem(tok + port))                # (B, 64, 15, 15)

        # ── Encoder ─────────────────────────────────────────────────────────
        for blk in self.enc_blocks:
            h = blk(h, cond)
        skip = h                                             # (B, 64, 15, 15) save skip

        h = self.down(h)                                     # (B, 128, 8, 8)

        # ── Bottleneck ──────────────────────────────────────────────────────
        for blk in self.bot_blocks:
            h = blk(h, cond)
        h = self.attn(h)

        # ── Decoder ─────────────────────────────────────────────────────────
        for blk in self.dec_blocks_bot:
            h = blk(h, cond)

        h = self.up(h, output_size=(B, h.shape[1], H, W))   # (B, 64, 15, 15)
        h = self.skip_conv(torch.cat([h, skip], dim=1))      # (B, 64, 15, 15)

        for blk in self.dec_blocks:
            h = blk(h, cond)

        # ── Output ──────────────────────────────────────────────────────────
        return self.out_conv(F.gelu(self.out_norm(h)))       # (B, 2, 15, 15)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# EMA wrapper
# ---------------------------------------------------------------------------

class EMA:
    """
    Exponential moving average of model weights, used for inference.

    Maintains a shadow copy of model weights: shadow = decay*shadow + (1-decay)*param.
    During evaluation, swap in the shadow weights; restore original for the next
    training step.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.model  = model
        self.decay  = decay
        self.shadow: dict = {}
        self._backup: dict = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone().float()

    @torch.no_grad()
    def update(self) -> None:
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    p.data.float(), alpha=1.0 - self.decay
                )

    def apply_shadow(self) -> None:
        """Copy EMA weights into model (for evaluation/sampling)."""
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self._backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name].to(p.dtype))

    def restore(self) -> None:
        """Restore original training weights."""
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self._backup:
                p.data.copy_(self._backup[name])
        self._backup.clear()

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict, device: torch.device) -> None:
        self.shadow = {k: v.to(device).float() for k, v in sd.items()}
