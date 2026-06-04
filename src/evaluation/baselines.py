"""
src/evaluation/baselines.py
============================
Phase 5 baseline models for PIXEL-2026 comparison.

Implements two deterministic inverse-design baselines:

  Det-CNN  — deterministic CNN inverse regressor.
             Input: y* ∈ R^(4×N_f)  →  Output: layout ∈ [0,1]^(15×15)
             Single forward pass; outputs a soft layout, threshold at 0.5.
             Represents the standard neural-network approach to inverse design
             (Liu et al. 2018, So et al. 2020).

  cVAE     — conditional VAE (Kingma & Welling 2013 + conditional variant).
             Encoder: (x, c_y) → (μ, log σ²)  ∈ R^d_z
             Decoder: (z, c_y) → layout ∈ [0,1]^(15×15)
             Sampling: z ~ N(0,I), decode conditioned on y*.
             Represents probabilistic baseline (Unni et al. 2021).

Both share the same SpectralEncoder for y* embedding (frozen or jointly trained).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spectral_encoder import SpectralEncoder


# ---------------------------------------------------------------------------
# Shared decoder: spectral embedding → 15×15 binary layout
# ---------------------------------------------------------------------------

class LayoutDecoder(nn.Module):
    """
    Fully-convolutional decoder: latent vector → soft binary layout (15×15).

    Architecture:
        Linear(latent_dim → 128*4*4) → reshape (128,4,4)
        → ConvTranspose 4→8 (128→64)
        → ConvTranspose 8→15 (64→32)    # careful output_size
        → Conv(32→1) → Sigmoid
    """

    def __init__(self, latent_dim: int = 256, base_ch: int = 64) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, base_ch * 2 * 4 * 4)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch), nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_ch // 2), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(base_ch // 2, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → layout: (B, 1, 15, 15)"""
        h = self.fc(z).view(z.shape[0], -1, 4, 4)   # (B, 128, 4, 4)
        h = self.up1(h)                               # (B, 64, 7, 7)  approx
        h = self.up2(h)                               # (B, 32, 13, 13) approx
        # Bilinear upsample to exact 15×15 before head
        h = F.interpolate(h, size=(15, 15), mode="bilinear", align_corners=False)
        return self.head(h)                           # (B, 1, 15, 15)


# ---------------------------------------------------------------------------
# Det-CNN: Deterministic inverse regressor
# ---------------------------------------------------------------------------

class DetCNN(nn.Module):
    """
    Deterministic CNN: y* → layout.

    SpectralEncoder produces c_y ∈ R^embed_dim.
    LayoutDecoder maps c_y → soft layout ∈ [0,1]^(15×15).
    Hard threshold at 0.5 during inference for binary layout.

    Trained with BCE against ground-truth layouts.
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim:   int = 256,
        base_ch:     int = 64,
    ) -> None:
        super().__init__()
        self.encoder = SpectralEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.decoder = LayoutDecoder(latent_dim=embed_dim, base_ch=base_ch)

    def forward(self, y_star: torch.Tensor) -> torch.Tensor:
        """y_star: (B, 4, N_f) → soft_layout: (B, 1, 15, 15)"""
        c_y = self.encoder(y_star)
        return self.decoder(c_y)

    @torch.no_grad()
    def predict(self, y_star: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """y_star: (B, 4, N_f) → binary_layout: (B, 15, 15) long"""
        soft = self.forward(y_star).squeeze(1)   # (B, 15, 15)
        return (soft > threshold).long()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# cVAE: Conditional Variational Autoencoder
# ---------------------------------------------------------------------------

class LayoutEncoder(nn.Module):
    """CNN encoder: layout + c_y → (μ, log σ²) in latent space."""

    def __init__(self, cond_dim: int = 256, latent_dim: int = 64, base_ch: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, base_ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.GELU(),  # 8×8
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1), nn.GELU(),  # 4×4
        )
        self.fc_in = base_ch * 4 * 4 * 4 + cond_dim
        self.fc_mu  = nn.Linear(self.fc_in, latent_dim)
        self.fc_lv  = nn.Linear(self.fc_in, latent_dim)

    def forward(self, x: torch.Tensor, c_y: torch.Tensor) -> tuple:
        """
        x:   (B, 1, 15, 15) float layout
        c_y: (B, cond_dim)
        Returns: mu, logvar — each (B, latent_dim)
        """
        h = self.conv(x).flatten(1)               # (B, base_ch*4*4*4)
        h = torch.cat([h, c_y], dim=1)
        return self.fc_mu(h), self.fc_lv(h)


class cVAE(nn.Module):
    """
    Conditional VAE for inverse EM design.

    Training:
        Encoder(x, c_y) → (μ, log σ²)
        z ~ μ + ε·exp(0.5·log σ²)
        Decoder(z, c_y) → x̂
        Loss = BCE(x̂, x) + β·KL(N(μ,σ²) || N(0,I))

    Inference:
        z ~ N(0, I)
        x̂ = Decoder(z, c_y)
        binary: threshold at 0.5
    """

    def __init__(
        self,
        in_channels:  int = 4,
        embed_dim:    int = 256,
        latent_dim:   int = 64,
        base_ch:      int = 64,
        enc_base_ch:  int = 32,
        beta:         float = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.spec_encoder  = SpectralEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.layout_encoder = LayoutEncoder(cond_dim=embed_dim, latent_dim=latent_dim,
                                            base_ch=enc_base_ch)
        self.decoder = LayoutDecoder(latent_dim=latent_dim + embed_dim, base_ch=base_ch)

    def encode(self, x: torch.Tensor, y_star: torch.Tensor):
        c_y = self.spec_encoder(y_star)
        mu, logvar = self.layout_encoder(x, c_y)
        return mu, logvar, c_y

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, c_y: torch.Tensor) -> torch.Tensor:
        zc = torch.cat([z, c_y], dim=1)
        return self.decoder(zc)   # (B, 1, 15, 15)

    def forward(self, x: torch.Tensor, y_star: torch.Tensor):
        """
        Returns: x_recon (B,1,15,15), mu (B,latent_dim), logvar (B,latent_dim)
        """
        mu, logvar, c_y = self.encode(x, y_star)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z, c_y)
        return x_recon, mu, logvar

    def loss(self, x: torch.Tensor, y_star: torch.Tensor) -> dict:
        """
        x: (B, 1, 15, 15) float layout ∈ {0,1}
        Returns dict with total loss and components.
        """
        x_recon, mu, logvar = self.forward(x, y_star)
        bce  = F.binary_cross_entropy(x_recon, x, reduction="mean")
        kl   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
        total = bce + self.beta * kl
        return {"loss": total, "bce": bce, "kl": kl}

    @torch.no_grad()
    def sample(self, y_star: torch.Tensor, n_samples: int = 1,
               threshold: float = 0.5) -> torch.Tensor:
        """
        Draw n_samples layouts per spec. Returns (B*n_samples, 15, 15) long.
        """
        B = y_star.shape[0]
        c_y = self.spec_encoder(y_star)                    # (B, embed_dim)
        c_y = c_y.repeat_interleave(n_samples, dim=0)      # (B*n, embed_dim)
        z   = torch.randn(B * n_samples, self.latent_dim,
                          device=y_star.device)
        soft = self.decode(z, c_y).squeeze(1)              # (B*n, 15, 15)
        return (soft > threshold).long()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
