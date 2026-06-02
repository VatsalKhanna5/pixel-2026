"""
src/losses/physics_losses.py
============================
Physics-constrained losses for surrogate training (PIXEL-2026 Phase 2).

All functions expect pred / target tensors of shape (B, 4, 100) where:
    pred[:, 0, :]  = S11_mag       ∈ [0, 1]   (linear magnitude)
    pred[:, 1, :]  = S21_mag       ∈ [0, 1]
    pred[:, 2, :]  = S11_phase/π   ∈ [-1, 1]  (normalised — un-normalise by ×π)
    pred[:, 3, :]  = S21_phase/π   ∈ [-1, 1]

Loss composition (weights from base_config.yaml):
    L_total = 1.0·L_mse + 0.10·L_pass + 0.005·L_kk + 0.02·L_smooth

KK weight is deliberately low (0.005): training data has a systematic ~0.30 KK
residual caused by the 2 ns FDTD time-window cap (τ_substrate ≈ 11 ns >> 2 ns).
The surrogate must reproduce the simulator output faithfully, not correct it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def passivity_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Enforce power conservation: |S11|² + |S21|² ≤ 1 at every frequency.

    Uses a mean hinge penalty over all (sample, frequency) violations.
    The surrogate dataset already has passivity enforced by the simulator,
    so this acts as a consistency regulariser that prevents the network from
    learning S-parameter combinations that violate physics.
    """
    s11 = pred[:, 0, :]   # (B, 100)  magnitudes in [0,1]
    s21 = pred[:, 1, :]
    power     = s11 ** 2 + s21 ** 2          # (B, 100)
    violation = torch.clamp(power - 1.0, min=0.0)
    return violation.mean()


def _hilbert_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the discrete Hilbert transform of x along its last dimension
    using the FFT analytic-signal method.

    For a real sequence x of length N, the analytic signal is computed via:
        1. X = FFT(x)
        2. Apply one-sided weight: h[0]=1, h[1..N/2-1]=2, h[N/2]=1, h[N/2+1..]=0
        3. analytic = IFFT(X * h)   →  imag part = H{x}

    KK relation:  Im[S(f)] = H{ Re[S(f)] }
    Residual:     Im[S] - H{Re[S]}  →  should be ≈ 0 for causal systems.
    """
    N  = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)                          # complex (B, N)

    h = torch.zeros(N, dtype=Xf.dtype, device=x.device)
    h[0] = 1.0
    h[1 : N // 2] = 2.0
    h[N // 2] = 1.0                                         # Nyquist bin

    analytic = torch.fft.ifft(Xf * h, dim=-1)              # complex (B, N)
    return analytic.imag                                    # H{x}  (B, N)  real


def kk_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Soft Kramers-Kronig regulariser: Im[S(f)] ≈ H{ Re[S(f)] }.

    Applied independently to S11 and S21, then averaged.

    λ should be 0.005 (see module docstring).  The loss measures how much
    the predicted complex S-parameters deviate from the causal relation,
    providing a gentle push toward physical causality without fighting the
    systematic FDTD windowing offset in the training labels.
    """
    s11_mag = pred[:, 0, :]
    s21_mag = pred[:, 1, :]
    s11_ph  = pred[:, 2, :] * math.pi     # un-normalise  →  radians
    s21_ph  = pred[:, 3, :] * math.pi

    re11 = s11_mag * torch.cos(s11_ph)    # (B, 100)  real parts
    im11 = s11_mag * torch.sin(s11_ph)    # (B, 100)  imaginary parts
    re21 = s21_mag * torch.cos(s21_ph)
    im21 = s21_mag * torch.sin(s21_ph)

    # KK residual: im - H{re} should be 0 for causal S-params
    res11 = (im11 - _hilbert_transform(re11)).pow(2).mean()
    res21 = (im21 - _hilbert_transform(re21)).pow(2).mean()
    return (res11 + res21) * 0.5


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """
    Penalise sharp adjacent-frequency jumps across all 4 output channels.

    Physical S-parameter curves are smooth functions of frequency.  This
    loss prevents the surrogate from predicting unphysical spectral spikes
    that could corrupt Phase 4 gradient guidance.
    """
    diff = pred[..., 1:] - pred[..., :-1]   # (B, 4, 99)
    return diff.pow(2).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def surrogate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_pass: float   = 0.10,
    lambda_kk: float     = 0.005,
    lambda_smooth: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Combined physics-constrained surrogate training loss.

    Args:
        pred:          (B, 4, 100)  model output
        target:        (B, 4, 100)  ground truth (same normalisation)
        lambda_pass:   weight for passivity loss
        lambda_kk:     weight for KK regulariser  (keep ≤ 0.005 — see module note)
        lambda_smooth: weight for spectral smoothness loss

    Returns:
        (total_loss, component_dict)   — component_dict is for logging only.
    """
    L_mse    = F.mse_loss(pred, target)
    L_pass   = passivity_loss(pred)
    L_kk     = kk_loss(pred)
    L_smooth = smoothness_loss(pred)

    total = (
        L_mse
        + lambda_pass   * L_pass
        + lambda_kk     * L_kk
        + lambda_smooth * L_smooth
    )

    components = {
        "loss/total":  total.item(),
        "loss/mse":    L_mse.item(),
        "loss/pass":   L_pass.item(),
        "loss/kk":     L_kk.item(),
        "loss/smooth": L_smooth.item(),
    }
    return total, components


# ---------------------------------------------------------------------------
# Per-channel MSE helpers  (used for val metrics)
# ---------------------------------------------------------------------------

def channel_mse(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """
    Return per-channel MSE for logging and checkpoint selection.

    Magnitudes are in [0,1]; phases are normalised to [-1,1].
    The primary checkpoint metric is the mean magnitude MSE
    (S11_mse + S21_mse) / 2.
    """
    with torch.no_grad():
        s11_mse   = F.mse_loss(pred[:, 0, :], target[:, 0, :]).item()
        s21_mse   = F.mse_loss(pred[:, 1, :], target[:, 1, :]).item()
        s11ph_mse = F.mse_loss(pred[:, 2, :], target[:, 2, :]).item()
        s21ph_mse = F.mse_loss(pred[:, 3, :], target[:, 3, :]).item()
    return {
        "val/s11_mag_mse":   s11_mse,
        "val/s21_mag_mse":   s21_mse,
        "val/s11_phase_mse": s11ph_mse,
        "val/s21_phase_mse": s21ph_mse,
        "val/mag_mse_mean":  (s11_mse + s21_mse) / 2.0,   # primary checkpoint metric
    }
