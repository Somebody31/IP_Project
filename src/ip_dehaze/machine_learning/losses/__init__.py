"""Differentiable loss terms for dehazing training.

The SSIM *loss* here is a compact differentiable approximation (used for
gradients only). All reported SSIM numbers come from
`ip_dehaze.evaluation.metrics`, which matches the traditional track.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    weights = torch.exp(-(coords**2) / (2 * sigma**2))
    weights = weights / weights.sum()
    return (weights[:, None] * weights[None, :]).to(dtype)


def ssim_index(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM over batch/channels; inputs are (B, C, H, W) in [0, 1].

    Computed in float32: the variance estimate `E[x^2] - E[x]^2` suffers
    catastrophic cancellation in fp16, which can drive the denominator
    through zero and poison the run with NaNs.
    """
    pred = pred.float()
    target = target.float()
    channels = pred.shape[1]
    window = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    window = window[None, None].repeat(channels, 1, 1, 1)
    pad = window_size // 2
    c1, c2 = 0.01**2, 0.03**2
    eps = 1e-8

    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)
    mu_pred_sq, mu_target_sq, mu_cross = mu_pred**2, mu_target**2, mu_pred * mu_target

    # clamp: sampling error can make these slightly negative
    var_pred = (F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq).clamp(min=0)
    var_target = (F.conv2d(target * target, window, padding=pad, groups=channels) - mu_target_sq).clamp(min=0)
    covariance = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_cross

    # Cauchy-Schwarz: |cov| <= sqrt(var_pred * var_target)
    covariance = covariance.clamp(
        min=-torch.sqrt(var_pred * var_target + eps), max=torch.sqrt(var_pred * var_target + eps)
    )

    numerator = (2 * mu_cross + c1) * (2 * covariance + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (var_pred + var_target + c2)
    return (numerator / denominator.clamp(min=eps)).mean()


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, ssim_weight: float = 0.2) -> torch.Tensor:
    """L1 plus a weighted differentiable SSIM term (computed in fp32)."""
    return F.l1_loss(pred.float(), target.float()) + ssim_weight * (1.0 - ssim_index(pred, target))
