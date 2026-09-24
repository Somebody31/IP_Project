"""Shared image-quality metrics.

Both the traditional and the machine-learning tracks must import these
functions so every reported number comes from identical code. Images are
float32 RGB in [0, 1] with shape (H, W, 3).
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(output: np.ndarray, target: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(target, output, data_range=1.0))


def ssim(output: np.ndarray, target: np.ndarray) -> float:
    return float(
        structural_similarity(target, output, data_range=1.0, channel_axis=2, win_size=11)
    )


def metrics(output: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {"psnr": psnr(output, target), "ssim": ssim(output, target)}
