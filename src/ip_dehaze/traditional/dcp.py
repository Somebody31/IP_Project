from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    image = _image(image)
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    channel_min = image.min(axis=2)
    horizontal = minimum_filter1d(channel_min, size=patch_size, axis=1, mode="nearest")
    return minimum_filter1d(horizontal, size=patch_size, axis=0, mode="nearest")


def estimate_atmospheric_light(
    image: np.ndarray, dark: np.ndarray, top_fraction: float = 0.001
) -> np.ndarray:
    image = _image(image)
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    flat_dark = dark.ravel()
    count = max(1, int(np.ceil(flat_dark.size * top_fraction)))
    indices = np.argpartition(flat_dark, -count)[-count:]
    candidates = image.reshape(-1, 3)[indices]
    return candidates[np.argmax(candidates.mean(axis=1))].astype(np.float32)


def estimate_transmission(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
) -> np.ndarray:
    image = _image(image)
    if not 0 <= omega <= 1:
        raise ValueError("omega must be in [0, 1]")
    normalized = image / np.maximum(atmospheric_light.reshape(1, 1, 3), 1e-6)
    channel_min = normalized.min(axis=2)
    horizontal = minimum_filter1d(channel_min, size=patch_size, axis=1, mode="nearest")
    dark = minimum_filter1d(horizontal, size=patch_size, axis=0, mode="nearest")
    return np.clip(1.0 - omega * dark, 0, 1)


def guided_filter(
    guide: np.ndarray, source: np.ndarray, radius: int = 20, epsilon: float = 1e-3
) -> np.ndarray:
    guide = np.asarray(guide, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    if guide.ndim == 3:
        guide = guide.mean(axis=2)
    if guide.shape != source.shape:
        raise ValueError("guide and source must have the same height and width")
    if radius < 1 or epsilon <= 0:
        raise ValueError("radius must be positive and epsilon must be positive")
    size = 2 * radius + 1
    mean_i = _box_filter(guide, size)
    mean_p = _box_filter(source, size)
    corr_i = _box_filter(guide * guide, size)
    corr_ip = _box_filter(guide * source, size)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + epsilon)
    b = mean_p - a * mean_i
    mean_a = _box_filter(a, size)
    mean_b = _box_filter(b, size)
    return np.clip(mean_a * guide + mean_b, 0, 1)


def recover_radiance(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    transmission: np.ndarray,
    transmission_floor: float = 0.1,
) -> np.ndarray:
    image = _image(image)
    if not 0 < transmission_floor <= 1:
        raise ValueError("transmission_floor must be in (0, 1]")
    t = np.maximum(transmission[..., None], transmission_floor)
    return np.clip((image - atmospheric_light.reshape(1, 1, 3)) / t + atmospheric_light.reshape(1, 1, 3), 0, 1)


def dehaze(
    image: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    top_fraction: float = 0.001,
    guided_radius: int = 20,
    guided_epsilon: float = 1e-3,
    transmission_floor: float = 0.1,
) -> np.ndarray:
    image = _image(image)
    dark = dark_channel(image, patch_size)
    atmosphere = estimate_atmospheric_light(image, dark, top_fraction)
    transmission = estimate_transmission(image, atmosphere, patch_size, omega)
    transmission = guided_filter(image, transmission, guided_radius, guided_epsilon)
    return recover_radiance(image, atmosphere, transmission, transmission_floor)


def _box_filter(image: np.ndarray, size: int) -> np.ndarray:
    return uniform_filter1d(
        uniform_filter1d(image, size=size, axis=1, mode="nearest"),
        size=size,
        axis=0,
        mode="nearest",
    )


def _image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 RGB array")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("image values must be finite and in [0, 1]")
    return image
