import numpy as np
import pytest

from ip_dehaze.traditional.dcp import (
    dark_channel,
    dehaze,
    estimate_atmospheric_light,
    guided_filter,
    recover_radiance,
)


def test_dcp_preserves_shape_and_returns_finite_rgb():
    rng = np.random.default_rng(12)
    image = rng.uniform(0.15, 0.9, size=(32, 40, 3)).astype(np.float32)
    result = dehaze(image, patch_size=5, guided_radius=3)
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1


def test_dark_channel_and_atmosphere_have_expected_shapes():
    image = np.full((9, 11, 3), 0.4, dtype=np.float32)
    dark = dark_channel(image, patch_size=3)
    atmosphere = estimate_atmospheric_light(image, dark)
    assert dark.shape == image.shape[:2]
    np.testing.assert_allclose(atmosphere, [0.4, 0.4, 0.4])


def test_guided_filter_preserves_constant_transmission():
    guide = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    source = np.full((10, 10), 0.6, dtype=np.float32)
    result = guided_filter(guide, source, radius=2)
    np.testing.assert_allclose(result, source, atol=1e-5)


def test_radiance_recovery_reverses_known_fog_formation():
    clear = np.linspace(0.1, 0.8, 12 * 13 * 3, dtype=np.float32).reshape(12, 13, 3)
    atmosphere = np.array([0.9, 0.85, 0.8], dtype=np.float32)
    transmission = np.full(clear.shape[:2], 0.6, dtype=np.float32)
    foggy = clear * transmission[..., None] + atmosphere * (1 - transmission[..., None])
    recovered = recover_radiance(foggy, atmosphere, transmission, transmission_floor=0.1)
    np.testing.assert_allclose(recovered, clear, atol=1e-6)
    assert np.mean((recovered - clear) ** 2) < np.mean((foggy - clear) ** 2)


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError, match="HxWx3"):
        dehaze(np.zeros((8, 8), dtype=np.float32))
