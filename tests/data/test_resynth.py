"""Sanity checks for the realistic-fog resynthesis renderer."""
import numpy as np

from ip_dehaze.data.resynth import SOURCE_BETA, render_fog


def test_render_fog_matches_scattering_model():
    clear = np.full((4, 4, 3), 0.2, dtype=np.float32)
    t_source = np.full((4, 4), 0.81, dtype=np.float32)  # t at beta=0.01
    airlight = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    hazy = render_fog(clear, t_source, 2 * SOURCE_BETA, airlight)  # doubling beta squares t
    t = 0.81**2
    np.testing.assert_allclose(hazy, 0.2 * t + 0.8 * (1 - t), rtol=1e-5)


def test_render_fog_identity_at_full_transmission():
    clear = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    hazy = render_fog(clear, np.ones((8, 8), np.float32), 0.03, np.array([0.9, 0.9, 0.9], np.float32))
    np.testing.assert_allclose(hazy, clear, atol=1e-6)
