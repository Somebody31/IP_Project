import numpy as np
import pytest
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/machine_learning"))

from evaluate_model_detection import dehaze_whole
from ip_dehaze.machine_learning.models.unet import build_unet


def test_flip_tta_on_symmetric_image():
    # Build a tiny U-Net model
    device = torch.device("cpu")
    model = build_unet("tiny", gated=True, gate_spatial=True).to(device)
    model.eval()

    # Create a horizontally symmetric image (H=64, W=64, 3)
    half = np.random.RandomState(42).rand(64, 32, 3).astype(np.float32)
    symm = np.concatenate([half, half[:, ::-1, :]], axis=1)

    out_plain = dehaze_whole(model, symm, device, tta="none")
    out_flip = dehaze_whole(model, symm, device, tta="flip")

    assert out_plain.shape == symm.shape
    assert out_flip.shape == symm.shape
    # For a horizontally symmetric image, flip TTA guarantees exact reflectional symmetry in the output
    np.testing.assert_allclose(out_flip, out_flip[:, ::-1, :], atol=1e-5)
    # And flip TTA stays very close to the standard pass (< 0.002 max diff from asymmetric kernel filters)
    np.testing.assert_allclose(out_flip, out_plain, atol=2e-3)
