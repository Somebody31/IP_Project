import numpy as np
import pytest
import torch
from PIL import Image

from ip_dehaze.data.dataset import BETA_CLASSES, FoggyPairs, beta_class
from ip_dehaze.machine_learning.losses import reconstruction_loss, ssim_index
from ip_dehaze.machine_learning.models.classifier import DensityClassifier
from ip_dehaze.machine_learning.models.unet import build_unet


def test_beta_class_mapping_covers_all_foggy_cityscapes_betas():
    assert [beta_class(b) for b in BETA_CLASSES.values()] == [0, 1, 2]
    with pytest.raises(ValueError):
        beta_class(0.03)


def test_unet_preserves_shape_and_range():
    model = build_unet("tiny")
    images = torch.rand(2, 3, 64, 64)
    output, density = model(images)
    assert output.shape == images.shape
    assert density is None
    assert float(output.min()) >= 0.0 and float(output.max()) <= 1.0


class _FixedDensityHead(torch.nn.Module):
    """Emits a constant logit so the gate saturates to 0 or 1."""

    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.full((features.shape[0], 1), self.logit, device=features.device)


def test_gated_output_is_convex_blend_of_input_and_restoration():
    """Gate at 1 -> raw restoration; gate at 0 -> untouched input."""
    torch.manual_seed(0)
    gated = build_unet("tiny", gated=True).eval()
    blind = build_unet("tiny", gated=False).eval()
    missing, unexpected = blind.load_state_dict(gated.state_dict(), strict=False)
    assert not missing
    assert unexpected  # only the density head is extra

    images = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        restoration, _ = blind(images)

        gated.density_head = _FixedDensityHead(20.0)
        blended_full, density_full = gated(images)
        gated.density_head = _FixedDensityHead(-20.0)
        blended_none, density_none = gated(images)

    assert float(density_full.min()) > 0.999
    assert float(density_none.max()) < 0.001
    assert torch.allclose(blended_full, restoration, atol=1e-3)
    assert torch.allclose(blended_none, images, atol=1e-3)


def test_classifier_outputs_three_logits():
    model = DensityClassifier()
    logits = model(torch.rand(4, 3, 64, 64))
    assert logits.shape == (4, 3)


def test_ssim_index_is_one_for_identical_and_lower_for_noise():
    image = torch.rand(1, 3, 64, 64)
    assert float(ssim_index(image, image)) == pytest.approx(1.0, abs=1e-4)
    noisy = image + 0.3 * torch.randn_like(image)
    assert float(ssim_index(noisy, image)) < float(ssim_index(image, image))


def test_ssim_index_is_finite_for_adversarial_constant_images():
    """Regression: constant/degenerate inputs must not produce NaN or inf."""
    constant_black = torch.zeros(2, 3, 64, 64)
    constant_white = torch.ones(2, 3, 64, 64)
    for pred, target in ((constant_black, constant_white), (constant_white, constant_black),
                         (constant_black, constant_black)):
        value = float(ssim_index(pred, target))
        assert np.isfinite(value), (pred.mean().item(), target.mean().item(), value)
    # fp16 inputs (as produced under autocast) must be safe too
    value = float(ssim_index(constant_black.half(), constant_white.half()))
    assert np.isfinite(value)


def test_reconstruction_loss_is_minimal_for_perfect_prediction():
    image = torch.rand(1, 3, 32, 32)
    assert float(reconstruction_loss(image, image)) == pytest.approx(0.0, abs=1e-5)


def _write_pair(root, split, city, image_id, beta):
    foggy_dir = root / "foggy" / "leftImg8bit_foggy" / split / city
    clear_dir = root / "clear" / "leftImg8bit" / split / city
    foggy_dir.mkdir(parents=True, exist_ok=True)
    clear_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash((image_id, beta))) % 2**32)
    clear = rng.integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
    hazy = np.clip(clear * 0.6 + 90, 0, 255).astype(np.uint8)
    Image.fromarray(clear).save(clear_dir / f"{image_id}_leftImg8bit.png")
    Image.fromarray(hazy).save(foggy_dir / f"{image_id}_leftImg8bit_foggy_beta_{beta}.png")


def test_dataset_returns_matching_crops_and_density_labels(tmp_path):
    for image_id in ("aachen_000001_000002", "aachen_000003_000004"):
        for beta in ("0.005", "0.01"):
            _write_pair(tmp_path, "train", "aachen", image_id, beta)
    dataset = FoggyPairs(
        str(tmp_path / "foggy"), str(tmp_path / "clear"), "train", crop=128, seed=7
    )
    assert len(dataset) == 4
    sample = dataset[0]
    assert sample["hazy"].shape == (3, 128, 128)
    assert sample["clear"].shape == (3, 128, 128)
    assert sample["hazy"].shape == sample["clear"].shape

    # Crops are deterministic per (seed, epoch, index) and vary across epochs.
    first = dataset[0]["hazy"].clone()
    assert torch.equal(first, dataset[0]["hazy"])
    dataset.set_epoch(1)
    assert not torch.equal(first, dataset[0]["hazy"])


def test_dataset_beta_filter_selects_one_density_class(tmp_path):
    _write_pair(tmp_path, "train", "aachen", "aachen_000001_000002", "0.005")
    _write_pair(tmp_path, "train", "aachen", "aachen_000003_000004", "0.02")
    dataset = FoggyPairs(
        str(tmp_path / "foggy"), str(tmp_path / "clear"), "train", crop=64, beta_filter="heavy"
    )
    assert len(dataset) == 1
    assert dataset[0]["density_class"].item() == 2


def test_gate_init_places_gate_at_target_for_both_heads():
    """gate_init=0.95 must give d=0.95 exactly at init, independent of the input."""
    images = torch.rand(2, 3, 64, 64)
    for gate_spatial in (False, True):
        model = build_unet("tiny", gated=True, gate_init=0.95, gate_spatial=gate_spatial)
        _, density = model(images)
        assert torch.allclose(density, torch.full_like(density, 0.95), atol=1e-4)
    with pytest.raises(ValueError):
        build_unet("tiny", gated=True, gate_init=1.5)


def test_gate_spatial_returns_per_pixel_density_map():
    """The spatial gate blends per pixel and reports one d per pixel."""
    images = torch.rand(2, 3, 64, 64)
    model = build_unet("tiny", gated=True, gate_spatial=True)
    output, density = model(images)
    assert output.shape == images.shape
    assert density.shape == (2, 1, 64, 64)
    assert float(density.min()) >= 0.0 and float(density.max()) <= 1.0


def test_gate_options_default_off_preserve_scalar_gate_contract():
    """Default flags must keep the original single-scalar gate and blend contract."""
    images = torch.rand(2, 3, 64, 64)
    output, density = build_unet("tiny", gated=True)(images)
    assert density.shape == (2, 1)
    assert output.shape == images.shape
    assert float(density.min()) >= 0.0 and float(density.max()) <= 1.0
