import csv
import json

import numpy as np
import pytest
import torch

from ip_dehaze.data.dataset import CachedCrops, CachedCropsWithContext


def build_cache(root, pairs=(("aachen_000001_000002", 0.005), ("aachen_000003_000004", 0.02)), crops_per_pair=2):
    rows = []
    hazy, clear = [], []
    for pair_index, (image_id, beta) in enumerate(pairs):
        for crop_index in range(crops_per_pair):
            rng = np.random.default_rng(pair_index * 10 + crop_index)
            clear_crop = rng.integers(0, 200, size=(64, 64, 3), dtype=np.uint8)
            hazy_crop = np.clip(clear_crop.astype(np.int16) + 80, 0, 255).astype(np.uint8)
            hazy.append(hazy_crop)
            clear.append(clear_crop)
            rows.append(
                {
                    "row": len(rows),
                    "split": "train",
                    "city": "aachen",
                    "image_id": image_id,
                    "beta": beta,
                    "density_class": 0 if beta == 0.005 else 2,
                    "beta_name": "light" if beta == 0.005 else "heavy",
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    np.save(root / "hazy.npy", np.stack(hazy))
    np.save(root / "clear.npy", np.stack(clear))
    (root / "crops_per_pair.txt").write_text(f"{crops_per_pair}\n")
    (root / "manifest.json").write_text(
        json.dumps({"crops_per_pair": crops_per_pair, "rows": len(rows), "complete": True})
    )
    with (root / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_cached_crops_length_matches_pairs_not_crops(tmp_path):
    build_cache(tmp_path, crops_per_pair=2)
    dataset = CachedCrops(tmp_path, split="train")
    assert len(dataset) == 2  # 2 pairs, 4 cached crops
    sample = dataset[0]
    assert sample["hazy"].shape == (3, 64, 64)
    assert sample["hazy"].dtype == torch.float32
    assert float(sample["hazy"].max()) <= 1.0


def test_cached_crops_rotate_between_epochs_and_keep_pairing(tmp_path):
    build_cache(tmp_path, crops_per_pair=2)
    dataset = CachedCrops(tmp_path, split="train")
    epoch0 = dataset[0]["hazy"].clone()
    dataset.set_epoch(1)
    epoch1 = dataset[0]["hazy"].clone()
    assert not torch.equal(epoch0, epoch1)  # different cached crop
    # hazy must stay brighter than its clear twin (the ASM invariant)
    assert float(sample_mean(dataset[0]) - sample_mean_clear(dataset[0])) > 0


def sample_mean(sample):
    return float(sample["hazy"].mean())


def sample_mean_clear(sample):
    return float(sample["clear"].mean())


def test_cached_crops_beta_filter(tmp_path):
    build_cache(tmp_path, crops_per_pair=1)
    dataset = CachedCrops(tmp_path, split="train", beta_filter="heavy")
    assert len(dataset) == 1
    assert dataset[0]["density_class"].item() == 2


def test_incomplete_cache_is_refused(tmp_path):
    """An interrupted build must never be readable as if complete."""
    build_cache(tmp_path, crops_per_pair=1)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        CachedCrops(tmp_path, split="train")


def build_context_cache(root, pairs=2, size=(64, 128)):
    """Whole-image context cache: one row per pair, aligned with build_cache."""
    root.mkdir(parents=True, exist_ok=True)
    rows = [np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(*size, 3) + pair for pair in range(pairs)]
    np.save(root / "hazy.npy", np.stack(rows))
    (root / "manifest.json").write_text(json.dumps({"crops_per_pair": 1, "rows": pairs, "complete": True}))


def test_context_dataset_pairs_each_crop_with_its_whole_image(tmp_path):
    build_cache(tmp_path / "crops", crops_per_pair=2)
    build_context_cache(tmp_path / "context", pairs=2)
    dataset = CachedCropsWithContext(tmp_path / "crops", tmp_path / "context", split="train")
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["context"].shape == (3, 64, 128)
    assert sample["context"].dtype == torch.float32
    # pair 0's context must be distinct from pair 1's (alignment, not repetition)
    assert not torch.equal(sample["context"], dataset[1]["context"])


def test_context_dataset_refuses_misaligned_or_incomplete_cache(tmp_path):
    build_cache(tmp_path / "crops", crops_per_pair=2)
    build_context_cache(tmp_path / "context", pairs=3)
    with pytest.raises(ValueError, match="not aligned"):
        CachedCropsWithContext(tmp_path / "crops", tmp_path / "context", split="train")
    build_context_cache(tmp_path / "context2", pairs=2)
    (tmp_path / "context2" / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        CachedCropsWithContext(tmp_path / "crops", tmp_path / "context2", split="train")
