import csv

import numpy as np
import torch

from ip_dehaze.data.dataset import CachedCrops


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
