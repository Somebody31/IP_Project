"""Paired Foggy Cityscapes dataset for dehazing training and validation.

Reuses `ip_dehaze.traditional.pairing.index_pairs` for scene/β matching so
the ML and traditional tracks agree on what a "pair" is. Pairing lives in
the traditional package but the logic is shared; import it, never copy it.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ip_dehaze.traditional.pairing import index_pairs

BETAS = (0.005, 0.01, 0.02)
BETA_CLASSES = {"light": 0.005, "medium": 0.01, "heavy": 0.02}
# Whole-scene size used by the gated model's context branch (width, height) —
# keep the training cache and every evaluation path on this single value.
CONTEXT_SIZE = (512, 256)


def beta_class(beta: float) -> int:
    """Map a Foggy Cityscapes β to its density class index (0/1/2)."""
    for index, value in enumerate(BETAS):
        if abs(beta - value) < 1e-9:
            return index
    raise ValueError(f"Unexpected beta {beta!r}; expected one of {BETAS}")


def class_name(index: int) -> str:
    return ("light", "medium", "heavy")[index]


class CachedCrops(Dataset):
    """Memmap-backed pre-extracted crops — removes PNG-decode I/O from training.

    Built by `scripts/machine_learning/build_crop_cache.py`. Rows are laid out
    pair-major (pair `p` owns rows `p*K .. p*K+K-1`); each epoch serves a
    different crop per pair, so length equals the number of pairs while crop
    diversity still varies across epochs.
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        beta_filter: float | str | None = None,
        hflip: bool = True,
        hazy_only: bool = False,
        seed: int = 1729,
    ) -> None:
        import csv
        import json

        self.dir = Path(cache_dir)
        manifest_path = self.dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{cache_dir} has no manifest.json — the cache is missing or was "
                "interrupted mid-build; rebuild it with scripts/machine_learning/build_crop_cache.py"
            )
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            raise ValueError(f"{cache_dir} is marked incomplete; rebuild it")
        self.crops_per_pair = int(manifest["crops_per_pair"])
        self.hazy = np.load(self.dir / "hazy.npy", mmap_mode="r")
        self.clear = np.load(self.dir / "clear.npy", mmap_mode="r")
        self.rows = list(csv.DictReader((self.dir / "index.csv").open()))
        if len(self.rows) != int(manifest["rows"]):
            raise ValueError(
                f"{cache_dir} index has {len(self.rows)} rows but manifest says "
                f"{manifest['rows']} — cache is inconsistent; rebuild it"
            )
        if split != self.rows[0]["split"]:
            raise ValueError(f"Cache holds split {self.rows[0]['split']!r}, requested {split!r}")

        wanted = None
        if beta_filter is not None:
            wanted = BETA_CLASSES[beta_filter] if isinstance(beta_filter, str) else beta_filter
        self.pair_indices = [
            index
            for index in range(len(self.rows) // self.crops_per_pair)
            if wanted is None or abs(float(self.rows[index * self.crops_per_pair]["beta"]) - wanted) < 1e-9
        ]
        if not self.pair_indices:
            raise ValueError("No pairs selected from cache; check split/beta_filter")
        self.hflip = hflip
        self.seed = seed
        # ponytail: fork-shared epoch counter — persistent DataLoader workers
        # never see main-process set_epoch(); a Value created before fork is
        # inherited by fork workers, so crop rotation reaches them.
        self._epoch = multiprocessing.Value("i", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = epoch

    @property
    def epoch(self) -> int:
        return self._epoch.value

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, index: int) -> dict:
        pair_index = self.pair_indices[index]
        row_index = pair_index * self.crops_per_pair + (self.epoch % self.crops_per_pair)
        meta = self.rows[row_index]
        hazy = np.array(self.hazy[row_index])
        clear = np.array(self.clear[row_index])
        if self.hflip and random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index).random() < 0.5:
            hazy, clear = hazy[:, ::-1], clear[:, ::-1]
        to_tensor = lambda array: torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return {
            "hazy": to_tensor(hazy.astype(np.float32) / 255.0),
            "clear": to_tensor(clear.astype(np.float32) / 255.0),
            "beta": torch.tensor(float(meta["beta"]), dtype=torch.float32),
            "density_class": torch.tensor(int(meta["density_class"]), dtype=torch.long),
            "scene_id": f"{meta['split']}/{meta['city']}/{meta['image_id']}",
        }


class CachedCropsWithDcpT(CachedCrops):
    """Crop cache plus DCP's per-crop transmission estimate as a 4th channel.

    The channel is the physical cue that transfers to real fog: at RTTS eval
    time the model can read a per-image DCP transmission estimate it never had
    before, while in-domain the network is free to ignore it. The channel is
    flipped with the same coin as hazy/clear so its gradients stay aligned.
    """

    def __init__(self, cache_dir: str, **kwargs) -> None:
        super().__init__(cache_dir, **kwargs)
        import json

        self.dcp_t = np.load(self.dir / "dcp_t.npy", mmap_mode="r")
        manifest = json.loads((self.dir / "manifest.json").read_text())
        if not manifest.get("dcp_t"):
            raise ValueError("cache was not built with --dcp-t; rebuild it")
        if self.dcp_t.shape[:3] != self.hazy.shape[:3]:
            raise ValueError("dcp_t.npy shape does not match the hazy cache")

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        pair_index = self.pair_indices[index]
        row_index = pair_index * self.crops_per_pair + (self.epoch % self.crops_per_pair)
        transmission = np.array(self.dcp_t[row_index]).astype(np.float32) / 255.0
        if self.hflip and random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index).random() < 0.5:
            transmission = transmission[:, ::-1]
        sample["dcp_t"] = torch.from_numpy(np.ascontiguousarray(transmission[None, :, :]))  # (1, H, W)
        return sample


class CachedCropsWithContext(CachedCrops):
    """Crop cache for restoration plus a whole-image context for the gate.

    A 256 crop carries a different field of view than the full scene, so a
    crop-trained gate reads a different density than the same scene at full
    resolution (measured: pooled d 0.80-0.88 on crops vs 0.72 full-image).
    `context_dir` is a resize cache built over the same split
    (`build_crop_cache.py --split train --resize 512x256 --out-name resize-512x256`)
    whose rows line up pair-for-pair with the crop cache.
    """

    def __init__(
        self,
        cache_dir: str,
        context_dir: str,
        split: str = "train",
        beta_filter: float | str | None = None,
        hflip: bool = True,
        seed: int = 1729,
    ) -> None:
        import json

        super().__init__(cache_dir, split=split, beta_filter=beta_filter, hflip=hflip, seed=seed)
        self.context_dir = Path(context_dir)
        manifest_path = self.context_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"{context_dir} has no manifest.json — build the whole-image context cache with "
                "scripts/machine_learning/build_crop_cache.py --resize 512x256 --out-name resize-512x256"
            )
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete"):
            raise ValueError(f"{context_dir} is marked incomplete; rebuild it")
        self.context = np.load(self.context_dir / "hazy.npy", mmap_mode="r")
        n_pairs = len(self.rows) // self.crops_per_pair
        if self.context.shape[0] != n_pairs:
            raise ValueError(
                f"context cache has {self.context.shape[0]} rows but the crop cache holds "
                f"{n_pairs} pairs — the two caches are not aligned"
            )

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        row = np.array(self.context[self.pair_indices[index]]).astype(np.float32) / 255.0
        sample["context"] = torch.from_numpy(np.ascontiguousarray(row.transpose(2, 0, 1)))
        return sample


def _resize(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Downscale to (width, height) — fog density is a global property."""
    image = Image.fromarray((array * 255).round().astype(np.uint8))
    return np.asarray(image.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


def load_rgb(path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


class FoggyPairs(Dataset):
    """Returns (hazy, clear) crops plus the pair's β and density class.

    Crops are deterministic given (seed, epoch, index) so a run can be
    reproduced exactly; call `set_epoch` between epochs to vary them.
    """

    def __init__(
        self,
        foggy_root: str,
        clear_root: str,
        split: str,
        crop: int | None = 256,
        resize: tuple[int, int] | None = None,
        beta_filter: float | str | None = None,
        refined_manifest: str | None = None,
        hflip: bool = True,
        hazy_only: bool = False,
        seed: int = 1729,
        stride: int = 1,
        offset: int = 0,
    ) -> None:
        self.pairs = index_pairs(foggy_root, clear_root, split, refined_manifest)
        if beta_filter is not None:
            wanted = BETA_CLASSES[beta_filter] if isinstance(beta_filter, str) else beta_filter
            self.pairs = [pair for pair in self.pairs if abs(pair.beta - wanted) < 1e-9]
        if stride > 1:
            self.pairs = self.pairs[offset::stride]
        elif offset:
            self.pairs = self.pairs[offset:]
        if not self.pairs:
            raise ValueError("No pairs selected; check split/beta_filter/stride")
        self.crop = crop
        self.resize = resize
        self.hflip = hflip
        self.hazy_only = hazy_only
        self.seed = seed
        self._epoch = multiprocessing.Value("i", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = epoch

    @property
    def epoch(self) -> int:
        return self._epoch.value

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        pair = self.pairs[index]
        hazy = load_rgb(pair.foggy_path)
        if self.hazy_only:
            # classification only needs the input image; skipping the clear
            # decode halves loader cost
            if self.crop is not None:
                rng = random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index)
                height, width = hazy.shape[:2]
                top = rng.randrange(height - self.crop + 1)
                left = rng.randrange(width - self.crop + 1)
                hazy = hazy[top : top + self.crop, left : left + self.crop]
                if self.hflip and rng.random() < 0.5:
                    hazy = hazy[:, ::-1]
            if self.resize is not None:
                hazy = _resize(hazy, self.resize)
            return {
                "hazy": torch.from_numpy(np.ascontiguousarray(hazy.transpose(2, 0, 1))),
                "clear": torch.zeros(3, 1, 1),
                "beta": torch.tensor(pair.beta, dtype=torch.float32),
                "density_class": torch.tensor(beta_class(pair.beta), dtype=torch.long),
                "scene_id": f"{pair.split}/{pair.city}/{pair.image_id}",
            }
        clear = load_rgb(pair.clear_path)
        if hazy.shape != clear.shape:
            raise ValueError(f"Size mismatch for {pair.foggy_path}")

        if self.resize is not None:
            hazy = _resize(hazy, self.resize)
            clear = _resize(clear, self.resize)

        if self.crop is not None:
            # Integer seed: random.seed rejects tuples on Python 3.11+.
            rng = random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index)
            height, width = hazy.shape[:2]
            if height < self.crop or width < self.crop:
                raise ValueError(f"Image smaller than crop for {pair.foggy_path}")
            top = rng.randrange(height - self.crop + 1)
            left = rng.randrange(width - self.crop + 1)
            hazy = hazy[top : top + self.crop, left : left + self.crop]
            clear = clear[top : top + self.crop, left : left + self.crop]
            if self.hflip and rng.random() < 0.5:
                hazy = hazy[:, ::-1]
                clear = clear[:, ::-1]

        to_tensor = lambda array: torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return {
            "hazy": to_tensor(hazy),
            "clear": to_tensor(clear),
            "beta": torch.tensor(pair.beta, dtype=torch.float32),
            "density_class": torch.tensor(beta_class(pair.beta), dtype=torch.long),
            "scene_id": f"{pair.split}/{pair.city}/{pair.image_id}",
        }
