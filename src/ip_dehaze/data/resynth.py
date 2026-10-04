"""On-the-fly realistic fog resynthesis over Cityscapes (Phase 2 of the RTTS track).

Motivation (val-comparison finding 15): with content held fixed, swapping only
the atmospheric-light *level* to real fog's (RTTS A ~ 0.86 vs ~0.93-0.96 in
Foggy Cityscapes) costs 3.8-5.4 dB, while real fog's transmission *structure*
costs nothing. Foggy Cityscapes also never shows camera artefacts that real
foggy photos carry (JPEG blocking, sensor noise, softness, small frames).

Render, per sample (clear target J is unchanged):
    t      = t_0.01 ** (beta / 0.01)          shipped Beer-Lambert map, re-densified
    A      = a * tint,  a ~ U[a_min, 1], tint ~ U[1-tint, 1+tint] per channel
    hazy   = J * t + A * (1 - t)
then degrade `hazy` only: optional downscale to RTTS-like frame sizes, blur,
Gaussian noise, JPEG.
"""
from __future__ import annotations

import io
import math
import multiprocessing
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

from ip_dehaze.traditional.pairing import index_pairs

ROOT = Path(__file__).resolve().parents[3]
TRANSMITTANCE_ROOT = ROOT / "data" / "transmittance" / "leftImg8bit_transmittance"
TRANSMITTANCE_ZIP = ROOT / "data" / "transmittance" / "leftImg8bit_trainvaltest_transmittance.zip"
SOURCE_BETA = 0.01


def render_fog(clear: np.ndarray, t_source: np.ndarray, beta: float, airlight: np.ndarray) -> np.ndarray:
    """Atmospheric scattering model with a re-densified shipped transmission map."""
    t = np.power(np.clip(t_source, 1e-4, 1.0), beta / SOURCE_BETA)[..., None]
    return clear * t + airlight[None, None, :] * (1.0 - t)


def degrade(hazy: np.ndarray, rng: random.Random) -> np.ndarray:
    """Camera-side artefacts real foggy photos carry and synthetic renders lack."""
    image = Image.fromarray((np.clip(hazy, 0, 1) * 255).round().astype(np.uint8))
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if rng.random() < 0.5:
        noise_rng = np.random.default_rng(rng.randrange(2**32))
        array = array + noise_rng.normal(0.0, rng.uniform(0.003, 0.02), array.shape).astype(np.float32)
    array = np.clip(array, 0, 1)
    if rng.random() < 0.7:
        buffer = io.BytesIO()
        Image.fromarray((array * 255).round().astype(np.uint8)).save(buffer, "JPEG",
                                                                     quality=rng.randint(30, 95))
        buffer.seek(0)
        array = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32) / 255.0
    return array


class ResynthPairs(Dataset):
    """(hazy, clear) crops with realistic fog rendered on the fly.

    Length is `repeats` x scenes so an epoch matches the 8,925-pair epochs of
    the original runs (each scene seen with `repeats` different fogs).
    """

    def __init__(self, split: str = "train", crop: int = 256, repeats: int = 3,
                 beta_range: tuple[float, float] = (0.003, 0.03),
                 airlight_min: float = 0.70, tint: float = 0.03,
                 downscale_prob: float = 0.5, downscale_range: tuple[float, float] = (0.25, 0.6),
                 degrade_prob: float = 0.8, heavy_fraction: float = 0.0, seed: int = 1729) -> None:
        pairs = index_pairs(ROOT / "data/foggy_cityscapes", ROOT / "data/cityscapes_clear", split)
        self.scenes = sorted({(p.city, p.image_id, str(p.clear_path)) for p in pairs})
        if not self.scenes:
            raise ValueError(f"no scenes for split {split!r}")
        self.split = split
        self.crop = crop
        self.repeats = repeats
        self.beta_range = beta_range
        self.airlight_min = airlight_min
        self.tint = tint
        self.downscale_prob = downscale_prob
        self.downscale_range = downscale_range
        self.degrade_prob = degrade_prob
        self.heavy_fraction = heavy_fraction
        self.seed = seed
        self._epoch = multiprocessing.Value("i", 0)
        self._zip: zipfile.ZipFile | None = None
        self._zip_pid: int | None = None

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = epoch

    def __len__(self) -> int:
        return len(self.scenes) * self.repeats

    def _open_transmittance(self, city: str, image_id: str) -> Image.Image:
        """Extracted file if present, else read from the shipped zip.

        Only the val maps are extracted on this machine; the train maps
        (0.72 GB at beta=0.01) are read in place to keep disk headroom. Each
        DataLoader worker opens its own ZipFile handle (not fork-safe to share).
        """
        name = f"leftImg8bit_transmittance/{self.split}/{city}/" \
               f"{image_id}_leftImg8bit_transmittance_beta_{SOURCE_BETA:g}.png"
        extracted = TRANSMITTANCE_ROOT.parent / name
        if extracted.is_file():
            return Image.open(extracted)
        pid = os.getpid()
        if self._zip is None or self._zip_pid != pid:
            self._zip = zipfile.ZipFile(TRANSMITTANCE_ZIP)
            self._zip_pid = pid
        return Image.open(io.BytesIO(self._zip.read(name)))

    def __getitem__(self, index: int) -> dict:
        city, image_id, clear_path = self.scenes[index % len(self.scenes)]
        rng = random.Random(self.seed * 1_000_003 + self._epoch.value * 10_007 + index)
        clear_img = Image.open(clear_path).convert("RGB")
        t_img = self._open_transmittance(city, image_id)
        if rng.random() < self.downscale_prob:
            factor = rng.uniform(*self.downscale_range)
            size = (max(self.crop, int(clear_img.width * factor)), max(self.crop, int(clear_img.height * factor)))
            clear_img = clear_img.resize(size, Image.BILINEAR)
            t_img = t_img.resize(size, Image.BILINEAR)
        width, height = clear_img.size
        top = rng.randrange(height - self.crop + 1)
        left = rng.randrange(width - self.crop + 1)
        box = (left, top, left + self.crop, top + self.crop)
        clear = np.asarray(clear_img.crop(box), dtype=np.float32) / 255.0
        t_source = np.asarray(t_img.crop(box), dtype=np.float32)
        t_source = (t_source[..., 0] if t_source.ndim == 3 else t_source) / 255.0

        if self.heavy_fraction > 0 and rng.random() < self.heavy_fraction:
            beta = rng.uniform(0.018, 0.024)
        else:
            low, high = self.beta_range
            beta = math.exp(rng.uniform(math.log(low), math.log(high)))
        level = rng.uniform(self.airlight_min, 1.0)
        airlight = np.clip(level * np.array([rng.uniform(1 - self.tint, 1 + self.tint) for _ in range(3)],
                                            dtype=np.float32), 0, 1)
        hazy = render_fog(clear, t_source, beta, airlight)
        if rng.random() < self.degrade_prob:
            hazy = degrade(hazy, rng)
        hazy = np.clip(hazy, 0, 1).astype(np.float32)
        if rng.random() < 0.5:
            hazy, clear = hazy[:, ::-1], clear[:, ::-1]
        to_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
        return {
            "hazy": to_tensor(hazy),
            "clear": to_tensor(clear),
            "beta": torch.tensor(beta, dtype=torch.float32),
            "airlight": torch.from_numpy(airlight),
            "scene_id": f"{self.split}/{city}/{image_id}",
        }
