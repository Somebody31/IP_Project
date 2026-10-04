"""RTTS detection crops for task-driven dehazer training.

Replaces the original `RttTrainSet` in `scripts/machine_learning/train_task_adapt.py`,
which had three alignment bugs (found 2026-10-03):
  * GT boxes were computed in full-image coordinates and never shifted by the
    crop offset, so they did not land on the crop;
  * each box drew its own flip coin, independent of the image's flip;
  * class labels were dropped.

Geometry contract (mirrors the evaluation protocol): the dehazer sees a crop at
the image's *native* resolution (eval runs the dehazer at native resolution),
and the detector sees that crop scaled by `scale = det_size / max(H, W)` — the
same factor YOLO's letterbox applies to the whole image at eval — so objects
reach the detector at the pixel size they will have at evaluation time.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# rtts/data.yaml names order -> COCO ids used by the frozen YOLOv8n
# (bicycle, bus, car, motorbike, person); must agree with
# configs/traditional/dcp_rtts_detection.yaml.
RTTS_TO_COCO = {0: 1, 1: 5, 2: 2, 3: 3, 4: 0}


def read_yolo_labels(path: Path, width: int, height: int) -> np.ndarray:
    """YOLO txt -> (N, 5) array [coco_cls, x1, y1, x2, y2] in native pixels."""
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = (float(v) for v in line.split()[:5])
            x1 = (cx - bw / 2) * width
            y1 = (cy - bh / 2) * height
            rows.append([RTTS_TO_COCO[int(cls)], x1, y1, x1 + bw * width, y1 + bh * height])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def crop_boxes(boxes: np.ndarray, top: int, left: int, crop: int, flip: bool,
               min_area: float = 8 * 8, min_keep: float = 0.4) -> np.ndarray:
    """Shift native-pixel boxes into crop coordinates, clip, optionally h-flip.

    A box survives if its clipped area is at least `min_area` px and at least
    `min_keep` of its original area (a sliver of a car is not a car).
    """
    if len(boxes) == 0:
        return boxes.reshape(0, 5)
    out = boxes.copy()
    out[:, [1, 3]] -= left
    out[:, [2, 4]] -= top
    area = (out[:, 3] - out[:, 1]) * (out[:, 4] - out[:, 2])
    out[:, 1:5] = np.clip(out[:, 1:5], 0, crop)
    clipped = (out[:, 3] - out[:, 1]) * (out[:, 4] - out[:, 2])
    keep = (clipped >= min_area) & (clipped >= min_keep * np.maximum(area, 1e-6))
    out = out[keep]
    if flip:
        x1 = crop - out[:, 3]
        x2 = crop - out[:, 1]
        out[:, 1], out[:, 3] = x1, x2
    return out


class RttsDetectionCrops(Dataset):
    """Native-resolution crops of RTTS images with crop-aligned COCO-class boxes.

    Returns `hazy` (3, crop, crop) in [0, 1], `boxes` (N, 5) [cls, x1, y1, x2, y2]
    in crop pixels, and `scale` (detector px per native px).
    Images smaller than the crop are reflect-padded (no boxes in the padding).
    With probability `object_bias` the crop is centred near a random GT box so
    most crops carry a task signal.
    """

    def __init__(self, images_dir: Path, labels_dir: Path, crop: int = 384,
                 det_size: int = 640, object_bias: float = 0.8,
                 limit: int = 0, seed: int = 1729,
                 exclude: set[str] | Path | None = None) -> None:
        exclude_set = set()
        if isinstance(exclude, (str, Path)):
            exclude_path = Path(exclude)
            if exclude_path.is_file():
                if exclude_path.suffix == ".csv":
                    import csv
                    with open(exclude_path) as f:
                        exclude_set = {r["name"] for r in csv.DictReader(f) if "name" in r}
                else:
                    exclude_set = {line.strip() for line in exclude_path.read_text().splitlines() if line.strip()}
        elif isinstance(exclude, (set, list, tuple)):
            exclude_set = set(exclude)

        self.paths = [p for p in sorted(Path(images_dir).glob("*.jpg")) if p.name not in exclude_set]
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError(f"no images under {images_dir}")
        self.labels_dir = Path(labels_dir)
        self.crop = crop
        self.det_size = det_size
        self.object_bias = object_bias
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        rng = random.Random(self.seed * 1_000_003 + self.epoch * 10_007 + index)
        with Image.open(path) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]
        boxes = read_yolo_labels(self.labels_dir / f"{path.stem}.txt", width, height)
        scale = self.det_size / max(height, width)

        crop = self.crop
        if height < crop or width < crop:
            rgb = np.pad(rgb, ((0, max(0, crop - height)), (0, max(0, crop - width)), (0, 0)),
                         mode="reflect")
        span_y = rgb.shape[0] - crop
        span_x = rgb.shape[1] - crop
        if len(boxes) and rng.random() < self.object_bias:
            cls, x1, y1, x2, y2 = boxes[rng.randrange(len(boxes))]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            jitter = crop // 4
            left = int(np.clip(cx - crop / 2 + rng.randint(-jitter, jitter), 0, span_x))
            top = int(np.clip(cy - crop / 2 + rng.randint(-jitter, jitter), 0, span_y))
        else:
            top = rng.randint(0, span_y)
            left = rng.randint(0, span_x)
        image = rgb[top : top + crop, left : left + crop]
        flip = rng.random() < 0.5  # ONE coin for the image and all its boxes
        if flip:
            image = image[:, ::-1]
        boxes = crop_boxes(boxes, top, left, crop, flip)
        return {
            "hazy": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "boxes": torch.from_numpy(boxes),
            "scale": float(scale),
            "name": path.name,
        }


def collate_detection(samples: list[dict]) -> dict:
    return {
        "hazy": torch.stack([s["hazy"] for s in samples]),
        "boxes": [s["boxes"] for s in samples],
        "scale": [s["scale"] for s in samples],
        "name": [s["name"] for s in samples],
    }


def to_detector_batch(restored: torch.Tensor, boxes: list[torch.Tensor], scales: list[float],
                      det_size: int = 640, pad_value: float = 114 / 255) -> tuple[torch.Tensor, dict]:
    """Differentiably paste each scaled crop onto a letterbox-grey canvas.

    Returns the (B, 3, det_size, det_size) canvas and an Ultralytics loss batch
    (`batch_idx`, `cls`, `bboxes` as normalised xywh on the canvas).
    """
    batch, _, crop, _ = restored.shape
    device = restored.device
    canvas = torch.full((batch, 3, det_size, det_size), pad_value, device=device, dtype=restored.dtype)
    idx, cls, xywh = [], [], []
    for b in range(batch):
        side = min(det_size, max(32, int(round(crop * scales[b]))))
        resized = torch.nn.functional.interpolate(
            restored[b : b + 1], size=(side, side), mode="bilinear", align_corners=False)
        canvas[b, :, :side, :side] = resized[0]
        if len(boxes[b]):
            bx = boxes[b].to(device).float()
            factor = side / crop
            x1, y1, x2, y2 = (bx[:, k] * factor for k in (1, 2, 3, 4))
            xywh.append(torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1) / det_size)
            cls.append(bx[:, 0:1])
            idx.append(torch.full((len(bx),), float(b), device=device))
    if idx:
        loss_batch = {"batch_idx": torch.cat(idx), "cls": torch.cat(cls), "bboxes": torch.cat(xywh)}
    else:
        loss_batch = {"batch_idx": torch.zeros(0, device=device), "cls": torch.zeros(0, 1, device=device),
                      "bboxes": torch.zeros(0, 4, device=device)}
    return canvas, loss_batch
