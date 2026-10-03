"""Alignment tests for the RTTS detection crops used by task-driven adaptation.

The v1 dataset never shifted boxes by the crop offset and flipped boxes with
an independent coin; these tests pin the geometry so that cannot recur.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ip_dehaze.data.rtts_detection import (
    RTTS_TO_COCO,
    RttsDetectionCrops,
    collate_detection,
    crop_boxes,
    to_detector_batch,
)


def _make_image(tmp_path: Path, width=600, height=400, box=(350, 120, 450, 220), cls=2):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    array = np.zeros((height, width, 3), dtype=np.uint8)
    x1, y1, x2, y2 = box
    array[y1:y2, x1:x2] = 255  # the object is a white rectangle
    Image.fromarray(array).save(images / "a.jpg", quality=100)
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    (labels / "a.txt").write_text(f"{cls} {cx} {cy} {(x2 - x1) / width} {(y2 - y1) / height}\n")
    return images, labels


def test_boxes_follow_crop_and_flip(tmp_path):
    images, labels = _make_image(tmp_path)
    dataset = RttsDetectionCrops(images, labels, crop=256, object_bias=1.0)
    flips_seen = set()
    for epoch in range(12):
        dataset.set_epoch(epoch)
        sample = dataset[0]
        boxes = sample["boxes"].numpy()
        assert len(boxes) == 1
        cls, x1, y1, x2, y2 = boxes[0]
        assert int(cls) == RTTS_TO_COCO[2]
        image = sample["hazy"].numpy().mean(0)
        inside = image[int(y1) + 2 : int(y2) - 2, int(x1) + 2 : int(x2) - 2]
        assert inside.size and inside.mean() > 0.9, "box does not cover the white object"
        mask = np.ones_like(image, dtype=bool)
        mask[max(0, int(y1) - 2) : int(y2) + 2, max(0, int(x1) - 2) : int(x2) + 2] = False
        assert image[mask].mean() < 0.05, "white object found outside the box"
        flips_seen.add(bool(x1 < 256 - x2))
    assert flips_seen == {True, False}, "12 epochs should exercise both flip orientations"


def test_crop_boxes_drops_slivers():
    boxes = np.array([[0, 0, 0, 100, 100], [0, 250, 0, 300, 50]], dtype=np.float32)
    out = crop_boxes(boxes, top=0, left=0, crop=256, flip=False)
    assert len(out) == 1  # the second box keeps only 6/50 of its width


def test_detector_batch_scales_boxes():
    restored = torch.rand(2, 3, 256, 256)
    boxes = [torch.tensor([[2.0, 64, 64, 128, 128]]), torch.zeros(0, 5)]
    canvas, batch = to_detector_batch(restored, boxes, [0.5, 1.0])
    assert canvas.shape == (2, 3, 640, 640)
    # scale 0.5: crop occupies 128 px; box (64..128) -> (32..64) px -> centre 48/640
    np.testing.assert_allclose(batch["bboxes"][0].numpy(), [48 / 640, 48 / 640, 32 / 640, 32 / 640], atol=1e-5)
    assert batch["batch_idx"].tolist() == [0.0]
    assert torch.allclose(canvas[0, :, 200, 200], torch.full((3,), 114 / 255))


def test_collate_keeps_variable_boxes(tmp_path):
    images, labels = _make_image(tmp_path)
    dataset = RttsDetectionCrops(images, labels, crop=256)
    batch = collate_detection([dataset[0], dataset[0]])
    assert batch["hazy"].shape == (2, 3, 256, 256)
    assert isinstance(batch["boxes"], list) and len(batch["boxes"]) == 2
