import json
import numpy as np
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
from ip_dehaze.evaluation.coco_bootstrap import image_records, average_precision


def test_paired_bootstrap_identical_run_is_zero():
    # Synthetic test data: identical records
    record = {
        "dtScores": np.array([0.95, 0.85]),
        "dtMatches": np.ones((10, 2)),
        "dtIgnore": np.zeros((10, 2), dtype=bool),
        "gtIgnore": np.zeros(2, dtype=bool),
    }
    rec_a = {1: {101: record, 102: record}}
    rec_b = {1: {101: record, 102: record}}

    img_ids = np.array([101, 102])
    rng = np.random.default_rng(1729)

    for _ in range(20):
        sampled = rng.choice(img_ids, size=len(img_ids), replace=True)
        ap_a = average_precision(rec_a[1], sampled, iou_count=10)
        ap_b = average_precision(rec_b[1], sampled, iou_count=10)
        np.testing.assert_allclose(ap_a, ap_b, atol=1e-12)
        diff = ap_a - ap_b
        np.testing.assert_allclose(diff, np.zeros(10), atol=1e-12)
