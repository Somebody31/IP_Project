import numpy as np

from ip_dehaze.evaluation.coco_bootstrap import average_precision, paired_image_bootstrap


def make_record(score, matched, gt_count=1):
    return {
        "dtScores": np.array([score]),
        "dtMatches": np.asarray([[1.0 if matched else 0.0]] * 10),
        "dtIgnore": np.zeros((10, 1), dtype=bool),
        "gtIgnore": np.zeros(gt_count, dtype=bool),
    }


def test_average_precision_returns_one_for_single_perfect_detection():
    records = {1: make_record(0.9, True)}
    ap = average_precision(records, np.array([1]), iou_count=10)
    np.testing.assert_allclose(ap, np.ones(10))


def test_paired_bootstrap_reports_zero_for_identical_conditions():
    normal = {1: {1: make_record(0.9, True), 2: make_record(0.8, True)}}
    dcp = {1: {1: make_record(0.9, True), 2: make_record(0.8, True)}}
    result = paired_image_bootstrap(normal, dcp, [1, 2], [1], resamples=50, seed=4)
    for metric in ("AP@0.5", "AP@0.5:0.95"):
        assert result["paired_image_bootstrap"][metric]["ci95"] == [0.0, 0.0]
        assert result["paired_image_bootstrap"][metric]["mean_delta"] == 0.0
