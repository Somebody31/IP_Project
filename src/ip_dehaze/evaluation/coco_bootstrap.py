from __future__ import annotations

import numpy as np


def image_records(evaluator, image_ids: list[int], category_ids: list[int]) -> dict[int, dict[int, dict]]:
    params = evaluator.params
    area_index = params.areaRngLbl.index("all")
    records = {category_id: {} for category_id in category_ids}
    for category_index, category_id in enumerate(params.catIds):
        if category_id not in records:
            continue
        for image_index, image_id in enumerate(params.imgIds):
            record = evaluator.evalImgs[category_index * len(params.areaRng) * len(params.imgIds)
                                        + area_index * len(params.imgIds) + image_index]
            if record is not None:
                records[category_id][image_id] = record
    return records


def average_precision(record_by_image: dict[int, dict], sampled_ids: np.ndarray,
                      iou_count: int, max_detections: int = 100) -> np.ndarray:
    multiplicities = np.bincount(sampled_ids, minlength=max(record_by_image, default=-1) + 1)
    selected = [(image_id, int(multiplicities[image_id])) for image_id in record_by_image
                if image_id < len(multiplicities) and multiplicities[image_id] > 0]
    if not selected:
        return np.full(iou_count, np.nan)

    gt_count = 0
    score_parts = []
    match_parts = []
    ignore_parts = []
    for image_id, multiplicity in selected:
        record = record_by_image[image_id]
        gt_ignore = np.asarray(record["gtIgnore"], dtype=bool)
        gt_count += int(np.count_nonzero(~gt_ignore)) * multiplicity
        scores = np.asarray(record["dtScores"][:max_detections], dtype=np.float64)
        matches = np.asarray(record["dtMatches"][:, :max_detections])
        ignored = np.asarray(record["dtIgnore"][:, :max_detections], dtype=bool)
        for _ in range(multiplicity):
            score_parts.append(scores)
            match_parts.append(matches)
            ignore_parts.append(ignored)
    if gt_count == 0:
        return np.full(iou_count, np.nan)
    if not score_parts or not any(len(scores) for scores in score_parts):
        return np.zeros(iou_count, dtype=np.float64)

    scores = np.concatenate(score_parts)
    matches = np.concatenate(match_parts, axis=1)
    ignored = np.concatenate(ignore_parts, axis=1)
    order = np.argsort(-scores, kind="mergesort")
    matches = matches[:, order]
    ignored = ignored[:, order]
    true_positive = (matches > 0) & ~ignored
    false_positive = (matches == 0) & ~ignored
    tp = np.cumsum(true_positive, axis=1, dtype=np.float64)
    fp = np.cumsum(false_positive, axis=1, dtype=np.float64)
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
    recall_points = np.linspace(0, 1, 101)
    sampled_precision = np.zeros((iou_count, len(recall_points)), dtype=np.float64)
    for threshold_index in range(iou_count):
        indices = np.searchsorted(recall[threshold_index], recall_points, side="left")
        valid = indices < precision.shape[1]
        sampled_precision[threshold_index, valid] = precision[threshold_index, indices[valid]]
    return sampled_precision.mean(axis=1)


def paired_image_bootstrap(normal_records: dict[int, dict[int, dict]],
                           dcp_records: dict[int, dict[int, dict]],
                           image_ids: list[int], category_ids: list[int],
                           iou_count: int = 10, resamples: int = 2000,
                           seed: int = 1729, max_detections: int = 100) -> dict:
    rng = np.random.default_rng(seed)
    image_ids_array = np.asarray(image_ids, dtype=np.int64)
    draw_indices = rng.integers(0, len(image_ids_array), size=(resamples, len(image_ids_array)))
    differences = {"AP@0.5": [], "AP@0.5:0.95": []}
    per_class = {category_id: {"AP@0.5": [], "AP@0.5:0.95": []}
                 for category_id in category_ids}

    for indices in draw_indices:
        sampled_ids = image_ids_array[indices]
        normal_class_ap = []
        dcp_class_ap = []
        for category_id in category_ids:
            normal_ap = average_precision(normal_records[category_id], sampled_ids,
                                          iou_count, max_detections)
            dcp_ap = average_precision(dcp_records[category_id], sampled_ids,
                                       iou_count, max_detections)
            if np.isfinite(normal_ap).all() and np.isfinite(dcp_ap).all():
                normal_class_ap.append(normal_ap)
                dcp_class_ap.append(dcp_ap)
                per_class[category_id]["AP@0.5"].append(float(dcp_ap[0] - normal_ap[0]))
                per_class[category_id]["AP@0.5:0.95"].append(float(dcp_ap.mean() - normal_ap.mean()))
        normal_mean = np.mean(normal_class_ap, axis=0)
        dcp_mean = np.mean(dcp_class_ap, axis=0)
        differences["AP@0.5"].append(float(dcp_mean[0] - normal_mean[0]))
        differences["AP@0.5:0.95"].append(float(dcp_mean.mean() - normal_mean.mean()))

    def interval(values: list[float]) -> dict:
        values_array = np.asarray(values, dtype=np.float64)
        return {
            "mean_delta": float(values_array.mean()),
            "ci95": [float(value) for value in np.quantile(values_array, [0.025, 0.975])],
            "resamples": len(values_array),
            "seed": seed,
        }

    return {
        "paired_image_bootstrap": {metric: interval(values) for metric, values in differences.items()},
        "per_class_paired_image_bootstrap": {
            str(category_id): {metric: interval(values) for metric, values in metric_values.items()}
            for category_id, metric_values in per_class.items()
        },
    }
