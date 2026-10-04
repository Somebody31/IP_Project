"""Fast paired bootstrap CIs for RTTS detection evaluations (image-level resampling).

Uses precomputed COCOeval image records to run 1,000+ resamples in seconds.
Supports paired comparison of all methods within a single run (e.g. hazy, dcp,
ours, routed) or comparing across two separate runs.

Usage:
  uv run python scripts/machine_learning/bootstrap_detection_ci.py \
    --run runs/machine_learning/paired-all-methods-valid-20261004 \
    --resamples 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ip_dehaze.evaluation.coco_bootstrap import average_precision, image_records


def evaluate_records(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
    detections: list[dict],
    iou_thresholds: list[float],
) -> dict[int, dict[int, dict]]:
    gt = COCO()
    gt.dataset = {
        "info": {"description": "RTTS Bootstrap Evaluation"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    gt.createIndex()
    if not detections:
        evaluator = COCOeval(gt, None, "bbox")
    else:
        dt = gt.loadRes(detections)
        evaluator = COCOeval(gt, dt, "bbox")
    evaluator.params.imgIds = [im["id"] for im in images]
    evaluator.params.catIds = [cat["id"] for cat in categories]
    evaluator.params.iouThrs = np.asarray(iou_thresholds, dtype=np.float64)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    cat_ids = [cat["id"] for cat in categories]
    img_ids = [im["id"] for im in images]
    return image_records(evaluator, img_ids, cat_ids)


def compute_ap_vector(
    records: dict[int, dict[int, dict]],
    sampled_ids: np.ndarray,
    category_ids: list[int],
    iou_count: int,
) -> np.ndarray:
    class_aps = []
    for cat_id in category_ids:
        ap = average_precision(records[cat_id], sampled_ids, iou_count=iou_count)
        if np.isfinite(ap).all():
            class_aps.append(ap)
    if not class_aps:
        return np.zeros(iou_count, dtype=np.float64)
    return np.mean(class_aps, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="run dir holding detections.json")
    parser.add_argument("--compare-run", type=Path, default=None, help="optional second run dir")
    parser.add_argument("--method-a", type=str, default="ours", help="primary method name")
    parser.add_argument("--method-b", type=str, default=None, help="secondary method name")
    parser.add_argument("--baseline", type=str, default="hazy", help="baseline method name for deltas")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    dump = json.loads((run_dir / "detections.json").read_text())
    images = dump["images"]
    annotations = dump["annotations"]
    categories = dump["categories"]
    detections: dict[str, list] = dict(dump["detections"])

    if args.compare_run:
        comp_dir = args.compare_run if args.compare_run.is_absolute() else ROOT / args.compare_run
        comp_dump = json.loads((comp_dir / "detections.json").read_text())
        for k, v in comp_dump["detections"].items():
            key_name = f"comp_{k}" if k in detections else k
            detections[key_name] = v

    category_ids = [cat["id"] for cat in categories]
    img_ids = np.array([im["id"] for im in images])
    iou_count = len(args.iou_thresholds)

    # 1. Compute evaluation records once per method
    records_by_method = {}
    print(f"Extracting evaluation records for {len(detections)} methods on {len(images)} images...")
    for name, dt_list in detections.items():
        print(f"  Indexing & evaluating {name} ({len(dt_list)} detections)...")
        records_by_method[name] = evaluate_records(
            images, annotations, categories, dt_list, args.iou_thresholds
        )

    # 2. Resample image IDs with replacement
    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.integers(0, len(img_ids), size=(args.resamples, len(img_ids)))

    metric_keys = ("AP@0.5", "AP@0.75", "AP@0.95", "AP@0.5:0.95")
    # Metric index mapping: 0 -> 0.50, 5 -> 0.75, 9 -> 0.95
    levels: dict[str, dict[str, list[float]]] = {
        name: {k: [] for k in metric_keys} for name in detections
    }

    print(f"Running {args.resamples} paired bootstrap resamples...")
    for iter_idx in range(args.resamples):
        sampled_ids = img_ids[sampled_indices[iter_idx]]
        for name in detections:
            ap_vec = compute_ap_vector(records_by_method[name], sampled_ids, category_ids, iou_count)
            levels[name]["AP@0.5"].append(float(ap_vec[0]))
            levels[name]["AP@0.75"].append(float(ap_vec[5]))
            levels[name]["AP@0.95"].append(float(ap_vec[9]))
            levels[name]["AP@0.5:0.95"].append(float(ap_vec.mean()))

    # 3. Compile output summary
    out: dict = {
        "resamples": args.resamples,
        "seed": args.seed,
        "n_images": len(images),
        "levels": {},
        "deltas": {},
    }

    # Define key delta pairs to report
    pairs_to_compute: list[Tuple[str, str, str]] = []
    # vs baseline (e.g. hazy)
    if args.baseline in detections:
        for name in detections:
            if name != args.baseline:
                pairs_to_compute.append((name, args.baseline, f"{name}_minus_{args.baseline}"))

    # Key head-to-head pairs
    if "ours" in detections and "dcp" in detections:
        if ("ours", "dcp", "ours_minus_dcp") not in pairs_to_compute:
            pairs_to_compute.append(("ours", "dcp", "ours_minus_dcp"))
    if "ours" in detections and "routed" in detections:
        if ("ours", "routed", "ours_minus_routed") not in pairs_to_compute:
            pairs_to_compute.append(("ours", "routed", "ours_minus_routed"))
    if "dcp" in detections and "routed" in detections:
        if ("dcp", "routed", "dcp_minus_routed") not in pairs_to_compute:
            pairs_to_compute.append(("dcp", "routed", "dcp_minus_routed"))

    if args.method_b and args.method_a in detections and args.method_b in detections:
        custom_key = f"{args.method_a}_minus_{args.method_b}"
        if (args.method_a, args.method_b, custom_key) not in pairs_to_compute:
            pairs_to_compute.append((args.method_a, args.method_b, custom_key))

    for k in metric_keys:
        print(f"\n--- {k} ---")
        print(f"{'Method':<12} {k:>12} {'95% CI':>24}")
        out["levels"][k] = {}
        for name in detections:
            arr = np.asarray(levels[name][k])
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out["levels"][k][name] = {"mean": float(arr.mean()), "ci95": [float(lo), float(hi)]}
            print(f"{name:<12} {arr.mean():>12.4f} {f'[{lo:.4f}, {hi:.4f}]':>24}")

        print(f"\n{'Pairwise Delta':<26} {'Mean \u0394':>10} {'95% CI':>24}  {'Excludes 0':>12}")
        out["deltas"][k] = {}
        for (m1, m2, delta_label) in pairs_to_compute:
            diffs = np.asarray(levels[m1][k]) - np.asarray(levels[m2][k])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            excludes = bool(lo > 0 or hi < 0)
            out["deltas"][k][delta_label] = {
                "mean": float(diffs.mean()),
                "ci95": [float(lo), float(hi)],
                "excludes_zero": excludes,
            }
            sig_marker = "* (p < 0.05)" if excludes else "No"
            print(f"{delta_label:<26} {diffs.mean():>10.4f} {f'[{lo:.4f}, {hi:.4f}]':>24}  {sig_marker:>12}")

    target = args.out or (ROOT / "results" / "machine_learning" / f"detection-bootstrap-{run_dir.name}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + "\n")
    try:
        shown = target.relative_to(ROOT)
    except ValueError:
        shown = target
    print(f"\nWrote full paired bootstrap analysis to {shown}")


if __name__ == "__main__":
    main()
