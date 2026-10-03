"""Bootstrap CIs for the RTTS detection deltas (image-level resampling).

The detection table quotes point estimates: ours -0.007 AP@0.5 vs hazy,
DCP +0.027, routed -0.013. Without a CI we cannot say whether a -0.007
difference is real. This resamples *images* with replacement, rebuilds the
COCO ground truth and detections for each resample, and recomputes AP@0.5,
giving a CI on both the levels and the deltas.

Input: the detections dump produced by
  evaluate_model_detection.py --save-detections
(scene-level AP cannot be averaged per image, so AP must be recomputed on
each resample — hence the dump.)

Usage:
  uv run python scripts/machine_learning/bootstrap_detection_ci.py \
    --run runs/machine_learning/rtts-test-bootstrap-20261002
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(ROOT / "scripts" / "traditional"))
from evaluate_dcp_detection import coco_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="run dir holding detections.json")
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--iou-thresholds", type=float, nargs="+",
                        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    dump = json.loads((run_dir / "detections.json").read_text())
    images = dump["images"]
    annotations = dump["annotations"]
    categories = dump["categories"]
    detections = dump["detections"]

    by_image: dict[int, list] = {image["id"]: [] for image in images}
    for name, rows in detections.items():
        for row in rows:
            by_image[row["image_id"]].append((name, row))
    anns_by_image: dict[int, list] = {image["id"]: [] for image in images}
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    ids = np.array([image["id"] for image in images])
    rng = np.random.default_rng(args.seed)
    metric_keys = ("AP@0.5", "AP@0.75", "AP@0.95", "AP@0.5:0.95")
    levels: dict[str, dict[str, list[float]]] = {
        name: {k: [] for k in metric_keys} for name in detections
    }

    for iteration in range(args.resamples):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        # Re-id resampled images so duplicated sources stay distinct entries.
        remap = {}
        new_images = []
        new_anns = []
        new_dets: dict[str, list] = {name: [] for name in detections}
        ann_id = 1
        for position, source_id in enumerate(sampled):
            source_id = int(source_id)
            new_id = position + 1
            remap[source_id] = new_id
            template = next(image for image in images if image["id"] == source_id)
            new_images.append({**template, "id": new_id})
            for ann in anns_by_image[source_id]:
                new_anns.append({**ann, "id": ann_id, "image_id": new_id})
                ann_id += 1
            for name, row in by_image[source_id]:
                new_dets[name].append({**row, "image_id": new_id})

        for name, rows in new_dets.items():
            res = coco_metrics(new_images, new_anns, categories, rows, args.iou_thresholds)
            for k in metric_keys:
                levels[name][k].append(res[k])
        if (iteration + 1) % 25 == 0:
            print(f"  {iteration + 1}/{args.resamples}", flush=True)

    out: dict = {"resamples": args.resamples, "seed": args.seed,
                 "n_images": len(images), "levels": {}, "deltas": {}}

    for k in metric_keys:
        print(f"\n--- {k} ---")
        print(f"{'column':<8} {k:>12} {'CI95':>20}")
        out["levels"][k] = {}
        for name in levels:
            array = np.asarray(levels[name][k])
            lo, hi = np.percentile(array, [2.5, 97.5])
            out["levels"][k][name] = {"mean": float(array.mean()), "ci95": [float(lo), float(hi)]}
            print(f"{name:<8} {array.mean():>12.4f} {f'[{lo:.4f}, {hi:.4f}]':>20}")

        print(f"\n{'delta':<22} {'mean':>8} {'CI95':>20}  {'excludes 0':>10}")
        out["deltas"][k] = {}
        for name in levels:
            if name == "hazy":
                continue
            diffs = np.asarray(levels[name][k]) - np.asarray(levels["hazy"][k])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            excludes = bool(lo > 0 or hi < 0)
            out["deltas"][k][f"{name}_minus_hazy"] = {
                "mean": float(diffs.mean()), "ci95": [float(lo), float(hi)], "excludes_zero": excludes}
            print(f"{name + '-hazy':<22} {diffs.mean():>8.4f} {f'[{lo:.4f}, {hi:.4f}]':>20}  {str(excludes):>10}")

    target = args.out or (ROOT / "results" / "machine_learning" /
                          f"detection-bootstrap-{run_dir.name}.json")
    target.write_text(json.dumps(out, indent=2) + "\n")
    try:
        shown = target.relative_to(ROOT)
    except ValueError:
        shown = target
    print(f"\nwrote {shown}")


if __name__ == "__main__":
    main()