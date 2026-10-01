#!/usr/bin/env python
"""Per-image detection-damage diagnosis on an RTTS *training* split.

Question: does our dehazer's harm on real fog localize to identifiable input
statistics? For each image we measure input stats (luminance, contrast, dark
channel, saturation) and per-image AP@0.5 for hazy / ours / DCP, then correlate
the AP deltas with the stats. Decision-shaping only: this runs on rtts/train
(rtts/valid stays clean for selection, rtts/test untouched).

Durable + resumable by design (a /tmp-hosted copy died silently twice and lost
its rows): per-image rows flush to a JSONL checkpoint every N images, and the
final summary + rows land in results/. Rows are only in memory within a chunk.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "machine_learning"))
sys.path.insert(0, str(ROOT / "scripts" / "traditional"))

from benchmark_model import load_dehazer  # noqa: E402
from evaluate_dcp_detection import coco_metrics, load_ground_truth  # noqa: E402
from evaluate_model_detection import dehaze_tiled, dehaze_whole  # noqa: E402
from ip_dehaze.traditional.dcp import dehaze as dcp_dehaze  # noqa: E402


def input_stats(rgb: np.ndarray) -> dict:
    """Distribution-stable input descriptors of the hazy image (float 0..1)."""
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark = ndimage.minimum_filter(rgb.min(axis=2), size=15)
    channel_max = rgb.max(axis=2)
    saturation = np.divide(rgb.max(axis=2) - rgb.min(axis=2), channel_max,
                           out=np.zeros_like(channel_max), where=channel_max > 1e-6)
    return {
        "lum_mean": float(luminance.mean()),
        "lum_std": float(luminance.std()),
        "dark_mean": float(dark.mean()),
        "dark_p90": float(np.percentile(dark, 90)),
        "saturation_mean": float(saturation.mean()),
    }


def per_image_ap50(image: dict, annotations: list[dict], categories: list[dict],
                   detections: list[dict], iou_thresholds: list[float]) -> float:
    """AP@0.5 restricted to one image (pycocotools on a single-image dataset).

    An empty detection set is AP 0 (coco_metrics handles it since 862374f).
    """
    metrics = coco_metrics([image], annotations, categories, detections, iou_thresholds)
    return metrics["AP@0.5"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/traditional/dcp_rtts_detection.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/checkpoints/unet-gated-context-20260929/best.pt")
    parser.add_argument("--split", default="train", choices=["train", "valid"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference", default="whole", choices=["whole", "tiled"])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flush-every", type=int, default=2,
                        help="flush the JSONL row checkpoint every N batches")
    parser.add_argument("--run-id", default="probe-rtts-damage")
    args = parser.parse_args()

    from ultralytics import YOLO

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = yaml.safe_load(args.config.read_text())
    dataset_root = (ROOT / config["dataset"]["root"]).parent / args.split  # rtts/<split>
    images_dir, labels_dir = dataset_root / "images", dataset_root / "labels"
    all_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not all_paths:
        raise ValueError(f"no images under {images_dir}")
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(len(all_paths)), min(args.limit, len(all_paths))))
    image_paths = [all_paths[i] for i in chosen]

    detector = config["detector"]
    class_map = detector["classes"]
    category_ids = sorted({int(v) for v in class_map.values()})
    categories = [{"id": cid, "name": next(n for n, c in class_map.items() if int(c) == cid)}
                  for cid in category_ids]
    images, annotations, image_lookup = load_ground_truth(image_paths, labels_dir, class_map)
    annotations_by_image = {image["id"]: [] for image in images}
    for annotation in annotations:
        annotations_by_image[annotation["image_id"]].append(annotation)
    if sum(1 for v in annotations_by_image.values() if v) < 0.9 * len(images):
        raise AssertionError("expected >90% of sampled images to carry at least one GT box")

    run_dir = ROOT / "runs" / "machine_learning" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "rows.jsonl"
    summary_path = run_dir / "summary.json"
    # Resume: skip images whose rows are already flushed.
    done_names: set[str] = set()
    if rows_path.is_file():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done_names.add(json.loads(line)["name"])
        print(f"resuming: {len(done_names)} rows already flushed", flush=True)
    remaining = [p for p in image_paths if p.name not in done_names]

    model, model_args = load_dehazer(args.checkpoint, device)
    yolo = YOLO(str(detector["weights"]))
    yolo.eval()
    iou_thresholds = config["evaluation"]["iou_thresholds"]

    detections: dict[str, list] = {name: [] for name in ("hazy", "ours", "dcp")}
    n_gt_total = 0
    started = time.time()
    rows_handle = rows_path.open("a")
    for start in range(0, len(remaining), args.batch):
        chunk_paths = remaining[start:start + args.batch]
        prepared = {"hazy": [], "ours": [], "dcp": []}
        stats_by_id = {}
        for path in chunk_paths:
            with Image.open(path) as handle:
                rgb = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
            stats_by_id[image_lookup[path]] = input_stats(rgb)
            restored = (dehaze_whole(model, rgb, device) if args.inference == "whole"
                        else dehaze_tiled(model, rgb, device))
            cleaned = dcp_dehaze(rgb, patch_size=config["dcp"]["patch_size"], omega=config["dcp"]["omega"],
                                 top_fraction=config["dcp"]["atmospheric_top_fraction"],
                                 guided_radius=config["dcp"]["guided_radius"],
                                 guided_epsilon=config["dcp"]["guided_epsilon"],
                                 transmission_floor=config["dcp"]["transmission_floor"])
            for name, array in (("hazy", rgb), ("ours", restored), ("dcp", cleaned)):
                prepared[name].append(Image.fromarray((array * 255).round().astype(np.uint8)))

        chunk_dets = {name: [] for name in prepared}
        for name, collection in prepared.items():
            outputs = yolo.predict(source=collection, imgsz=detector["image_size"],
                                   conf=detector["confidence"], iou=detector["nms_iou"],
                                   max_det=detector["max_detections"], classes=category_ids,
                                   device=0 if device.type == "cuda" else "cpu",
                                   verbose=False, stream=False)
            for path, output in zip(chunk_paths, outputs):
                if output.boxes is None:
                    continue
                for box, score, class_id in zip(output.boxes.xyxy.cpu().numpy(),
                                                output.boxes.conf.cpu().numpy(),
                                                output.boxes.cls.cpu().numpy().astype(int)):
                    x1, y1, x2, y2 = map(float, box)
                    detection = {"image_id": image_lookup[path], "category_id": int(class_id),
                                 "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                                 "score": float(score)}
                    chunk_dets[name].append(detection)
                    detections[name].append(detection)

        for path in chunk_paths:
            image_id = image_lookup[path]
            image = images[image_id - 1]
            ground_truth = annotations_by_image[image_id]
            n_gt_total += len(ground_truth)
            row = {"name": path.name, "n_gt": len(ground_truth), **stats_by_id[image_id]}
            for name in ("hazy", "ours", "dcp"):
                own = [d for d in chunk_dets[name] if d["image_id"] == image_id]
                row[f"ap50_{name}"] = (float("nan") if not ground_truth
                                       else per_image_ap50(image, ground_truth, categories, own, iou_thresholds))
            rows_handle.write(json.dumps(row) + "\n")
        if (start // args.batch + 1) % args.flush_every == 0 or start + args.batch >= len(remaining):
            rows_handle.flush()
            import os
            os.fsync(rows_handle.fileno())
        print(f"[{time.time() - started:7.1f}s] {min(start + len(chunk_paths), len(remaining))}"
              f"/{len(remaining)} (total {len(done_names) + min(start + len(chunk_paths), len(remaining))}"
              f"/{len(image_paths)})", flush=True)

    rows_handle.close()

    # Reload all rows from the checkpoint so the summary covers every image ever
    # flushed, including rows written by previous (interrupted) invocations.
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    scored = [row for row in rows if not (isinstance(row.get("ap50_hazy"), float) and np.isnan(row["ap50_hazy"]))]
    for row in scored:
        for name in ("hazy", "ours", "dcp"):
            ap = row[f"ap50_{name}"]
            assert not np.isnan(ap) and 0.0 <= ap <= 1.0, f"AP out of range: {row}"
        row["delta_ours"] = row["ap50_ours"] - row["ap50_hazy"]
        row["delta_dcp"] = row["ap50_dcp"] - row["ap50_hazy"]

    def correlation(stat: str, delta: str) -> dict:
        a = np.array([row[stat] for row in scored])
        b = np.array([row[delta] for row in scored])
        if a.std() > 0 and b.std() > 0:
            pearson = float(np.corrcoef(a, b)[0, 1])
            spearman = float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
        else:
            pearson = spearman = float("nan")
        return {"pearson": pearson, "spearman": spearman}

    stats_names = ["lum_mean", "lum_std", "dark_mean", "dark_p90", "saturation_mean"]
    dataset_ap = {name: coco_metrics(images, annotations, categories, detections[name], iou_thresholds)
                  for name in detections}
    summary = {
        "n_images": len(scored),
        "n_gt_boxes": len(annotations),
        "dataset_ap_from_partial_detections": {k: round(v["AP@0.5"], 4) for k, v in dataset_ap.items()},
        "mean_delta_ours": float(np.mean([row["delta_ours"] for row in scored])),
        "mean_delta_dcp": float(np.mean([row["delta_dcp"] for row in scored])),
        "frac_worsened_ours": float(np.mean([row["delta_ours"] < 0 for row in scored])),
        "frac_worsened_dcp": float(np.mean([row["delta_dcp"] < 0 for row in scored])),
        "corr_delta_ours": {s: correlation(s, "delta_ours") for s in stats_names},
        "corr_delta_dcp": {s: correlation(s, "delta_dcp") for s in stats_names},
        "stat_quantiles": {s: {q: float(np.percentile([row[s] for row in scored], q))
                               for q in (10, 50, 90)} for s in stats_names},
        "elapsed_seconds": round(time.time() - started, 1),
        "checkpoint": str(args.checkpoint),
        "inference": args.inference,
        "device": str(device),
        "split": f"rtts/{args.split}",
        "sampling": {"limit": len(image_paths), "seed": args.seed, "available": len(all_paths)},
        "rows_path": str(rows_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    (ROOT / "results" / "machine_learning" / f"{args.run_id}.json").parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(summary_path, ROOT / "results" / "machine_learning" / f"{args.run_id}.json")
    print(json.dumps(summary, indent=2))
    print(f"rows: {rows_path}")
    print(f"summary: {ROOT / 'results' / 'machine_learning' / f'{args.run_id}.json'}")


if __name__ == "__main__":
    main()
