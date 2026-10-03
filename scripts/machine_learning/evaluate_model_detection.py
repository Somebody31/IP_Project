#!/usr/bin/env python
"""Frozen-detector AP on RTTS: hazy vs DCP vs our dehazer.

The paper's headline claim is detection mAP, not PSNR, so this is the metric on
which an adaptive dehazer can win even when PSNR only ties. RTTS is real fog
with no clear counterpart, so only detection AP is measurable here — it doubles
as the out-of-domain test (our models were trained on synthetic Cityscapes fog).

Inference must match the training scale. The restoration network was trained on
256x256 crops, so whole-image inference at 1920x1080 would shift the input
distribution the same way the crop/full-image mismatch broke the gate (see
docs/reports/val-comparison.md). Tiles are therefore 256 with 50% overlap,
averaged in the overlap; the *gate* still receives the whole downscaled scene,
exactly as in training and in the val benchmark.

The COCO AP machinery is imported from the traditional track's script so both
paths score identically — same wrapping, same thresholds, same report shape.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "machine_learning"))
sys.path.insert(0, str(ROOT / "scripts" / "traditional"))

from ip_dehaze.data.dataset import CONTEXT_SIZE, _resize  # noqa: E402
from ip_dehaze.traditional.dcp import dehaze as dcp_dehaze  # noqa: E402
from benchmark_model import classify, load_classifier, load_dehazer  # noqa: E402
# Shared COCO path: scores our dehazer exactly as the DCP baseline was scored.
from evaluate_dcp_detection import coco_metrics, load_ground_truth, sha256  # noqa: E402


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    """Window offsets covering [0, length), last window flush to the edge."""
    if length <= tile:
        return [0]
    offsets = list(range(0, length - tile + 1, stride))
    if offsets[-1] != length - tile:
        offsets.append(length - tile)
    return offsets


@torch.no_grad()
def dehaze_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile: int = 256,
    stride: int = 128,
    batch: int = 32,
) -> np.ndarray:
    """Restore a whole image with 256-tiled inference, averaged in overlaps."""
    height, width = image.shape[:2]
    context = None
    dcp_channel = None
    if getattr(model, "dcp_t", False):
        from ip_dehaze.traditional.dcp import (
            dark_channel,
            estimate_atmospheric_light,
            estimate_transmission,
        )

        # Compute once on the whole image (same sequence as dehaze()), crop per tile.
        dark = dark_channel(image, 15)
        t = estimate_transmission(image, estimate_atmospheric_light(image, dark))
        dcp_channel = t
    if model.context_stem is not None:
        whole = _resize(image, CONTEXT_SIZE)
        context = torch.from_numpy(np.ascontiguousarray(whole.transpose(2, 0, 1)))[None].to(device)

    output = np.zeros_like(image, dtype=np.float32)
    counts = np.zeros((height, width, 1), dtype=np.float32)
    windows = [(top, left) for top in tile_positions(height, tile, stride)
               for left in tile_positions(width, tile, stride)]
    for start in range(0, len(windows), batch):
        chunk = windows[start : start + batch]
        patches = []
        for top, left in chunk:
            patch = image[top : top + tile, left : left + tile]
            if patch.shape[0] != tile or patch.shape[1] != tile:  # image smaller than a tile
                padded = np.zeros((tile, tile, image.shape[2]), dtype=image.dtype)
                padded[: patch.shape[0], : patch.shape[1]] = patch
                patch = padded
            patches.append(np.ascontiguousarray(patch.transpose(2, 0, 1)))
        tensor = torch.from_numpy(np.stack(patches)).to(device)
        if dcp_channel is not None:
            channels = []
            for top, left in chunk:
                t_patch = dcp_channel[top : top + tile, left : left + tile]
                if t_patch.shape != (tile, tile):
                    padded = np.zeros((tile, tile), dtype=t_patch.dtype)
                    padded[: t_patch.shape[0], : t_patch.shape[1]] = t_patch
                    t_patch = padded
                channels.append(t_patch)
            tensor = torch.cat(
                [tensor, torch.from_numpy(np.ascontiguousarray(
                    np.stack(channels)[:, None, :, :])).to(device)], dim=1)
        ctx = context.repeat(len(chunk), 1, 1, 1) if context is not None else None
        restored, _ = model(tensor, ctx)
        restored = restored.clamp(0, 1).cpu().numpy()
        for offset, (top, left) in enumerate(chunk):
            patch = restored[offset].transpose(1, 2, 0)
            output[top : top + tile, left : left + tile] += patch
            counts[top : top + tile, left : left + tile] += 1.0
    return output / np.maximum(counts, 1.0)


@torch.no_grad()
def dehaze_whole(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Single forward pass at native resolution — the protocol the val benchmark used.

    The network is fully convolutional, so one pass is valid; padding to a
    multiple of 16 keeps the four downsamplings aligned. This is the faithful
    mode for small images (RTTS is 480x320 up to 896x504), where tiling would
    shrink the field of view below anything seen in training.
    """
    height, width = image.shape[:2]
    context = None
    if model.context_stem is not None:
        whole = _resize(image, CONTEXT_SIZE)
        context = torch.from_numpy(np.ascontiguousarray(whole.transpose(2, 0, 1)))[None].to(device)
    pad_h, pad_w = (-height) % 16, (-width) % 16
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)))[None].to(device)
    if getattr(model, "dcp_t", False):
        from ip_dehaze.traditional.dcp import (
            dark_channel,
            estimate_atmospheric_light,
            estimate_transmission,
        )

        dark = dark_channel(image, 15)
        t = estimate_transmission(image, estimate_atmospheric_light(image, dark))
        channel = torch.from_numpy(np.ascontiguousarray(t[None, None, :, :])).to(device)
        tensor = torch.cat([tensor, channel], dim=1)
    restored, _ = model(tensor, context)
    restored = restored.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return restored[:height, :width]


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen-detector AP on RTTS: hazy vs DCP vs ours")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/traditional/dcp_rtts_detection.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-split", default="",
                        help="override the config's RTTS split dir (e.g. valid); test stays frozen")
    parser.add_argument("--routed-classifier", type=Path,
                        help="classifier checkpoint; adds the paper-style routed column")
    parser.add_argument("--routed-branches", nargs=3, type=Path, metavar=("LIGHT", "MEDIUM", "HEAVY"),
                        help="light/medium/heavy branch checkpoints for the routed column")
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("model-rtts-%Y%m%d-%H%M%S"))
    parser.add_argument("--limit", type=int, default=0, help="smoke-test image limit")
    parser.add_argument("--save-detections", action="store_true",
                        help="also dump per-image detections for bootstrap CIs on the deltas")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--inference", default="whole", choices=["whole", "tiled"],
                        help="whole = single native-resolution pass (the val protocol); "
                             "tiled = 256 windows averaged in overlaps")
    parser.add_argument("--skip-dcp", action="store_true",
                        help="skip the DCP column (it costs ~10 min of CPU)")
    parser.add_argument("--skip-hazy", action="store_true", help="skip the untreated baseline column")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from importlib.metadata import version

    import ultralytics
    from ultralytics import YOLO

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset_root = (ROOT / config["dataset"]["root"]).resolve()
    dataset_split = config["dataset"]["split"]
    if args.dataset_split:
        dataset_root = (ROOT / "rtts" / args.dataset_split).resolve()
        dataset_split = args.dataset_split
    images_dir, labels_dir = dataset_root / "images", dataset_root / "labels"
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise ValueError(f"No images found under {images_dir}")

    detector = config["detector"]
    class_map = detector["classes"]
    category_ids = sorted({int(v) for v in class_map.values()})
    categories = [
        {"id": cid, "name": next(n for n, c in class_map.items() if int(c) == cid)} for cid in category_ids
    ]
    images, annotations, image_lookup = load_ground_truth(image_paths, labels_dir, class_map)

    model, model_args = load_dehazer(args.checkpoint, device)
    routed = None
    routing_counts: dict[int, int] = {}
    if args.routed_classifier is not None:
        if not args.routed_branches:
            raise SystemExit("--routed-classifier requires --routed-branches LIGHT MEDIUM HEAVY")
        routed = (
            load_classifier(args.routed_classifier, device),
            [load_dehazer(path, device)[0] for path in args.routed_branches],
        )
    detector_weights = str(detector["weights"])
    yolo = YOLO(detector_weights)
    yolo.eval()

    run_dir = ROOT / "runs" / "machine_learning" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())) + "\n"
    )

    collections = {
        "hazy": [] if not args.skip_hazy else None,
        "dcp": [] if not args.skip_dcp else None,
        "ours": [],
        "routed": [] if routed else None,
    }
    batch_size = int(detector["batch_size"])
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        prepared = {"hazy": [], "dcp": [], "ours": [], "routed": []}
        for path in batch_paths:
            with Image.open(path) as handle:
                rgb = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
            if collections["hazy"] is not None:
                prepared["hazy"].append(Image.fromarray((rgb * 255).round().astype(np.uint8)))
            if collections["dcp"] is not None:
                restored = dcp_dehaze(
                    rgb,
                    patch_size=config["dcp"]["patch_size"],
                    omega=config["dcp"]["omega"],
                    top_fraction=config["dcp"]["atmospheric_top_fraction"],
                    guided_radius=config["dcp"]["guided_radius"],
                    guided_epsilon=config["dcp"]["guided_epsilon"],
                    transmission_floor=config["dcp"]["transmission_floor"],
                )
                prepared["dcp"].append(Image.fromarray((restored * 255).round().astype(np.uint8)))
            ours = (
                dehaze_whole(model, rgb, device)
                if args.inference == "whole"
                else dehaze_tiled(model, rgb, device, tile=args.tile, stride=args.stride)
            )
            prepared["ours"].append(Image.fromarray((ours * 255).round().astype(np.uint8)))
            if routed:
                (classifier, classifier_size), branches = routed
                chosen = classify(classifier, rgb, device, size=classifier_size)
                routing_counts[chosen] = routing_counts.get(chosen, 0) + 1
                routed_output = dehaze_whole(branches[chosen], rgb, device)
                prepared["routed"].append(
                    Image.fromarray((routed_output * 255).round().astype(np.uint8)))

        for name, collection in collections.items():
            if collection is None:
                continue
            outputs = yolo.predict(
                source=prepared[name],
                imgsz=detector["image_size"],
                conf=detector["confidence"],
                iou=detector["nms_iou"],
                max_det=detector["max_detections"],
                device=0 if device.type == "cuda" else "cpu",
                classes=category_ids,
                verbose=False,
                stream=False,
            )
            for path, output in zip(batch_paths, outputs):
                if output.boxes is None:
                    continue
                boxes = output.boxes.xyxy.cpu().numpy()
                scores = output.boxes.conf.cpu().numpy()
                classes = output.boxes.cls.cpu().numpy().astype(int)
                for box, score, class_id in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = map(float, box)
                    collection.append({
                        "image_id": image_lookup[path],
                        "category_id": int(class_id),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    })
        print(f"evaluated {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}", flush=True)

    thresholds = config["evaluation"]["iou_thresholds"]
    metrics = {name: coco_metrics(images, annotations, categories, coll, thresholds)
               for name, coll in collections.items() if coll is not None}

    def delta(a: str, b: str) -> dict | None:
        if a not in metrics or b not in metrics:
            return None
        return {k: metrics[a][k] - metrics[b][k] for k in ("AP@0.5", "AP@0.5:0.95")}

    payload = {
        "status": "completed_exploratory_test_evaluation",
        "method": f"Dehazer {model_args.get('model', '?')} "
                  f"(gate_context={bool(model_args.get('gate_context'))}) + frozen YOLOv8n",
        "test_usage_note": (
            "Exploratory comparison on RTTS test; do not tune on this split. "
            "RTTS has no clear counterpart, so dehazing quality is only measurable through detection AP."
        ),
        "split": dataset_split,
        "n_images": len(image_paths),
        "n_ground_truth_boxes": len(annotations),
        "inference": {"mode": args.inference, "tile": args.tile, "stride": args.stride,
                      "context": list(CONTEXT_SIZE), "batched_tiles": batch_size},
        "checkpoint": str(args.checkpoint),
        "comparison": {
            **{name: value for name, value in metrics.items()},
            "delta_ours_minus_hazy": delta("ours", "hazy"),
            "delta_dcp_minus_hazy": delta("dcp", "hazy"),
            "delta_ours_minus_dcp": delta("ours", "dcp"),
            "delta_routed_minus_hazy": delta("routed", "hazy"),
            "delta_routed_minus_ours": delta("routed", "ours"),
        },
        "routed": None if not routed else {
            "classifier": str(args.routed_classifier),
            "branches": {"light": str(args.routed_branches[0]), "medium": str(args.routed_branches[1]),
                         "heavy": str(args.routed_branches[2])},
            "routing_counts": {str(k): v for k, v in sorted(routing_counts.items())},
        },
        "detector": {
            "weights": detector_weights,
            "sha256": sha256(Path(detector_weights)) if Path(detector_weights).is_file() else None,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "device": str(device),
            "frozen": True,
            "image_size": detector["image_size"],
            "confidence": detector["confidence"],
            "nms_iou": detector["nms_iou"],
            "max_detections": detector["max_detections"],
            "class_mapping_rtts_to_coco": class_map,
        },
        "evaluation": {"iou_thresholds": thresholds,
                       "pycocotools_version": version("pycocotools")},
        "provenance": {
            "git_revision": None,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "dataset_manifest_sha256": hashlib.sha256(
                "\n".join(f"{p.name},{p.stat().st_size}" for p in image_paths).encode()
            ).hexdigest(),
        },
    }
    result_path = ROOT / "results" / "machine_learning" / f"{args.run_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    if args.save_detections:
        # Per-image detections, so the AP deltas can be bootstrapped over
        # images by an analysis pass (point estimates alone cannot say whether
        # a -0.007 AP difference is real). Same schema as the COCO inputs.
        run_dir = ROOT / "runs/machine_learning" / args.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            "images": images,
            "categories": categories,
            "annotations": annotations,
            "detections": {name: coll for name, coll in collections.items() if coll is not None},
        }
        (run_dir / "detections.json").write_text(json.dumps(dump))
        print(f"detections: {run_dir.relative_to(ROOT)}/detections.json")
    for name, value in metrics.items():
        print(f"{name:>6}: AP@0.5 {value['AP@0.5']:.4f} | AP@0.5:0.95 {value['AP@0.5:0.95']:.4f} "
              f"| {value['n_detections']} detections")
    print(f"summary: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
