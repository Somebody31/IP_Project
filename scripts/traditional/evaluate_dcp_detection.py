from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ip_dehaze.traditional.dcp import dehaze
from ip_dehaze.evaluation.rtts_labels import CLASS_NAMES, yolo_label_to_coco


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_ground_truth(image_paths: list[Path], labels_dir: Path, class_map: dict[str, int]):
    images = []
    annotations = []
    image_lookup = {}
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, 1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing RTTS label file: {label_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        image_lookup[image_path] = image_id
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                annotation = yolo_label_to_coco(line.split(), width, height, class_map)
            except (ValueError, KeyError) as error:
                raise ValueError(f"Malformed YOLO label {label_path}:{line_number}: {error}") from error
            annotations.append({"id": annotation_id, "image_id": image_id, **annotation})
            annotation_id += 1
    return images, annotations, image_lookup


def coco_metrics(images, annotations, categories, predictions, iou_thresholds):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        # pycocotools' loadRes indexes anns[0] and raises IndexError on an empty
        # set; no detections at all is a legitimate all-zero result (AP 0), and
        # it happens in per-image scoring and in heavily restored conditions.
        return {"AP@0.5:0.95": 0.0, "AP@0.5": 0.0, "n_detections": 0}

    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {"description": "RTTS test subset", "version": "1"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ground_truth.createIndex()
    detections = ground_truth.loadRes(predictions)
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = [image["id"] for image in images]
    evaluator.params.catIds = [category["id"] for category in categories]
    evaluator.params.iouThrs = np.asarray(iou_thresholds, dtype=np.float64)
    evaluator.params.maxDets = [1, 10, int(evaluator.params.maxDets[-1])]
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    all_area_index = evaluator.params.areaRngLbl.index("all")
    max_det_index = evaluator.params.maxDets.index(100)
    precision_all = precision[:, :, :, all_area_index, max_det_index]
    valid = precision_all[precision_all > -1]
    ap5095 = float(valid.mean()) if valid.size else 0.0

    def ap_at_iou(iou_target: float) -> float:
        idx = int(np.argmin(np.abs(evaluator.params.iouThrs - iou_target)))
        sub = precision_all[idx]
        val = sub[sub > -1]
        return float(val.mean()) if val.size else 0.0

    ap50 = ap_at_iou(0.50)
    ap75 = ap_at_iou(0.75)
    ap95 = ap_at_iou(0.95)
    return {
        "AP@0.5:0.95": ap5095,
        "AP@0.5": ap50,
        "AP@0.75": ap75,
        "AP@0.95": ap95,
        "n_detections": len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen-detector AP on RTTS with/without DCP")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/traditional/dcp_rtts_detection.yaml")
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("dcp-rtts-test-%Y%m%d-%H%M%S"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="optional smoke-test image limit")
    args = parser.parse_args()

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise SystemExit("Install detection dependencies with `uv sync --extra detection`") from error

    from importlib.metadata import version

    pycocotools_version = version("pycocotools")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset_root = (ROOT / config["dataset"]["root"]).resolve()
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise ValueError(f"No images found under {images_dir}")

    detector = config["detector"]
    class_map = detector["classes"]
    category_ids = sorted(set(int(value) for value in class_map.values()))
    categories = [{"id": category_id, "name": next(name for name, cid in class_map.items() if int(cid) == category_id)}
                  for category_id in category_ids]
    images, annotations, image_lookup = load_ground_truth(image_paths, labels_dir, class_map)
    device = detector["device"]
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    weights_arg = args.weights or Path(detector["weights"])
    weights_path = str(weights_arg if weights_arg.is_absolute() else weights_arg)
    model = YOLO(weights_path)
    model.eval()
    model_names = model.names
    available_ids = set(model_names.keys())
    if not set(category_ids).issubset(available_ids):
        raise ValueError(f"Requested COCO classes {category_ids} missing from detector class IDs {sorted(available_ids)}")

    run_dir = ROOT / "runs" / "traditional" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    (run_dir / "config.yaml").write_text(config_path.read_text())
    normal_predictions = []
    dcp_predictions = []
    batch_size = int(detector["batch_size"])
    dcp_config = config["dcp"]
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        normal_inputs = []
        dcp_inputs = []
        for path in batch_paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            normal_inputs.append(Image.fromarray(rgb))
            restored = dehaze(
                rgb.astype(np.float32) / 255.0,
                patch_size=dcp_config["patch_size"],
                omega=dcp_config["omega"],
                top_fraction=dcp_config["atmospheric_top_fraction"],
                guided_radius=dcp_config["guided_radius"],
                guided_epsilon=dcp_config["guided_epsilon"],
                transmission_floor=dcp_config["transmission_floor"],
            )
            dcp_inputs.append(Image.fromarray(np.round(restored * 255).astype(np.uint8)))

        for inputs, collection in ((normal_inputs, normal_predictions), (dcp_inputs, dcp_predictions)):
            outputs = model.predict(
                source=inputs,
                imgsz=detector["image_size"],
                conf=detector["confidence"],
                iou=detector["nms_iou"],
                max_det=detector["max_detections"],
                device=device,
                classes=category_ids,
                verbose=False,
                stream=False,
            )
            for path, output in zip(batch_paths, outputs):
                image_id = image_lookup[path]
                if output.boxes is None:
                    continue
                boxes = output.boxes.xyxy.cpu().numpy()
                scores = output.boxes.conf.cpu().numpy()
                class_ids = output.boxes.cls.cpu().numpy().astype(int)
                for box, score, class_id in zip(boxes, scores, class_ids):
                    x1, y1, x2, y2 = map(float, box)
                    collection.append({
                        "image_id": image_id,
                        "category_id": int(class_id),
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    })
        print(f"evaluated {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}", flush=True)

    metrics_config = config["evaluation"]
    normal_metrics = coco_metrics(images, annotations, categories, normal_predictions,
                                  metrics_config["iou_thresholds"])
    dcp_metrics = coco_metrics(images, annotations, categories, dcp_predictions,
                               metrics_config["iou_thresholds"])
    model_path = Path(getattr(model, "ckpt_path", weights_path)).expanduser()
    model_hash = sha256(model_path) if model_path.is_file() else None
    manifest_content = "\n".join(f"{p.name},{p.stat().st_size}" for p in image_paths)
    manifest_hash = hashlib.sha256(manifest_content.encode()).hexdigest()
    payload = {
        "status": "completed_exploratory_test_evaluation",
        "method": "Frozen COCO-pretrained YOLOv8n",
        "test_usage_note": "Exploratory comparison; do not tune DCP or detector settings on this test set. Freeze the protocol before any confirmatory test claim.",
        "split": config["dataset"]["split"],
        "n_images": len(image_paths),
        "n_ground_truth_boxes": len(annotations),
        "comparison": {
            "normal": normal_metrics,
            "dcp": dcp_metrics,
            "delta_dcp_minus_normal": {
                key: dcp_metrics[key] - normal_metrics[key]
                for key in ("AP@0.5", "AP@0.5:0.95")
            },
        },
        "detector": {
            "weights": str(model_path),
            "sha256": model_hash,
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
        "evaluation": {
            "iou_thresholds": metrics_config["iou_thresholds"],
            "max_detections_per_image": metrics_config["max_detections_per_image"],
            "pycocotools_version": pycocotools_version,
        },
        "dcp_parameters": dcp_config,
        "dataset_manifest_sha256": manifest_hash,
        "config_sha256": sha256(config_path),
        "source_revision": git_revision(),
        "command": " ".join(sys.argv),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "started_at": started_at,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (run_dir / "predictions.json").write_text(json.dumps({"normal": normal_predictions, "dcp": dcp_predictions},
                                                          separators=(",", ":")) + "\n")
    result_path = ROOT / "results" / "traditional" / f"{args.run_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["comparison"], indent=2))
    print(f"run: {run_dir.relative_to(ROOT)}")
    print(f"summary: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
