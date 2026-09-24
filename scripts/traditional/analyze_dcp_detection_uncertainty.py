from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ip_dehaze.evaluation.coco_bootstrap import image_records, paired_image_bootstrap
from ip_dehaze.evaluation.rtts_labels import CLASS_NAMES, yolo_label_to_coco


def build_ground_truth(image_paths: list[Path], labels_dir: Path, class_map: dict[str, int]):
    images = []
    annotations = []
    image_ids = {}
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths, 1):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        with Image.open(image_path) as image:
            width, height = image.size
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        image_ids[image_path.name] = image_id
        for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
            if line.strip():
                annotation = yolo_label_to_coco(line.split(), width, height, class_map)
                annotations.append({"id": annotation_id, "image_id": image_id, **annotation})
                annotation_id += 1
    return images, annotations, image_ids


def evaluate(images, annotations, categories, predictions, iou_thresholds):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {"description": "RTTS test", "version": "1"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ground_truth.createIndex()
    detections = ground_truth.loadRes(predictions)
    result = COCOeval(ground_truth, detections, "bbox")
    result.params.imgIds = [image["id"] for image in images]
    result.params.catIds = [category["id"] for category in categories]
    result.params.iouThrs = np.asarray(iou_thresholds, dtype=np.float64)
    result.params.maxDets = [1, 10, 100]
    result.evaluate()
    result.accumulate()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-bootstrap and per-class analysis of saved DCP detections")
    parser.add_argument("--run-id", default="dcp-rtts-test-20260923")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.resamples < 1:
        raise ValueError("resamples must be positive")

    run_dir = ROOT / "runs" / "traditional" / args.run_id
    summary = json.loads((run_dir / "summary.json").read_text())
    predictions = json.loads((run_dir / "predictions.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    dataset_root = ROOT / config["dataset"]["root"]
    image_paths = sorted(path for path in (dataset_root / "images").iterdir()
                         if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    images, annotations, image_id_map = build_ground_truth(
        image_paths, dataset_root / "labels", config["detector"]["classes"]
    )
    if len(images) != summary["n_images"]:
        raise ValueError("Reconstructed test image count does not match the detector run")
    category_ids = sorted(set(int(value) for value in config["detector"]["classes"].values()))
    categories = [{"id": category_id,
                   "name": next(name for name, value in config["detector"]["classes"].items()
                                if int(value) == category_id)} for category_id in category_ids]
    expected_ids = set(range(1, len(images) + 1))
    for condition in ("normal", "dcp"):
        prediction_ids = {record["image_id"] for record in predictions[condition]}
        if not prediction_ids.issubset(expected_ids):
            raise ValueError(f"Saved {condition} predictions contain unknown image IDs")

    iou_thresholds = config["evaluation"]["iou_thresholds"]
    normal_eval = evaluate(images, annotations, categories, predictions["normal"], iou_thresholds)
    dcp_eval = evaluate(images, annotations, categories, predictions["dcp"], iou_thresholds)
    normal_records = image_records(normal_eval, list(expected_ids), category_ids)
    dcp_records = image_records(dcp_eval, list(expected_ids), category_ids)
    bootstrap = paired_image_bootstrap(
        normal_records,
        dcp_records,
        list(expected_ids),
        category_ids,
        iou_count=len(iou_thresholds),
        resamples=args.resamples,
        seed=args.seed,
        max_detections=config["evaluation"]["max_detections_per_image"],
    )

    def per_class_ap(evaluator):
        precision = evaluator.eval["precision"]
        all_index = evaluator.params.areaRngLbl.index("all")
        max_index = evaluator.params.maxDets.index(100)
        result = {}
        for index, category_id in enumerate(evaluator.params.catIds):
            current = precision[:, :, index, all_index, max_index]
            valid = current[current > -1]
            at50 = current[0]
            valid50 = at50[at50 > -1]
            result[str(category_id)] = {
                "name": next(category["name"] for category in categories if category["id"] == category_id),
                "AP@0.5": float(valid50.mean()) if valid50.size else 0.0,
                "AP@0.5:0.95": float(valid.mean()) if valid.size else 0.0,
            }
        return result

    normal_per_class = per_class_ap(normal_eval)
    dcp_per_class = per_class_ap(dcp_eval)
    per_class = {}
    for category_id in map(str, category_ids):
        per_class[category_id] = {
            "name": normal_per_class[category_id]["name"],
            "normal": normal_per_class[category_id],
            "dcp": dcp_per_class[category_id],
            "delta_dcp_minus_normal": {
                metric: dcp_per_class[category_id][metric] - normal_per_class[category_id][metric]
                for metric in ("AP@0.5", "AP@0.5:0.95")
            },
            "paired_bootstrap": bootstrap["per_class_paired_image_bootstrap"][category_id],
        }

    payload = {
        "status": "completed_exploratory_test_analysis",
        "source_run": args.run_id,
        "n_images": len(images),
        "n_ground_truth_boxes": len(annotations),
        "bootstrap_unit": "RTTS test image; normal and DCP conditions sampled together",
        "bootstrap_resamples": args.resamples,
        "bootstrap_seed": args.seed,
        "aggregate": {
            "normal": summary["comparison"]["normal"],
            "dcp": summary["comparison"]["dcp"],
            "delta_dcp_minus_normal": summary["comparison"]["delta_dcp_minus_normal"],
            "paired_bootstrap": bootstrap["paired_image_bootstrap"],
        },
        "per_class": per_class,
        "test_usage_note": summary["test_usage_note"],
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_path = ROOT / "results" / "traditional" / f"{args.run_id}-uncertainty.json"
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (run_dir / "uncertainty.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"aggregate": payload["aggregate"], "per_class": per_class}, indent=2))
    print(f"analysis: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
