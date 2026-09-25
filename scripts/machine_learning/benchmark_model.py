"""Benchmark a trained dehazer on the paired val split.

Scores model outputs with the same shared metric + clustered-bootstrap code
the traditional track uses, so ML and DCP numbers are directly comparable.

Modes:
  blind / gated : one checkpoint for every image
  routed        : paper-style system — a 3-class classifier picks which
                  per-class branch checkpoint processes each image
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ip_dehaze.data.dataset import BETA_CLASSES, FoggyPairs, _resize, beta_class, load_rgb  # noqa: E402
from ip_dehaze.evaluation.metrics import metrics as image_metrics  # noqa: E402
from ip_dehaze.evaluation.pair_benchmark import summarize  # noqa: E402
from ip_dehaze.evaluation.provenance import environment, git_revision, sha256  # noqa: E402
from ip_dehaze.machine_learning.models.classifier import DensityClassifier  # noqa: E402
from ip_dehaze.machine_learning.models.unet import build_unet  # noqa: E402

CLASS_ORDER = ("light", "medium", "heavy")


def load_dehazer(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    state = torch.load(path, map_location=device, weights_only=False)
    args = state.get("args", {})
    model = build_unet(args.get("size", "medium"), gated=args.get("model") == "gated").to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, args


def load_classifier(path: Path, device: torch.device) -> tuple[torch.nn.Module, tuple[int, int] | None]:
    """Returns the classifier plus its whole-image input size (None = crop voting).

    A checkpoint trained on whole downscaled images must be *routed* the same
    way it was trained; otherwise routing accuracy silently degrades.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model = DensityClassifier().to(device)
    model.load_state_dict(state["model"])
    model.eval()
    resize = (state.get("args") or {}).get("resize") or ""
    size = None
    if resize:
        width, height = (int(value) for value in resize.lower().split("x"))
        size = (width, height)
    return model, size


@torch.no_grad()
def classify(
    model: torch.nn.Module,
    hazy: np.ndarray,
    device: torch.device,
    size: tuple[int, int] | None = None,
    crops: int = 5,
) -> int:
    """Whole downscaled image if the classifier was trained that way, else crop voting."""
    if size is not None:
        small = _resize(hazy, size)
        tensor = torch.from_numpy(np.ascontiguousarray(small.transpose(2, 0, 1)))[None].to(device)
        return int(model(tensor).argmax(1))
    tensor = torch.from_numpy(np.ascontiguousarray(hazy.transpose(2, 0, 1)))[None].to(device)
    _, _, height, width = tensor.shape
    size = 256
    if height < size or width < size:
        logits = model(tensor)
        return int(logits.argmax(1))
    top, left = (height - size) // 2, (width - size) // 2
    positions = [(top, left), (0, 0), (0, width - size), (height - size, 0), (height - size, width - size)]
    votes: list[int] = []
    for index in range(max(1, crops)):
        y, x = positions[index % len(positions)]
        patch = tensor[:, :, y : y + size, x : x + size]
        votes.append(int(model(patch).argmax(1)))
    return max(set(votes), key=votes.count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a trained dehazer on paired val images")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["blind", "gated", "routed"], required=True)
    parser.add_argument("--checkpoint", type=Path, help="blind/gated checkpoint")
    parser.add_argument("--classifier", type=Path, help="routed mode: classifier checkpoint")
    parser.add_argument("--branch-light", type=Path)
    parser.add_argument("--branch-medium", type=Path)
    parser.add_argument("--branch-heavy", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--routing", default="classifier", choices=["classifier", "oracle"],
                        help="routed mode: use the learned classifier, or true beta as an upper bound")
    parser.add_argument("--workers", type=int, default=1,
                        help="threads for load/forward/metric; the serial path is ~5x slower")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(6)  # leave cores for any concurrent training job
    print(f"device: {device}", flush=True)
    pairs = FoggyPairs("data/foggy_cityscapes", "data/cityscapes_clear", args.split, crop=None).pairs
    if args.limit:
        pairs = pairs[: args.limit]

    if args.mode == "routed":
        for name in ("branch_light", "branch_medium", "branch_heavy"):
            if getattr(args, name) is None:
                raise SystemExit(f"--{name.replace('_', '-')} is required in routed mode")
        if args.routing == "classifier" and args.classifier is None:
            raise SystemExit("--classifier is required with --routing classifier")
        classifier, classifier_size = (
            load_classifier(args.classifier, device) if args.routing == "classifier" else (None, None)
        )
        branches = {
            name: load_dehazer(path, device)[0]
            for name, path in zip(
                CLASS_ORDER, (args.branch_light, args.branch_medium, args.branch_heavy)
            )
        }
        model = None
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required")
        model, _ = load_dehazer(args.checkpoint, device)
        classifier, branches = None, {}

    run_dir = ROOT / "runs" / "machine_learning" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "config.yaml"
    config_path.write_text("\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())) + "\n")
    started_at = dt.datetime.now().isoformat(timespec="seconds")

    gpu_lock = threading.Lock()
    progress = {"done": 0}
    progress_lock = threading.Lock()

    def score_pair(pair) -> dict:
        hazy = load_rgb(pair.foggy_path)
        clear = load_rgb(pair.clear_path)
        tensor = torch.from_numpy(np.ascontiguousarray(hazy.transpose(2, 0, 1)))[None].to(device)
        chosen = None
        with gpu_lock, torch.no_grad():
            if args.mode == "routed":
                chosen = (
                    CLASS_ORDER[beta_class(pair.beta)]
                    if args.routing == "oracle"
                    else CLASS_ORDER[classify(classifier, hazy, device, size=classifier_size)]
                )
                output, _ = branches[chosen](tensor)
            else:
                output, _ = model(tensor)
        output = output.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        hazy_scores = image_metrics(hazy, clear)
        model_scores = image_metrics(output, clear)
        with progress_lock:
            progress["done"] += 1
            if progress["done"] % 100 == 0 or progress["done"] == len(pairs):
                print(f"scored {progress['done']}/{len(pairs)}", flush=True)
        return {
            "split": pair.split,
            "city": pair.city,
            "image_id": pair.image_id,
            "scene_id": f"{pair.split}/{pair.city}/{pair.image_id}",
            "beta": pair.beta,
            "foggy_path": str(pair.foggy_path.relative_to(ROOT)),
            "clear_path": str(pair.clear_path.relative_to(ROOT)),
            "clear_reference_psnr": None,
            "clear_reference_ssim": 1.0,
            "hazy_psnr": hazy_scores["psnr"],
            "hazy_ssim": hazy_scores["ssim"],
            "dehazed_psnr": model_scores["psnr"],
            "dehazed_ssim": model_scores["ssim"],
            "delta_psnr": model_scores["psnr"] - hazy_scores["psnr"],
            "delta_ssim": model_scores["ssim"] - hazy_scores["ssim"],
            "routed_class": chosen or "",
        }

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(score_pair, pairs))
    else:
        rows = [score_pair(pair) for pair in pairs]
    rows.sort(key=lambda row: (row["scene_id"], row["beta"]))

    routing_counts = {name: 0 for name in CLASS_ORDER}
    for row in rows:
        if row["routed_class"]:
            routing_counts[row["routed_class"]] += 1

    with (run_dir / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, args.bootstrap_resamples, args.seed, condition="dehazed")
    payload = {
        "status": "completed",
        "command": " ".join(sys.argv),
        "method": {
            "blind": "Blind U-Net",
            "gated": "Density-gated U-Net",
            "routed": "Paper-style routed system (classifier + 3 branches)",
        }[args.mode],
        "split": args.split,
        "evaluation_scope": "full_split",
        "n_images": len(rows),
        "n_scenes": len({row["scene_id"] for row in rows}),
        "beta_counts": {
            beta: sum(1 for row in rows if f"{row['beta']:.3f}" == beta)
            for beta in sorted({f"{row['beta']:.3f}" for row in rows})
        },
        "comparison": ["clear_reference", "hazy", "dehazed"],
        "metrics": ["PSNR", "SSIM"],
        "summary": summary,
        "routing": args.routing if args.mode == "routed" else None,
        "routing_counts": routing_counts if args.mode == "routed" else None,
        "checkpoints": {
            "model": None if args.checkpoint is None else str(args.checkpoint),
            "classifier": None if args.classifier is None else str(args.classifier),
            "branches": (
                None
                if args.mode != "routed"
                else {
                    name: str(path)
                    for name, path in zip(
                        CLASS_ORDER, (args.branch_light, args.branch_medium, args.branch_heavy)
                    )
                }
            ),
        },
        "config_sha256": sha256(config_path),
        "dataset_manifest_sha256": sha256(ROOT / "data/foggy_cityscapes/foggy_trainval_refined_filenames.txt"),
        "source_revision": git_revision(ROOT),
        "environment": environment(),
        "started_at": started_at,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    result_path = ROOT / "results" / "machine_learning" / f"{args.run_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary["pooled"], indent=2))
    print(f"run: {run_dir.relative_to(ROOT)}")
    print(f"summary: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
