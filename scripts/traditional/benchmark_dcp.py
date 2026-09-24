from __future__ import annotations

import argparse
import csv
import datetime as dt
import json

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ip_dehaze.evaluation.metrics import metrics  # noqa: E402
from ip_dehaze.evaluation.pair_benchmark import summarize  # noqa: E402
from ip_dehaze.evaluation.provenance import environment, git_revision, sha256  # noqa: E402
from ip_dehaze.traditional.dcp import dehaze  # noqa: E402
from ip_dehaze.traditional.pairing import index_pairs  # noqa: E402


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def evaluate_pair(pair, method: dict) -> dict:
    foggy = load_rgb(pair.foggy_path)
    clear = load_rgb(pair.clear_path)
    if foggy.shape != clear.shape:
        raise ValueError(f"Image size mismatch: {pair.foggy_path} and {pair.clear_path}")
    output = dehaze(
        foggy,
        patch_size=method["patch_size"],
        omega=method["omega"],
        top_fraction=method["atmospheric_top_fraction"],
        guided_radius=method["guided_radius"],
        guided_epsilon=method["guided_epsilon"],
        transmission_floor=method["transmission_floor"],
    )
    hazy_scores = metrics(foggy, clear)
    dcp_scores = metrics(output, clear)
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
        "dcp_psnr": dcp_scores["psnr"],
        "dcp_ssim": dcp_scores["ssim"],
        "delta_psnr": dcp_scores["psnr"] - hazy_scores["psnr"],
        "delta_ssim": dcp_scores["ssim"] - hazy_scores["ssim"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired DCP vs untreated fog benchmark")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/traditional/dcp.yaml")
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("dcp-val-%Y%m%d-%H%M%S"))
    parser.add_argument("--foggy-root", type=Path)
    parser.add_argument("--clear-root", type=Path)
    parser.add_argument("--split")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset = config["dataset"]
    foggy_root = (args.foggy_root or ROOT / dataset["foggy_root"]).resolve()
    clear_root = (args.clear_root or ROOT / dataset["clear_root"]).resolve()
    split = args.split or dataset["split"]
    method = config["method"]
    manifest = dataset.get("refined_manifest") if dataset.get("refined_only", False) else None
    manifest_path = (ROOT / manifest).resolve() if manifest else None
    workers = int(config.get("workers", 1))
    if workers < 1:
        raise ValueError("workers must be at least one")
    bootstrap = config["bootstrap"]
    pairs = index_pairs(foggy_root, clear_root, split, manifest_path)

    run_dir = ROOT / "runs" / "traditional" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started_at = dt.datetime.now().isoformat(timespec="seconds")
    (run_dir / "config.yaml").write_text(config_path.read_text())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(lambda pair: evaluate_pair(pair, method), pairs))
    print(f"processed {len(rows)}/{len(pairs)}", flush=True)

    fields = list(rows[0])
    with (run_dir / "per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows, bootstrap["resamples"], bootstrap["seed"])
    payload = {
        "status": "completed",
        "command": " ".join(sys.argv),
        "method": "Dark Channel Prior",
        "split": split,
        "evaluation_scope": "refined_manifest_subset" if manifest_path else "full_split",
        "manifest": ({"path": str(manifest_path.relative_to(ROOT)), "sha256": sha256(manifest_path)}
                     if manifest_path else None),
        "n_images": len(rows),
        "n_scenes": len({row["scene_id"] for row in rows}),
        "beta_counts": {beta: sum(1 for row in rows if f"{row['beta']:.3f}" == beta)
                        for beta in sorted({f"{row['beta']:.3f}" for row in rows})},
        "comparison": ["clear_reference", "hazy", "dcp"],
        "metrics": ["PSNR", "SSIM"],
        "summary": summary,
        "parameters": method,
        "dataset_roots": {"foggy": str(foggy_root), "clear": str(clear_root)},
        "config_sha256": sha256(config_path),
        "source_revision": git_revision(ROOT),
        "environment": environment(),
        "started_at": started_at,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    result_path = ROOT / "results" / "traditional" / f"{args.run_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"run: {run_dir.relative_to(ROOT)}")
    print(f"summary: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
