"""Pre-extract 256x256 crops into memmap caches (one-time, ~10 min).

Training straight from 2048x1024 PNGs is I/O bound: measured 0.73 s per
batch of 16 just to decode, versus 0.34 s for the GPU step. Caching crops
as uint8 memmaps removes decoding entirely from the training loop.

Usage: uv run python scripts/machine_learning/build_crop_cache.py \
           --split train --crops-per-pair 4
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ip_dehaze.data.dataset import _resize, beta_class, class_name, load_rgb  # noqa: E402
from ip_dehaze.traditional.pairing import index_pairs  # noqa: E402

CACHE_ROOT = ROOT / "data" / "derived" / "crops"


def crop_one(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    foggy_path, clear_path, size, seed, resize = args
    hazy = load_rgb(foggy_path)
    clear = load_rgb(clear_path)
    if resize is not None:
        # whole image, downscaled: global statistics for density classification
        return (
            (_resize(hazy, resize) * 255).round().astype(np.uint8),
            (_resize(clear, resize) * 255).round().astype(np.uint8),
        )
    height, width = hazy.shape[:2]
    rng = random.Random(seed)
    top = rng.randrange(height - size + 1)
    left = rng.randrange(width - size + 1)
    return (
        (hazy[top : top + size, left : left + size] * 255).round().astype(np.uint8),
        (clear[top : top + size, left : left + size] * 255).round().astype(np.uint8),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build memmap crop caches")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--crops-per-pair", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--resize", default="", help="WxH: store whole downscaled images instead of crops")
    parser.add_argument("--out-name", default="", help="override output subdirectory")
    parser.add_argument("--force", action="store_true", help="overwrite an existing cache")
    args = parser.parse_args()

    resize = None
    if args.resize:
        width, height = (int(value) for value in args.resize.lower().split("x"))
        resize = (width, height)
        args.crops_per_pair = 1  # one whole image per pair

    pairs = index_pairs(
        ROOT / "data/foggy_cityscapes", ROOT / "data/cityscapes_clear", args.split
    )
    # Never clobber a cache by accident: resize caches get their own
    # directory name, and an existing cache needs --force.
    out_dir = CACHE_ROOT / (args.out_name or (f"resize-{args.resize}" if args.resize else args.split))
    index_path = out_dir / "index.csv"
    if index_path.is_file() and not args.force:
        raise SystemExit(
            f"{out_dir.relative_to(ROOT)} already exists ({sum(1 for _ in index_path.open()) - 1} rows); "
            "pass --force to overwrite or --out-name to write elsewhere"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate any previous cache first: an interrupted build must never
    # look complete to a reader (index.csv alone is not proof of completion).
    for stale in ("index.csv", "manifest.json"):
        (out_dir / stale).unlink(missing_ok=True)
    total = len(pairs) * args.crops_per_pair
    shape = (total, resize[1], resize[0], 3) if resize else (total, args.size, args.size, 3)
    hazy_store = np.lib.format.open_memmap(out_dir / "hazy.npy", mode="w+", dtype=np.uint8, shape=shape)
    clear_store = np.lib.format.open_memmap(out_dir / "clear.npy", mode="w+", dtype=np.uint8, shape=shape)
    print(f"{len(pairs)} pairs x {args.crops_per_pair} crops = {total} crops, {shape}", flush=True)

    jobs = [
        (
            str(pair.foggy_path),
            str(pair.clear_path),
            args.size,
            args.seed * 1_000_003 + pair_index * 100 + crop_index,
            resize,
        )
        for pair_index, pair in enumerate(pairs)
        for crop_index in range(args.crops_per_pair)
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job_index, (hazy_crop, clear_crop) in enumerate(pool.map(crop_one, jobs, chunksize=16)):
            hazy_store[job_index] = hazy_crop
            clear_store[job_index] = clear_crop
            pair = pairs[job_index // args.crops_per_pair]
            rows.append(
                {
                    "row": job_index,
                    "split": pair.split,
                    "city": pair.city,
                    "image_id": pair.image_id,
                    "beta": pair.beta,
                    "density_class": beta_class(pair.beta),
                    "beta_name": class_name(beta_class(pair.beta)),
                }
            )
            if job_index % 2000 == 0:
                print(f"  {job_index}/{total}", flush=True)

    hazy_store.flush()
    clear_store.flush()
    (out_dir / "crops_per_pair.txt").write_text(f"{args.crops_per_pair}\n")
    with (out_dir / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import json

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "split": args.split,
                "crops_per_pair": args.crops_per_pair,
                "rows": total,
                "shape": list(shape),
                "resize": args.resize or None,
                "seed": args.seed,
                "complete": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"cache: {out_dir.relative_to(ROOT)} ({total} crops, shape {shape})")


if __name__ == "__main__":
    main()
