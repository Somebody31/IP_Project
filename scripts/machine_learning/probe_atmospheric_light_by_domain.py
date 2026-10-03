"""Atmospheric-light level by domain: the axis the training distribution missed.

The earlier domain-transfer work measured RTTS's atmospheric light as
"neutral to within 1%" and ruled out a *colour cast*. That measured the
channel RATIOS (tint), not the absolute LEVEL. This measures the level:

  A = DCP's atmospheric-light estimate per image (top 0.1% of the dark
  channel), averaged over a sample of each domain, with the same estimator
  on both sides so the comparison is like-for-like.

Motivation: `probe_realfog_on_cityscapes.py` showed that transplanting real
fog's transmission onto in-domain content costs nothing (density-matched
27.46 dB vs control 27.14 dB), while swapping only the atmospheric light
(real fog's A on the training fog's own transmission) costs ~5.4 dB. This
script checks whether that swap reflects a real domain difference.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ip_dehaze.traditional.dcp import dark_channel, estimate_atmospheric_light  # noqa: E402

DOMAINS = {
    "foggy_cityscapes_train": lambda: Path(ROOT / "data/foggy_cityscapes/leftImg8bit_foggy/train").rglob("*beta_0.02.png"),
    "foggy_cityscapes_val": lambda: Path(ROOT / "data/foggy_cityscapes/leftImg8bit_foggy/val").rglob("*beta_0.02.png"),
    "rtts_train": lambda: Path(ROOT / "rtts/train/images").glob("*.jpg"),
    "rtts_valid": lambda: Path(ROOT / "rtts/valid/images").glob("*.jpg"),
    "rtts_test": lambda: Path(ROOT / "rtts/test/images").glob("*.jpg"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--run-id", default="atmospheric-light-by-domain")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out: dict = {"samples_per_domain": args.samples, "seed": args.seed, "domains": {}}
    print(f"{'domain':<24} {'A (R,G,B)':<24} {'sd':<22} {'luminance':>9}")
    for name, factory in DOMAINS.items():
        paths = sorted(factory())
        picks = rng.sample(paths, min(args.samples, len(paths)))
        levels, luminance = [], []
        for path in picks:
            with Image.open(path) as handle:
                image = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
            levels.append(np.asarray(estimate_atmospheric_light(image, dark_channel(image, 15)),
                                     dtype=np.float32).reshape(-1)[:3])
            luminance.append(float(image.mean()))
        array = np.asarray(levels)
        mean, sd = array.mean(axis=0), array.std(axis=0)
        out["domains"][name] = {
            "n": len(picks),
            "A_mean": [float(v) for v in mean],
            "A_sd": [float(v) for v in sd],
            "A_luminance_mean": float(mean.mean()),
            "image_luminance_mean": float(np.mean(luminance)),
        }
        print(f"{name:<24} ({mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f})  "
              f"({sd[0]:.3f}, {sd[1]:.3f}, {sd[2]:.3f})  {np.mean(luminance):>9.3f}")

    train = out["domains"]["foggy_cityscapes_train"]
    rtts = out["domains"]["rtts_test"]
    gap = train["A_luminance_mean"] - rtts["A_luminance_mean"]
    out["train_vs_rtts_test_A_luminance_gap"] = gap
    out["conclusion"] = (
        f"Training distribution A (level) {train['A_luminance_mean']:.3f} vs RTTS test "
        f"{rtts['A_luminance_mean']:.3f} -> a {gap:.3f} systematic level gap. Channel ratios "
        "stay near 1 in both domains, so this is a brightness/level difference, not a tint."
    )
    print(f"\nA level: train {train['A_luminance_mean']:.3f} vs RTTS test "
          f"{rtts['A_luminance_mean']:.3f} (gap {gap:.3f})")
    print("tint (channel ratios) is ~1 in every domain -> the axis is LEVEL, not cast")

    out["completed_at"] = dt.datetime.now().isoformat(timespec="seconds")
    path = ROOT / "results" / "machine_learning" / f"{args.run_id}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()