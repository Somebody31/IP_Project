"""Pre-training sanity gates (docs/plan.md P3.2).

Gate 1: loader visual check — writes hazy/clear pairs for eyeballing.
Gate 2: overfit smoke — a U-Net must memorise 32 pairs (>30 dB) or the
        model/loss/optimiser is broken.
Gate 3: copy-baseline — the identity model must reproduce the hazy PSNR
        recorded by the traditional track, proving one shared eval path.

Run: uv run python scripts/machine_learning/sanity_gates.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ip_dehaze.data.dataset import FoggyPairs  # noqa: E402
from ip_dehaze.evaluation.metrics import metrics as image_metrics  # noqa: E402
from ip_dehaze.machine_learning.losses import reconstruction_loss  # noqa: E402
from ip_dehaze.machine_learning.models.unet import build_unet  # noqa: E402


def gate_loader_visual(seed: int) -> None:
    out_dir = ROOT / "outputs" / "figures" / "loader_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_set = FoggyPairs("data/foggy_cityscapes", "data/cityscapes_clear", "train", crop=256, seed=seed)
    for index in (0, 1, 2, 3):
        sample = train_set[index]
        hazy = (sample["hazy"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        clear = (sample["clear"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        panel = np.concatenate([hazy, clear], axis=1)
        name = sample["scene_id"].split("/")[-1]
        Image.fromarray(panel).save(out_dir / f"{index}_{name}_beta{sample['beta']:.3f}.png")
    print(f"[gate 1] wrote 4 pair panels to {out_dir.relative_to(ROOT)}")
    print(f"[gate 1] sample scene ids: {[train_set[i]['scene_id'] for i in range(4)]}")


def gate_overfit(seed: int, steps: int, threshold_db: float, cache_dir: Path | None = None) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cache_dir and (cache_dir / "hazy.npy").is_file():
        from ip_dehaze.data.dataset import CachedCrops

        source = CachedCrops(cache_dir, split="train", seed=seed)
        print(f"[gate 2] using crop cache {cache_dir.relative_to(ROOT)}", flush=True)
    else:
        source = FoggyPairs("data/foggy_cityscapes", "data/cityscapes_clear", "train", crop=256, seed=seed)
    subset = Subset(source, list(range(32)))
    loader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    model = build_unet("medium", gated=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    step = 0
    while step < steps:
        for batch in loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output, _ = model(hazy)
            loss = reconstruction_loss(output, clear)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                print(f"[gate 2] step {step} loss {loss.item():.4f}", flush=True)
            if step >= steps:
                break

    model.eval()
    scores = []
    with torch.no_grad():
        for index in range(32):
            sample = subset[index]
            hazy = sample["hazy"][None].to(device)
            clear = sample["clear"][None].to(device)
            output, _ = model(hazy)
            scores.append(
                image_metrics(
                    output[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(),
                    clear[0].permute(1, 2, 0).cpu().numpy(),
                )["psnr"]
            )
    mean_psnr = float(np.mean(scores))
    passed = mean_psnr > threshold_db
    print(f"[gate 2] overfit PSNR {mean_psnr:.2f} dB (threshold {threshold_db} dB) -> {'PASS' if passed else 'FAIL'}")
    return {"psnr": mean_psnr, "passed": passed}


def gate_copy_baseline() -> dict:
    """Identity model must reproduce the traditional track's hazy PSNR exactly."""
    reference = ROOT / "runs" / "traditional" / "dcp-refined-val-20260923" / "per_image.csv"
    if not reference.is_file():
        print("[gate 3] SKIP: no reference per_image.csv")
        return {"passed": None}
    rows = list(csv.DictReader(reference.open()))[:5]
    worst = 0.0
    for row in rows:
        hazy = np.asarray(Image.open(ROOT / row["foggy_path"]).convert("RGB"), dtype=np.float32) / 255.0
        clear = np.asarray(Image.open(ROOT / row["clear_path"]).convert("RGB"), dtype=np.float32) / 255.0
        recomputed = image_metrics(hazy, clear)["psnr"]
        worst = max(worst, abs(recomputed - float(row["hazy_psnr"])))
    passed = worst < 1e-9
    print(f"[gate 3] identity-vs-recorded hazy PSNR max |diff| {worst:.2e} -> {'PASS' if passed else 'FAIL'}")
    return {"max_diff": worst, "passed": passed}


def gate_cache_integrity(cache_dir: Path) -> dict:
    """Cached crops must be foggy/clear twins: fog raises brightness (ASM)."""
    if not (cache_dir / "hazy.npy").is_file():
        print("[gate 4] SKIP: no crop cache yet")
        return {"passed": None}
    from ip_dehaze.data.dataset import CachedCrops

    dataset = CachedCrops(cache_dir, split="train")
    hazy = np.asarray(dataset.hazy[:32], dtype=np.float32).mean(axis=(1, 2, 3))
    clear = np.asarray(dataset.clear[:32], dtype=np.float32).mean(axis=(1, 2, 3))
    brighter = float((hazy > clear).mean())
    passed = brighter > 0.9
    print(
        f"[gate 4] {len(dataset)} pairs in cache; hazy brighter than clear in "
        f"{brighter:.0%} of sampled crops -> {'PASS' if passed else 'FAIL'}"
    )
    return {"fraction_hazy_brighter": brighter, "passed": passed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-training sanity gates")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--steps", type=int, default=800,
                        help="measured: 32 crops reach >30 dB around step 800; 300 steps only reaches ~25 dB")
    parser.add_argument("--threshold-db", type=float, default=30.0)
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/derived/crops/train")
    args = parser.parse_args()

    gate_loader_visual(args.seed)
    overfit = (
        {"passed": True}
        if args.skip_overfit
        else gate_overfit(args.seed, args.steps, args.threshold_db, args.cache_dir)
    )
    copy = gate_copy_baseline()
    cache = gate_cache_integrity(args.cache_dir)
    if not overfit["passed"] or copy["passed"] is False or cache["passed"] is False:
        raise SystemExit("sanity gates failed; do not start full training")


if __name__ == "__main__":
    main()
