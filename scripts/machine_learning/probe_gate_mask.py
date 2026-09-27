#!/usr/bin/env python
"""Is the per-pixel gate actually per-pixel, or has it collapsed?

A spatial gate (`--gate-spatial`) predicts a mask over the image instead of one
scalar per image, so its failure mode is silent: if the mask saturates to 1 it is
just the plain restoration, and if it saturates to 0 it is the identity — in
both cases training still produces plausible loss curves and a PSNR that looks
like a result. This prints the mask statistics needed to tell the difference,
per density class, and returns a verdict.

The mask is captured with a forward hook on `spatial_head`, so no model change
is needed and any `--gate-spatial` checkpoint can be probed:

    uv run python scripts/machine_learning/probe_gate_mask.py \
        --checkpoint outputs/checkpoints/unet-gated-spatial-<date>/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "machine_learning"))

from ip_dehaze.data.dataset import BETA_CLASSES, CONTEXT_SIZE, FoggyPairs, _resize  # noqa: E402
from benchmark_model import load_dehazer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a per-pixel gate for mask collapse")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes-per-beta", type=int, default=10)
    parser.add_argument("--crop", type=int, default=512, help="input window used for the probe")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    model, model_args = load_dehazer(args.checkpoint, device)
    if getattr(model, "spatial_head", None) is None:
        raise SystemExit(
            "this checkpoint has no spatial gate (spatial_head is None) — "
            "probe a --gate-spatial run, or read the scalar gate's d_mean from its log"
        )

    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):  # hook on the 1x1 conv that emits mask logits
        captured["logits"] = output

    handle = model.spatial_head.register_forward_hook(capture)
    print(f"checkpoint: {args.checkpoint}")
    print(f"config: model={model_args.get('model')} gate_spatial={model_args.get('gate_spatial')} "
          f"gate_context={model_args.get('gate_context')}")
    print(f"\n{'class':>7} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} {'frac>0.9':>9} {'frac<0.1':>9}")

    pooled: list[np.ndarray] = []
    try:
        for name in BETA_CLASSES:
            dataset = FoggyPairs(
                ROOT / "data" / "foggy_cityscapes", ROOT / "data" / "cityscapes_clear", "val",
                crop=args.crop, beta_filter=name, seed=1729,
            )
            per_scene = []
            for index in range(min(args.scenes_per_beta, len(dataset))):
                sample = dataset[index]
                tensor = sample["hazy"][None].to(device)
                context = None
                if model.context_stem is not None:  # match training: whole downscaled scene
                    whole = _resize(sample["hazy"].permute(1, 2, 0).numpy(), CONTEXT_SIZE)
                    context = torch.from_numpy(
                        np.ascontiguousarray(whole.transpose(2, 0, 1))
                    )[None].to(device)
                with torch.no_grad():
                    model(tensor, context)
                mask = torch.sigmoid(captured["logits"])[0].cpu().numpy().ravel()
                per_scene.append(mask)
                pooled.append(mask)
            stacked = np.concatenate(per_scene)
            print(f"{name:>7} {stacked.mean():>7.3f} {stacked.std():>7.3f} {stacked.min():>7.3f} "
                  f"{stacked.max():>7.3f} {(stacked > 0.9).mean():>9.3f} {(stacked < 0.1).mean():>9.3f}")
    finally:
        handle.remove()

    everything = np.concatenate(pooled)
    mean, std = float(everything.mean()), float(everything.std())
    print(f"\npooled: mean {mean:.3f} std {std:.3f} | >0.9: {(everything > 0.9).mean():.3f} "
          f"| <0.1: {(everything < 0.1).mean():.3f}")
    if std < 0.01 and mean > 0.9:
        print("VERDICT: COLLAPSED to restoration (mask ≈ 1 everywhere) — the gate is doing nothing")
    elif std < 0.01 and mean < 0.05:
        print("VERDICT: COLLAPSED to identity (mask ≈ 0 everywhere) — nothing is being restored")
    elif std < 0.02:
        print("VERDICT: nearly uniform — spatial variation is negligible; report it as a scalar gate")
    else:
        print("VERDICT: genuinely spatial — mask varies within images; the PSNR number is meaningful")


if __name__ == "__main__":
    main()
