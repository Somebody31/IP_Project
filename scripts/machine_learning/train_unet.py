"""Train a dehazing U-Net.

One script covers all three learned-dehazer runs:
  --model blind                plain U-Net (control column + loop validation)
  --model gated                density-gated U-Net (self-supervised gate)
  --beta-class light|medium|heavy   paper-style routed branch (subset only)

Writes a run record under runs/machine_learning/<run-id>/ and checkpoints
under outputs/checkpoints/<run-id>/ (kept out of source control).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ip_dehaze.data.dataset import (  # noqa: E402
    BETA_CLASSES,
    CONTEXT_SIZE,
    CachedCrops,
    CachedCropsWithContext,
    CachedCropsWithDcpT,
    FoggyPairs,
    _resize,
)
from ip_dehaze.evaluation.metrics import metrics as image_metrics  # noqa: E402
from ip_dehaze.evaluation.provenance import environment, git_revision, sha256  # noqa: E402
from ip_dehaze.machine_learning.losses import reconstruction_loss  # noqa: E402
from ip_dehaze.machine_learning.models.unet import build_unet  # noqa: E402


def build_val_set(val_scenes_per_beta: int, seed: int) -> list[FoggyPairs]:
    """Full-resolution validation pairs, balanced across the three β levels."""
    sets = []
    for name in BETA_CLASSES:
        probe = FoggyPairs("data/foggy_cityscapes", "data/cityscapes_clear", "val", crop=None, beta_filter=name)
        stride = max(1, len(probe) // val_scenes_per_beta)
        sets.append(
            FoggyPairs(
                "data/foggy_cityscapes",
                "data/cityscapes_clear",
                "val",
                crop=None,
                beta_filter=name,
                stride=stride,
                seed=seed,
            )
        )
    return sets


def _prepare_input(model: torch.nn.Module, hazy: torch.Tensor,
                   dcp_t: torch.Tensor | None) -> torch.Tensor:
    """Concatenate the DCP transmission channel when the model was built with it."""
    if getattr(model, "dcp_t", False):
        if dcp_t is None:
            raise ValueError("model expects a DCP transmission channel; pass dcp_t=")
        return torch.cat([hazy, dcp_t], dim=1)
    return hazy


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_sets: list[FoggyPairs], device: torch.device) -> dict:
    model.eval()
    per_beta: dict[str, list[dict[str, float]]] = defaultdict(list)
    densities: list[float] = []
    for val_set in val_sets:
        name = ("light", "medium", "heavy")[int(val_set[0]["density_class"])]
        for index in range(len(val_set)):
            sample = val_set[index]
            dcp_t = None
            if getattr(model, "dcp_t", False):
                # FoggyPairs (full-res val) carries no dcp_t; compute it from
                # the un-cropped hazy image with the same call sequence as
                # dehaze() — identical to the eval-time protocol.
                from ip_dehaze.traditional.dcp import (
                    dark_channel,
                    estimate_atmospheric_light,
                    estimate_transmission,
                )

                hazy_np = sample["hazy"].permute(1, 2, 0).numpy()
                dark = dark_channel(hazy_np, 15)
                transmission = estimate_transmission(hazy_np, estimate_atmospheric_light(hazy_np, dark))
                dcp_t = torch.from_numpy(np.ascontiguousarray(transmission[None, None, :, :])).to(device)
            hazy = _prepare_input(model, sample["hazy"][None].to(device), dcp_t)
            clear = sample["clear"][None].to(device)
            context = None
            if model.context_stem is not None:
                whole = _resize(sample["hazy"].permute(1, 2, 0).numpy(), CONTEXT_SIZE)
                context = torch.from_numpy(np.ascontiguousarray(whole.transpose(2, 0, 1)))[None].to(device)
            output, density = model(hazy, context)
            output = output.clamp(0, 1)
            scores = image_metrics(
                output[0].permute(1, 2, 0).cpu().numpy(), clear[0].permute(1, 2, 0).cpu().numpy()
            )
            per_beta[name].append(scores)
            if density is not None:
                densities.append(float(density.mean()))
    pooled = [row for rows in per_beta.values() for row in rows]
    result = {
        "psnr": float(np.mean([row["psnr"] for row in pooled])),
        "ssim": float(np.mean([row["ssim"] for row in pooled])),
        "n_images": len(pooled),
        "by_beta": {
            name: {
                "psnr": float(np.mean([row["psnr"] for row in rows])),
                "ssim": float(np.mean([row["ssim"] for row in rows])),
                "n_images": len(rows),
            }
            for name, rows in sorted(per_beta.items())
        },
    }
    if densities:
        result["density"] = {
            "mean": float(np.mean(densities)),
            "min": float(np.min(densities)),
            "max": float(np.max(densities)),
        }
    model.train()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dehazing U-Net (blind, gated, or routed branch)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", choices=["blind", "gated"], default="blind")
    parser.add_argument("--frozen-gate", action="store_true",
                        help="gated architecture with the gate forced to d=1 (the d==1 ablation)")
    parser.add_argument("--gate-init", type=float, default=None,
                        help="gate starting value in (0,1); d~1 keeps the blend a no-op at init")
    parser.add_argument("--gate-spatial", action="store_true",
                        help="per-pixel gate instead of one scalar per image")
    parser.add_argument("--gate-context", action="store_true",
                        help="gate reads a whole downscaled scene (needs --context-dir)")
    parser.add_argument("--dcp-t", action="store_true",
                        help="4th input channel: DCP's per-crop transmission estimate "
                             "(needs a cache built with build_crop_cache.py --dcp-t); "
                             "the model receives cat([hazy, dcp_t]) and in_ch widens")
    parser.add_argument("--context-dir", type=Path, default=None,
                        help="whole-image resize cache aligned with --cache-dir")
    parser.add_argument("--beta-class", choices=list(BETA_CLASSES), help="restrict to one density class (branch)")
    parser.add_argument("--size", default="medium", help="U-Net capacity preset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--val-scenes-per-beta", type=int, default=20)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=0, help="smoke-test: cap training pairs")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/derived/crops/train",
                        help="memmap crop cache (removes PNG-decode I/O)")
    parser.add_argument("--amp", action="store_true",
                        help="enable fp16 autocast — measured 4x slower and NaN-prone on this GPU; off by default")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        torch.set_num_threads(4)  # leave cores for any concurrent training job

    if args.gate_spatial and args.gate_context:
        raise SystemExit("--gate-spatial and --gate-context are mutually exclusive")
    if args.gate_context and args.context_dir is None:
        raise SystemExit("--gate-context requires --context-dir (a whole-image resize cache)")
    if args.cache_dir and Path(args.cache_dir).is_dir():
        if args.gate_context:
            train_set = CachedCropsWithContext(
                args.cache_dir, args.context_dir, split="train",
                beta_filter=args.beta_class, seed=args.seed,
            )
            print(f"using whole-image context {args.context_dir}", flush=True)
        elif args.dcp_t:
            train_set = CachedCropsWithDcpT(
                args.cache_dir, split="train", beta_filter=args.beta_class, seed=args.seed,
            )
            print("using DCP transmission as a 4th input channel", flush=True)
        else:
            train_set = CachedCrops(args.cache_dir, split="train", beta_filter=args.beta_class, seed=args.seed)
        if args.limit:
            train_set.pair_indices = train_set.pair_indices[: args.limit]
        print(f"using crop cache {args.cache_dir}", flush=True)
    else:
        if args.gate_context:
            raise SystemExit("--gate-context needs the crop cache (--cache-dir); the PNG path has no context")
        train_set = FoggyPairs(
            "data/foggy_cityscapes",
            "data/cityscapes_clear",
            "train",
            crop=args.crop,
            beta_filter=args.beta_class,
            seed=args.seed,
        )
        if args.limit:
            train_set.pairs = train_set.pairs[: args.limit]
    val_sets = build_val_set(args.val_scenes_per_beta, args.seed)

    loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = build_unet(
        args.size, gated=args.model == "gated", frozen_gate=args.frozen_gate,
        gate_init=args.gate_init, gate_spatial=args.gate_spatial,
        gate_context=args.gate_context, dcp_t=args.dcp_t,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_dir = ROOT / "runs" / "machine_learning" / args.run_id
    ckpt_dir = ROOT / "outputs" / "checkpoints" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    config_text = "\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())) + "\n"

    start_epoch = 0
    best_psnr = -math.inf
    last_checkpoint = ckpt_dir / "last.pt"
    if args.resume and last_checkpoint.is_file():
        state = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_psnr = state["best_psnr"]
        print(f"resumed from epoch {start_epoch}", flush=True)

    if start_epoch >= args.epochs:
        # A RESUME=1 re-invocation of an already-complete chain must not
        # rewrite its run record — that would clobber the real training
        # timestamps and source revision with the re-run's.
        print(f"run {args.run_id} is already complete ({start_epoch}/{args.epochs} epochs); nothing to do", flush=True)
        return

    config_path.write_text(config_text)
    log_path = run_dir / "log.csv"
    log_stream = log_path.open("a", newline="")
    writer = csv.writer(log_stream)
    if start_epoch == 0 and log_path.stat().st_size == 0:
        writer.writerow(
            ["epoch", "train_loss", "lr", "seconds", "val_psnr", "val_ssim", "d_mean", "d_min", "d_max"]
        )

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    n_parameters = sum(p.numel() for p in model.parameters())
    counts = {name: len(probe) for name, probe in zip(BETA_CLASSES, val_sets)}
    print(
        f"train pairs {len(train_set)} | val {counts} | params {n_parameters:,} | device {device} | amp {use_amp}",
        flush=True,
    )

    model.train()
    for epoch in range(start_epoch, args.epochs):
        train_set.set_epoch(epoch)
        epoch_start = time.time()
        running, seen = 0.0, 0
        for batch in loader:
            hazy = batch["hazy"].to(device, non_blocking=True)
            clear = batch["clear"].to(device, non_blocking=True)
            context = batch["context"].to(device, non_blocking=True) if "context" in batch else None
            dcp_t = batch["dcp_t"].to(device, non_blocking=True) if "dcp_t" in batch else None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output, _ = model(_prepare_input(model, hazy, dcp_t), context)
                loss = reconstruction_loss(output, clear, args.ssim_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * hazy.shape[0]
            seen += hazy.shape[0]
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_start, 1),
        }
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            with torch.amp.autocast("cuda", enabled=False):
                val = evaluate(model, val_sets, device)
            row["val_psnr"] = val["psnr"]
            row["val_ssim"] = val["ssim"]
            density = val.get("density", {})
            row["d_mean"] = density.get("mean", "")
            row["d_min"] = density.get("min", "")
            row["d_max"] = density.get("max", "")
            if val["psnr"] > best_psnr:
                best_psnr = val["psnr"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val": val,
                        "args": vars(args),
                        "n_parameters": n_parameters,
                    },
                    ckpt_dir / "best.pt",
                )
            print(
                f"epoch {epoch + 1}/{args.epochs} loss {row['train_loss']:.4f} "
                f"val_psnr {val['psnr']:.2f} val_ssim {val['ssim']:.4f}"
                + (f" d[{density['min']:.3f},{density['max']:.3f}]" if density else ""),
                flush=True,
            )
        else:
            print(f"epoch {epoch + 1}/{args.epochs} loss {row['train_loss']:.4f}", flush=True)
        writer.writerow([row.get(key, "") for key in
                         ["epoch", "train_loss", "lr", "seconds", "val_psnr", "val_ssim", "d_mean", "d_min", "d_max"]])
        log_stream.flush()
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_psnr": best_psnr,
            },
            last_checkpoint,
        )

    log_stream.close()
    best_path = ckpt_dir / "best.pt"
    best_state = torch.load(best_path, map_location="cpu", weights_only=False) if best_path.is_file() else None
    payload = {
        "status": "completed",
        "command": " ".join(sys.argv),
        "method": {"blind": "Blind U-Net", "gated": "Density-gated U-Net"}[args.model]
        + (" (gate frozen to d=1)" if args.frozen_gate else "")
        + (f" ({args.beta_class} branch)" if args.beta_class else ""),
        "config": {key: (str(value) if isinstance(value, Path) else value)
                   for key, value in vars(args).items()},
        "train_pairs": len(train_set),
        "n_parameters": n_parameters,
        "best": {
            "epoch": None if best_state is None else best_state["epoch"],
            "val": None if best_state is None else best_state["val"],
        },
        "config_sha256": sha256(config_path),
        "source_revision": git_revision(ROOT),
        "environment": environment(),
        "checkpoint": str(best_path.relative_to(ROOT)),
        "started_at": started_at,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"run: {run_dir.relative_to(ROOT)}")
    print(f"best checkpoint: {best_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
