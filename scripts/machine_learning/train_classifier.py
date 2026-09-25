"""Train the 3-class haze-density classifier (paper-style router front end).

Labels are free: Foggy Cityscapes filenames encode β, which maps to
light/medium/heavy. Writes runs/machine_learning/<run-id>/ and
outputs/checkpoints/<run-id>/best.pt.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ip_dehaze.data.dataset import BETA_CLASSES, CachedCrops, FoggyPairs  # noqa: E402
from ip_dehaze.evaluation.provenance import environment, git_revision, sha256  # noqa: E402
from ip_dehaze.machine_learning.models.classifier import DensityClassifier  # noqa: E402

CLASS_NAMES = ("light", "medium", "heavy")


def vote_density_class(
    model: torch.nn.Module,
    hazy: torch.Tensor,
    device: torch.device,
    crop: int,
    votes: int = 5,
) -> int:
    """Majority vote over centre + corner crops; matches routed inference."""
    height, width = hazy.shape[-2:]
    if height < crop or width < crop:
        return int(model(hazy[None].to(device)).argmax(1))
    top, left = (height - crop) // 2, (width - crop) // 2
    positions = [
        (top, left),
        (0, 0),
        (0, width - crop),
        (height - crop, 0),
        (height - crop, width - crop),
    ]
    tallies: list[int] = []
    for index in range(max(1, votes)):
        y, x = positions[index % len(positions)]
        patch = hazy[:, y : y + crop, x : x + crop][None].to(device)
        tallies.append(int(model(patch).argmax(1)))
    return max(set(tallies), key=tallies.count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 3-class haze-density classifier")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--crop", type=int, default=256, help="crop size for crop mode (ignored with --resize)")
    parser.add_argument("--resize", default="", help="WxH whole-image downscale; empty = train on cached crops")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--val-per-beta", type=int, default=50, help="val images per density class")
    parser.add_argument("--votes", type=int, default=5, help="crops voted per image at validation/inference")
    parser.add_argument("--stride", type=int, default=1,
                        help="sample every Nth training pair (spread over the split without a cache)")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not args.resize and args.crop < 32:
        raise SystemExit("--crop must be at least 32 in crop mode (use --resize for whole-image mode)")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    if device.type == "cpu":
        torch.set_num_threads(4)  # leave cores for any concurrent training job

    width, height = (
        (int(value) for value in args.resize.lower().split("x")) if args.resize else (0, 0)
    )
    cache_dir = ROOT / "data/derived" / ("crops" / f"resize-{args.resize}" if args.resize else "crops/train")
    if not args.resize and cache_dir.is_dir():
        train_set = CachedCrops(cache_dir, split="train", seed=args.seed)
        if args.limit:
            train_set.pair_indices = train_set.pair_indices[: args.limit]
        print(f"using crop cache {cache_dir.relative_to(ROOT)}", flush=True)
    else:
        train_set = FoggyPairs(
            "data/foggy_cityscapes",
            "data/cityscapes_clear",
            "train",
            crop=None if args.resize else args.crop,
            resize=(width, height) if args.resize else None,
            stride=args.stride,
            seed=args.seed,
        )
        if args.limit:
            train_set.pairs = train_set.pairs[: args.limit]
        print(f"input: {'whole image ' + args.resize if args.resize else str(args.crop) + 'px crops on the fly'}", flush=True)
    val_sets = [
        FoggyPairs(
            "data/foggy_cityscapes",
            "data/cityscapes_clear",
            "val",
            crop=None,
            resize=(width, height) if args.resize else None,
            beta_filter=name,
            stride=max(1, 500 // args.val_per_beta),
            seed=args.seed,
        )
        for name in BETA_CLASSES
    ]
    loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
    )

    model = DensityClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    run_dir = ROOT / "runs" / "machine_learning" / args.run_id
    ckpt_dir = ROOT / "outputs" / "checkpoints" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    config_path.write_text("\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())) + "\n")

    log_stream = (run_dir / "log.csv").open("a", newline="")
    writer = csv.writer(log_stream)
    writer.writerow(["epoch", "train_loss", "train_acc", "val_acc", "seconds"])

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    best_accuracy, best_epoch = -1.0, -1
    for epoch in range(args.epochs):
        model.train()
        train_set.set_epoch(epoch)
        epoch_start = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for batch in loader:
            images = batch["hazy"].to(device, non_blocking=True)
            labels = batch["density_class"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
            correct += int((logits.argmax(1) == labels).sum())
            seen += images.shape[0]
        scheduler.step()

        model.eval()
        with torch.no_grad():
            hits, total = 0, 0
            per_class = {}
            for class_index, val_set in enumerate(val_sets):
                class_hits = 0
                for index in range(len(val_set)):
                    sample = val_set[index]
                    if args.resize:
                        image = sample["hazy"][None].to(device)
                        prediction = int(model(image).argmax(1))
                    else:
                        prediction = vote_density_class(
                            model, sample["hazy"], device, args.crop, args.votes
                        )
                    class_hits += int(prediction == class_index)
                per_class[CLASS_NAMES[class_index]] = class_hits / max(len(val_set), 1)
                hits += class_hits
                total += len(val_set)
        val_accuracy = hits / max(total, 1)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "train_acc": correct / max(seen, 1),
            "val_acc": val_accuracy,
            "seconds": round(time.time() - epoch_start, 1),
        }
        writer.writerow([row[k] for k in ["epoch", "train_loss", "train_acc", "val_acc", "seconds"]])
        log_stream.flush()
        print(
            f"epoch {epoch + 1}/{args.epochs} loss {row['train_loss']:.4f} "
            f"train_acc {row['train_acc']:.4f} val_acc {val_accuracy:.4f}",
            flush=True,
        )
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "val_accuracy": val_accuracy,
                 "per_class": per_class, "args": vars(args)},
                ckpt_dir / "best.pt",
            )

    log_stream.close()
    payload = {
        "status": "completed",
        "command": " ".join(sys.argv),
        "method": "3-class haze-density classifier (router front end)",
        "config": {key: (str(value) if isinstance(value, Path) else value)
                   for key, value in vars(args).items()},
        "train_pairs": len(train_set),
        "best": {"epoch": best_epoch, "val_accuracy": best_accuracy},
        "class_names": CLASS_NAMES,
        "config_sha256": sha256(config_path),
        "source_revision": git_revision(ROOT),
        "environment": environment(),
        "checkpoint": str((ckpt_dir / "best.pt").relative_to(ROOT)),
        "started_at": started_at,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"best val accuracy {best_accuracy:.4f} at epoch {best_epoch}")
    print(f"run: {run_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
