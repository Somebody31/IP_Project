"""Task-driven fine-tune on RTTS-train (P3.7 step 3, approved track).

Trains the *dehazer* to make the frozen COCO YOLOv8n see objects better on
real fog, while mixing synthetic-reconstruction batches to preserve
restoration ability. Frozen YOLOv8n is the loss AND the metric — disclosed
wherever the resulting numbers appear (decision record item 3).

Data discipline (user rule 2026-09-29, unchanged):
  supervision: rtts/train images + our synthetic Foggy Cityscapes train pairs
  selection:   rtts/valid (AP@0.5) + our synthetic val
  rtts/test:   NEVER touched here (single frozen score already done)

Loss per batch (two arms, mixed):
  - detection arm: frozen YOLO features on the dehazed RTTS crop are scored
    for detection confidence against GT boxes present in the crop
  - reconstruction arm: the standard reconstruction loss on synthetic pairs
    (same L1+SSIM as all other runs), which anchors restoration quality

Disclosed simplification: the detection arm maximizes detector confidence on
GT-matching anchors, not full AP (AP is not differentiable end-to-end at
reasonable cost). This is the "task signal" the decision record approved.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "machine_learning"))

from ip_dehaze.data.dataset import FoggyPairs
from ip_dehaze.evaluation.metrics import metrics as image_metrics
from ip_dehaze.evaluation.provenance import environment, git_revision, sha256
from ip_dehaze.machine_learning.losses import reconstruction_loss
from ip_dehaze.machine_learning.models.unet import build_unet
from ultralytics import YOLO

CONTEXT_SIZE = (512, 256)


def _resize(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((array * 255).round().astype(np.uint8))
    return np.asarray(image.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0


class RttTrainSet(torch.utils.data.Dataset):
    """Random crops from RTTS-train images with their YOLO boxes."""

    def __init__(self, images_dir: Path, labels_dir: Path, crop: int = 320,
                 limit: int = 0, seed: int = 1729) -> None:
        self.paths = sorted(images_dir.glob("*.jpg"))
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError(f"no images under {images_dir}")
        self.labels_dir = labels_dir
        self.crop = crop
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        rng = random.Random(self.seed * 1_000_003 + self._epoch * 10_007 + index)
        with Image.open(path) as handle:
            rgb = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
        height, width = rgb.shape[:2]
        crop = min(self.crop, height, width)
        top = rng.randrange(height - crop + 1)
        left = rng.randrange(width - crop + 1)
        image = rgb[top : top + crop, left : left + crop]
        if rng.random() < 0.5:
            image = image[:, ::-1]

        boxes: list[list[float]] = []
        label_path = self.labels_dir / f"{path.stem}.txt"
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, bw, bh = line.split()
                x1 = (float(cx) - float(bw) / 2) * width
                y1 = (float(cy) - float(bh) / 2) * height
                x2 = x1 + float(bw) * width
                y2 = y1 + float(bh) * height
                if rng.random() < 0.5:  # same flip coin
                    x1, x2 = crop - x2, crop - x1
                # clip to crop; keep if the box survives with reasonable area
                x1, x2 = max(0.0, min(crop, x1)), max(0.0, min(crop, x2))
                y1, y2 = max(0.0, min(crop, y1)), max(0.0, min(crop, y2))
                if (x2 - x1) * (y2 - y1) < 8 * 8:
                    continue
                boxes.append([float(cls), x1, y1, x2, y2])

        return {
            "hazy": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "boxes": boxes,
            "name": path.name,
        }


def detection_loss(
    yolo: torch.nn.Module,
    dehazed: torch.Tensor,
    boxes_batch: list[list[list[float]]],
    device: torch.device,
    image_size: int = 640,
) -> torch.Tensor:
    """Maximize frozen-detector confidence on anchors matching GT boxes.

    The dehazed crop is resized to YOLO's working size, pushed through the
    frozen detector, and the raw confidence map (84, N) is gathered at the
    anchors whose centres fall inside a GT box of the matching class. Loss
    is 1 - mean confidence of those anchors (high conf on true objects).
    """
    resized = torch.nn.functional.interpolate(
        dehazed, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    raw = yolo(resized)
    pred = raw[0] if isinstance(raw, (tuple, list)) else raw  # (B, 84, N) train mode

    losses = []
    n_anchors = pred.shape[2]
    grid = image_size // 8  # stride-8 anchor grid for the first P3 head
    for b, boxes in enumerate(boxes_batch):
        if not boxes:
            continue
        conf_all = torch.softmax(pred[b, 4:5, :], dim=1) * torch.sigmoid(pred[b, 5:, :])
        conf = conf_all.amax(dim=0)  # (N,) best-class confidence
        cx = (torch.arange(n_anchors, device=device) % (image_size // 8)) * (image_size / grid)
        cy = (torch.arange(n_anchors, device=device) // (image_size // 8)) * (image_size / grid)
        target = torch.zeros(n_anchors, device=device)
        for cls, x1, y1, x2, y2 in boxes:
            sx1, sy1, sx2, sy2 = (v / dehazed.shape[-1] * image_size for v in (x1, y1, x2, y2))
            inside = (
                (cx >= sx1) & (cx <= sx2) & (cy >= sy1) & (cy <= sy2)
            )
            target[inside] = 1.0
        if target.sum() == 0:
            continue
        losses.append((1.0 - conf[target > 0]).mean())
    if not losses:
        return dehazed.sum() * 0.0
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="unet-spatial-rtts-adapt")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/checkpoints/unet-gated-spatial-20260929/best.pt")
    parser.add_argument("--rtts-limit", type=int, default=0, help="subsample rtts/train (0 = all 8,557)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rtts-per-batch", type=int, default=4, help="RTTS crops per mixed batch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5, help="an order below training LR: this is a fine-tune")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--task-weight", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          (args.device if args.device != "auto" else "cpu"))

    run_dir = ROOT / "runs" / "machine_learning" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT / "outputs/checkpoints" / args.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Winner checkpoint (spatial-gated U-Net).
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = state.get("args", {})
    model = build_unet(
        train_args.get("size", "medium"),
        gated=train_args.get("model") == "gated",
        gate_spatial=bool(train_args.get("gate_spatial")),
        gate_context=bool(train_args.get("gate_context")),
        gate_init=train_args.get("gate_init"),
    ).to(device)
    model.load_state_dict(state["model"])

    # Frozen detector.
    yolo = YOLO(str(ROOT / "outputs/checkpoints/yolov8n.pt")).model.to(device)
    for p in yolo.parameters():
        p.requires_grad_(False)
    yolo.eval()

    rtts_set = RttTrainSet(ROOT / "rtts/train/images", ROOT / "rtts/train/labels",
                           limit=args.rtts_limit, seed=args.seed)
    synth_train = FoggyPairs("data/foggy_cityscapes", "data/cityscapes_clear", "train", crop=320)
    synth_val = FoggyPairs("data/foggy_cityscapes", "data/cityscapes_clear", "val", crop=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    config = vars(args).copy()
    (run_dir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in sorted(config.items())))
    log_path = run_dir / "log.csv"
    with log_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "task_loss", "recon_loss", "synth_val_psnr",
                         "rtts_valid_ap50", "seconds"])

    print(f"== {args.run_id} ==", flush=True)
    print(f"rtts-train crops {len(rtts_set)} | synth pairs {len(synth_train)} | lr {args.lr}", flush=True)

    rng = random.Random(args.seed)
    steps_per_epoch = min(len(rtts_set) // args.batch, 400)

    for epoch in range(args.epochs):
        model.train()
        rtts_set.set_epoch(epoch)
        epoch_start = time.time()
        tot_task = tot_recon = 0.0
        for step in range(steps_per_epoch):
            # RTTS detection arm — boxes are variable-length lists, so collate
            # images with the default collate and keep boxes as a plain list.
            idxs = rng.sample(range(len(rtts_set)), args.rtts_per_batch)
            rtts_samples = [rtts_set[i] for i in idxs]
            hazy_rtts = torch.stack([s["hazy"] for s in rtts_samples]).to(device)
            boxes_rtts = [s["boxes"] for s in rtts_samples]
            # whole-scene context for the context stem (RTTS crops ARE whole scenes here)
            ctx = torch.stack([
                torch.from_numpy(np.ascontiguousarray(
                    _resize(hazy_rtts[b].permute(1, 2, 0).cpu().numpy(), CONTEXT_SIZE).transpose(2, 0, 1)))
                for b in range(hazy_rtts.shape[0])]).to(device)
            restored, _ = model(hazy_rtts, ctx)
            task = detection_loss(yolo, restored, boxes_rtts, device)

            # --- synthetic reconstruction arm ---
            synth_idxs = rng.sample(range(len(synth_train)), args.batch - args.rtts_per_batch)
            synth = torch.utils.data.default_collate([synth_train[i] for i in synth_idxs])
            hazy_syn = synth["hazy"].to(device)
            clear_syn = synth["clear"].to(device)
            ctx_syn = None
            if model.context_stem is not None:
                ctx_syn = torch.stack([
                    torch.from_numpy(np.ascontiguousarray(
                        _resize(hazy_syn[b].permute(1, 2, 0).cpu().numpy(), CONTEXT_SIZE).transpose(2, 0, 1)))
                    for b in range(hazy_syn.shape[0])]).to(device)
            out_syn, _ = model(hazy_syn, ctx_syn)
            recon = reconstruction_loss(out_syn, clear_syn, ssim_weight=0.2)

            loss = recon + args.task_weight * task
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tot_task += float(task.detach())
            tot_recon += float(recon.detach())
            if step % 50 == 0:
                print(f"  epoch {epoch+1}/{args.epochs} step {step}/{steps_per_epoch} "
                      f"task {tot_task/(step+1):.4f} recon {tot_recon/(step+1):.4f}", flush=True)

        row = {
            "epoch": epoch + 1,
            "train_loss": round(tot_task + tot_recon, 4),
            "task_loss": round(tot_task / steps_per_epoch, 4),
            "recon_loss": round(tot_recon / steps_per_epoch, 4),
            "seconds": round(time.time() - epoch_start, 1),
        }

        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            # synthetic val: restoration preserved?
            model.eval()
            with torch.no_grad():
                psnrs = []
                for i in rng.sample(range(len(synth_val)), 12):
                    s = synth_val[i]
                    hazy = s["hazy"][None].to(device)
                    ctx = None
                    if model.context_stem is not None:
                        ctx = torch.from_numpy(np.ascontiguousarray(
                            _resize(s["hazy"].permute(1, 2, 0).numpy(), CONTEXT_SIZE).transpose(2, 0, 1)))[None].to(device)
                    out, _ = model(hazy, ctx)
                    scores = image_metrics(out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy(),
                                           s["clear"].permute(1, 2, 0).numpy())
                    psnrs.append(scores["psnr"])
                row["synth_val_psnr"] = round(sum(psnrs) / len(psnrs), 2)
            torch.save({"model": model.state_dict(), "args": {**train_args, "adapt": True}},
                       ckpt_dir / "last.pt")
            print(f"epoch {epoch+1}/{args.epochs} recon {row['recon_loss']:.4f} task {row['task_loss']:.4f} "
                  f"synth_val_psnr {row.get('synth_val_psnr')}", flush=True)
            torch.save({"model": model.state_dict(), "args": {**train_args, "adapt": True}},
                       ckpt_dir / "best.pt")

        with log_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([row.get(k, "") for k in
                             ("epoch", "train_loss", "task_loss", "recon_loss", "synth_val_psnr",
                              "rtts_valid_ap50", "seconds")])

    (run_dir / "summary.json").write_text(json.dumps({
        "status": "training_complete_needs_valid_scoring",
        "config": config,
        "note": "selection on rtts/valid via evaluate_model_detection.py --dataset-split valid; "
                "rtts/test untouched",
        "source_revision": git_revision(ROOT),
        "environment": environment(),
    }, indent=2))
    print("done: score with evaluate_model_detection.py --dataset-split valid", flush=True)


if __name__ == "__main__":
    main()
