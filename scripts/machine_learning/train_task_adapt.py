"""Task-driven fine-tune on RTTS-train (P3.7 step 3, approved track) — v2.

Trains the *dehazer* to make the frozen COCO YOLOv8n see objects better on
real fog, while mixing synthetic-reconstruction batches to preserve
restoration ability. Frozen YOLOv8n is the loss AND the metric — disclosed
wherever the resulting numbers appear (decision record item 3).

Data discipline (user rule 2026-09-29, unchanged):
  supervision: rtts/train images + our synthetic Foggy Cityscapes train pairs
  selection:   rtts/valid (AP@0.5) + our synthetic val
  rtts/test:   NEVER touched here

v2 (2026-10-03) replaces the v1 task loss, which carried almost no signal:
  * v1 treated YOLOv8's class-probability rows as logits plus an objectness
    row (YOLOv8 has none): it softmaxed the *person* row across all 8,400
    anchors (~1/8400 each), so confidence was ~1e-4 and the loss sat at
    0.9999 — exactly the logged "0.9999 -> 0.9899 plateau";
  * v1 placed all 8,400 anchors on the stride-8 grid (anchors 6,400+ belong
    to the stride-16/32 heads) and its GT boxes were neither crop-shifted nor
    flipped with the image (see ip_dehaze.data.rtts_detection).
v2 uses Ultralytics' own v8DetectionLoss (TaskAligned assignment, box + cls
+ DFL) on the frozen detector, crop-aligned COCO-class boxes, and selects the
checkpoint on rtts/valid AP@0.5 with a synthetic-PSNR guard.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "machine_learning"))
sys.path.insert(0, str(ROOT / "scripts" / "traditional"))

from ip_dehaze.data.dataset import CachedCrops  # noqa: E402
from ip_dehaze.data.rtts_detection import (  # noqa: E402
    RttsDetectionCrops,
    collate_detection,
    to_detector_batch,
)
from ip_dehaze.evaluation.metrics import metrics as image_metrics  # noqa: E402
from ip_dehaze.evaluation.provenance import environment, git_revision  # noqa: E402
from ip_dehaze.machine_learning.losses import reconstruction_loss  # noqa: E402
from benchmark_model import load_dehazer  # noqa: E402
from evaluate_dcp_detection import coco_metrics, load_ground_truth  # noqa: E402
from evaluate_model_detection import dehaze_whole  # noqa: E402
from train_unet import build_val_set  # noqa: E402

CLASS_MAP = {"bicycle": 1, "bus": 5, "car": 2, "motorbike": 3, "person": 0}
CATEGORIES = [{"id": cid, "name": name} for name, cid in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]
IOU_THRESHOLDS = [0.5 + 0.05 * k for k in range(10)]


def freeze_batchnorm(model: torch.nn.Module) -> None:
    """Fine-tune with frozen BN statistics: small mixed-domain batches would
    otherwise drag the running stats away from what the synthetic val needs."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


class RttsScorer:
    """AP@0.5 on a fixed list of RTTS images through the exact eval path
    (dehaze_whole at native resolution -> uint8 -> YOLO.predict letterbox)."""

    def __init__(self, paths: list[Path], labels_dir: Path, device: torch.device) -> None:
        from ultralytics import YOLO

        self.paths = paths
        self.images, self.annotations, self.lookup = load_ground_truth(paths, labels_dir, CLASS_MAP)
        self.yolo = YOLO(str(ROOT / "outputs/checkpoints/yolov8n.pt"))
        self.device = device
        self.rgb = []
        for path in paths:
            with Image.open(path) as handle:
                self.rgb.append(np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0)

    def _detect(self, images: list[np.ndarray]) -> list[dict]:
        detections = []
        for start in range(0, len(images), 8):
            chunk = [Image.fromarray((im * 255).round().astype(np.uint8)) for im in images[start : start + 8]]
            outputs = self.yolo.predict(source=chunk, imgsz=640, conf=0.001, iou=0.7, max_det=300,
                                        device=0 if self.device.type == "cuda" else "cpu",
                                        classes=sorted(CLASS_MAP.values()), verbose=False)
            for path, output in zip(self.paths[start : start + 8], outputs):
                if output.boxes is None:
                    continue
                for box, score, cls in zip(output.boxes.xyxy.cpu().numpy(), output.boxes.conf.cpu().numpy(),
                                           output.boxes.cls.cpu().numpy().astype(int)):
                    x1, y1, x2, y2 = map(float, box)
                    detections.append({"image_id": self.lookup[path], "category_id": int(cls),
                                       "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                                       "score": float(score)})
        return detections

    def score_images(self, images: list[np.ndarray]) -> float:
        dets = self._detect(images)
        return coco_metrics(self.images, self.annotations, CATEGORIES, dets, IOU_THRESHOLDS)["AP@0.5"]

    @torch.no_grad()
    def score_model(self, model: torch.nn.Module) -> float:
        model.eval()
        restored = [dehaze_whole(model, rgb, self.device) for rgb in self.rgb]
        return self.score_images(restored)


@torch.no_grad()
def synth_psnr(model: torch.nn.Module, val_sets, device: torch.device) -> float:
    model.eval()
    scores = []
    for val_set in val_sets:
        for index in range(len(val_set)):
            sample = val_set[index]
            out = dehaze_whole(model, sample["hazy"].permute(1, 2, 0).numpy(), device)
            scores.append(image_metrics(out, sample["clear"].permute(1, 2, 0).numpy())["psnr"])
    return float(np.mean(scores))


def infinite(loader):
    while True:
        yield from loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", "--init-checkpoint", dest="checkpoint", type=Path,
                        default=ROOT / "outputs/checkpoints/unet-gated-spatial-20260929/best.pt")
    parser.add_argument("--synth-cache", type=Path, default=ROOT / "data/derived/crops/train",
                        help="crop cache for the reconstruction arm (ignored with --resynth)")
    parser.add_argument("--resynth", action="store_true",
                        help="reconstruction arm renders realistic fog on the fly (ip_dehaze.data.resynth)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=400)
    parser.add_argument("--rtts-batch", type=int, default=4)
    parser.add_argument("--synth-batch", type=int, default=4)
    parser.add_argument("--crop", type=int, default=384, help="native-resolution RTTS crop for the dehazer")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--task-weight", type=float, default=0.01,
                        help="weight on the v8 detection loss (box*7.5+cls*0.5+dfl*1.5, ~5-7 at start)")
    parser.add_argument("--distill-weight", type=float, default=0.0,
                        help="distillation loss weight against frozen pre-adaptation model")
    parser.add_argument("--heavy-fraction", type=float, default=0.0,
                        help="fraction of synthetic crops sampled specifically from heavy fog")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="exponential moving average decay for model weights (e.g. 0.999)")
    parser.add_argument("--select-dir", type=Path, default=ROOT / "rtts/select",
                        help="selection split directory holding images/ and labels/")
    parser.add_argument("--valid-subset", type=int, default=300, help="selection split images scored per epoch")
    parser.add_argument("--psnr-guard", type=float, default=1.0,
                        help="epochs whose synthetic PSNR drops more than this (dB) are not selectable")
    parser.add_argument("--rtts-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / "runs" / "machine_learning" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir = ROOT / "outputs/checkpoints" / args.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items())) + "\n")

    model, train_args = load_dehazer(args.checkpoint, device)

    teacher_model = None
    if args.distill_weight > 0.0:
        import copy
        teacher_model = copy.deepcopy(model)
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        teacher_model.eval()

    ema_model = None
    if args.ema_decay > 0.0:
        import copy
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    detector = YOLO(str(ROOT / "outputs/checkpoints/yolov8n.pt")).model.to(device)
    detector.args = get_cfg()  # default hyp: box 7.5, cls 0.5, dfl 1.5
    for p in detector.parameters():
        p.requires_grad_(False)
    detector.eval()

    select_csv = ROOT / "data/manifests/rtts_select.csv"
    rtts_set = RttsDetectionCrops(ROOT / "rtts/train/images", ROOT / "rtts/train/labels",
                                  crop=args.crop, limit=args.rtts_limit, seed=args.seed,
                                  exclude=select_csv if select_csv.is_file() else None)
    rtts_loader = DataLoader(rtts_set, batch_size=args.rtts_batch, shuffle=True, num_workers=args.workers,
                             collate_fn=collate_detection, drop_last=True, persistent_workers=False)
    if args.resynth:
        from ip_dehaze.data.resynth import ResynthPairs

        synth_set = ResynthPairs(split="train", heavy_fraction=args.heavy_fraction, seed=args.seed)
    else:
        synth_set = CachedCrops(args.synth_cache, split="train", seed=args.seed)
    synth_loader = DataLoader(synth_set, batch_size=args.synth_batch, shuffle=True, num_workers=args.workers,
                              drop_last=True, persistent_workers=False)

    select_img_dir = args.select_dir / "images" if args.select_dir.is_dir() else ROOT / "rtts/valid/images"
    select_lbl_dir = args.select_dir / "labels" if args.select_dir.is_dir() else ROOT / "rtts/valid/labels"
    select_paths = sorted(select_img_dir.glob("*.jpg"))
    subset = sorted(random.Random(args.seed).sample(select_paths, min(args.valid_subset, len(select_paths))))
    scorer = RttsScorer(subset, select_lbl_dir, device)
    val_sets = build_val_set(10, args.seed)

    hazy_ap = scorer.score_images(scorer.rgb)
    start_ap = scorer.score_model(model)
    start_psnr = synth_psnr(model, val_sets, device)
    print(f"== {args.run_id} == rtts-train {len(rtts_set)} | synth {len(synth_set)} | "
          f"select subset {len(subset)}: hazy AP50 {hazy_ap:.4f}, start AP50 {start_ap:.4f}, "
          f"start synth PSNR {start_psnr:.2f}", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * args.steps_per_epoch)
    log_path = run_dir / "log.csv"
    fields = ["epoch", "task_loss", "task_loss_hazy", "recon_loss", "synth_psnr", "valid_ap50", "selectable",
              "seconds"]
    with log_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(fields)
        csv.writer(handle).writerow([0, "", "", "", round(start_psnr, 3), round(start_ap, 4), True, 0])

    best = {"epoch": 0, "valid_ap50": start_ap, "synth_psnr": start_psnr}
    torch.save({"model": model.state_dict(), "args": {**train_args, "adapt": "v3"}, "epoch": 0},
               ckpt_dir / "best.pt")
    rtts_iter, synth_iter = infinite(rtts_loader), infinite(synth_loader)
    for epoch in range(1, args.epochs + 1):
        rtts_set.set_epoch(epoch)
        if hasattr(synth_set, "set_epoch"):
            synth_set.set_epoch(epoch)
        model.train()
        freeze_batchnorm(model)
        started = time.time()
        tot_task = tot_hazy = tot_recon = 0.0
        for step in range(args.steps_per_epoch):
            batch = next(rtts_iter)
            hazy = batch["hazy"].to(device, non_blocking=True)
            restored, _ = model(hazy, None)
            canvas, loss_batch = to_detector_batch(restored, batch["boxes"], batch["scale"])
            det_loss, _ = detector.loss(loss_batch, detector(canvas))
            task = det_loss.sum() / hazy.shape[0]
            with torch.no_grad():
                hazy_canvas, _ = to_detector_batch(hazy, batch["boxes"], batch["scale"])
                hazy_loss, _ = detector.loss(loss_batch, detector(hazy_canvas))

            synth = next(synth_iter)
            out_syn, _ = model(synth["hazy"].to(device, non_blocking=True), None)
            recon = reconstruction_loss(out_syn, synth["clear"].to(device, non_blocking=True), 0.2)

            loss = recon + args.task_weight * task
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_out, _ = teacher_model(synth["hazy"].to(device, non_blocking=True), None)
                distill_loss = torch.nn.functional.l1_loss(out_syn, teacher_out)
                loss = loss + args.distill_weight * distill_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if ema_model is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                        p_ema.data.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

            tot_task += float(task.detach())
            tot_hazy += float(hazy_loss.sum()) / hazy.shape[0]
            tot_recon += float(recon.detach())
            if step % 100 == 0:
                n = step + 1
                print(f"  epoch {epoch} step {step}/{args.steps_per_epoch} task {tot_task / n:.3f} "
                      f"(hazy {tot_hazy / n:.3f}) recon {tot_recon / n:.4f}", flush=True)

        ap = scorer.score_model(model)
        psnr = synth_psnr(model, val_sets, device)
        selectable = psnr >= start_psnr - args.psnr_guard
        n = args.steps_per_epoch
        row = [epoch, round(tot_task / n, 4), round(tot_hazy / n, 4), round(tot_recon / n, 5),
               round(psnr, 3), round(ap, 4), selectable, round(time.time() - started, 1)]
        with log_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(row)
        print(f"epoch {epoch}: task {row[1]} (hazy {row[2]}) recon {row[3]} | synth PSNR {psnr:.2f} | "
              f"valid-subset AP50 {ap:.4f} (hazy {hazy_ap:.4f}) {'' if selectable else '[guard]'}", flush=True)
        state = {"model": model.state_dict(), "args": {**train_args, "adapt": "v4"}, "epoch": epoch}
        torch.save(state, ckpt_dir / "last.pt")
        if ema_model is not None:
            ema_state = {"model": ema_model.state_dict(), "args": {**train_args, "adapt": "v4_ema"}, "epoch": epoch}
            torch.save(ema_state, ckpt_dir / "last_ema.pt")
            torch.save(ema_state, ckpt_dir / "ema.pt")
        if selectable and ap > best["valid_ap50"]:
            best = {"epoch": epoch, "valid_ap50": ap, "synth_psnr": psnr}
            torch.save(state, ckpt_dir / "best.pt")
            if ema_model is not None:
                torch.save(ema_state, ckpt_dir / "best_ema.pt")

    (run_dir / "summary.json").write_text(json.dumps({
        "status": "training_complete_needs_full_valid_scoring",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "valid_subset": {"n": len(subset), "hazy_ap50": hazy_ap, "start_ap50": start_ap},
        "start_synth_psnr": start_psnr,
        "best": best,
        "checkpoint": str((ckpt_dir / "best.pt").relative_to(ROOT)),
        "note": "selection on a fixed rtts/valid subset; score best.pt on full rtts/valid with "
                "evaluate_model_detection.py --dataset-split valid; rtts/test untouched",
        "source_revision": git_revision(ROOT),
        "environment": environment(),
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }, indent=2) + "\n")
    print(f"best: {best}", flush=True)


if __name__ == "__main__":
    main()
