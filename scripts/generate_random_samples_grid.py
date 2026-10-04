#!/usr/bin/env python
"""Generate systematic comparison grids for randomly sampled images from rtts/valid."""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/machine_learning"))
sys.path.insert(0, str(ROOT / "scripts/traditional"))

import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from benchmark_model import classify, load_classifier, load_dehazer
from evaluate_model_detection import dehaze_whole
from ip_dehaze.traditional.dcp import dehaze as dcp_dehaze


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    clf = load_classifier(
        Path("outputs/checkpoints/density-classifier-20260924/best.pt"), device
    )[0]
    paper_branches = [
        load_dehazer(
            Path("outputs/checkpoints/branch-light-20260924/best.pt"), device
        )[0],
        load_dehazer(
            Path("outputs/checkpoints/branch-medium-20260924/best.pt"), device
        )[0],
        load_dehazer(
            Path("outputs/checkpoints/branch-heavy-20260924/best.pt"), device
        )[0],
    ]
    our_model = load_dehazer(
        Path("outputs/checkpoints/adapt-v3-resynth-w0.01-20261003/best.pt"), device
    )[0]
    yolo = YOLO("outputs/checkpoints/yolov8n.pt")

    imgs = sorted(list(Path("rtts/valid/images").glob("*.jpg")))
    random.seed(42)
    sampled = random.sample(imgs, 4)

    out_dir = Path("docs/figures/presentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        "Raw Hazy",
        "Reference Paper",
        "Classical DCP",
        "Our Method",
    ]

    cell_w, cell_h = 320, 240
    header_h = 28
    grid_img = Image.new(
        "RGB", (cell_w * 4, (cell_h + header_h) * 4), (255, 255, 255)
    )
    draw = ImageDraw.Draw(grid_img)

    for row_idx, img_path in enumerate(sampled):
        scene_name = img_path.name.split(".")[0][:18]
        with Image.open(img_path) as im:
            raw_pil = im.convert("RGB")
            rgb = np.asarray(raw_pil, dtype=np.float32) / 255.0

        dcp_np = dcp_dehaze(rgb)
        dcp_pil = Image.fromarray(
            (np.clip(dcp_np, 0, 1) * 255).round().astype(np.uint8)
        )

        density = classify(clf, rgb, device)
        paper_np = dehaze_whole(paper_branches[density], rgb, device)
        paper_pil = Image.fromarray(
            (np.clip(paper_np, 0, 1) * 255).round().astype(np.uint8)
        )

        our_np = dehaze_whole(our_model, rgb, device)
        our_pil = Image.fromarray(
            (np.clip(our_np, 0, 1) * 255).round().astype(np.uint8)
        )

        processed = [raw_pil, paper_pil, dcp_pil, our_pil]

        y_base = row_idx * (cell_h + header_h)
        for col_idx, m_label in enumerate(methods):
            x_base = col_idx * cell_w
            header_text = (
                f"{m_label} | {scene_name}" if row_idx == 0 else scene_name
            )
            draw.rectangle(
                [x_base, y_base, x_base + cell_w, y_base + header_h],
                fill=(240, 243, 246),
            )
            draw.text(
                (x_base + 8, y_base + 6),
                header_text,
                fill=(30, 30, 30) if row_idx > 0 else (20, 50, 100),
            )

            cell_thumb = processed[col_idx].resize(
                (cell_w, cell_h), Image.Resampling.BILINEAR
            )
            grid_img.paste(cell_thumb, (x_base, y_base + header_h))

            draw.rectangle(
                [
                    x_base,
                    y_base,
                    x_base + cell_w,
                    y_base + header_h + cell_h,
                ],
                outline=(210, 215, 220),
                width=1,
            )

    grid_img.save(out_dir / "random_samples_grid.png", quality=95)
    print(f"Saved random samples grid to {out_dir / 'random_samples_grid.png'}")


if __name__ == "__main__":
    main()
