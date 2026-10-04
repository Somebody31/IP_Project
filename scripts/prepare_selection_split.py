#!/usr/bin/env python
"""Carve an honest 500-image model selection split from rtts/train.

Uses a fixed random seed (42) to sample 500 images and their labels, creates
rtts/select with relative symlinks (saving disk space), and records full
manifests for both rtts/select and the remaining training images.
"""
from __future__ import annotations

import csv
import hashlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def count_boxes(label_path: Path) -> int:
    if not label_path.is_file():
        return 0
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def main() -> None:
    train_img_dir = ROOT / "rtts" / "train" / "images"
    train_lbl_dir = ROOT / "rtts" / "train" / "labels"
    select_dir = ROOT / "rtts" / "select"
    select_img_dir = select_dir / "images"
    select_lbl_dir = select_dir / "labels"
    manifests_dir = ROOT / "data" / "manifests"

    select_img_dir.mkdir(parents=True, exist_ok=True)
    select_lbl_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(train_img_dir.glob("*.jpg"))
    if len(all_images) != 8557:
        print(f"Warning: expected 8557 images in rtts/train, found {len(all_images)}")

    random.seed(42)
    selected_images = sorted(random.sample(all_images, 500), key=lambda p: p.name)
    selected_set = {p.name for p in selected_images}
    remaining_images = [p for p in all_images if p.name not in selected_set]

    # Create symlinks in rtts/select
    for img_path in selected_images:
        dest_img = select_img_dir / img_path.name
        if dest_img.is_symlink() or dest_img.exists():
            dest_img.unlink()
        # Relative symlink
        dest_img.symlink_to(Path("..") / ".." / "train" / "images" / img_path.name)

        lbl_name = f"{img_path.stem}.txt"
        src_lbl = train_lbl_dir / lbl_name
        dest_lbl = select_lbl_dir / lbl_name
        if dest_lbl.is_symlink() or dest_lbl.exists():
            dest_lbl.unlink()
        if src_lbl.exists():
            dest_lbl.symlink_to(Path("..") / ".." / "train" / "labels" / lbl_name)

    # Save selection manifest
    select_manifest_path = manifests_dir / "rtts_select.csv"
    with open(select_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "size_bytes", "sha256", "n_boxes"]
        )
        writer.writeheader()
        for p in selected_images:
            lbl_p = train_lbl_dir / f"{p.stem}.txt"
            writer.writerow({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "n_boxes": count_boxes(lbl_p),
            })

    # Save remaining train manifest
    train_manifest_path = manifests_dir / "rtts_train_remaining.csv"
    with open(train_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "size_bytes", "sha256", "n_boxes"]
        )
        writer.writeheader()
        for p in remaining_images:
            lbl_p = train_lbl_dir / f"{p.stem}.txt"
            writer.writerow({
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
                "n_boxes": count_boxes(lbl_p),
            })

    print(f"Successfully created rtts/select with {len(selected_images)} images.")
    print(f"Remaining training set: {len(remaining_images)} images.")
    print(f"Manifest written: {select_manifest_path}")
    print(f"Manifest written: {train_manifest_path}")


if __name__ == "__main__":
    main()
