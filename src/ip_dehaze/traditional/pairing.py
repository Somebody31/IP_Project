from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

FOGGY_PATTERN = re.compile(r"^(?P<image_id>.+)_leftImg8bit_foggy_beta_(?P<beta>0\.005|0\.01|0\.02)\.png$")


@dataclass(frozen=True)
class ImagePair:
    split: str
    city: str
    image_id: str
    beta: float
    foggy_path: Path
    clear_path: Path


def index_pairs(
    foggy_root: str | Path,
    clear_root: str | Path,
    split: str,
    refined_manifest: str | Path | None = None,
) -> list[ImagePair]:
    foggy_split = Path(foggy_root) / "leftImg8bit_foggy" / split
    clear_split = Path(clear_root) / "leftImg8bit" / split
    if not foggy_split.is_dir() or not clear_split.is_dir():
        raise FileNotFoundError(f"Missing Foggy Cityscapes split {split!r}")

    refined_ids = None
    if refined_manifest is not None:
        manifest_path = Path(refined_manifest)
        refined_ids = {
            tuple(line.strip().split("/", 2)[1:])
            for line in manifest_path.read_text().splitlines()
            if line.strip() and line.strip().split("/", 1)[0] == split
        }

    pairs = []
    seen = set()
    missing = []
    for foggy_path in sorted(foggy_split.glob("*/*.png")):
        match = FOGGY_PATTERN.fullmatch(foggy_path.name)
        if not match:
            raise ValueError(f"Unexpected Foggy Cityscapes filename: {foggy_path}")
        image_id = match.group("image_id")
        city = foggy_path.parent.name
        key = (city, image_id, float(match.group("beta")))
        if refined_ids is not None and (city, image_id) not in refined_ids:
            continue
        if key in seen:
            raise ValueError(f"Duplicate foggy image for {key}")
        seen.add(key)
        clear_path = clear_split / city / f"{image_id}_leftImg8bit.png"
        if not clear_path.is_file():
            missing.append(str(clear_path))
            continue
        pairs.append(ImagePair(split, city, image_id, float(match.group("beta")), foggy_path, clear_path))
    if missing:
        raise FileNotFoundError(f"{len(missing)} clear twins missing; first: {missing[0]}")
    if not pairs:
        raise ValueError(f"No paired Foggy Cityscapes images found in split {split!r}")
    return pairs
