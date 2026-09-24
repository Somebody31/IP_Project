from pathlib import Path

import pytest

from ip_dehaze.traditional.pairing import index_pairs


def make_pair(root: Path, split: str, city: str, foggy_name: str, clear_name: str):
    foggy = root / "foggy" / "leftImg8bit_foggy" / split / city
    clear = root / "clear" / "leftImg8bit" / split / city
    foggy.mkdir(parents=True, exist_ok=True)
    clear.mkdir(parents=True, exist_ok=True)
    (foggy / foggy_name).touch()
    (clear / clear_name).touch()
    return root / "foggy", root / "clear"


def test_pairs_match_same_split_city_and_beta(tmp_path):
    foggy_root, clear_root = make_pair(
        tmp_path,
        "val",
        "aachen",
        "aachen_000001_000002_leftImg8bit_foggy_beta_0.005.png",
        "aachen_000001_000002_leftImg8bit.png",
    )
    pairs = index_pairs(foggy_root, clear_root, "val")
    assert len(pairs) == 1
    assert pairs[0].beta == 0.005
    assert pairs[0].city == "aachen"
    assert pairs[0].split == "val"


def test_refined_manifest_filters_by_split_city_and_image(tmp_path):
    foggy_root = tmp_path / "foggy"
    clear_root = tmp_path / "clear"
    for image_id in ("aachen_000001_000002", "aachen_000003_000004"):
        make_pair(
            tmp_path,
            "val",
            "aachen",
            f"{image_id}_leftImg8bit_foggy_beta_0.01.png",
            f"{image_id}_leftImg8bit.png",
        )
    manifest = tmp_path / "refined.txt"
    manifest.write_text("val/aachen/aachen_000001_000002\ntrain/aachen/aachen_000003_000004\n")
    pairs = index_pairs(foggy_root, clear_root, "val", manifest)
    assert [(pair.city, pair.image_id) for pair in pairs] == [("aachen", "aachen_000001_000002")]


def test_missing_clear_twin_fails(tmp_path):
    foggy = tmp_path / "foggy/leftImg8bit_foggy/val/aachen"
    clear = tmp_path / "clear/leftImg8bit/val/aachen"
    foggy.mkdir(parents=True, exist_ok=True)
    clear.mkdir(parents=True, exist_ok=True)
    (foggy / "aachen_000001_000002_leftImg8bit_foggy_beta_0.02.png").touch()
    with pytest.raises(FileNotFoundError, match="clear twins missing"):
        index_pairs(tmp_path / "foggy", tmp_path / "clear", "val")


def test_unexpected_beta_is_rejected(tmp_path):
    foggy = tmp_path / "foggy/leftImg8bit_foggy/val/aachen"
    clear = tmp_path / "clear/leftImg8bit/val/aachen"
    foggy.mkdir(parents=True, exist_ok=True)
    clear.mkdir(parents=True, exist_ok=True)
    (foggy / "aachen_000001_000002_leftImg8bit_foggy_beta_0.03.png").touch()
    with pytest.raises(ValueError, match="Unexpected Foggy Cityscapes filename"):
        index_pairs(tmp_path / "foggy", tmp_path / "clear", "val")
