import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_selection_split_integrity():
    select_csv = ROOT / "data" / "manifests" / "rtts_select.csv"
    train_csv = ROOT / "data" / "manifests" / "rtts_train_remaining.csv"
    assert select_csv.is_file(), "rtts_select.csv must exist"
    assert train_csv.is_file(), "rtts_train_remaining.csv must exist"

    with open(select_csv) as f:
        select_rows = list(csv.DictReader(f))
    with open(train_csv) as f:
        train_rows = list(csv.DictReader(f))

    assert len(select_rows) == 500, f"Expected 500 selected rows, got {len(select_rows)}"
    assert len(train_rows) == 8057, f"Expected 8057 training rows, got {len(train_rows)}"

    select_names = {r["name"] for r in select_rows}
    train_names = {r["name"] for r in train_rows}

    # 1. Zero overlap between select and train
    overlap = select_names.intersection(train_names)
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping images between select and train"

    # 2. Check files on disk
    select_img_dir = ROOT / "rtts" / "select" / "images"
    select_lbl_dir = ROOT / "rtts" / "select" / "labels"
    assert select_img_dir.is_dir()
    assert select_lbl_dir.is_dir()

    disk_images = sorted(p.name for p in select_img_dir.glob("*.jpg"))
    assert len(disk_images) == 500
    assert set(disk_images) == select_names

    for name in disk_images:
        img_p = select_img_dir / name
        assert img_p.resolve().is_file(), f"Symlink target {img_p} must resolve to a valid file"

    # 3. Strictly disjoint from valid and test
    valid_names = {p.name for p in (ROOT / "rtts" / "valid" / "images").glob("*.jpg")}
    test_names = {p.name for p in (ROOT / "rtts" / "test" / "images").glob("*.jpg")}

    assert len(select_names.intersection(valid_names)) == 0, "Select split cannot overlap with valid"
    assert len(select_names.intersection(test_names)) == 0, "Select split cannot overlap with test"
