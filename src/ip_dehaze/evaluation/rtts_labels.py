from __future__ import annotations

CLASS_NAMES = ("bicycle", "bus", "car", "motorbike", "person")


def yolo_label_to_coco(parts: list[str], width: int, height: int, class_map: dict[str, int]) -> dict:
    if len(parts) != 5:
        raise ValueError("YOLO label row must have five fields")
    class_index = int(parts[0])
    if not 0 <= class_index < len(CLASS_NAMES):
        raise ValueError(f"Unknown RTTS class {class_index}")
    center_x, center_y, box_w, box_h = map(float, parts[1:])
    if not all(0 <= value <= 1 for value in (center_x, center_y, box_w, box_h)):
        raise ValueError("Normalized YOLO box values must be in [0, 1]")
    x = (center_x - box_w / 2) * width
    y = (center_y - box_h / 2) * height
    w = box_w * width
    h = box_h * height
    return {
        "category_id": int(class_map[CLASS_NAMES[class_index]]),
        "bbox": [x, y, w, h],
        "area": w * h,
        "iscrowd": 0,
    }
