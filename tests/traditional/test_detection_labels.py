import pytest

from ip_dehaze.evaluation.rtts_labels import yolo_label_to_coco


CLASS_MAP = {"bicycle": 1, "bus": 5, "car": 2, "motorbike": 3, "person": 0}


def test_rtts_yolo_box_maps_to_coco_pixels():
    result = yolo_label_to_coco("2 0.5 0.5 0.2 0.4".split(), 100, 50, CLASS_MAP)
    assert result["category_id"] == 2
    assert result["bbox"] == [40.0, 15.0, 20.0, 20.0]
    assert result["area"] == 400.0


def test_unknown_class_is_rejected():
    with pytest.raises(ValueError, match="Unknown RTTS class"):
        yolo_label_to_coco("5 0.5 0.5 0.2 0.4".split(), 100, 50, CLASS_MAP)
