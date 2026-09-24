from pathlib import Path


def test_method_tracks_and_shared_evaluation_exist():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "src/ip_dehaze/traditional",
        "src/ip_dehaze/machine_learning",
        "src/ip_dehaze/evaluation",
        "configs/traditional",
        "configs/machine_learning",
    ):
        assert (root / relative).is_dir()
