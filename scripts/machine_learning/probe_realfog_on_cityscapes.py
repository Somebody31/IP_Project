"""Real-fog stats over in-domain content: the missing cell of the 2x2.

The failure explanation says our model misreads RTTS *content* as fog. The
complementary experiment: keep content in-distribution (Cityscapes clear)
and transplant fog measured from *real* fog images (RTTS), so only the fog
statistics change from training. If the model still over-removes, the cause
is fog realism, not content; if it behaves, content was the cause.

Arms (same scenes, paired against the known clear reference):
  1. rtts_t      : I = J·t_rtts + 1·(1−t_rtts)      real transmission, white A
  2. rtts_tA     : I = J·t_rtts + A_rtts·(1−t_rtts) real transmission AND A
  3. shipped     : I = J·t_ship + 1·(1−t_ship)      training-distribution fog (control)

Per arm, per scene: PSNR/SSIM of the restored image vs the clear reference,
plus the luminance ratio restored/input (the over-removal signature: healthy
in-domain restorations sit at 0.48–0.52; RTTS over-removal measured 0.449).

RTTS transmission is estimated with the same DCP routine used everywhere
else in this project, then resized to the Cityscapes frame. Caveat recorded
in the output: resizing changes the absolute spatial frequency of fog
structure (real fog is low-frequency, so the structure survives, but this is
a transplant, not a capture).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "machine_learning"))
sys.path.insert(0, str(ROOT / "scripts" / "traditional"))

from ip_dehaze.evaluation.metrics import metrics as image_metrics  # noqa: E402
from ip_dehaze.traditional.dcp import (  # noqa: E402
    dark_channel,
    estimate_atmospheric_light,
    estimate_transmission,
)
from ip_dehaze.traditional.pairing import index_pairs  # noqa: E402
from benchmark_model import load_dehazer  # noqa: E402
from evaluate_model_detection import dehaze_whole  # noqa: E402

TRANSMITTANCE_ROOT = ROOT / "data" / "transmittance" / "leftImg8bit_transmittance"


def resize_to(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8))
    return np.asarray(image.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0


def haze_lift(image: np.ndarray) -> float:
    """Dark-channel mean: the standard single-image fog-density proxy used here."""
    return float(dark_channel(image, 15).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="realfog-stats-on-cityscapes")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/checkpoints/unet-gated-spatial-20260929/best.pt")
    parser.add_argument("--scenes", type=int, default=60)
    parser.add_argument("--rtts-split", default="valid")
    parser.add_argument("--source-beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _ = load_dehazer(args.checkpoint, device)

    pairs = [p for p in index_pairs(ROOT / "data/foggy_cityscapes",
                                    ROOT / "data/cityscapes_clear", "val")
             if abs(p.beta - args.source_beta) < 1e-9]
    pairs.sort(key=lambda p: (p.city, p.image_id))
    rng = np.random.default_rng(args.seed)
    picks = [pairs[i] for i in rng.choice(len(pairs), size=min(args.scenes, len(pairs)), replace=False)]

    rtts_images = sorted((ROOT / "rtts" / args.rtts_split / "images").glob("*.jpg"))
    print(f"scenes {len(picks)} | rtts pool {len(rtts_images)} ({args.rtts_split})", flush=True)

    rows = []
    for index, pair in enumerate(picks):
        clear = np.asarray(Image.open(pair.clear_path).convert("RGB"), dtype=np.float32) / 255.0
        shape = clear.shape[:2]

        # --- real fog, measured from an RTTS image ---
        rtts_path = rtts_images[int(rng.integers(len(rtts_images)))]
        rtts = np.asarray(Image.open(rtts_path).convert("RGB"), dtype=np.float32) / 255.0
        dark = dark_channel(rtts, 15)
        a_rtts = estimate_atmospheric_light(rtts, dark)
        t_rtts = estimate_transmission(rtts, a_rtts)
        t_real = resize_to(np.clip(t_rtts, 0.0, 1.0), shape)

        # --- training-distribution fog (control) ---
        t_path = TRANSMITTANCE_ROOT / "val" / pair.city / \
            f"{pair.image_id}_leftImg8bit_transmittance_beta_{args.source_beta:g}.png"
        t_ship = np.asarray(Image.open(t_path), dtype=np.float32) / 255.0

        # --- density-matched real fog: only the fog FIELD statistics differ ---
        # t' = t^gamma, gamma solved so the dark-channel lift matches the
        # shipped arm's, so PSNR differences isolate structure, not density.
        # lift is evaluated on a 1/8-scale copy (density is scale-robust) and
        # the branch direction is: larger gamma -> less transmission -> more fog.
        def lift_at_scale(image: np.ndarray) -> float:
            small = resize_to(image, (image.shape[0] // 8, image.shape[1] // 8))
            return haze_lift(small)

        target = lift_at_scale(np.clip(clear * t_ship[:, :, None] + (1.0 - t_ship[:, :, None]), 0, 1))
        lo, hi = 0.1, 6.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            t_mid = np.power(np.clip(t_real, 1e-6, 1.0), mid)
            if lift_at_scale(np.clip(clear * t_mid[:, :, None] + (1.0 - t_mid[:, :, None]), 0, 1)) > target:
                hi = mid  # too much fog -> smaller gamma
            else:
                lo = mid
        t_match = np.power(np.clip(t_real, 1e-6, 1.0), 0.5 * (lo + hi))

        a_vec = np.asarray(a_rtts, dtype=np.float32).reshape(-1)[:3]
        arms = {
            "rtts_t": (clear * t_real[:, :, None] + (1.0 - t_real[:, :, None])),
            "rtts_matched": (clear * t_match[:, :, None] + (1.0 - t_match[:, :, None])),
            "rtts_matched_A": (clear * t_match[:, :, None] + a_vec[None, None, :] * (1.0 - t_match[:, :, None])),
            "rtts_tA": (clear * t_real[:, :, None] + a_vec[None, None, :] * (1.0 - t_real[:, :, None])),
            # shipped transmission + real A: isolates the atmospheric-light
            # level (real fog ~0.90 vs the 1.0 our training renders used)
            "ship_A": (clear * t_ship[:, :, None] + a_vec[None, None, :] * (1.0 - t_ship[:, :, None])),
            "shipped": (clear * t_ship[:, :, None] + (1.0 - t_ship[:, :, None])),
        }
        entry = {"scene": f"{pair.city}/{pair.image_id}", "rtts_source": rtts_path.name,
                 "a_rtts": [float(v) for v in np.asarray(a_rtts).reshape(-1)[:3]], "arms": {}}
        for name, hazy in arms.items():
            hazy = np.clip(hazy, 0.0, 1.0).astype(np.float32)
            restored = dehaze_whole(model, hazy, device)
            scores = image_metrics(restored, clear)
            entry["arms"][name] = {
                "psnr": scores["psnr"], "ssim": scores["ssim"],
                "hazy_lift": haze_lift(hazy),
                "lum_ratio": float(restored.mean() / hazy.mean()),
                "hazy_psnr": image_metrics(hazy, clear)["psnr"],
            }
        rows.append(entry)
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(picks)}", flush=True)

    summary = {}
    for name in ("rtts_t", "rtts_matched", "rtts_matched_A", "rtts_tA", "ship_A", "shipped"):
        psnr = np.array([r["arms"][name]["psnr"] for r in rows])
        ratio = np.array([r["arms"][name]["lum_ratio"] for r in rows])
        lift = np.array([r["arms"][name]["hazy_lift"] for r in rows])
        hazy_psnr = np.array([r["arms"][name]["hazy_psnr"] for r in rows])
        summary[name] = {
            "psnr_mean": float(psnr.mean()), "psnr_std": float(psnr.std()),
            "hazy_psnr_mean": float(hazy_psnr.mean()),
            "lum_ratio_mean": float(ratio.mean()), "lum_ratio_std": float(ratio.std()),
            "haze_lift_mean": float(lift.mean()),
            "gain_over_hazy": float((psnr - hazy_psnr).mean()),
        }

    print(f"\n{'arm':<10} {'restored':>9} {'hazy':>7} {'gain':>7} {'lum ratio':>10} {'haze lift':>10}")
    for name, values in summary.items():
        print(f"{name:<10} {values['psnr_mean']:>9.2f} {values['hazy_psnr_mean']:>7.2f} "
              f"{values['gain_over_hazy']:>+7.2f} {values['lum_ratio_mean']:>10.3f} "
              f"{values['haze_lift_mean']:>10.3f}")
    print("\nreference points: in-domain healthy restoration ratio 0.48-0.52; "
          "RTTS over-removal measured 0.449")

    out = {
        "run_id": args.run_id, "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "scenes": len(rows), "rtts_split": args.rtts_split, "seed": args.seed,
        "note": "RTTS transmission estimated by DCP then resized to the Cityscapes "
                "frame; a transplant of real-fog statistics onto in-domain content, "
                "not a captured pair.",
        "summary": summary, "rows": rows,
        "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    path = ROOT / "results" / "machine_learning" / f"{args.run_id}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
