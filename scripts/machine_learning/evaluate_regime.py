#!/usr/bin/env python
"""Score every method at the paper's fog levels, rendered on *our* val images.

Our dataset ships β ∈ {0.005, 0.01, 0.02}; the paper's is {0.03, 0.06, 0.09}.
Comparing across datasets is meaningless (different scenes, different synthesis),
so this renders heavier fog on the same val images instead, keeping the
comparison within-project and changing only the regime.

Because transmission is Beer-Lambert, `t = exp(-β·d)`, a heavier level follows
directly from the shipped transmittance map:

    t_new = t_old ** (β_new / β_old)

with the atmospheric-scattering model `I = J·t + A·(1 - t)`. Measured on the
val split: the fitted A is exactly (1, 1, 1) and re-rendering the *shipped* β
reproduces the shipped foggy image to 0.28/255 mean absolute error — so the
render below is a faithful extension, not a new synthesis. The source β is 0.02
because depth resolution in `d = -ln(t)/β` improves with β.

Every model was trained at β ≤ 0.02, so this regime is out-of-distribution for
all of them equally; any conclusion is about *transfer beyond training density*.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "machine_learning"))

from ip_dehaze.data.dataset import CONTEXT_SIZE, _resize, load_rgb  # noqa: E402
from ip_dehaze.evaluation.metrics import metrics  # noqa: E402
from ip_dehaze.evaluation.pair_benchmark import summarize  # noqa: E402
from ip_dehaze.traditional.dcp import dehaze as dcp_dehaze  # noqa: E402
from ip_dehaze.traditional.pairing import index_pairs  # noqa: E402
from benchmark_model import classify, load_classifier, load_dehazer  # noqa: E402
from evaluate_model_detection import dehaze_whole  # noqa: E402

TRANSMITTANCE_ROOT = ROOT / "data" / "transmittance" / "leftImg8bit_transmittance"


def transmittance_path(split: str, city: str, image_id: str, beta: float) -> Path:
    return TRANSMITTANCE_ROOT / split / city / f"{image_id}_leftImg8bit_transmittance_beta_{beta:g}.png"


def render_fog(clear: np.ndarray, t_old: np.ndarray, beta_old: float, beta_new: float) -> np.ndarray:
    """Denser fog from an existing pair: I = J·t + A·(1−t), t = exp(−β·d)."""
    t_new = np.power(np.clip(t_old, 1e-6, 1.0), beta_new / beta_old)
    return np.clip(clear * t_new[:, :, None] + (1.0 - t_new[:, :, None]), 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate every method at rendered heavier fog")
    parser.add_argument("--run-id", default=dt.datetime.now().strftime("regime-%Y%m%d-%H%M%S"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--source-beta", type=float, default=0.02,
                        help="β of the shipped pair used as the depth source (best resolution)")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.03, 0.06, 0.09],
                        help="target fog levels to render (the paper's regime)")
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="name=path for each dehazer to evaluate")
    parser.add_argument("--routed-classifier", type=Path,
                        help="classifier checkpoint; enables the paper-style routed column")
    parser.add_argument("--routed-branches", nargs=3, type=Path, metavar=("LIGHT", "MEDIUM", "HEAVY"))
    parser.add_argument("--skip-dcp", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="scene limit for smoke runs")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    dcp_config = yaml.safe_load((ROOT / "configs" / "traditional" / "dcp.yaml").read_text())["method"]

    pairs = index_pairs(
        ROOT / "data" / "foggy_cityscapes", ROOT / "data" / "cityscapes_clear", args.split
    )
    sources = [p for p in pairs if abs(p.beta - args.source_beta) < 1e-9]
    sources.sort(key=lambda p: (p.city, p.image_id))
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"no pairs at β={args.source_beta} in split {args.split!r}")

    dehazers: dict[str, torch.nn.Module] = {}
    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--checkpoint expects name=path, got {spec!r}")
        dehazers[name], _ = load_dehazer(Path(path), device)
    routed = None
    if args.routed_classifier is not None:
        if not args.routed_branches:
            raise SystemExit("--routed-classifier requires --routed-branches LIGHT MEDIUM HEAVY")
        routed = (
            load_classifier(args.routed_classifier, device),
            [load_dehazer(path, device)[0] for path in args.routed_branches],
        )
    methods = list(dehazers) + (["routed"] if routed else []) + ([] if args.skip_dcp else ["dcp"])

    run_dir = ROOT / "runs" / "machine_learning" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    results: dict[str, dict] = {}
    render_errors: list[float] = []
    rows: dict[str, dict[float, list[dict]]] = {m: {b: [] for b in args.betas} for m in methods}
    started = time.time()
    for target in args.betas:
        for index, pair in enumerate(sources, 1):
            t_path = transmittance_path(pair.split, pair.city, pair.image_id, pair.beta)
            clear = load_rgb(pair.clear_path)
            t_old = np.asarray(Image.open(t_path).convert("L"), dtype=np.float32) / 255.0
            if len(render_errors) < 5:  # validate the render against the shipped image, once per scene
                shipped = load_rgb(pair.foggy_path)
                check = render_fog(clear, t_old, pair.beta, pair.beta)
                render_errors.append(float(np.abs(check - shipped).mean()))
            hazy = render_fog(clear, t_old, pair.beta, target)
            hazy_scores = metrics(hazy, clear)
            scene_id = f"{pair.split}/{pair.city}/{pair.image_id}"
            outputs: dict[str, np.ndarray] = {}
            if "dcp" in methods:
                outputs["dcp"] = dcp_dehaze(
                    hazy,
                    patch_size=dcp_config["patch_size"],
                    omega=dcp_config["omega"],
                    top_fraction=dcp_config["atmospheric_top_fraction"],
                    guided_radius=dcp_config["guided_radius"],
                    guided_epsilon=dcp_config["guided_epsilon"],
                    transmission_floor=dcp_config["transmission_floor"],
                )
            for name, model in dehazers.items():
                outputs[name] = dehaze_whole(model, hazy, device)
            if routed:
                (classifier, classifier_size), branches = routed
                chosen = classify(classifier, hazy, device, size=classifier_size)
                outputs["routed"] = dehaze_whole(branches[chosen], hazy, device)
            for name, output in outputs.items():
                scores = metrics(output, clear)
                rows[name][target].append({
                    "beta": target,
                    "scene_id": scene_id,
                    f"{name}_psnr": scores["psnr"],
                    f"{name}_ssim": scores["ssim"],
                    "hazy_psnr": hazy_scores["psnr"],
                    "hazy_ssim": hazy_scores["ssim"],
                    "delta_psnr": scores["psnr"] - hazy_scores["psnr"],
                    "delta_ssim": scores["ssim"] - hazy_scores["ssim"],
                })
            if index % 50 == 0:
                print(f"β={target}: {index}/{len(sources)} scenes "
                      f"({time.time() - started:.0f}s)", flush=True)
    for name in methods:
        results[name] = summarize(
            [row for target in args.betas for row in rows[name][target]],
            args.resamples, args.seed, condition=name,
        )
    # Fail loudly *before* writing a payload: summarize() groups by row["beta"], so a
    # summary missing a level means the beta loop was mis-structured (as it was once,
    # silently writing only the last level after 110 min of compute).
    expected = {f"{beta:.3f}" for beta in args.betas}
    for name in methods:
        if set(results[name]["by_beta"]) != expected:
            raise SystemExit(
                f"{name}: by_beta incomplete "
                f"({sorted(results[name]['by_beta'])} != {sorted(expected)})"
            )


    payload = {
        "status": "completed_regime_extension",
        "purpose": "Evaluate every method beyond its training fog density, on rendered heavier fog.",
        "split": args.split,
        "source_beta": args.source_beta,
        "target_betas": args.betas,
        "n_scenes": len(sources),
        "render": {
            "model": "atmospheric scattering, I = J·t + A·(1−t), A = 1, t_new = t_old^(β_new/β_old)",
            "source": "data/transmittance/leftImg8bit_transmittance (val split)",
            "validation_mean_abs_error_255": float(np.mean(render_errors)) * 255 if render_errors else None,
            "validation_note": "re-rendering the shipped β reproduces the shipped image; see validation_mean_abs_error_255",
        },
        "results": results,
        "provenance": {
            "git_revision": None,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "checkpoints": args.checkpoint,
            "normalized_context_size": list(CONTEXT_SIZE),
            "resamples": args.resamples,
            "seed": args.seed,
            "dcp_config_sha256": hashlib.sha256(
                (ROOT / "configs" / "traditional" / "dcp.yaml").read_bytes()
            ).hexdigest(),
        },
    }
    result_path = ROOT / "results" / "machine_learning" / f"{args.run_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nrender validation: {np.mean(render_errors) * 255:.2f}/255 mean abs error vs shipped fog")
    for name, block in results.items():
        parts = []
        for beta in args.betas:
            entry = block["by_beta"][f"{beta:.3f}"]
            parts.append(f"β={beta:g}: {entry[f'{name}_psnr']:.2f} dB ({entry[f'{name}_minus_hazy_psnr']['mean']:+.2f})")
        print(f"{name:>8}: " + " | ".join(parts))
    print(f"summary: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
