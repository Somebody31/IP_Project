"""Shared paired-benchmark summary code.

Extracted so the traditional (DCP) and machine-learning tracks produce
identical result schemas: pooled and per-β means plus scene-clustered
bootstrap intervals on the paired difference against the untreated hazy
input. Behaviour must not change — the traditional run records depend on it.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def clustered_interval(rows: list[dict], field: str, resamples: int, seed: int) -> dict:
    """95% interval for a per-image field, resampling whole scenes."""
    by_scene: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_scene[row["scene_id"]].append(row[field])
    scene_values = np.asarray([np.mean(v) for v in by_scene.values()], dtype=np.float64)
    if not len(scene_values):
        raise ValueError("Cannot bootstrap an empty result set")
    rng = np.random.default_rng(seed)
    samples = rng.choice(scene_values, size=(resamples, len(scene_values)), replace=True).mean(axis=1)
    return {
        "mean": float(scene_values.mean()),
        "ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
        "n_scenes": int(len(scene_values)),
        "resamples": resamples,
        "seed": seed,
    }


def summarize(rows: list[dict], resamples: int, seed: int, condition: str = "dcp") -> dict:
    """Pooled + per-β summary. `condition` names the processed columns in `rows`."""
    by_beta: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_beta[f"{row['beta']:.3f}"].append(row)

    def condition_summary(selected: list[dict]) -> dict:
        result = {
            "n_images": len(selected),
            "n_scenes": len({row["scene_id"] for row in selected}),
            "clear_reference": {"identity": True, "psnr": "infinity", "ssim": 1.0},
        }
        for name in ("hazy", condition):
            for metric in ("psnr", "ssim"):
                key = f"{name}_{metric}"
                result[key] = float(np.mean([row[key] for row in selected]))
        for metric in ("psnr", "ssim"):
            result[f"{condition}_minus_hazy_{metric}"] = clustered_interval(
                selected, f"delta_{metric}", resamples, seed
            )
        return result

    return {
        "pooled": condition_summary(rows),
        "by_beta": {beta: condition_summary(values) for beta, values in sorted(by_beta.items())},
    }
