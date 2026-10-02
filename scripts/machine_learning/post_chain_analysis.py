#!/usr/bin/env python
"""Post-chain readout: seed variance + spatial gate + hybrid comparison.

Run after the follow-up chain completes (chain-summary.txt exists). Pure
consumer over committed result JSONs — no training, no GPU. Verifies artifact
completeness before computing anything (the regime-eval lesson: a partial
write must never look like a finished stage).

Outputs a compact table + the seed-variance CI verdict to stdout and
results/machine_learning/post-chain-analysis-<date>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "machine_learning"


def pooled_gated(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text())
        summary = payload.get("summary", {})
        pooled = summary.get("pooled")
        if pooled is None or "dehazed_psnr" not in pooled:
            return None
        return {
            "pooled_psnr": pooled["dehazed_psnr"],
            "hazy_psnr": pooled.get("hazy_psnr"),
            "status": payload.get("status"),
            "n_images": payload.get("n_images"),
            "by_beta": summary.get("by_beta"),
        }
    except (json.JSONDecodeError, OSError):
        return None


def bootstrap_ci(values: list[float], iterations: int = 10_000, seed: int = 1729) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return (float("nan"), float("nan"))
    means = rng.choice(array, size=(iterations, array.size)).mean(axis=1)
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=RESULTS / "unet-gated-context-full-val-20260929.json",
                        help="the seed-1729 reference run (post-rotation-fix)")
    parser.add_argument("--out", type=Path,
                        default=RESULTS / f"post-chain-analysis-{date.today().isoformat()}.json")
    args = parser.parse_args()

    seed_paths = {seed: RESULTS / f"unet-gated-context-full-val-seed{seed}.json" for seed in (43, 7)}
    baseline = pooled_gated(args.baseline)
    seeds = {seed: pooled_gated(path) for seed, path in seed_paths.items()}
    spatial = pooled_gated(RESULTS / "unet-gated-spatial-full-val-20260929.json")
    hybrid = pooled_gated(RESULTS / "unet-hybrid-dcpt-full-val-20261002.json")

    missing = [name for name, value in
               [("baseline", baseline), ("spatial", spatial), ("hybrid", hybrid)]
               + list(seeds.items()) if value is None]
    print(f"present: baseline={baseline is not None}, seeds={[s for s, v in seeds.items() if v is not None]},"
          f" spatial={spatial is not None}, hybrid={hybrid is not None}")
    if missing:
        print(f"INCOMPLETE — missing artifacts (nothing fabricated): {missing}")
        print("Seed-variance verdict requires baseline + both seeds. Hybrid/spatial rows are optional here.")
        # Still write what exists, marked incomplete, so the consumer is verifiable.
        complete = False
    else:
        complete = True

    out: dict = {"complete": complete, "rows": {}}

    def row(name: str, data: dict | None) -> None:
        out["rows"][name] = None if data is None else {
            "pooled_psnr": round(data["pooled_psnr"], 3),
            "hazy_psnr": None if data["hazy_psnr"] is None else round(data["hazy_psnr"], 3),
            "status": data["status"],
            "n_images": data["n_images"],
        }

    row("baseline_seed1729", baseline)
    for seed, value in seeds.items():
        row(f"seed{seed}", value)
    row("spatial_gate", spatial)
    row("hybrid_dcp_t", hybrid)

    if baseline is not None and all(v is not None for v in seeds.values()):
        values = [baseline["pooled_psnr"]] + [seeds[s]["pooled_psnr"] for s in sorted(seeds)]
        lo, hi = bootstrap_ci(values)
        spread = max(values) - min(values)
        out["seed_variance"] = {
            "values": [round(v, 3) for v in values],
            "bootstrap_ci95": [round(lo, 3), round(hi, 3)],
            "spread_db": round(spread, 3),
            "verdict": ("spread < 0.1 dB: parity claim is defensible"
                        if spread < 0.1 else
                        "spread >= 0.1 dB: seed variance is NOT negligible; parity wording must change"),
        }
        # Does the (small) in-domain edge over blind survive seeds? Baseline vs
        # blind 29.46 -> edge = baseline - 29.46; compare to the seed spread.
        edge = baseline["pooled_psnr"] - 29.46
        out["in_domain_edge_over_blind"] = {
            "edge_db": round(edge, 3),
            "vs_seed_spread_db": round(spread, 3),
            "survives_seeds": edge > spread,
        }

    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
