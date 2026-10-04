#!/usr/bin/env python
"""Accurate inference latency & memory benchmark for Our Method vs Reference Paper.

Measures:
  1. Our Unified Model (7.8M params)
  2. Paper's Density Classifier (1.2M params)
  3. Paper's Dehazing Branches (8.4M params each)
  4. Paper's End-to-End Routed Pipeline (Classifier + Routed Branch)

Across:
  - Standard Perception Resolution (640x480, typical RTTS scene)
  - Native Automotive Resolution (2048x1024, Foggy Cityscapes)

Uses torch.cuda.Event for hardware-level GPU microsecond timing after warm-up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/machine_learning"))

from benchmark_model import load_classifier, load_dehazer


def time_model(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup: int = 15,
    iterations: int = 50,
) -> dict[str, float]:
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    # Timing with CUDA Events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(iterations):
            start_event.record()
            _ = model(input_tensor)
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))  # ms

    timings = np.asarray(timings, dtype=np.float64)
    return {
        "mean_ms": float(timings.mean()),
        "median_ms": float(np.median(timings)),
        "std_ms": float(timings.std()),
        "p95_ms": float(np.percentile(timings, 95)),
        "min_ms": float(timings.min()),
        "max_ms": float(timings.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA device required for accurate GPU latency benchmarking.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Benchmarking on {gpu_name} ({args.iterations} iterations, {args.warmup} warmups)...")

    # Load models
    our_ckpt = Path("outputs/checkpoints/adapt-v3-resynth-w0.01-20261003/best.pt")
    clf_ckpt = Path("outputs/checkpoints/density-classifier-20260924/best.pt")
    branch_light_ckpt = Path("outputs/checkpoints/branch-light-20260924/best.pt")
    branch_med_ckpt = Path("outputs/checkpoints/branch-medium-20260924/best.pt")
    branch_heavy_ckpt = Path("outputs/checkpoints/branch-heavy-20260924/best.pt")

    our_model, _ = load_dehazer(our_ckpt, device)
    clf_model, clf_size = load_classifier(clf_ckpt, device)
    branch_light, _ = load_dehazer(branch_light_ckpt, device)
    branch_med, _ = load_dehazer(branch_med_ckpt, device)
    branch_heavy, _ = load_dehazer(branch_heavy_ckpt, device)

    resolutions = [
        ("RTTS Typical (640x480)", 640, 480),
        ("Foggy Cityscapes (2048x1024)", 2048, 1024),
    ]

    results = {
        "device": gpu_name,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "resolutions": {},
    }

    for res_name, width, height in resolutions:
        print(f"\n==========================================")
        print(f"Resolution: {res_name} ({width}x{height})")
        print(f"==========================================")
        dummy_img = torch.randn(1, 3, height, width, device=device)
        clf_input = (
            torch.randn(1, 3, clf_size[1], clf_size[0], device=device)
            if clf_size is not None
            else dummy_img
        )

        torch.cuda.reset_peak_memory_stats()
        t_our = time_model(our_model, dummy_img, args.warmup, args.iterations)
        vram_our = torch.cuda.max_memory_allocated() / (1024 * 1024)

        torch.cuda.reset_peak_memory_stats()
        t_clf = time_model(clf_model, clf_input, args.warmup, args.iterations)
        t_light = time_model(branch_light, dummy_img, args.warmup, args.iterations)
        t_med = time_model(branch_med, dummy_img, args.warmup, args.iterations)
        t_heavy = time_model(branch_heavy, dummy_img, args.warmup, args.iterations)
        vram_paper = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Average branch latency
        avg_branch_ms = (t_light["mean_ms"] + t_med["mean_ms"] + t_heavy["mean_ms"]) / 3.0
        paper_pipeline_ms = t_clf["mean_ms"] + avg_branch_ms

        results["resolutions"][res_name] = {
            "our_model": {**t_our, "peak_vram_mb": vram_our},
            "paper_classifier": t_clf,
            "paper_branches_avg_ms": avg_branch_ms,
            "paper_pipeline_total_ms": paper_pipeline_ms,
            "paper_peak_vram_mb": vram_paper,
            "speedup_vs_paper": paper_pipeline_ms / max(t_our["mean_ms"], 1e-6),
        }

        print(f"{'Method / Component':<32} {'Mean Latency':>15} {'Median':>12} {'Std':>10}")
        print("-" * 72)
        print(f"{'Our Unified Model':<32} {t_our['mean_ms']:>12.2f} ms {t_our['median_ms']:>9.2f} ms {t_our['std_ms']:>7.2f} ms")
        print(f"{'Paper: Classifier':<32} {t_clf['mean_ms']:>12.2f} ms {t_clf['median_ms']:>9.2f} ms {t_clf['std_ms']:>7.2f} ms")
        print(f"{'Paper: Dehazer Branch (Avg)':<32} {avg_branch_ms:>12.2f} ms")
        print(f"{'Paper: Full Pipeline (Clf+Branch)':<32} {paper_pipeline_ms:>12.2f} ms")
        print(f"Speedup of Our Method: {paper_pipeline_ms / t_our['mean_ms']:.2f}x faster")
        print(f"Peak VRAM during inference: Ours {vram_our:.1f} MB vs Paper {vram_paper:.1f} MB")

    out_file = args.out or (ROOT / "results" / "machine_learning" / "latency_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved latency benchmark results to {out_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
