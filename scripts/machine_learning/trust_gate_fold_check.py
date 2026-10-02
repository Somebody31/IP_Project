"""Trust-gate fold check: does ANY existing per-image signal separate
'ours worsened detection' from 'ours helped' on RTTS?

Signals tested (all already computed, no GPU):
  - probe-rtts-damage rows (per-image AP50 hazy/ours/dcp + input stats)
  - gate/density estimates if present in full-val artifacts

Verdict criterion: an abstention rule needs a signal that flags the
worsened images with high recall at reasonable precision. We report the
best achievable (precision, recall) over each signal's ROC sweep, plus
AUC. AUC <= 0.6 means 'no viable signal' -> trust gate closes as
measured-not-viable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import math

rows = [json.loads(line) for line in
        (ROOT / "runs/machine_learning/probe-rtts-damage/rows.jsonl").read_text().splitlines()]
rows = [r for r in rows if not (math.isnan(r["ap50_ours"]) or math.isnan(r["ap50_hazy"]))]
print(f"rows: {len(rows)} (NaN-AP rows dropped)")

# Harm definition: ours' per-image AP50 delta vs hazy. Note per-image AP on a
# single image is AP over that image's detections (0/1-ish resolution);
# delta < 0 = ours hurt that image.
signals = {}
deltas = []
for r in rows:
    deltas.append(r["ap50_ours"] - r["ap50_hazy"])
    for key in ("lum_mean", "lum_std", "dark_mean", "dark_p90", "saturation_mean"):
        signals.setdefault(key, []).append(r[key])
    # transmission proxy: dark channel (inversely related to transmission)
    signals.setdefault("neg_dark_mean", []).append(-r["dark_mean"])

harm = [d < 0 for d in deltas]
n_harm = sum(harm)
print(f"worsened: {n_harm}/{len(rows)} ({n_harm/len(rows):.1%}) | mean delta {sum(deltas)/len(deltas):+.4f}")


def auc(scores: list[float], labels: list[bool]) -> float:
    """Rank AUC (Mann-Whitney). Labels True = positive = 'will harm'."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def best_operating_point(scores: list[float], labels: list[bool]) -> tuple[float, float, float]:
    """Sweep thresholds; return (best F1, precision, recall) with recall >= 0.5."""
    pairs = sorted(zip(scores, labels), key=lambda t: -t[0])
    total_pos = sum(labels)
    best = (0.0, 0.0, 0.0)
    tp = 0
    fp = 0
    for idx, (_, lab) in enumerate(pairs, 1):
        if lab:
            tp += 1
        else:
            fp += 1
        precision = tp / idx
        recall = tp / total_pos
        if recall >= 0.5:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            if f1 > best[0]:
                best = (f1, precision, recall)
    return best


print(f"\n{'signal':<16} {'AUC':>6} {'best-F1':>8} {'prec':>6} {'rec':>6}")
verdict_viable = False
for name, values in sorted(signals.items()):
    a = auc(values, harm)
    f1, p, rec = best_operating_point(values, harm)
    flag = " <-- viable" if a > 0.6 else ""
    if a > 0.6:
        verdict_viable = True
    print(f"{name:<16} {a:>6.3f} {f1:>8.3f} {p:>6.3f} {rec:>6.3f}{flag}")

# Also: does OUR OUTPUT's PSNR-vs-input (available for full-val, not RTTS)
# correlate? Not computable here - RTTS has no clean reference. Note it.
print("\nnote: PSNR-vs-input is not computable on RTTS (no clean reference); "
      "gate-value d per image is not in the probe rows - the probe's signals "
      "are input statistics only.")

print("\nVERDICT:", "a signal separates harm -> trust gate viable as abstention rule"
      if verdict_viable else
      "no signal reaches AUC > 0.6 -> trust gate closes as measured-not-viable "
      "(harm not predictable from any computed per-image statistic)")
