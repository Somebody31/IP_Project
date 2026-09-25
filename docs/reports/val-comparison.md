# Validation comparison — Foggy Cityscapes val

Generated 2026-09-24. Every number comes from the shared evaluation path
(`src/ip_dehaze/evaluation/{metrics,pair_benchmark}.py`), scene-clustered
bootstrap with 2,000 resamples, seed 1729. No test data was touched.

## Setup

- Data: Foggy Cityscapes `val`, 500 scenes × 3 β = 1,500 pairs, evaluated at
  native resolution (2048×1024) against the matching clear Cityscapes image.
- Methods compared: untreated hazy input, DCP (classical prior), blind U-Net
  (one model, all 8,925 training pairs), and the paper-style **routed** system
  (3-class density classifier + three per-class branches, each trained on its
  own 2,975-pair subset).
- Two routing variants for the routed system: **oracle** (true β, the upper
  bound of the design) and **learned** (the trained classifier).

## Pooled results (1,500 images)

| Method | hazy PSNR | dehazed PSNR | ΔPSNR (CI95) | ΔSSIM (CI95) |
|---|---|---|---|---|
| DCP (classical prior) | 14.65 | 15.79 | +1.14 [0.92, 1.33] | −0.1212 [−0.1264, −0.1159] |
| **Blind U-Net** | 14.65 | **29.46** | **+14.81** [14.53, 15.08] | **+0.1414** [0.1380, 0.1446] |
| Routed, oracle routing | 14.65 | 27.19 | +12.54 [12.24, 12.81] | +0.1288 [0.1256, 0.1319] |
| Routed, learned routing | 14.65 | 26.42 | +11.77 [11.48, 12.03] | +0.1233 [0.1198, 0.1264] |

All four rows are the same 1,500 images. The 52-scene refined-subset run
(`results/traditional/dcp-refined-val-20260923.json`, +1.25 dB, SSIM −0.138)
agrees with the full-split result, confirming the refined subset was not
skewing the classical baseline.

## Per-β (ΔPSNR, dB)

| Method | β=0.005 (light) | β=0.01 (medium) | β=0.02 (heavy) |
|---|---|---|---|
| DCP | −2.79 (SSIM −0.219) | +1.47 (SSIM −0.127) | +4.73 (SSIM −0.018) |
| Blind U-Net | +12.55 | +15.42 | +16.45 |
| Routed, oracle | +9.37 | +13.22 | +15.01 |
| Routed, learned | +9.38 | +11.69 | +14.23 |

DCP **degrades** light fog (−2.79 dB, and SSIM collapses by 0.22) and only
becomes useful at heavier densities, where its SSIM damage also shrinks. The
learned models show the opposite pattern — gains *increase* with fog density —
and improve SSIM everywhere.

## Routing behaviour (learned variant)

Routing accuracy on val: **895/1500 = 59.7%**.

| true density | routed light | routed medium | routed heavy |
|---|---|---|---|
| light (β=0.005) | **453** | 32 | 15 |
| medium (β=0.01) | 324 | **82** | 94 |
| heavy (β=0.02) | 60 | 80 | **360** |

The classifier confuses medium with light (324/500) — the dominant error.
Note the trained classifier reached 0.5533 val accuracy on random 256 crops
versus 83% on centre crops, and cropped-input training never learned
(train accuracy stuck ≈0.55) whereas whole-image training climbed immediately
(0.67 → 0.73). Fog density is a global property; a 256-pixel patch often does
not carry it.

## Findings

1. **Learned dehazing dominates the classical prior** on this data: +14.81 dB
   versus +1.25 dB over hazy, with SSIM improving where DCP *degrades* it.
2. **Density routing does not beat a single model.** With *perfect* routing the
   routed system reaches 27.19 dB against 29.46 dB for one model trained on all
   pairs — a 2.27 dB shortfall. Comparing each branch with the single model on
   the *same* density class (10-image selection, noisy): light 28.82 vs 31.11,
   medium 28.96 vs 29.89, heavy 27.61 vs 27.38. Specialization only breaks even
   at the heaviest fog.
3. **Likely cause: per-branch data starvation**, inherent to the design. Each
   branch trains on 2,975 pairs (one β) while the single model sees all 8,925;
   the routing design pays that cost regardless of routing quality. This is a
   fairness caveat of the comparison, not a bug: it is what the paper's design
   implies when applied to this dataset.
4. **Routing quality caps the learned variant**: 59.7% accuracy explains its
   further 0.77 dB drop (27.19 → 26.42 dB).

## Caveats

- This is a **mechanism reimplementation** of ADAM-Dehaze (U-Net branches
  instead of CORUN, L1+SSIM instead of their 3-term loss, fitted thresholds
  instead of unpublished ones). No reproduction claim — see
  `docs/paper_notes.md` §5 for why the paper is not reproducible as published.
- Fog regimes differ (this project β ∈ {0.005, 0.01, 0.02}; the paper
  β ∈ {0.03, 0.06, 0.09}), so absolute values are not comparable to theirs.
- Per-density branch comparisons use a 10-image selection per β and are noisy;
  the pooled 1,500-image numbers are the reliable ones.
- Detection results (frozen YOLOv8n on RTTS) are reported separately in
  `results/traditional/dcp-rtts-*` and are out of scope for this table.
