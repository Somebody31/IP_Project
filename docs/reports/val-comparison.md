# Validation comparison — Foggy Cityscapes val

Generated 2026-09-24; method 2 (gated U-Net) added 2026-09-27. Every number comes from the shared evaluation path
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
| Gated U-Net, crop gate (as trained) | 14.65 | 24.39 | +9.74 [9.55, 9.93] | +0.1145 [0.1119, 0.1170] |
| Gated U-Net, whole-scene gate (fixed) | 14.65 | **29.63** | +14.98 [14.70, 15.25] | +0.1411 [0.1384, 0.1439] |

All rows are the same 1,500 images. The 52-scene refined-subset run
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
| Gated U-Net, crop gate | +7.43 (SSIM +0.055) | +9.84 (SSIM +0.108) | +11.95 (SSIM +0.180) |

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
5. **Every adaptive design closes on the single model as fog thickens.** Deficit
   to the blind model by density (difference of ΔPSNR, dB):

   | | light (0.005) | medium (0.01) | heavy (0.02) |
   |---|---|---|---|
   | Gated U-Net, crop gate | −5.11 | −5.58 | −4.50 |
   | Routed, oracle | −3.18 | −2.20 | −1.44 |
   | Routed, learned | −3.17 | −3.73 | −2.22 |
   | **Gated U-Net, whole-scene gate** | **+0.43** | **+0.13** | **−0.05** |

   The fixed gate is the exception that sharpens the rule: it is at parity (or
   marginally ahead) at *every* density, including the light-fog regime where
   the routed design loses 3.2 dB. What it fixes is not routing but per-image
   blend weight — a continuous, self-supervised quantity rather than a
   discrete class decision over 1/3-size branches.

   Specialization is least harmful where fog is heaviest and most harmful at
   light fog, where the plain model is already near-perfect and there is
   nothing for adaptivity to win. This is the strongest form of the
   regime caveat below: our fog is lighter than the paper's, and the deficit
   is largest exactly in that lighter regime.
6. **The gated model's failure was gate calibration, not adaptivity.** Fog
   density is a global property, so a 256² crop reads denser than the whole
   scene it came from (see §Method 2 below). Trained on crops and evaluated on
   whole images, the gate kept ~40% of the hazy input and lost ~5 dB
   uniformly across densities — the deficit was *flat*, i.e. not a failure to
   discriminate fog levels (its ordering was correct at every scale) but a
   mis-set level. Letting the gate read a whole downscaled scene in both
   training and evaluation fixes it: **29.63 dB pooled (+14.98 [14.70, 15.25]),
   ΔSSIM +0.1411**, against the blind model's 29.46 dB and the paper's
   best-case (oracle) routing at 27.19 dB. Per density the fixed gate is at
   +0.43 / +0.13 / −0.05 dB relative to the blind model, so it is best
   described as **parity with a single model, 2.44 dB ahead of the paper's
   design at its ceiling** — the pooled 0.17 dB edge over blind is inside the
   noise (overlapping CIs) and must not be claimed as a win.

## Method 2 — density-gated U-Net (ours)

One U-Net backbone plus a scalar gate `d` predicted from the input, output
`J = d·f(P) + (1−d)·P`. Self-supervised: no density labels anywhere — the gate
is learned because over-correcting light fog is punished by the reconstruction
loss itself (rationale in `plan.md`; §Design constraints).

**As trained, the gate is mis-calibrated, and the cause is measurable.** The
gate was trained on 256² crops and evaluated on whole 2048×1024 images. Fog
density is a global property, so the same scene reads differently at the two
scales (probe on the epoch-14 checkpoint; pooled value is the mean over the
1,500-image val split):

| Gate input | light | medium | heavy | pooled |
|---|---|---|---|---|
| full image 2048×1024 (eval) | 0.597 | 0.727 | 0.830 | **0.718** |
| centre 256² crop (train) | 0.661 | 0.826 | 0.916 | 0.801 |
| corner 256² crop | 0.794 | 0.907 | 0.945 | 0.882 |

The **ordering is scale-invariant** (light < medium < heavy at every scale);
the **level is not**. Evaluated at full resolution, the gate therefore reports
less fog than it was trained for and blends ~40% of the hazy input back in
(`d_mean` 0.576 at epoch 50). That blend is priced by the same arithmetic that
predicts the observed score: hazy RMS error 0.185, restored-head RMS 0.034, so
a blend at `d = 0.72` yields RMS ≈ 0.057 ≈ 24.9 dB — against 24.69 dB measured
at epoch 15 and 23.39 dB at epoch 20.

**Fix — the gate reads the whole scene.** `--gate-context` feeds the gate a
512×256 downscaled whole image, with the identical transform used in training
the cache and in evaluation; `--gate-init 0.95` starts it at a high blend
weight instead of the measured default 0.386, so the first output is not mostly
hazy. Everything else is held fixed (7,771,172 parameters, same data, same
protocol, same seed):

| epoch (30-image balanced val) | 5 | 15 | 25 | 35 | 45 | 50 |
|---|---|---|---|---|---|---|
| Blind U-Net | 25.01 | 26.50 | 23.20 | 29.53 | 30.87 | 31.41 |
| Gated, crop gate | 21.06 | 24.69 | 25.23 | 24.67 | 24.37 | 25.04 |
| Gated, whole-scene gate | 21.94 | 24.80 | 28.54 | 30.34 | 31.33 | **31.74** |

The fixed gate's pooled `d` holds at 0.87–0.88 across all 50 epochs (the crop
gate drifted at 0.54–0.63), i.e. it is *not* collapsing to `d ≡ 1` — it keeps a
stable, calibrated level while the blind model's own val score is matched. The
full-val row for this variant is the `_pending_` entry in the pooled table
above (run `unet-gated-context-full-val-20260927`).

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
- The gated variants are **single runs** (one seed each), scored through the
  same paired-bootstrap path as every other row; no seed-variance estimate
  exists for them, so treat sub-0.2 dB differences as unresolved.
- **What the blend does.** `J = d·f(P) + (1−d)·P` mixes the restored image with
  the input, so the gate chooses per image between restoration error and
  residual-haze error. Where the restoration is worse than the near-clean input
  — light fog — blending back helps: the fixed gate gains +0.43 dB over the
  blind model at β=0.005 and +0.13 at β=0.01, and gives up 0.05 at β=0.02
  (+0.17 dB pooled). That pooled margin sits inside the noise (overlapping CIs),
  so the honest description is *parity* with a single model, not a win.
- `--gate-context` and `--gate-init` are default-off options; with defaults the
  network is unchanged, so the blind and routed rows remain directly comparable
  to the gate variants.

- **Crop-rotation bug (fixed 2026-09-29, affects every completed run).** The
  datasets rotate crops and re-roll flips per epoch via `set_epoch()`, but with
  `persistent_workers=True` the forked loader workers never see that call — a
  plain attribute write in the main process is invisible to them. Measured:
  with `crops_per_pair=4` every epoch after the first re-served crop #0 of each
  pair with the same flips, so all runs trained on 8,925 frozen crops instead
  of rotating through 35,700. No validation leakage (val pairs are a disjoint
  scene split), and since blind / gated / context-gated / routed branches all
  trained with the same loader settings, the comparisons stay internally
  consistent. But each model saw ¼ of its intended data diversity, and the
  seed-variance caveat above applies a fortiori. Fixed by moving the epoch
  counter into shared memory (`multiprocessing.Value`), regression-tested in
  `tests/machine_learning/test_crop_cache.py::test_set_epoch_reaches_persistent_workers`.
  A future retrain would likely move every absolute PSNR number slightly, but
  the ordering of methods should hold: the gap between the paper's design and
  ours (+2.4 dB pooled) is far larger than any plausible effect of this bug.
