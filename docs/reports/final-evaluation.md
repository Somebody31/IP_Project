# Final evaluation — frozen protocol (P4.1)

Executed 2026-10-02 in a single pass after P3.7 froze the winner selection.
Two tracks, one run each, no reruns. All claims below re-run under the
frozen protocol only; the test splits are burned for claims.

## Track 1 — Foggy Cityscapes test (dehazing)

`results/machine_learning/spatial-test-frozen-20261002.json` — the
valid-selected winner (spatial-gated U-Net, `unet-gated-spatial-20260929`,
best in-domain 30.40 dB and +0.94 over the blind model), 1,525 test scenes
× 3 β = 4,575 pairs, native resolution, same shared eval path as every val
number (scene-clustered bootstrap, 2,000 resamples, seed 1729).

| β | dehazed PSNR |
|---|---|
| 0.005 (light) | 33.35 |
| 0.010 (medium) | 32.08 |
| 0.020 (heavy) | 29.39 |
| **pooled** | **31.61 dB** (Δ vs hazy +16.82 [16.66, 16.98], ΔSSIM +0.1552 [0.1529, 0.1574]) |

Same split, same protocol, the two non-learned/paper comparators:

| method | pooled test PSNR | Δ vs hazy |
|---|---|---|
| hazy | 14.79 | — |
| DCP | 16.25 | +1.46 [1.31, 1.62] (ΔSSIM −0.125) |
| paper-style routed | 27.55 | +12.76 |
| **ours (spatial gate)** | **31.61** | **+16.82** |

The test number sits 1.2 dB above the val pooled result (30.40) — test
scenes are marginally easier, the method's ordering and per-β pattern are
unchanged, and no hyperparameter was touched between selection and this
score.

## Track 2 — RTTS test (detection)

`results/machine_learning/spatial-rtts-test-frozen-20261002.json` — same
winner, 432-image test split, frozen COCO YOLOv8n, whole-image protocol,
single touch.

| system | AP@0.5 | Δ vs hazy |
|---|---|---|
| hazy (no dehazing) | 0.4923 | — |
| DCP | 0.5194 | **+0.027** — only method that helps |
| **ours (spatial-gated winner, frozen)** | **0.4850** | −0.007 |
| ours + real-fog adaptation (separate track) | 0.4508 (valid) | −0.002 (valid) |
| paper's routed design, frozen | 0.4789 | −0.013 |

The adaptation row is scored on `rtts/valid` (its selection split) and is a
**separate labelled track** — the paper trains on synthetic fog only, so
merging it into the paper-comparison table would be protocol asymmetry.

## Track 1 (paper's method) — Foggy Cityscapes test

`results/machine_learning/routed-test-frozen-20261002.json` — the paper-style
routed system (3-class classifier + three per-class branches), same frozen
protocol, 4,575 test pairs: **pooled 27.55 dB** (29.23 / 27.29 / 26.15) vs
our spatial gate's 31.61 dB. The **+4.06 dB gap is larger on test than on
val (+2.4 dB)**, so the ordering is not a val-split artifact.

## The two-track answer

- **In-domain (synthetic fog, paired references): ours dominates.** +16.82 dB
  over hazy on the frozen test split; the paper-style routed system's best
  case is 2.4 dB below our val result at every density (val finding 5), and
  the spatial gate is the only adaptive variant that *improves* over a plain
  model (finding 11, +0.94 dB over blind, beyond the 0.127 dB seed spread).
- **Real fog (detection, no references): no learned method helps.** Ours
  −0.007 (test) / −0.008 (valid), the paper's design −0.013 (test) / −0.011
  (valid); DCP +0.027/+0.018. The transferability limitation (finding 9:
  content/scene-domain shift, every fog-physics axis measured and ruled out)
  stands on the frozen protocol for **both** methods.
- **Adaptation is the one lever that moves in the right direction**, but not
  far enough: RTTS-train fine-tuning with the frozen detector lifts valid
  AP@0.5 from 0.4451 to 0.4508 (+0.0057) — still below hazy (0.4527) and DCP
  (0.4706), so its pre-registered acceptance fails. Reported as its own
  labelled track.
- The DCP-transmission hybrid (finding 13, −4.51 dB in-domain) and the
  trust gate (measured-not-viable, AUC < 0.6) close the two cheap levers;
  task-driven RTTS-train adaptation (finding 14, +0.0057 but acceptance
  fails) closes the third. Every measured lever is now spent: the honest
  close is that real-fog transfer needs paired real-fog supervision, which
  RTTS does not provide (no clean references) — a different project, not
  another variant of this one.

## Provenance

| artifact | run | revision |
|---|---|---|
| Foggy test dehazing | `spatial-test-frozen-20261002` | `3351e4f`+ |
| Foggy test, paper's method | `routed-test-frozen-20261002` | `3d6ffb0` |
| RTTS test detection (ours) | `spatial-rtts-test-frozen-20261002` | `3351e4f` |
| RTTS test detection (paper's) | `routed-rtts-test-frozen-20261002` | `d18d09a` |
| RTTS valid adaptation track | `adapted-rtts-valid-20261002` | `d18d09a` |
| winner selection | `unet-gated-spatial-full-val-20260929` + post-chain analysis | `e87f669` |
