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
| **ours (spatial-gated winner)** | **0.4850** | −0.007 |

## The two-track answer

- **In-domain (synthetic fog, paired references): ours dominates.** +16.82 dB
  over hazy on the frozen test split; the paper-style routed system's best
  case is 2.4 dB below our val result at every density (val finding 5), and
  the spatial gate is the only adaptive variant that *improves* over a plain
  model (finding 11, +0.94 dB over blind, beyond the 0.127 dB seed spread).
- **Real fog (detection, no references): no learned method helps.** Ours
  −0.007, the paper's design −0.011 on valid (finding 12); DCP +0.027. The
  transferability limitation (finding 9: content/scene-domain shift, every
  fog-physics axis measured and ruled out) stands on the frozen protocol.
- The DCP-transmission hybrid (finding 13, −4.51 dB in-domain) and the
  trust gate (measured-not-viable, AUC < 0.6) close the two cheap levers;
  the task-driven RTTS-train adaptation (approved,
  `docs/decisions/2026-10-01-rtts-train-adaptation.md`) is the remaining
  lever and runs as a separate labelled track.

## Provenance

| artifact | run | revision |
|---|---|---|
| Foggy test dehazing | `spatial-test-frozen-20261002` | `3351e4f`+ |
| RTTS test detection | `spatial-rtts-test-frozen-20261002` | `3351e4f` |
| winner selection | `unet-gated-spatial-full-val-20260929` + post-chain analysis | `e87f669` |
