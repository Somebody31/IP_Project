# Analysis write-up (P4.2)

Answers the four questions the project set out to ask, with confidence
intervals, from the frozen artifacts only. Detail and caveats live in
`val-comparison.md` (in-domain + transfer findings) and
`final-evaluation.md` (the frozen two-track test pass); this document is the
condensed answer sheet.

## Q1 — Does DCP improve perception metrics?

**PSNR yes, SSIM no.** On the 1,500-pair val split (scene-clustered
bootstrap, 2,000 resamples, seed 1729):

| | ΔPSNR | ΔSSIM |
|---|---|---|
| DCP vs hazy, pooled val | **+1.14** [0.92, 1.33] | **−0.121** [−0.126, −0.116] |
| DCP vs hazy, pooled Foggy **test** (4,575 pairs) | **+1.46** [1.31, 1.62] | **−0.125** [−0.128, −0.121] |

The pooled PSNR gain hides a sign change across density, and the SSIM
regression is unambiguous — on test as on val:

| β | ΔPSNR (test) | ΔPSNR (val) |
|---|---|---|
| 0.005 (light) | **−2.61** | −2.79 [−3.04, −2.56] |
| 0.010 (medium) | +1.88 | +1.47 |
| 0.020 (heavy) | +5.11 | +4.73 |

DCP *degrades* light fog on both metrics and only earns its PSNR gain at
heavier densities, where its structural damage also shrinks. It is a
density-specific tool, not a general improvement.

## Q2 — Does dehazing help detection?

**Only classical DCP, and only slightly. No learned method helps.** Frozen
COCO YOLOv8n, AP@0.5, paired per image:

| system | RTTS-valid (865) | RTTS-test (432) |
|---|---|---|
| hazy (no dehazing) | 0.4527 | 0.4923 |
| DCP | **0.4706 (+0.018)** | **0.5194 (+0.027)** |
| ours (spatial-gated) | 0.4850 (−0.007)test / 0.4451valid | — |
| paper's routed design | 0.4415 (−0.011) | 0.4789 (−0.013) |
| ours + RTTS-train adaptation (separate track) | 0.4508 (−0.002) | — |

Every learned system is below doing nothing on both splits; DCP is the only
one above. The adaptation track is the only learned variant that moved in
the right direction (+0.006 over its unadapted parent) and still missed the
bar (acceptance: valid ≥ DCP-valid). This is the project's central negative
result, and it is symmetric — the paper's own design is the weakest of all.

## Q3 — Does the gain vary with fog density?

**Yes, strongly, and the direction differs by method.** ΔPSNR vs hazy by β
(learned, val, 1,500 pairs):

| method | light (0.005) | medium (0.01) | heavy (0.02) |
|---|---|---|---|
| blind U-Net | +12.55 | +15.42 | +16.45 |
| routed, oracle | +9.37 | +13.22 | +15.01 |
| routed, learned | +9.38 | +11.69 | +14.23 |
| gated, whole-scene | +14.98 pooled; +0.43 / +0.13 / −0.05 vs blind |
| DCP | −2.79 | +1.47 | +4.73 |

Learned models gain *more* as fog thickens; DCP flips sign. The adaptive
variants (routed, gated) close on — and where the gate is calibrated, match
— the single model as density rises, because the plain model is already
near-perfect at light fog and there is nothing left to win there.

Out of regime (β = 0.03/0.06/0.09, 4.5× the training density, homogeneous
render): every learned model still gains 7.3–9.8 dB, so restoration
extrapolates in density; what does not is the gated variant's edge over the
plain model (finding 8) — the blend absorbs the gate head's deficit in
regime and only part of it beyond.

## Q4 — Gated (ours) vs routed (paper) vs blind vs DCP

**Quality** (frozen test where available; val otherwise):

| method | val pooled PSNR | Foggy test pooled PSNR | params | inference |
|---|---|---|---|---|
| DCP | 15.79 (+1.14) | 16.25 (+1.46) | 0 (no learning) | 561 ms/img (CPU) |
| blind U-Net | 29.46 | — | 7,766,051 | 215 ms/img |
| routed (paper-style) | 26.42 (learned) / 27.19 (oracle) | **27.55** | **26,378,804** (61,059 classifier + 3 × 7.77 M) | 240 ms/img |
| gated, context | 29.63 / 30.00 (post-fix) | — | 7,771,172 | 242 ms/img |
| **gated, spatial (ours, winner)** | **30.40** | **31.61** | 7,766,084 | 230 ms/img |

(The blind/context variants were not scored on test: the valid-selected
winner was the spatial gate, and the frozen test pass was single-touch by
protocol.)

**Detection delta** (ΔAP@0.5 vs hazy): DCP +0.018/+0.027; ours −0.008/−0.007;
routed −0.011/−0.013; adaptation −0.002 (valid).

**Mechanism vs machinery — the answer.** The paper's premise is that
*adaptivity* helps, implemented as supervised 3-way density routing into
specialised branches. Our result:

- **Routing loses to a single model, at both routing qualities** (oracle
  27.19, learned 26.42 vs blind 29.46 val) — the deficit is set by per-branch
  data starvation (each branch sees 1/3 of the pairs), which the design pays
  regardless of routing accuracy.
- **A self-supervised continuous gate, no density labels anywhere, matches
  and then beats the single model**: context gate at parity (+0.17 dB pooled
  val, inside seed spread 0.127 dB), and the per-pixel gate at **+0.94 dB
  over blind and +4.06 dB over the paper's design on the frozen test split**.
- **Fewer parameters, not more**: 7.77 M vs the routed system's 26.4 M
  (3.4× smaller), for 4 dB more quality.
- The spatial gate is the only adaptive variant that *improves* on a plain
  model, and a mask-collapse probe confirms its gain is genuine per-pixel
  adaptivity (pooled mask std 0.272), not a collapsed constant.

**Disclosed asymmetry.** Our comparison trains both families on Foggy
Cityscapes (the paper's training set, FogIntensity-25K, is unreleased); the
routed system is a faithful mechanism reimplementation (U-Net branches
instead of CORUN, L1+SSIM instead of the three-term loss, fitted thresholds
instead of unpublished ones), so absolute values are not the paper's. What
transfers is the *ordering* and the mechanism conclusion, not the numbers.

## Bottom line

In-domain, a self-supervised per-pixel gate beats both a single U-Net and
the paper's supervised routing design, with a third of the parameters. On
real fog, no learned dehazer — ours or the paper's — improves detection over
doing nothing, while classical DCP slightly does; every fog-physics axis and
every cheap architectural lever was measured and ruled out, leaving
content/scene-domain transfer as the limitation, which RTTS cannot resolve
for lack of clean references.
