# Validation comparison — Foggy Cityscapes val

Generated 2026-09-24; method 2 (gated U-Net) added 2026-09-27; seeds,
spatial gate, routed-on-RTTS-valid, and the hybrid negative result added
2026-10-02. Every number comes from the shared evaluation path
(`src/ip_dehaze/evaluation/{metrics,pair_benchmark}.py`), scene-clustered
bootstrap with 2,000 resamples, seed 1729. RTTS test was scored once,
post-freeze (finding 12, step 5 / P4.1): all claims now re-runnable under
the frozen protocol.

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
| Gated U-Net, spatial gate (`--gate-spatial`) | 14.65 | **30.40** | +15.75 [15.46, 16.03] | +0.1468 [0.1434, 0.1500] |
| Gated U-Net, context gate (seed 43) | 14.65 | 29.87 | +15.22 [14.95, 15.49] | +0.1426 [0.1392, 0.1458] |
| Gated U-Net, context gate (seed 7) | 14.65 | 29.95 | +15.30 [15.02, 15.57] | +0.1427 [0.1394, 0.1459] |
| Hybrid, DCP-transmission 4th channel | 14.65 | 25.49 | +10.84 [10.66, 11.01] | +0.1243 [0.1217, 0.1267] |

(The seed rows and the hybrid are context-gate variants; the spatial gate is
the same backbone with a per-pixel gate head. `post-chain-analysis-2026-10-02.json`
holds all rows in one place.)

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
   noise (overlapping CIs) and must not be claimed as a win. This parity is a
   *within-regime* result: at densities above the training range the gate loses
   to the plain model (finding 8).

7. **The paired-data advantage reverses on real fog.** On the 432-image RTTS
   test split with the frozen COCO YOLOv8n (the paper's own headline metric,
   paired conditions per image), DCP is the *only* method that helps detection:
   AP@0.5 hazy 0.4923 → DCP 0.5194 (+0.027), while both learned models hurt —
   blind 0.4786 (−0.014), gated (whole-scene) 0.4740 (−0.018). The failure is
   the restoration itself, not the gate: blind and gated degrade equally
   within noise, and the gate's per-image ordering is correct in-domain. Its
   failure mode looks like mild **over-removal**: on real RTTS images the gate
   reads heavy fog (`d` ≈ 0.87 on every image probed) and the shared head drives
   output to 0.449× the input luminance on average (per-image 0.21–0.55 over 20
   images) against the 0.48–0.52 that our own well-restored renders produce —
   about 8% darker than a healthy restoration in the mean, with a long dark tail.
   The ratio alone is weak evidence, since a correct restoration is itself about
   half the luminance of its hazy input; the detection loss is the stronger
   signal, and it too is modest (−0.018 AP) rather than catastrophic.
   The training regime (synthetic β ≤ 0.02, atmospheric light exactly white)
   never covers real fog. Two *suggestive but confounded* measurements point at
   the gap being narrower than assumed: RTTS's atmospheric light is neutral to
   within 1% (A ≈ (0.902, 0.895, 0.887), tint (1.008, 1.000, 0.991)), so it is
   not the grey/blue cast this report first hypothesized; and both RTTS's global
   luminance (0.507) and its dark-channel haze lift (0.303) fall *below* our
   β = 0.02 render (0.592 / 0.435) — consistent with real fog there being
   milder, not denser, than our training fog. Neither comparison is independent
   evidence and they do not support each other: RTTS's clear-scene content is
   unknown, and content differences alone move both statistics (our own *clear*
   images score the highest haze-lift spread of any set, 0.709, precisely
   because haze compresses range). The unconfounded statements are the
   within-scene results in finding 8. In-domain PSNR and real-fog detection are
   different questions; the honest summary is that our +2.44 dB pooled advantage
   does not transfer, and the axes this leaves open are taken up in finding 8.

8. **Out of regime, the restoration is density-robust and our gate is not.**
   Re-rendering the val split at the paper's fog levels from the shipped
   transmittance maps — homogeneous fog, atmospheric light exactly 1, so
   *only* density changes — 500 scenes, ΔPSNR vs hazy with clustered 95% CIs:

   | method | β=0.03 | β=0.06 | β=0.09 | pooled |
   |---|---|---|---|---|
   | blind | 25.12 (+16.55) | 18.35 (+12.10) | 14.14 (+8.89) | +12.51 [12.27, 12.76] |
   | routed (paper) | 24.34 (+15.77) | 18.53 (+12.27) | 15.00 (+9.75) | +12.60 [12.37, 12.82] |
   | gated (ours) | 24.42 (+15.85) | 16.62 (+10.36) | 12.58 (+7.33) | +11.18 [10.93, 11.44] |
   | dcp | 14.51 (+5.94) | 11.20 (+4.94) | 9.08 (+3.83) | +4.90 [4.75, 5.07] |

   (hazy 8.57 / 6.26 / 5.25 dB; render validation 1.77/255 mean absolute error
   against the shipped fog, so the levels really are the ones claimed.) Every
   learned model still gains 7.3–9.8 dB at 4.5× the training density: the
   *restoration* extrapolates in density, and DCP — the only method that helps
   detection on RTTS — is by far the weakest here. What does not extrapolate is
   our gate's edge. In regime the gated variant edged blind (+0.17 dB,
   overlapping CIs) and beat the paper's design by 2.44 dB; out of regime it is
   the weakest learned column, losing to blind at all three levels (pooled
   −1.33 dB), while the routed system edges *ahead* of blind at the heavy
   levels. The ordering reverses, so the "parity with a single model" claim in
   finding 6 is a within-regime statement.

   **Mechanism: the blend is right, the head is the weak term.** The gate's `d`
   is flat at ≈0.87 across densities (0.877 / 0.869 / 0.865 at β = 0.02 / 0.06 /
   0.09) and sits within 0.004 of the oracle scalar blend for the model's *own*
   head at every level (0.865 / 0.868 / 0.863), so out-of-regime mis-calibration
   is not the cause — the gate already reports the level a scalar blend should
   use, and a resynthesis that only extends the β range has nothing to fix there.
   Two further arms on the same scenes isolate where the loss sits (dB):

   | β | blind | gated head alone (`d ≡ 1`) | gated blend (natural) |
   |---|---|---|---|
   | 0.02 | 28.09 | 24.86 | 28.02 |
   | 0.06 | 18.81 | 15.98 | 16.99 |
   | 0.09 | 13.41 | 11.76 | 12.38 |

   The head standing alone is 1.6–3.2 dB *below* the plain model at *every*
   density, so this is not a head that decays out of regime — it is a weaker
   restoration head throughout, having only ever been trained as one factor of a
   blend. What changes at the regime boundary is what the blend can absorb: in
   regime it recovers the entire 3.2 dB deficit (parity, −0.07 dB), beyond it
   only part (1.0 of 2.8 dB at β = 0.06; 0.6 of 1.7 dB at β = 0.09). The
   out-of-regime deficit against the plain model is that unabsorbed remainder —
   which is also why forcing `d ≡ 1` scores *worse* than the natural blend: the
   blend is doing real work, just not enough of it.

   *Diagnostic detail:* 4 scenes on CPU, every arm on the same scenes with the
   same whole-scene protocol; two independent code paths agree exactly on the
   head column (blend inversion vs a forced-`d ≡ 1` forward, both 11.76 at
   β = 0.09), and the small-sample columns track the 500-scene table above
   (blend 16.99 vs 16.62 at β = 0.06, 12.38 vs 12.58 at β = 0.09; `d` = 0.877 at
   β = 0.02 against the documented in-regime 0.87–0.88).

9. **What the RTTS failure is not.** Four probes, each measured rather than
   inferred, remove every fog-physics axis we can vary:

   | probe | measurement | removes |
   |---|---|---|
   | density to 4.5× training β (val, 500 scenes) | learned models still gain +7.3…+9.8 dB; gate `d` within 0.004 of oracle | density-range mismatch |
   | RTTS atmospheric light | neutral to ±1%, A ≈ (0.902, 0.895, 0.887) | colour cast |
   | model on fog-free real photos | 29.73 dB (blind) / 31.08 dB (gated) vs identity, lum ratio 0.93 / 0.94 | blanket domain shift |
   | patchy-β and patchy-A renders (6 scenes, mean preserved) | both hurt learned models ~3× more than DCP (β: −0.77 vs −0.29; A: −4.19 vs −1.37 dB) but **neither reproduces RTTS's signature** | spatial fog variation as the mechanism |

   The last row's second half matters as much as its first: sensitivity to
   spatial fog variation is real and asymmetric — the learned restoration
   degrades ~3× more than the classical prior under identical perturbation — but
   it fails in the *opposite direction* to RTTS. The patchy-A arm, the most
   damaging perturbation tested, *under*-removes (luminance ratio rises to
   0.54/0.57 from 0.48/0.52, i.e. it leaves fog in), whereas RTTS over-removes.
   A perturbation that breaks the model the wrong way does not explain the
   failure. The observational counterpart agrees: RTTS's haze-lift spread (0.505)
   sits *below* what our own β = 0.02 renders predict at the same lift (≈0.57),
   so there is no sign of real fog carrying anomalous spatial structure.

   The residual explanation is **content/scene-domain shift** — the learned
   inverse physics is tuned to synthetic-fog-on-Cityscapes statistics and
   overshoots on a different real-world content distribution. A fifth probe
   leans that way without settling it: given DCP-dehazed RTTS imagery — the same
   scenes with most of the fog removed — the model still applies substantial
   restoration (18.82 dB against its own input, luminance ratio 0.742), whereas
   on fog-free Cityscapes photos it preserved the input (29.73 / 31.08 dB,
   ratio 0.93 / 0.94). That is consistent with the model reading RTTS scene
   statistics as fog. It is not conclusive: DCP's output keeps residual haze and
   carries its own artifacts (halos, saturation), so part of that response may be
   legitimate restoration of what DCP left behind. Making it conclusive needs
   paired real-fog data with clean references, which RTTS does not provide —
   hence "inference by elimination", not a demonstrated mechanism. The practical
   consequence is that the RTTS gap is a **transferability limitation rather than
   a resynthesis target**: the axes a heavier-β or tinted-A resynthesis would
   tune are precisely the ones measured here and found not to bind.

10. **Seed variance: the parity claim survives, reworded.** Three seeds of the
    context-gated variant (1729 / 43 / 7, identical config apart from the seed,
    all with the crop-rotation fix) score 29.999 / 29.872 / 29.947 dB pooled —
    spread **0.127 dB**. The pre-registered "parity within noise" wording must
    become "parity within seed spread (±0.13 dB)": seed noise is not negligible
    at this effect size. The context gate's +0.54 dB edge over the blind model
    is 4× the seed spread and survives (`post-chain-analysis-2026-10-02.json`,
    `seed_variance` block).

11. **The spatial gate is a genuine improvement, and the mask probe proves the
    mechanism.** Training the gate per-pixel (`--gate-spatial`, one scalar per
    pixel instead of one per image) reaches **30.40 dB pooled, +0.40 dB over
    the context gate and +0.94 over the blind model** — beyond seed spread
    (0.127 dB), consistent at every density (+0.29 / +0.38 / +0.53 dB at
    light / medium / heavy). The mask-collapse diagnostic was run *before*
    trusting the number, and it passes: pooled mask mean 0.550, std **0.272**,
    12.5% of pixels > 0.9, 5.7% < 0.1 — the mask varies within images at every
    density (per-density rows in `outputs/logs/spatial-mask-probe.log`), so
    the gain is a real spatial adaptivity signal, not a collapsed constant.
    This is the only adaptivity variant that improved over a plain model
    in-domain.

12. **The paper's routed design is the *worst* learned system on real fog.**
    On RTTS-valid (865 images, frozen COCO YOLOv8n, whole-image protocol):
    hazy 0.4527, DCP 0.4706 (**+0.018, only method that helps**), ours
    (context-gated) 0.4451 (−0.008), routed (learned routing) **0.4415**
    (−0.011). The routed column closes the fairness gap in finding 7: the
    failure on real fog is not specific to our gate — every learned variant
    hurts while the classical prior helps, so the content/scene-domain
    transferability limitation (finding 9) applies to the paper's own design
    as well.
    **The single post-freeze test score (step 5, P4.1) confirms it
    on the untouched test split**: the valid-selected winner (spatial-gated
    U-Net, `spatial-rtts-test-frozen-20261002.json`) scores **0.4850 AP@0.5**
    vs hazy 0.4923 (−0.007) and DCP 0.5194 (+0.027). The paper's design was
    then scored on the same frozen test split (`routed-rtts-test-frozen-20261002.json`):
    **0.4789 (−0.0134)** — the weakest learned system on test as on valid,
    so the ordering is symmetric across both RTTS splits. The learned model
    still trails doing nothing; DCP remains the only method that helps. The
    honest conclusion is the transferability limitation, stated with the
    frozen split's numbers for both methods.

    **Bootstrapped over images** (400 resamples,
    `detection-bootstrap-rtts-test-bootstrap-20261002.json`), the frozen
    test deltas are:

    | delta vs hazy | mean | CI95 | excludes 0 |
    |---|---|---|---|
    | DCP | **+0.0273** | [0.0124, 0.0424] | **yes** |
    | ours | −0.0076 | [−0.0207, 0.0086] | no |
    | routed | −0.0135 | [−0.0308, 0.0040] | no |

    So the defensible statement is **"no learned method improves detection,
    and their deficits are not statistically distinguishable from zero"** —
    not "the learned models are significantly worse". DCP's gain is the only
    one the data separates from no-op.

13. **The DCP-transmission hybrid is a decisive negative.** Feeding DCP's
    per-image transmission estimate as a 4th input channel (the "physical
    prior as input feature" hypothesis, step 2 of P3.7) trains fine but scores
    **25.49 dB pooled — 4.51 dB *below* the same-gate baseline (29.999)** on
    full val, failing the pre-registered acceptance (≥ 29.4) by ~3.9 dB. The
    gate itself behaves (d mean 0.63, not pinned), so the deficit is the
    channel degrading the learned representation, not gate misuse. Combined
    with finding 12, the mechanism evidence now says: DCP helps RTTS only as
    a *per-image estimator applied to the image itself* (classical DCP), and
    its estimate helps neither as a residual base nor as an input channel.
    The negative result closes the cheap physical-prior lever and motivates
    the remaining one: task-driven adaptation on RTTS-train (approved,
    `docs/decisions/2026-10-01-rtts-train-adaptation.md`), which targets the
    metric directly instead of importing physics.

14. **Task-driven adaptation moves real-fog detection the right way, but not
    far enough.** The approved track (P3.7 step 3,
    `docs/decisions/2026-10-01-rtts-train-adaptation.md`): fine-tune the
    spatial-gated winner with the frozen COCO YOLOv8n in the loss — maximize
    detector confidence on GT-matching anchors — mixed with synthetic
    reconstruction batches (L1+SSIM) to hold restoration. 10 epochs,
    RTTS-train crops + synthetic pairs, selection on `rtts/valid`
    (`adapted-rtts-valid-20261002.json`):

    | RTTS-valid AP@0.5 | value |
    |---|---|
    | hazy (no dehazing) | 0.4527 |
    | DCP | **0.4706** |
    | ours, unadapted (context gate) | 0.4451 |
    | **ours, adapted** | **0.4508** |

    Adaptation is the **only lever measured in this project that moves
    real-fog detection in the right direction** (+0.0057 over the unadapted
    model) — every other learned variant, and the paper's design, sits
    further below hazy. But it still lands **below doing nothing** (−0.0019)
    and far below DCP (−0.0198), so the pre-registered acceptance
    (`valid ≥ DCP-valid`) **fails**. The task arm engaged only late (loss
    0.9999 → 0.9899 by epoch 8, plateauing ~0.99) while reconstruction held
    (~0.034), i.e. the detector signal is weak per epoch at this learning
    rate and weight. **Labelled separately** ("ours + real-fog adaptation");
    never merged into the paper-comparison table, since the paper trains on
    synthetic fog only (protocol asymmetry). The frozen detector is both the
    loss and the metric — a general detection-benefit claim would need a
    second detector.

15. **Real fog over in-domain content — and the atmospheric-light *level*
    as a binding axis.** The failure explanation in finding 9 says the model
    misreads unfamiliar content as fog. The complementary experiment keeps
    content in-distribution (Cityscapes clear) and transplants fog *measured
    from real fog* (RTTS's own DCP transmission, resized to the Cityscapes
    frame, and its own atmospheric light). 60 scenes, paired against the known
    clear reference (`realfog-stats-on-cityscapes-20261003*.json`):

    | arm (Cityscapes content) | restored PSNR | gain vs hazy | lum ratio | fog lift |
    |---|---|---|---|---|
    | shipped fog (training distribution, A=1) | 27.14 | +16.90 | 0.553 | 0.504 |
    | real transmission, density-matched, A=1 | **27.46** | **+17.26** | 0.555 | 0.513 |
    | real transmission, density-matched, **real A** | **23.34** | +11.20 | 0.556 | 0.454 |
    | real transmission at its own density, A=1 | 25.26 | +17.19 | 0.486 | 0.614 |
    | shipped transmission + **real A** | **21.73** | +9.55 | 0.548 | 0.446 |
    | real transmission + real A | 20.01 | +9.98 | 0.466 | 0.530 |

    The 2×2 at **matched density** isolates the two axes cleanly:

    | matched-density arm | restored PSNR | Δ vs control |
    |---|---|---|
    | shipped transmission, A=1 (control) | 27.14 | — |
    | **real** transmission, A=1 | 27.46 | **+0.32** (nothing) |
    | **real** transmission, real A | 23.34 | **−3.80** |
    | shipped transmission, real A | 21.73 | **−5.41** |

    **Two readings.** (a) *Real fog's transmission structure costs nothing*
    when the content is familiar: density-matched, it matches or slightly
    beats the training-distribution control (27.46 vs 27.14 dB). Fog-field
    realism is not what breaks the model. (b) *The atmospheric-light level is
    a binding axis*: at matched density, swapping only `A` to real fog's level
    costs **3.8–5.4 dB** and roughly a third of the restoration gain.

    **Why this was missed.** The earlier probe measured RTTS's atmospheric
    light as "neutral to within 1%" — that was the channel *ratios* (colour
    cast) and remains correct. The absolute *level* is a different axis:

    | domain | A level | channel ratios |
    |---|---|---|
    | Foggy Cityscapes train (training distribution) | 0.929 (sd 0.113) | ~1.00 |
    | Foggy Cityscapes val | 0.960 | ~1.00 |
    | RTTS valid | 0.855 | ~1.00 |
    | RTTS test | **0.872** | ~1.00 |

    Real fog is ~6–7% dimmer in `A` than the fog this model trained on, with
    neither tint nor density accounting for it. This **corrects the P3.6
    recommendation** in `plan.md`: a heavier-β or tinted-A resynthesis still
    has nothing to fix, but an **A-level resynthesis** (render training fog
    with `A` drawn from the real range) is the first resynthesis variant in
    this project with a measured reason to exist.

    *Caveats.* The transplant resizes the transmission map (structure is
    low-frequency, but this is a transplant, not a captured pair); `A` and
    apparent density are entangled in the "shipped transmission + real A" arm
    (its haze lift rises to 0.446 from 0.504), which is why a
    matched-transmission + real-A arm is included above to separate them. No
    in-domain arm reproduces RTTS's full over-removal signature (all ratios
    ≥ 0.466 vs RTTS's 0.449), so content still contributes on top of the A
    level.

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

- The out-of-regime table (finding 8) varies **density only**: the render is
  homogeneous fog with atmospheric light exactly 1, so it isolates β by
  construction but cannot speak to spatially varying fog, tint, or scene
  statistics — the axes that finding 7's measurements leave standing. It is also
  PSNR against a clean reference we generated, whereas the RTTS verdict is
  detection AP without any reference, so the two are not directly comparable.
- The mechanism result in finding 8 rests on a **4-scene CPU diagnostic**, not
  the 500-scene grid, because the training chain owned the GPU. It is cross-checked
  against that grid (blend column within 0.2–0.4 dB, gate level reproducing the
  documented 0.87–0.88) and two independent code paths agree exactly on the head
  column, but small-sample absolutes should not be quoted as results.

- This is a **mechanism reimplementation** of ADAM-Dehaze (U-Net branches
  instead of CORUN, L1+SSIM instead of their 3-term loss, fitted thresholds
  instead of unpublished ones). No reproduction claim — see
  `docs/paper_notes.md` §5 for why the paper is not reproducible as published.
- Fog regimes differ (this project β ∈ {0.005, 0.01, 0.02}; the paper
  β ∈ {0.03, 0.06, 0.09}), so absolute values are not comparable to theirs.
- Per-density branch comparisons use a 10-image selection per β and are noisy;
  the pooled 1,500-image numbers are the reliable ones.
- Detection results (frozen YOLOv8n on RTTS) come from the same eval path as
  the dehazing table but a different protocol (no reference image): RTTS-valid
  865-image columns for all four systems are in finding 12; the RTTS-test
  score is deferred to the single post-freeze run (step 5, P4.1).
- The gated variants' single-run caveat is **resolved 2026-10-02**: three
  seeds of the context gate show 0.127 dB spread (finding 10); the spatial
  gate's +0.40 dB margin over the context gate is 3× that spread. Sub-0.2 dB
  differences between *different* gate variants remain unresolved until each
  has seed replication.
- **Hybrid negative-result caveat.** The hybrid trained with the same loader
  settings and cache conventions as every other row; its val crash (missing
  `dcp_t` in the full-res val set) was fixed by computing the channel
  on-the-fly in validation with the exact `dehaze()` call sequence
  (`485bde1`), i.e. train and eval use the same protocol. The -4.51 dB is
  not an artifact of that fix: the fix was validated on full-res forwards
  before resume, and the resumed run's loss curve continued the pre-crash
  trajectory.
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
  **Measured 2026-09-29 (post-fix retrain, seed 1729, context-gated): 29.999 dB
  pooled vs 29.632 pre-fix — +0.37 dB.** So the bug's cost was real but small,
  and the ordering claim holds; the pre-fix numbers remain usable for method
  comparison, with absolute values understated by ~0.4 dB.
