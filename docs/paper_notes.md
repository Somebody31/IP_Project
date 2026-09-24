# Reference paper notes — ADAM-Dehaze

Source: `assets/2506.15837v1.pdf` (arXiv:2506.15837v1, 8 pages + references).
Verified by reading the extracted full text on 2026-09-24. Every claim below
carries a page/section reference into the source PDF.

---

## 1. Bibliographic record

- **Title:** ADAM-Dehaze: Adaptive Density-Aware Multi-Stage Dehazing for
  Improved Object Detection in Foggy Conditions (p. 1, title block)
- **Authors:** Fatmah AlHindaassi¹*, Mohammed Talha Alam¹*, Fakhri Karray¹
  (* equal contribution) (p. 1, title block + footnote)
- **Affiliation:** Mohamed bin Zayed University of Artificial Intelligence,
  UAE (p. 1, footnote 1)
- **Version:** arXiv v1, dated 16 Jun 2025 (p. 1, sidebar stamp)
- **Citation:** AlHindaassi, F., Alam, M. T., & Karray, F. (2025).
  ADAM-Dehaze: Adaptive Density-Aware Multi-Stage Dehazing for Improved
  Object Detection in Foggy Conditions. arXiv:2506.15837v1 [cs.CV].

## 2. Problem statement and claimed contributions

**Problem.** Fog degrades images (contrast loss, detail obscuring, color
distortion), and detectors trained on clear weather miss objects or fire
false alarms in fog (§I, p. 1). Two specific gaps the paper targets:

1. Existing dehazing is "one-size-fits-all": a fixed-capacity network is
   applied regardless of how foggy the image is — wasted compute on light
   fog, under-processing on heavy fog (§I, p. 1).
2. Classic priors like DCP partially restore clarity but leave residual
   haze or halos, especially in dense fog (§I, p. 1; §II, p. 1–2).

**Claimed contributions (§I, "contributions" list, p. 1):**

1. **FogIntensity-25K dataset** — 25,000 synthetic paired hazy/clear images,
   stratified into light/medium/heavy fog via the atmospheric scattering
   model, built from Cityscapes and Synscapes with depth-based fog
   simulation (§III-A1, p. 2; Table I, p. 2).
2. **Adaptive dehazing networks** — three CORUN variants (2 / 4 / 6 stages,
   the 6-stage one adding attention) specialized for light / medium / heavy
   fog (§I contributions; §III-B2, p. 2).
3. **Modulated loss function** — a fog-intensity–weighted combination of a
   physical-coherence term, a perceptual term, and a density term (§III-B3,
   p. 2).
4. **Improved detection under fog** — up to +13 mAP on real-world foggy
   images with 20% faster inference (§I contributions, p. 1; abstract, p. 1).

## 3. Methods, with equations and implementation details

### 3.1 Physics background (needed to read the rest)

The atmospheric scattering model (ASM) the paper uses (Eq. 1, §III-A1, p. 2):

    P(x) = J(x)·t(x) + A·(1 − t(x))

- `P(x)` — observed hazy image, `J(x)` — clear scene radiance,
  `A` — global atmospheric light, `t(x)` — transmission map.
- Transmission follows Beer–Lambert (Eq. 2): `t(x) = exp(−β·d(x))` where
  `d(x)` is scene depth and β is the fog density coefficient.
- β = 0.03 / 0.06 / 0.09 label light / medium / heavy fog throughout the
  paper (§III-A1, p. 2; Fig. 2 caption, p. 3).

Beginner note: this is the same model our DCP baseline inverts — DCP
estimates `t(x)` and `A` from the image itself, then recovers
`J = (P − A(1−t))/max(t, t₀)`.

### 3.2 FogIntensity-25K dataset (§III-A1, Table I, p. 2)

Synthesized with the ASM over **Cityscapes and Synscapes**, using depth maps
so distant objects fade more. Composition:

| Fog level | β    | #Images | Description                  |
|-----------|------|---------|------------------------------|
| Light     | 0.03 | 8,333   | Mild degradation, visibility preserved |
| Medium    | 0.06 | 8,333   | Moderate degradation         |
| Heavy     | 0.09 | 8,334   | Severe visibility loss       |

Enables full-reference evaluation (PSNR, SSIM, LPIPS). Real-world RTTS
(~4,000 images, §III-A2, p. 2) has no clear ground truth, so only
non-reference metrics are used there: FADE, BRISQUE, NIMA (§III-A2, p. 2).

### 3.3 Pipeline (§III-B, p. 2; Fig. 2, p. 3; Algorithm 1, p. 3)

Step 1 — **Haze Density Estimation Network (HDEN)** maps an image to a
continuous density score `d = D_θD(P) ∈ [0,1]` (Eq. 3, p. 2).

Step 2 — **Branch routing** via two thresholds α and β (Eq. 4, p. 2):

    J_out = f_Light(P)    if d < α
          = f_Medium(P)   if α ≤ d ≤ β
          = f_Complex(P)  if d > β

- Light: 2-stage unfolding, for mildly foggy scenes.
- Medium: 4-stage unfolding, balances detail vs. cost.
- Complex: 6-stage unfolding **plus transformer-based attention** for dense
  fog (§III-B2, p. 2).

The branches are CORUN-style cooperative unfolding networks — "deep
unfolding" embeds the iterative optimization steps of a model-based dehazing
solver into network layers (§II "Deep Unfolding", p. 2; CORUN cited as [12]
in the paper's related work, which is an internal citation inconsistency —
see §5 below).

Step 3 — **Adaptive loss** (Eq. 5, p. 2):

    L_adaptive = γ(d)·L_coh + (1 − γ(d))·L_contraRec + L_dens

with γ(d) fixed per regime: 0.3 (light), 0.6 (medium), 0.9 (heavy).

- **Coherence loss** (Eq. 6): L1 between the reconstruction recomposed
  through the physical model and the hazy input:
  `L_coh = ‖J_out ⊙ T_out + (1 − T_out) − P‖₁`. Beginner note: this says
  "your dehazed image, when re-fogged through the ASM with your estimated
  transmission, must look like the input" — a physics-consistency check.
- **Perceptual loss** (Eq. 7): weighted L1 between VGG-19 feature maps of
  the ground truth and the output (`ϕ_i` = layer-i features, `τ_i` = weights).
- **Density loss** (Eq. 8): `L_dens = ‖D_θD(J_out)‖₁` — push the HDEN score
  of the *dehazed* output toward zero, i.e. penalize residual haze.

Step 4 — **Detection integration** (§III-B4, p. 2): the dehazed output feeds
YOLOv8n; detection losses can back-propagate into the dehazing branches.

### 3.4 Training details (§IV-A, p. 3)

- Dehazing network: PyTorch, NVIDIA A100, mixed precision (APEX), Adam,
  LR 2e-4, weight decay 1e-4, batch 16, 100 epochs, LR ×0.1 at epochs 50
  and 75.
- Detector: YOLOv8n pretrained on clear weather, **fine-tuned on
  FogIntensity-25K** to align its feature distribution with dehazed inputs;
  batch 32, LR 1e-4, mosaic augmentation.
- Paper states all code, weights, and FogIntensity-25K "will be publicly
  released" (§IV-A, p. 3) — as of this v1, not yet available; treat
  reproduction claims accordingly.

## 4. Datasets, splits, baselines, metrics, reported values

### 4.1 Haze-density classification (Table II, §IV-B, p. 3)

Ten backbones compared on a 5,000-image test set. DenseNet121 wins:
99.80% accuracy / precision / recall / F1, test loss 0.0065, ahead of
ResNet50 (99.00%), VGG16 (96.67%), ConvNeXt-L (94.67%).

### 4.2 Synthetic full-reference dehazing (Table IV, §IV-C1, p. 4)

Only the paper's own numbers are broken down by fog level (comparison to
baselines for PSNR/SSIM/LPIPS is not given per level in Table IV):

| Fog level | SSIM ↑ | PSNR ↑ (dB) | LPIPS ↓ |
|-----------|--------|-------------|---------|
| Light (0.03) | 0.9188 | 23.95 | 0.0585 |
| Medium (0.06) | 0.8761 | 21.78 | 0.0929 |
| Heavy (0.09) | 0.8060 | 19.39 | 0.1456 |

No-reference tables (Table III): BRISQUE ↓ / NIMA ↑ across light/medium/
complex — Ours 10.78/9.82/8.55 BRISQUE and 4.97/5.13/5.26 NIMA, beating
DAD, MBDN, RIDCP, PSD. (Table III lists RIDCP's BRISQUE as 21.28/18.23/
17.52 — an internal inconsistency with the abstract's claim, see §5.)

### 4.3 Real-world non-reference (Table V, §IV-E, p. 5)

FADE ↓ / BRISQUE ↓ / NIMA ↑ on RTTS "under light fog conditions" (the
table's own caption) + detection mAP per class:

| Method | FADE | BRISQUE | NIMA | Bicycle | Bus | Car | Motor | Person | Mean |
|--------|------|---------|------|---------|-----|-----|-------|--------|------|
| MBDN   | 1.363 | 27.672 | 4.529 | 0.54 | 0.27 | 0.63 | 0.43 | 0.75 | 0.52 |
| DADv1  | 1.130 | 32.241 | 4.312 | 0.52 | 0.29 | 0.65 | 0.38 | 0.74 | 0.51 |
| DADv2  | 1.358 | 33.210 | 4.484 | 0.54 | 0.28 | 0.64 | 0.42 | 0.75 | 0.52 |
| PSD    | 0.920 | 27.713 | 4.598 | 0.52 | 0.25 | 0.63 | 0.42 | 0.74 | 0.51 |
| RIDCP  | 0.944 | 17.293 | 4.965 | 0.57 | 0.32 | 0.66 | 0.47 | 0.76 | 0.55 |
| **Ours** | **0.828** | **11.961** | **5.346** | 0.59 | 0.31 | 0.67 | 0.48 | 0.77 | **0.56** |

Note: the prose (§IV-C2, p. 4) quotes FADE 0.824 / BRISQUE 11.956 / NIMA
5.342, but Table V records 0.828 / 11.961 / 5.346 — small internal mismatch
(see §5).

### 4.4 Detection claims (abstract, §I, §IV-E)

- Abstract (p. 1): "improves object detection mAP by up to 13 points" and
  "cutting inference time by 20%".
- §IV-E (p. 4): mAP 0.59 (Bicycle), 0.68 (Car), 0.49 (Motor), overall mean
  0.57 — **these numbers do not match Table V** (0.59/0.31/0.67/0.48/0.77,
  mean 0.56); see §5.
- Conclusion (p. 6): "boosting YOLOv8 mAP by 3–5% compared to
  state-of-the-art baselines, all with 20% faster inference in light haze".

### 4.5 Complexity (Table VI, §IV-F, p. 5) and ablations (§V, p. 5–6)

Inference per branch (batch 1): Light 18 ms / 45 GFLOPs, Medium 38 ms /
80 G, Complex 50 ms / 150 G, fixed 6-stage 45 ms / 120 G, adaptive average
30 ms / 92 G. Routing cuts latency 20–40% vs. always-6-stage (abstract
rounds this to 20%).

Ablations (Tables VII–VIII): removing the coherence loss costs the most
(PSNR 23.95 → 21.45, SSIM 0.9188 → 0.8762); removing density loss → 23.12 /
0.9113; perceptual loss → 22.10 / 0.9025; removing CPMM (Cooperative
Proximal Mapping Module) → 22.50 / 0.8992.

## 5. Inconsistencies and verification findings

Cross-checked against the extracted text of the PDF. **Do not quote the
abstract's headline numbers without these caveats.**

1. **"+13 mAP" (abstract, §I) vs. tables.** The largest per-class gain over
   the best baseline in Table V is +0.02 (bicycle: 0.59 vs. RIDCP 0.57),
   and the mean gain is +0.01 over RIDCP (0.56 vs. 0.55). "Up to 13 points"
   appears nowhere else in the paper and is not supported by any table.
   Similarly "+2.1 dB PSNR" (abstract) exceeds any table delta shown
   (Table VII's full-model 23.95 dB is the paper's own light-fog number,
   with no baseline PSNR printed alongside it).
2. **"30% lower FADE" (abstract) vs. tables.** Table V: ours 0.828 vs.
   next-best PSD 0.920 → 10.0% lower, not 30%. §IV-C2's prose claims
   "exceed RIDCP's performance by 12.7% in FADE and 30.8% in BRISQUE":
   vs. RIDCP 0.944 → 12.3% in FADE (≈12.7% only if computed on the prose's
   0.824), and vs. RIDCP 17.293 → 30.9% in BRISQUE (≈30.8% checks out).
   The "30%" in the abstract matches the BRISQUE comparison, not FADE.
3. **"reduces average inference time by 20%" (abstract) vs. 33% (§IV-F,
   p. 5) vs. "20–40%" (Table VI text, p. 5) vs. "23% FLOPs saved".** Three
   different savings figures for the same routing idea; the abstract's 20%
   is the most conservative of them.
4. **§IV-E prose vs. Table V.** Prose: "mAP values of 0.59 (Bicycle), 0.68
   (Car), and 0.49 (Motor) with an overall mean mAP of 0.57". Table V
   (same section): Bicycle 0.59, Bus 0.31, Car 0.67, Motor 0.48, Person
   0.77, Mean 0.56. Car 0.68 vs. 0.67 and mean 0.57 vs. 0.56 disagree.
5. **Prose metric values vs. Table V.** §IV-C2 says FADE 0.824, BRISQUE
   11.956, NIMA 5.342; Table V says 0.828, 11.961, 5.346.
6. **Table V caption vs. content.** Caption says "under light fog
   conditions" and "leveraging the CORUN-Light branch", yet the paper's
   core selling point is per-density routing — mixing a branch-specific
   result into the headline real-world table without stating which branch
   each baseline used, or how RTTS images were density-routed, weakens the
   comparison. RTTS has no ground-truth density labels (§III-A2), so the
   routing thresholds on real data are undefined in the paper.
7. **CORUN citation mismatch.** §II says "CORUN [12] introduced cooperative
   unfolding modules" but [12] is MSBDN (Dong et al., multi-scale boosted
   dehazing). The actual CORUN paper (Mou et al., CVPR 2022, "Deep
   general unfolding networks") is [21], which is cited for a different
   claim. Minor but real bibliography error.
8. **Ablation tables reference an undefined module.** Table VIII studies
   "CPMM" with prose explaining its role, but CPMM never appears in §III's
   method description or Algorithm 1 — the architecture section and the
   ablation section describe different-looking models.
9. **Reproducibility gap.** Code, weights, and FogIntensity-25K are promised
   (§IV-A) but not included in v1; thresholds α and β of Eq. 4 are never
   given numeric values, so the routing rule is not reproducible from the
   paper. HDEN's architecture is also unspecified beyond "lightweight".
10. **Detection evaluation is under-specified.** No IoU range, confidence
    threshold, input resolution, or val/test split is given for the mAP
    numbers in Table V; the detector is fine-tuned on FogIntensity-25K
    (synthetic) but evaluated on RTTS (real) with no mention of which RTTS
    split or whether RTTS labels were used for anything.

**Bottom line for our project:** the paper's *architecture* is clearly
described (HDEN router + 2/4/6-stage CORUN branches + 3-term adaptive loss)
and is reproducible in spirit, but its *headline numbers* are not internally
consistent. Any comparison we run must be against the tables, not the
abstract. Our DCP-vs-hazy and DCP-vs-detection protocol (same paired images,
CIs on deltas) is methodologically stricter than the paper's own detection
evaluation.

## 6. Differences between the paper setup and this project

| Aspect | Paper | This project | Consequence |
|---|---|---|---|
| Synthetic fog | ASM with β ∈ {0.03, 0.06, 0.09} on Cityscapes + Synscapes depth | Foggy Cityscapes β ∈ {0.005, 0.01, 0.02} (Sakaridis-style synthesis) | Fog regimes do not overlap; absolute metrics are not comparable, only within-project deltas |
| Detector | YOLOv8n **fine-tuned** on FogIntensity-25K | YOLOv8n **frozen COCO-pretrained** (no fine-tuning) | Our detection deltas measure dehazing alone, uncontaminated by joint training; not comparable to the paper's mAP values |
| Detection metric | mAP per class, split/thresholds unspecified | COCO AP@0.5 and AP@0.5:0.95 via pycocotools, paired bootstrap CIs | Ours is stricter and reports uncertainty; the paper reports none |
| Dehazing metric | PSNR/SSIM/LPIPS + BRISQUE/NIMA/FADE | PSNR + SSIM only | We lack perceptual metrics; adding LPIPS is optional future work |
| HDEN routing | Density classifier picks one of 3 branches | No routing — one fixed DCP configuration for all images | We test the paper's premise ("adaptivity matters") from the opposite direction: does even a cheap classical method help detection, and does the gain vary with fog level (our β breakdown mirrors their light/medium/heavy axis) |
| Data availability | FogIntensity-25K promised, not released | Foggy Cityscapes + RTTS locally available | We cannot reproduce their training set; we benchmark on overlapping-but-different data |

## 7. Verification status

- [x] Full text extracted from `assets/2506.15837v1.pdf` (all 8 pages) and
      read in full on 2026-09-24.
- [x] Every headline claim traced to a table, equation, or section; ten
      internal inconsistencies recorded in §5.
- [ ] No reproduction attempted yet — per README and docs/protocol.md, this
      project makes no reproduction claim until the protocol is frozen and
      both baselines are evaluated under it.
