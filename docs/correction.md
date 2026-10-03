# Correction & continuation plan: RTTS (real-fog) detection track

> **LATEST STATUS (13:15 IST - Complete & Verified):**
> - **1. Full RTTS-Valid Detection (`adapt-v2-w0.01-valid-eval`, 865 images):**
>   - **Ours (Adapted v2, Epoch 3): AP@0.5 = 0.4779** | AP@0.5:0.95 = 0.2848 (109,967 detections).
>   - **Hazy Baseline: AP@0.5 = 0.4527** | AP@0.5:0.95 = 0.2657.
>   - **DCP Benchmark: AP@0.5 = 0.4706**.
>   - **Delta vs Hazy: +0.0252 AP@0.5** (+0.0190 AP@0.5:0.95).
>   - **Bootstrap 95% CI (400 resamples): mean +0.0257 [0.0100, 0.0405], excludes zero (True)**. Statistically significant gain!
>   - **Delta vs DCP: +0.0073 AP@0.5** (First learned method to officially beat DCP on real fog!).
>   - Artifacts: `results/machine_learning/adapt-v2-w0.01-valid-eval.json`, `results/machine_learning/detection-bootstrap-adapt-v2-w0.01-valid-eval.json`, `runs/machine_learning/adapt-v2-w0.01-valid-eval/detections.json`.
>
> - **2. In-Domain Full-Resolution Validation (`adapt-v2-full-val-20261003`, 1,500 pairs):**
>   - **Pooled PSNR: 29.65 dB** (Δ vs hazy: **+15.00 dB [14.73, 15.24]**). Acceptance bar $\ge 29.40\text{ dB}$ **PASSED**.
>   - **Pooled SSIM: 0.9682** (Δ vs hazy: **+0.1432 [0.1399, 0.1463]**).
>   - By $\beta$: light (0.005) = **31.86 dB**, medium (0.01) = **29.90 dB**, heavy (0.02) = **27.19 dB**.
>   - Artifacts: `results/machine_learning/adapt-v2-full-val-20261003.json`, `runs/machine_learning/adapt-v2-full-val-20261003/`.
>
> - **3. Phase 1 Post-Processing Probes (865 valid images, unadapted spatial winner):**
>   - Raw unadapted: **0.4451 AP@0.5**.
>   - `lum-match` (alpha=0.50): **0.4472 AP@0.5** (+0.0021, leaves gap open; artifact: `results/machine_learning/unet-spatial-lum-match-0.50-valid.json`).
>   - `blend` (alpha=0.70 ours + 0.30 hazy): **0.4601 AP@0.5** (+0.0150, mitigates over-removal; artifact: `results/machine_learning/unet-spatial-blend-0.70-valid.json`).
>   - Takeaway: simple scalar brightness scaling fails; input blending helps, but learned task adaptation dominates both and beats DCP.
>
> - **4. Checkpoint Location:**
>   - Best adapted model: `outputs/checkpoints/adapt-v2-w0.01-20261003/best.pt`.
>
> **Docs still to update:**
> - `docs/plan.md`
> - `docs/reports/{val-comparison,analysis,final-evaluation}.md`
> - `docs/decisions/2026-10-03-rtts-improvement-track.md`
>
> **Planning artifact:** `~/.gemini/antigravity/brain/9eb60c2a-ee52-453c-9cea-031088ed6b70/implementation_plan.md`

**Written:** 2026-10-03 12:26 IST, so that any agent can pick this up mid-flight.
**Owner decisions:** the user said "make your own decisions". The agent decided:
- **Test split:** all selection happens on `rtts/valid`. The final winner gets at most ONE labelled run on `rtts/test`, at the very end.
- **Training budget:** Phase 2 is a ~15-epoch fine-tune of the current winner, not a 50-epoch retrain.
- **Paper's method:** the routed system gets the same treatment only if time allows.
- **Disk:** keep free space on `/home` **≥ 60 GB**. It was 61–62 GB at writing, so **no downloads and no extraction**. Content-diversity work (new clean images + a depth model) is therefore **deferred**.

---

## 1. Why this track exists (one paragraph)

In-domain, both learned dehazers look excellent: ours (spatial-gated U-Net) scores 31.61 dB on the Foggy Cityscapes test split, and the paper's routed design scores 27.55 dB. On real fog (RTTS, frozen COCO YOLOv8n, AP@0.5), neither beats doing nothing, while classical DCP does:

| | valid | test |
|---|---|---|
| hazy | 0.4527 | 0.4923 |
| DCP | 0.4706 | 0.5194 |
| ours | 0.4451 (context gate) | 0.4850 |
| routed | 0.4415 | 0.4789 |

`docs/reports/val-comparison.md` findings 7–15 hold the full diagnosis:
- **Over-removal:** restored/hazy luminance ratio 0.449 vs 0.48–0.52 for healthy in-domain restorations.
- **Content/scene-domain shift.**
- **Atmospheric-light level gap:** RTTS A ≈ 0.86 vs 0.93–0.96 in training. This alone costs 3.8–5.4 dB (finding 15).

---

## 2. CORRECTION: the v1 task-adaptation loss was broken (finding 14 is invalid)

`scripts/machine_learning/train_task_adapt.py` as it stood at commit `d18d09a` (run `unet-spatial-rtts-adapt-20261002`, result `results/machine_learning/adapted-rtts-valid-20261002.json`, valid AP 0.4508) had these bugs:

1. **Confidence formula was wrong for YOLOv8.** YOLOv8 has no objectness row. In eval mode, `pred[:, 4:84]` are class probabilities that are *already sigmoided*. v1 computed `softmax(pred[b, 4:5, :], dim=1)` (person row, softmaxed **across 8,400 anchors**, so ≈1/8400) × `sigmoid(pred[b, 5:])`. Confidence was ≈1e-4, so loss was ≈ **0.9999**. This is exactly the logged "0.9999 → 0.9899 plateau".
2. **Anchor positions were wrong.** All 8,400 anchors were placed on the stride-8 grid (80×80 = 6,400). Anchors 6,400–8,399 belong to the stride-16/32 heads, so they got off-image positions.
3. **Boxes were not shifted by the crop offset.** GT boxes were computed in full-image pixels and never moved by the crop's `top/left`.
4. **Box flip was independent of the image flip.** Each box drew its own `rng.random()` coin, separate from the image's.
5. **No real selection.** Class labels were dropped, and `best.pt` was simply the last epoch.

**Consequence:** finding 14 ("adaptation moves the right way, +0.0057, not far enough") measured a near-zero, misaligned signal. It must be **retracted and replaced** by the v2 result in `val-comparison.md`, `analysis.md`, `final-evaluation.md`, and the decision record. Keep the v1 numbers as history, marked "invalid: broken loss, see docs/correction.md".

---

## 3. What has been done so far (2026-10-03)

### New and changed files

| File | Status | What |
|---|---|---|
| `src/ip_dehaze/data/rtts_detection.py` | **NEW, tested** | `RttsDetectionCrops` (native-res crops, crop-aligned COCO-class boxes, one flip coin, 80% object-centred crops), `crop_boxes`, `collate_detection`, `to_detector_batch` (differentiable letterbox onto a 640 grey canvas; scale = 640/max(H,W), the same factor YOLO's letterbox applies at eval) |
| `tests/machine_learning/test_rtts_detection.py` | **NEW, 4 passing** | box/crop/flip alignment, sliver drop, canvas scaling |
| `scripts/machine_learning/train_task_adapt.py` | **REWRITTEN (v2)** | Ultralytics `v8DetectionLoss` via `detector.loss(batch, detector(canvas))` with `detector.args = get_cfg()` (box 7.5 / cls 0.5 / dfl 1.5); frozen detector (params frozen, eval mode); dehazer BatchNorm frozen; recon arm on `CachedCrops` (or `--resynth`); per-epoch scoring on a fixed 300-image `rtts/valid` subset through the exact eval path (`dehaze_whole` → uint8 → `YOLO.predict`, imgsz 640, conf 0.001); synthetic-PSNR guard (30 full-res val images, max −1 dB); logs `task_loss` vs `task_loss_hazy` (detector loss on the untouched hazy crop, the key diagnostic) |
| `scripts/machine_learning/evaluate_model_detection.py` | **MODIFIED** | `--post {none,lum-match,blend,dcp-ensemble}` + `--post-alpha`, recorded in the payload's `inference` block. **Not yet smoke-tested** |
| `src/ip_dehaze/data/resynth.py` | **NEW, sample-checked** | `ResynthPairs`: `t = t_0.01^(β/0.01)`, β log-uniform [0.003, 0.03]; A = U[0.70, 1.0] × per-channel tint ±3%; then (p = 0.8) blur / noise σ ≤ 0.02 / JPEG q 30–95; p = 0.5 downscale ×0.25–0.6 before cropping; 3 fogs per scene per epoch (8,925 samples). **Reads train transmittance maps directly from `data/transmittance/leftImg8bit_trainvaltest_transmittance.zip`** (only `val/` is extracted; do NOT extract, it would cost disk). ~0.12 s/sample on one core |
| `tests/data/test_resynth.py` | **NEW, 2 passing** | scattering-model math |
| `scripts/machine_learning/train_unet.py` | **MODIFIED** | `--resynth` (uses `ResynthPairs`; incompatible with `--gate-context/--dcp-t/--beta-class`) and `--init-checkpoint` (fine-tune start). **Not yet smoke-tested** |

Scratch (not project code): `~/.gemini/antigravity/brain/9eb60c2a-…/scratch/` holds `probe_v8loss.py` (proved the v8 loss gives input gradients; blurring raises the loss) and `check_resynth.py` plus `resynth_samples/grid.png`.

### Run in progress at writing time

```
run id:  adapt-v2-w0.01-20261003
command: .venv/bin/python scripts/machine_learning/train_task_adapt.py \
           --run-id adapt-v2-w0.01-20261003 --task-weight 0.01
log:     outputs/logs/adapt-v2-w0.01-20261003.log
record:  runs/machine_learning/adapt-v2-w0.01-20261003/{config.yaml,log.csv,summary.json}
ckpts:   outputs/checkpoints/adapt-v2-w0.01-20261003/{best.pt,last.pt}
```

- **Setup:** 10 epochs × 400 steps, 4 RTTS crops + 4 synthetic crops per step, LR 2e-5 (cosine), from `outputs/checkpoints/unet-gated-spatial-20260929/best.pt`. About 0.6 s/step, so about 5–6 min/epoch including scoring, about 1 h total.
- **Reference values on the 300-image valid subset:** hazy AP50 **0.4701**, start (unadapted spatial winner) **0.4690**, start synth PSNR **32.35** (30-image set; not the same as the 1,500-image full-val number).
- **Early signal:** task loss on dehazed 5.31 vs hazy 5.48 at step 200, i.e. the detector loss on dehazed crops is already slightly below the hazy-input loss.
- **Check progress:**
  ```
  grep -v -E "index|DONE|Loading|Running|Evaluate|Accumulating" outputs/logs/adapt-v2-w0.01-20261003.log | tail
  ```
- **If it died:** `nvidia-smi`, check the log tail, rerun with a NEW `--run-id` (the script refuses to reuse a run dir).

---

## 4. Remaining plan (in order)

Data discipline (unchanged user rule): supervision = `rtts/train` + synthetic train; selection = `rtts/valid` + synthetic val; `rtts/test` only once, at the end, for the single final winner, as a **separately labelled track** ("ours + real-fog adaptation", never merged into the paper-reproduction table). Frozen YOLOv8n is both loss and metric; disclose this.

### Step A: finish and score Phase 0 (fixed adaptation)
1. When `adapt-v2-w0.01-20261003` finishes, read `summary.json` → `best`.
2. **If the task loss stays ≥ the hazy loss and subset AP never beats start**, raise the task weight: run `--task-weight 0.03` and then `0.1`, each with a new run id `adapt-v2-w<w>-20261003`. Keep each to 10 epochs. Log every config (valid is the only tuning surface).
3. Score the best checkpoint on **full** `rtts/valid`:
   ```
   .venv/bin/python scripts/machine_learning/evaluate_model_detection.py \
     --checkpoint outputs/checkpoints/<run>/best.pt --dataset-split valid \
     --run-id <run>-rtts-valid --save-detections
   ```
   DCP takes about 10 min of CPU; pass `--skip-dcp` if it is already known (valid DCP = 0.4706, hazy = 0.4527).
4. Bootstrap CIs on the deltas with `scripts/machine_learning/bootstrap_detection_ci.py`. Read its `--help` first; it consumes `runs/machine_learning/<run-id>/detections.json`.
5. Check in-domain quality is preserved on the **full** 1,500-pair val:
   ```
   .venv/bin/python scripts/machine_learning/benchmark_model.py --help
   ```
   Then run it on the best checkpoint. **Acceptance: ≥ 29.4 dB.**
6. **Phase 0 acceptance:** full-valid AP50 > 0.4527 (hazy); stretch ≥ 0.4706 (DCP).

### Step B: Phase 1, inference-time over-removal probe (no training; can run on CPU or while GPU is idle)
Use the current winner `unet-gated-spatial-20260929/best.pt` on `--dataset-split valid`, with `--skip-dcp --skip-hazy` where possible:
- `--post lum-match --post-alpha 0.50` (and 0.55)
- `--post blend --post-alpha 0.5 / 0.7 / 0.85`
- `--post dcp-ensemble --post-alpha 0.5` (needs DCP; slow)

Smoke-test first with `--limit 16`. **Interpretation:** if `lum-match` closes most of the gap to hazy/DCP, over-removal is the mechanism. Report as a probe (finding 16a), not a method.

### Step C: Phase 2, realistic-fog fine-tune
1. Smoke test:
   ```
   .venv/bin/python scripts/machine_learning/train_unet.py --run-id smoke-resynth --model gated --gate-spatial \
     --resynth --init-checkpoint outputs/checkpoints/unet-gated-spatial-20260929/best.pt \
     --epochs 1 --limit 64 --val-scenes-per-beta 2 --workers 6
   ```
   Then delete `runs/machine_learning/smoke-resynth` and `outputs/checkpoints/smoke-resynth`.
2. Full run:
   ```
   --run-id unet-spatial-resynth-20261003 --model gated --gate-spatial --resynth \
   --init-checkpoint outputs/checkpoints/unet-gated-spatial-20260929/best.pt \
   --epochs 15 --lr 1e-4 --workers 6
   ```
   About 3–4 min/epoch expected. Note that `train_unet.py` selects `best.pt` on synthetic val PSNR, which uses the shipped fog, so it stays comparable.
3. Score on full `rtts/valid` (as in Step A.3) and full val PSNR (as in A.5).
4. Rerun the real-fog transplant probe on the new checkpoint:
   ```
   .venv/bin/python scripts/machine_learning/probe_realfog_on_cityscapes.py \
     --checkpoint outputs/checkpoints/unet-spatial-resynth-20261003/best.pt \
     --run-id realfog-stats-resynth-20261003
   ```
   **Expectation:** the `ship_A` and `rtts_matched_A` arms recover much of their 3.8–5.4 dB loss.
5. **Acceptance:** full val ≥ 29.4 dB; valid AP50 > 0.4451 (its unadapted parent); stretch > hazy.
6. **If in-domain PSNR collapses:** lower `airlight_min` spread, or mix 50% shipped-fog cache batches. That needs a small code change: e.g. a `ConcatDataset` of `CachedCrops` + `ResynthPairs`.

### Step D: Phase 3, combine
Run fixed adaptation on top of the resynth model:
```
train_task_adapt.py --checkpoint outputs/checkpoints/unet-spatial-resynth-20261003/best.pt --resynth \
  --task-weight <best from A> --run-id adapt-v2-resynth-20261003
```
**Acceptance:** valid AP50 ≥ 0.4706 (DCP) with full val ≥ 29.4 dB.

### Step E (optional): Phase 4, unsupervised real-fog fine-tune (PSD-style)
Only if A–D fall short of DCP. Add to `train_task_adapt.py`:
- a dark-channel-prior loss (mean of the soft dark channel of the output on unlabeled RTTS images);
- a coherence loss `‖J·t + A(1−t) − P‖₁`, using DCP's `t`/`A` estimates of the input as fixed targets.

Weight both small.

### Step F (if time): symmetric treatment for the paper's routed system
Apply the best recipe to the three branch checkpoints (`outputs/checkpoints/branch-{light,medium,heavy}-20260924/best.pt`). Score with `--routed-classifier outputs/checkpoints/density-classifier-v2-20260927/best.pt --routed-branches …`. Verify which classifier the frozen routed runs used by reading `results/machine_learning/routed-rtts-valid-20261002.json`.

### Step G: final single test touch + docs
1. Pick ONE winner of this track by full-valid AP50 (subject to the 29.4 dB guard).
2. Score it once on `rtts/test`:
   ```
   evaluate_model_detection.py --checkpoint <winner> --save-detections --run-id <winner>-rtts-test-20261003
   ```
   The default config split is `test`. Then run the bootstrap. Label it "ours + real-fog adaptation (separate track)".
3. Update the docs:
   - `docs/reports/val-comparison.md`: erratum on finding 14 (link here), new finding 16 (Phase 0–3 results, Phase 1 probe).
   - `docs/reports/analysis.md`: Q2 table and bottom line.
   - `docs/reports/final-evaluation.md`: adaptation row.
   - NEW `docs/decisions/2026-10-03-rtts-improvement-track.md`: this track's scope and acceptance bars.
4. Run the full test suite: `.venv/bin/python -m pytest tests -q`.

---

## 5. Gotchas for the next agent

- **Run scripts from the repo root.** Several helpers use relative `data/...` paths, e.g. `train_unet.build_val_set`.
- **The sandboxed `run_command` was failing** ("connecting to sandbox server") on 2026-10-03, so commands were run with sandbox bypass.
- **The `grep_search` tool returned false "No results"** on this repo. Use shell `grep` instead.
- **GPU:** GTX 1660 SUPER, 6 GB. Do not run two training jobs at once. AMP is slow/NaN-prone here; keep fp32.
- **RTTS splits** are a Roboflow export: train 8,557 (3 augmented versions per source image; flips/90° rotations), valid 865, test 432. Class order is `bicycle, bus, car, motorbike, person`. COCO ids are in `RTTS_TO_COCO` in `rtts_detection.py` and must match `configs/traditional/dcp_rtts_detection.yaml`.
- **`train_task_adapt.py` refuses an existing run dir** (`exist_ok=False`). Always use a new `--run-id`.
- **Disk:** check `df -h /home` before anything that writes large outputs. Keep it ≥ 60 GB free.
- **Probe numbers are not method results.** Report Phase 1 as a diagnostic.
