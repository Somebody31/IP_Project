# Machine-learning method track

Status: implemented 2026-09-24. Two learned methods are trained and compared
against each other, against the classical baseline (DCP), and against the
untreated hazy input.

## 1. What is trained

| Run | Script | What it is |
|---|---|---|
| Blind U-Net | `scripts/machine_learning/train_unet.py --model blind` | Plain U-Net, hazy → clear. Control: isolates "learning" from "adaptivity"; also validates the training loop |
| **Method 1 — paper-style routed** | `train_classifier.py` + `train_unet.py --beta-class <name>` | 3-class density classifier routes each image to one of three per-class U-Net branches (graded capacity: light/medium/heavy) |
| **Method 2 — density-gated (ours)** | `train_unet.py --model gated` | One U-Net + a density head predicting a scalar `d`; output is the blend `J = d·f(P) + (1−d)·P`. No density labels: the gate is learned from the reconstruction loss alone |
| `d ≡ 1` ablation | (flag on the gated model) | Same architecture with the gate disabled — proves whether the gate (not the backbone) earns any gain |

## 2. Data

- **Train:** Foggy Cityscapes `train`, 8,925 pairs (2,975 scenes × 3 β).
  Method 2 and the blind U-Net see all pairs; each routed branch sees only
  its β-subset (2,975 pairs) — inherent to the design, disclosed in the report.
- **Validation:** Foggy Cityscapes `val`, full-resolution, balanced across β.
  Used for checkpoint selection only.
- **Test:** untouched until the protocol's final evaluation.
- Pairing reuses `ip_dehaze.traditional.pairing.index_pairs` — one definition
  of a pair for both tracks.

## 3. Training configuration

- Crops 256×256 (images are 2048×1024; full-resolution batches do not fit in
  6 GB), random crop + horizontal flip; deterministic per (seed, epoch, index).
- Adam, LR 2e-4 (classifier 1e-3), weight decay 1e-4, cosine schedule.
- Loss: L1 + 0.2·(1 − SSIM). The SSIM *term* is a compact differentiable
  approximation used for gradients only (`machine_learning/losses/`);
  every reported SSIM comes from `evaluation/metrics.py`, shared with DCP.
- Batch 16, 50 epochs (classifier: 12), AMP, gradient clipping at 1.0.
- U-Net capacity presets: light `base=24,depth=3`, medium `base=32,depth=4`,
  heavy `base=48,depth=5`.
- Checkpoints: `outputs/checkpoints/<run-id>/best.pt` (best val PSNR),
  `last.pt` (resumable). Run records: `runs/machine_learning/<run-id>/`.

## 4. Evaluation

`scripts/machine_learning/benchmark_model.py` scores a checkpoint (or the
routed system) on the full val split at native resolution and writes the same
result schema as the traditional track, using the same shared metric and
scene-clustered bootstrap code (`evaluation/pair_benchmark.py`). Reported
quantities are therefore directly comparable across tracks:
`dehazed_psnr/ssim`, and `dehazed_minus_hazy_{psnr,ssim}` with 95% CIs.

## 5. Deviations from the reference paper (ADAM-Dehaze)

Method 1 reimplements the paper's *mechanism*, not its exact implementation,
because the publication is not reproducible as written (see
`docs/paper_notes.md` §5): no code, weights or dataset released, routing
thresholds never stated numerically, and the branch architecture (CORUN) is a
cited external paper rather than something specified in theirs.

| Component | Paper | This project |
|---|---|---|
| Density estimator | "lightweight" HDEN, DenseNet121-class | small CNN classifier (`models/classifier.py`) |
| Routing | thresholds α, β on a continuous score (values unpublished) | argmax over 3 classes; thresholds therefore implicit |
| Branches | CORUN 2/4/6-stage unfolding + attention | U-Net with graded capacity |
| Loss | 3-term adaptive (coherence + perceptual + density) | L1 + SSIM |
| Detector coupling | YOLOv8n fine-tuned on synthetic fog, detection loss back-props | frozen COCO YOLOv8n, no coupling |
| Density labels | supervised fog-intent labels | free from filenames (β), supervised for method 1; **none** for method 2 |

## 6. Config files

Training configuration lives in the CLI plus the per-run `config.yaml` the
scripts write into `runs/machine_learning/<run-id>/` (hashed into the run
record). Separate `configs/machine_learning/*.yaml` files are deliberately
not used — a single source of truth avoids the two drifting apart.
