# Project plan — Dehazing: traditional vs. machine learning

**Last updated:** 2026-09-24 · **Owner:** fill in · **Status legend:** ☐ todo · ◐ in progress · ✅ done

This is the single forward-looking checklist for the project. One row = one
concrete, verifiable action. When a step finishes, tick it and add the run ID
or file path that proves it. If reality diverges from a step, update the step
here first, then do the work.

---

## Where we are (snapshot 2026-09-24)

Done so far:

- ✅ Project scaffold: `src/ip_dehaze/{data,evaluation,traditional}`, configs,
  scripts, tests, uv environment.
- ✅ DCP baseline implemented and benchmarked on the 52-scene refined val
  subset (156 pairs, β ∈ {0.005, 0.01, 0.02}) — PSNR +1.25 dB (CI95
  [0.79, 1.68]), SSIM −0.138 (CI95 excludes 0). See
  `results/traditional/dcp-refined-val-20260923.json`.
- ✅ Exploratory RTTS detection eval: frozen COCO YOLOv8n, DCP helps
  (AP@0.5 +0.027, CI95 [+0.012, +0.043]; AP@0.5:0.95 +0.011, CI95
  [+0.002, +0.022]). Per-class: car/person/bus significant; bicycle/motorbike
  positive but unresolved (61 and 116 GT boxes — too rare to conclude).
  See `results/traditional/dcp-rtts-test-20260923{,-uncertainty}.json`.
- ✅ Reference paper read and verified:
  `docs/paper_notes.md` (ADAM-Dehaze, arXiv:2506.15837v1). 10 internal
  inconsistencies recorded; **quote its tables, never its abstract.**

Not done yet: protocol not frozen; ML track completely empty; full-split
traditional validation; final test evaluation; reports.

**Audit findings (2026-09-24, all checked against the code on disk;**
**runtime claims re-verified by actually executing the pipeline):**

- ❌ **No git repository exists.** Every run record has
  `"source_revision": null` because there is nothing to read. Fix in P0.1.
- ❌ **Bare `python3` cannot run the project** (no scipy installed
  system-wide). Every command in this plan must run under `uv run`.
  Tests pass under `uv run` (14/14, 2026-09-24).
- ⚠️ Leftover empty dirs `src/ip_dehaze/machine_learning/{models,routing,losses}`
  from the deleted run. Decide in P3.1 whether to keep this layout; delete
  if the chosen design doesn't need it.
- ✅ **Verified solid by execution (2026-09-24):** `index_pairs` finds
  exactly 1,500 full-split val pairs; DCP + PSNR/SSIM pipeline runs on
  real images at 1.24 s/pair (→ ~3–5 min for P2.1 with 12 workers);
  detection deps import and CUDA works (GTX 1660 SUPER 6 GB);
  `pairing.py` regex matching has 4 unit tests; DCP core provably
  reverses known fog formation (unit test); paired-bootstrap code is
  shared in `src/ip_dehaze/evaluation/`; both benchmark scripts hash
  configs and manifests into run records.
- ⚠️ **Found during verification:** `torch` is not a declared dependency
  (present only transitively via the `detection` extra) — P3.3 must add
  a `training` extra. Full-res images are 2048×1024; 6 GB VRAM forces
  crop-based training (see resolution trap in P3.4). The refined manifest
  contains 498 train + 52 val scenes (550 total).

---

## Phase 0 — Housekeeping (quick wins, ~½ day)

- ☐ **P0.1 Initialize version control** (blocks everything else):
  `git init && git add` code/docs/configs (`.gitignore` already excludes
  data/outputs), first commit. Every later run record then gets a real
  `source_revision`.
- ☐ **P0.2 Delete dead weight in `decks/`**: keep one deck template,
  remove `project_deck (Copy).pptx` duplicates (ask before deleting).
- ☐ **P0.3 Record dataset provenance** in `data/README.md`: Foggy Cityscapes
  version/source, Cityscapes license notes, RTTS Roboflow version
  (`CC BY 4.0`, already noted) — one paragraph each, plus file counts
  per split (verified 2026-09-24: train 2,975 clear / 8,925 foggy,
  val 500 / 1,500, test 1,525 / 4,575; RTTS test 432 images, 4,134 boxes:
  bicycle 61, bus 272, car 2610, motorbike 116, person 1075).
- ☐ **P0.4 Wire `source_revision` for real:** after P0.1, scripts'
  `git_revision()` starts returning hashes automatically — verify with one
  throwaway run, then make `uv run` the documented way to execute
  everything (bare `python3` lacks scipy).

## Phase 1 — Freeze the protocol (gates everything confirmatory)

- ☐ **P1.1 Freeze `docs/protocol.md`.** It must define, verifiably:
  - Dataset versions and manifest SHA-256s (already recorded per run —
    formalize the list here).
  - **Methods under comparison:** (a) untreated hazy input, (b) DCP
    (fixed params: patch 15, ω 0.95, top 0.1%, guided r=20, ε=1e-3,
    t₀=0.1), (c) ML model(s) as they land, (d) clear reference.
  - **Metrics:** PSNR, SSIM (exact implementations: `skimage.metrics`,
    data_range 255, gaussian_weights true — match
    `benchmark_dcp.py`); detection: COCO AP@0.5, AP@0.5:0.95 via
    pycocotools, max 100 dets/img, class mapping per
    `configs/traditional/dcp_rtts_detection.yaml`.
  - **Split discipline:** development + model selection on *train* and
    *val only*. `test` splits (Foggy Cityscapes test, RTTS test) are
    touched exactly once, after this file is frozen, for the final table.
    Note the RTTS-test exception: already used exploratory (2026-09-23,
    recorded in run files) — declare that run invalid for confirmatory
    claims and re-run identically-configured eval after freezing if the
    number is to be quoted as final.
  - **Seeds & selection rules:** bootstrap seed 1729, 2000 resamples;
    ML checkpoint selection = best val PSNR, one run per config, no
    test peeking.
  - **Per-run provenance:** config SHA-256, source revision, dataset
    manifest hash, env versions, status — the schema the existing run
    records already use.
- ☐ **P1.2 Write decision record** `docs/decisions/2026-09-xx-protocol-v1.md`
  (one page: what was frozen and why). Update `README.md` workflow when done.
- **Exit criterion:** protocol.md has no "not yet frozen" banner; every
  subsequent run cites it.

## Phase 2 — Complete the traditional track

- ☐ **P2.1 Full-split DCP validation:** set `refined_only: false` in
  `configs/traditional/dcp.yaml`, run
  `scripts/traditional/benchmark_dcp.py --run-id dcp-full-val-<date>`
  under `uv run`. **Verified 2026-09-24:** `index_pairs` finds exactly
  1,500 val pairs at full split; measured 1.24 s/pair serial → with the
  config's `workers: 12` expect **~3–5 min** (budget 15). Sanity-check
  after: pooled hazy PSNR should stay in the ~15 dB ballpark of the
  refined subset; a wild shift means the refined subset was skewing,
  which is itself a finding to record. Spot check on the full split
  already reproduces the known pattern (frankfurt β=0.005: DCP 15.3 dB
  < hazy 20.7 dB — DCP hurts light fog).
- ☐ **P2.2 (Optional, cheap) second traditional method** for a less barren
  comparison column — e.g. guided-filter–only enhancement or CLAHE-based
  pre-dehaze. Skip if time-constrained; DCP alone is a valid baseline.
- ☐ **P2.3 Confirmatory RTTS detection eval** (post-freeze, same frozen
  config): `scripts/traditional/evaluate_dcp_detection.py` with a new run
  ID + paired-bootstrap uncertainty analysis. If any setting changed since
  the exploratory run, justify the change from val data only.
- **Exit criterion:** `results/traditional/` contains a full-val dehazing
  run and a post-freeze detection run, both with CIs, both citing protocol v1.

## Phase 3 — ML baseline (the main remaining work)

**Implementation status (2026-09-24):** code built and gates passed. What
exists now:
`src/ip_dehaze/data/dataset.py` (FoggyPairs + memmap `CachedCrops`),
`machine_learning/models/{unet,classifier}.py`,
`machine_learning/losses/`, `evaluation/{metrics,provenance,pair_benchmark}.py`,
and scripts `train_unet.py`, `train_classifier.py`, `benchmark_model.py`,
`sanity_gates.py`, `build_crop_cache.py`, `run_paper_method.sh`.
Sanity gates **all pass**: overfit 31.37 dB (>30), identity-vs-recorded hazy
PSNR diff 0.00e+00, cache integrity 100%, loader panels written to
`outputs/figures/loader_check/`. The training chain is running as the
service `paper_method_training` (blind → classifier → 3 branches).
Remaining: the gated model (ours) and its `d≡1` ablation, then the routed
val evaluation.

Design constraints (from paper notes §6): our fog β ∈ {0.005, 0.01, 0.02}
differs from the paper's {0.03, 0.06, 0.09}; absolute numbers are not
comparable to the paper — only our own within-project deltas are.

**Novelty (ML headline, revised 2026-09-24): self-supervised density-gated
dehazing network (method 2; method 1 is the paper-style routed system
below).** The contribution must be an ML method — so the
adaptive-strength idea becomes architectural. One U-Net backbone, two
heads: restoration head `f(P)` and a lightweight density head predicting a
continuous scalar `d ∈ [0,1]`. Output: `J = d·f(P) + (1−d)·P`. Training:
plain L1 + SSIM on the synthetic train pairs, end-to-end, **no density
labels at all** — the gate is learned because overcorrecting light fog is
punished by the reconstruction loss itself. Motivation (measured before
the method existed): fixed-strength DCP hurts light fog (β=0.005:
−2.96 dB) while helping heavy (β=0.02: +5.13 dB).

Positioning (revised 2026-09-24): **extend, don't oppose.** The method
extends the paper's density-adaptivity premise (their §III-B routing) in
two ways: (1) continuous scalar instead of their supervised 3-class
HDEN + 3 separate networks; (2) *self-supervised* — no fog labels, the
gate emerges from the plain reconstruction loss. Fair claim if it works:
"we extend ADAM-Dehaze's density awareness into a label-free,
single-network form." ADAM-Dehaze itself uses a *supervised 3-class*
classifier routing to *3 separate networks* with a *custom 3-term loss*;
ours is *one* network, *one continuous self-supervised* scalar, *standard*
loss. No privileged inputs anywhere (no β labels, no transmittance maps
at train or test); RTTS detection is the out-of-domain proof.

Known risk (write down now): density-head collapse — network cheats with
`d≈1` everywhere and lets `f` stay conservative. Counter-measures if seen:
(a) monotonicity penalty (pooled `d` should track mean dark-channel
intensity — still label-free), or (b) warm-start `d` with train-only β
pseudo-labels. Diagnose via the d-frozen ablation below, not by tuning
on val.

Experiment matrix (all through the same eval path):
hazy / DCP (done) → blind U-Net (control: does learning help?) →
**theirs, paper-style routed** (supervised 3-class classifier + 3
per-class branches + discrete routing — the paper's mechanism on our
data) → **ours, density-gated** (does learned self-supervised adaptivity
help beyond theirs and beyond plain learning?) → **ours with d frozen to
1** (kill-shot ablation: is the gate doing the work?). Final head-to-head:
DCP vs. blind U-Net vs. theirs vs. ours.

Secondary finding (zero extra work): SSIM drops while detection AP rises
under DCP — "perceptual metrics don't predict task utility under fog";
report with CIs.

Demoted, clearly-labeled ablations only (never headline):
Adaptive-DCP (`ω = clip(a·d + b)`, `a,b` fit on train — motivation
experiment for the report, not the contribution); transmission-supervised
U-Net using `data/transmittance/` GT maps (8,944 train / 1,504 val / 4,582
test in `leftImg8bit_trainvaltest_transmittance.zip`) — benchmark-privileged,
ablation column only.

- ☐ **P3.1 Model ladder (build/run order):**
  0. **Blind plain U-Net** (hazy→clear, L1+SSIM) — *infrastructure +
     control*, not a headline method: it validates the training loop
     (smoke, P3.2) before any expensive run and provides the
     non-adaptive ML comparison column. Build first, cheap.
  1. **Theirs — paper-style routed system (method 1)** — small 3-class
     density classifier (supervised; labels free from β in filenames)
     routing to 3 per-class U-Net branches (light/medium/heavy capacity,
     each trained only on its β-subset, 2,975 pairs); thresholds fit on
     train. Build classifier first (cheap, independently checkable via
     val accuracy ~30 min), then the 3 branches.
     **Deviations from the paper, recorded in the methods doc:** U-Net
     backbones instead of CORUN (their branch architecture is a cited
     external paper, not specified in theirs), standard L1+SSIM instead
     of their 3-term loss, thresholds fitted (theirs were never
     published). Frame as *mechanism reimplementation*, never
     "reproduction."
  2. **Ours — density-gated U-Net (method 2, the novelty)** — same
     backbone as rung 0 + density head + output gate
     `J = d·f(P) + (1−d)·P`; same loss; no labels.
  3. **Kill-shot ablation** — identical to rung 2 but `d ≡ 1`
     (architecture unchanged; if rung2 > rung3 at light fog, the gate is
     proven, else it's dead weight — report whichever).
  4. (Optional) transmittance-supervised ablation — labeled
     privileged-GT column, never headline.
  Record the choice + rationale in `docs/methods/machine_learning.md`
  (replace the placeholder) and a decision file in `docs/decisions/`.
  Decide there whether the leftover `models/routing/losses` skeleton stays.
- ☐ **P3.2 Sanity gate BEFORE real training** (this is what failed last
  time — do not skip):
  1. Visual check: save 3–4 (foggy, clear) pairs from the loader to
     `outputs/figures/loader_check/` and eyeball alignment.
  2. Loader test: assert hazy/clear filename pairs match scene + β for a
     random 50-pair sample.
  3. Overfit smoke test: train on ~32 pairs for ~800 steps; the model must
     reach near-perfect PSNR on those pairs (>30 dB). If it can't, the
     model/loss/optimizer is broken — stop and fix, do not launch the full
     run. **Measured 2026-09-24** (32 crops, medium U-Net, 0.195 s/step):
     PSNR 23.4 → 27.4 → 29.4 → 30.2 → 32.9 dB at steps 200/400/600/800/1000,
     so the step count matters — 300 steps reaches only ~25 dB and is not
     evidence of breakage.
  4. Baseline check: a model that only copies its input must score exactly
     the hazy baseline (15.14 dB pooled) on val — proves the eval path
     scores ML outputs identically to DCP's.
- ☐ **P3.3 Train.** Suggested starting config
  (`configs/machine_learning/<model>.yaml`): Adam 2e-4, batch 8–16, ~50
  epochs, **L1 + SSIM loss** (pure L1 blurs and scores badly — a likely
  cause of the previous failed run; skip the paper's 3-term adaptive loss
  for v1, it's their contribution not ours). Train on the 8,925-pair train
  split, select on the 500-scene val split. Log to
  `runs/machine_learning/<run-id>/`. If val PSNR after epoch ~50 is below
  DCP's 16.39 dB, treat as setup bug: revisit P3.2 gates before scaling up
  epochs or model size.
  **Verified 2026-09-24:** uv env has torch 2.14.0+cu130 with CUDA working
  (GTX 1660 SUPER, 6 GB, ~5.3 GB free) — AOD-Net-class training fits
  comfortably, but train on **256×256 random crops** (full images are
  2048×1024; full-res batches will not fit in 6 GB). Note: `torch` is not
  in `pyproject.toml` deps — it's present transitively via the
  `detection` extra; **add a `training` extra to pyproject** in P3.3 so
  the env is declarative, don't rely on the transitive install.
  **Training budget — MEASURED 2026-09-24 (replaces earlier estimates):**
  on the GTX 1660 SUPER, batch 16 at 256²: fp32 GPU step **0.343 s**;
  fp16 AMP step **1.296 s — 3.8× slower and NaN on every step**
  (GradScaler scale collapsed 65536→64), so **AMP is disabled** and
  `--amp` is opt-in only. Straight-from-PNG loading measured **0.731 s
  per batch of 16** (I/O bound, 2 × 2048×1024 decodes per sample), so
  training now reads a one-time **memmap crop cache**
  (`data/derived/crops/train`, 35,700 crops, 14 GB, 23.6 min to build):
  ~557 steps/epoch × ~0.35 s ≈ **3.5 min/epoch → ~3 h per 50-epoch run**.
  Full matrix (blind + classifier + 3 branches) ≈ **10–12 h ≈ two
  overnights**. Fairness note for the report: our gated model trains on
  all 8,925 pairs; routed branches see only their class's 2,975 —
  inherent to the designs, disclose it. **Run order (decided
  2026-09-24):** blind U-Net (loop validation) → **theirs/routed (method
  1)** → **ours/gated (method 2)** → `d≡1` ablation.
  **Local vs. Kaggle (decided 2026-09-24, updated):** train locally —
  the full matrix (~15–16 h) splits across two overnight runs on the
  1660 Super. Kaggle only if (a) a parallel sweep is wanted or (b) a
  single run must exceed ~4 h locally. If training off-machine: train
  remote, **evaluate locally** through the same eval path, and record
  the remote env in the run record. Do not upload Foggy Cityscapes
  -derived data to Kaggle (Cityscapes non-commercial terms — see
  data/README.md); remote training would require a re-licensed or
  synthetic data source, so in practice local-only for now.
- ☐ **P3.4 Evaluate like-for-like.** **Implementation note (verified
  2026-09-24):** `benchmark_dcp.py` calls `dehaze()` by name inside
  `evaluate_pair` — the lazy way to reuse it is to refactor that one call
  site into a pluggable `method` (a small `--method {dcp,model}` flag or a
  callable from the config), keeping `metrics`, `summarize`,
  `clustered_interval`, and the run-record schema untouched. ML outputs
  then flow through the *same* PSNR/SSIM/clustered-bootstrap code with
  zero reimplementation. One result JSON per run in
  `results/machine_learning/`.
  **Go/no-go gate:** if the ML model does not beat DCP on val PSNR *and*
  SSIM after one honest tuning round, record that negative result in
  `docs/reports/` and pivot the story to "DCP is a strong cheap baseline;
  learned dehazing underperforms at this data scale" — a valid, defensible
  finding, not a failure to hide. Do not quietly iterate test-set-adjacent.
  ⚠️ **Resolution trap (verified):** training crops at 256×256 but
  evaluating on full 2048×1024 images is a distribution shift for
  convolutional models with global receptive-field assumptions. AOD-Net
  is fully convolutional so it *runs* at any size, but if val PSNR looks
  inexplicably bad, first evaluate on 512×512 center crops of val before
  diagnosing the model — and if crop-eval differs strongly from full-res
  eval, adopt tiled/overlap inference for the final protocol and write
  that into `docs/methods/machine_learning.md`.
- ☐ **P3.5 ML detection eval, both methods.** Dehaze RTTS test
  (post-freeze) with **(a) the density-gated model and (b) the
  paper-style routed system** → frozen YOLOv8n → same COCO AP comparison
  as P2.3. For the routed system, RTTS images have no β labels — the
  3-class classifier must route each real image first (that routing
  accuracy on unlabeled real data is itself a reportable number). This
  gives the final head-to-head: DCP vs. ours (gated) vs. theirs (routed)
  for detection.
- **Exit criterion:** `results/machine_learning/` has val dehazing runs +
  RTTS detection runs for **both** ML methods (gated + routed) plus the
  `d≡1` ablation, same schema as the traditional ones.

## Phase 4 — Final evaluation & reporting

- ☐ **P4.1 Final test evaluation, once:** both tracks on Foggy Cityscapes
  test (dehazing PSNR/SSIM vs. clear targets — requires generating foggy
  test pairs or using the dataset's own test fog) and RTTS test (detection).
  Single pass, no reruns, results in `docs/reports/final-evaluation.md`.
- ☐ **P4.2 Analysis write-up:** answer the project's questions with CIs —
  - Does DCP improve perception metrics? (val says: PSNR yes at β ≥ 0.01,
    SSIM no.)
  - Does dehazing help detection? (exploratory says: yes, +2.7 AP@0.5.)
  - Does the gain vary with fog density? (yes — larger at heavier fog.)
  - **Gated (ours) vs. routed (paper-style) vs. blind U-Net vs. DCP:**
    quality, speed, params, and detection delta — with the
    mechanism-vs-machinery conclusion (does self-supervised gating match
    supervised 3-way routing?) and the disclosed data-asymmetry note.
- ☐ **P4.5 Freeze the traditional track before final eval:** after P2, tag
  the code revision and pin the DCP config — no parameter edits between the
  val benchmark and the test eval, or the test number measures something
  different from what val promised.
- ☐ **P4.3 Update the deck** (`decks/project_deck.pptx`) with the final
  tables and 2–3 qualitative figure panels
  (`outputs/figures/` — hazy vs. DCP vs. blind U-Net vs. ours (gated) vs.
  paper-style (routed) vs. GT).
- ☐ **P4.4 Sanity-check claims vs. paper:** no sentence in the report may
  cite the paper's abstract numbers (see paper notes §5).

## Optional / explicitly out of scope for now

- **Fine-tuned detector ablation** ("does dehazing still help a fog-adapted
  detector?") — only after P4, only if time remains; needs train-split-only
  tuning and a protocol addendum.
- **FogIntensity-25K reproduction** — dataset not released in paper v1;
  impossible now, note in report limitations.
- **Perceptual metrics (LPIPS/FADE/BRISQUE)** — add only if a reviewer-style
  objection demands them; PSNR+SSIM+CIs is a defensible core.

## Risks & watchpoints

| Risk | Mitigation |
|---|---|
| Test-set contamination (RTTS test already touched) | Declared exploratory; re-run post-freeze before any final claim (P1.1, P2.3) |
| ML overfits val through repeated selection | Cap val-based decisions; log every config tried in `runs/machine_learning/` |
| GPU runtime surprises in P3 | Smoke-run 2 images first (the DCP scripts already support `--limit`) |
| Metric implementation drift between tracks | Both tracks must call the *same* eval functions in `src/ip_dehaze/evaluation/` — no re-implementations |
| Scope creep beyond the agreed matrix (e.g., CORUN branch reimplementation from its own paper, HDEN at full paper fidelity) | The P3.1 matrix is the agreed scope; escalations need a decision-record entry first |
| Broken environment surprises mid-run | Always `uv run`; bare `python3` fails on scipy (verified 2026-09-24) |
| Untracked code invalidates provenance | P0.1 git init first; no benchmark run before the first commit |
| Late discovery that ML method doesn't work | P3.2 sanity gates catch setup bugs before the full run; P3.4 defines the honest negative-result exit — the plan survives a failing model, it doesn't survive a silent one |

## Non-negotiables (read before every phase)

1. **Never touch test splits before P4.1** — no visualization, no EDA, no
   "just checking" on Foggy Cityscapes test or RTTS test.
2. **Every run gets a record** — config hash, manifest hash, source
   revision, status; the existing scripts do this automatically, keep it that way.
3. **One shared evaluation path** — ML outputs go through the same
   `benchmark`/eval code as DCP, not a fresh reimplementation.
4. **Sanity gates before scale-ups** — loader check, overfit test, and
   copy-baseline check precede any long training (P3.2).
5. **A negative result is a result** — record it; don't iterate into
   silent test-set leakage.
