# Decision record — 2026-10-01: RTTS-train adaptation track

**Status: DRAFT — awaiting user approval.** Nothing below is executed until
the user marks it approved in this file.

## Context

The project's measured failure on real fog (RTTS): our learned dehazer is
*worse than doing nothing* for detection (AP@0.5 0.4740 vs hazy 0.4923 vs DCP
0.5194 on the 432-image test split, exploratory). Every fog-physics axis is
measured and does not bind (P3.6); the residual cause is content/scene-domain
shift of a model trained on synthetic fog over a single content world. The
paper's own design has the same blind-transfer vulnerability and reports an
unverifiable +13 mAP with an unspecified RTTS protocol (paper_notes §6).

RTTS ships with **train (8,557) / valid (865) / test (432)** splits, fully
disjoint (verified 2026-09-29: no filename or stem collisions), with
detection labels on all three.

## Decision (proposed)

1. **Task-driven adaptation on RTTS-train is approved / rejected** (user to
   mark). If approved: fine-tune the best dehazer with the frozen COCO
   YOLOv8n in the loss, on a fixed subsample of `rtts/train`, mixed with
   synthetic-reconstruction batches to preserve restoration ability.
2. The adapted system is reported as a **separate labelled track** —
   "ours + real-fog adaptation" — never merged into the paper-reproduction
   comparison table (the paper trains on FogIntensity-25K and adapts
   nowhere; same test set + different training distribution is protocol
   asymmetry).
3. The frozen COCO YOLOv8n is both the loss and the metric; this is
   disclosed wherever the adapted track's numbers appear, and any claim of
   general detection benefit requires a second detector.
4. Data discipline (user rule, 2026-09-29) applies unchanged: supervision on
   `rtts/train` + our synthetic pairs only; all selection on `rtts/valid`;
   `rtts/test` touched **once**, post-freeze (P4.1).

## Consequences

- The headline table gains a row that is *not* a paper reproduction; it must
  be labelled as such in every report.
- The valid split is the only tuning surface; repeated selection on it is
  capped by logging every config in `runs/machine_learning/`.
- If approved, the run is scheduled after the follow-up chain frees the GPU.

## References

- P3.7 steps and acceptance criteria (`docs/plan.md`).
- Step-0 diagnosis: `results/machine_learning/probe-rtts-damage.json` —
  harm concentrated (87% of harm mass in 10% of images) but not localizable
  by simple input statistics.
