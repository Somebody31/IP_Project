#!/usr/bin/env bash
# Full follow-up chain, in the user-approved order (2026-09-29):
#   1. RTTS blind control (gated sweep already done)
#   2. Heavier-β regime eval (existing checkpoints, no retraining)
#   3. Context-gated retrain seed 1729 (crop-rotation fix) + full-val score
#   4. Seed variance: 2 extra seeds, chained train + score each
#   5. Per-pixel gate: train, score, then the mask-collapse probe
# Fog resynthesis is deliberately NOT here: go/no-go decided by the RTTS +
# regime results, per the user's instruction.
# Each stage writes results/*.json; the unit stops on first failure.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p outputs/logs

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- 1. RTTS blind control ------------------------------------------------
if [[ ! -f results/machine_learning/unet-blind-rtts-full-20260929.json ]]; then
  log "stage 1: RTTS blind control"
  OMP_NUM_THREADS=4 uv run python scripts/machine_learning/evaluate_model_detection.py \
    --run-id unet-blind-rtts-full-20260929 \
    --checkpoint outputs/checkpoints/unet-blind-20260924/best.pt \
    --inference whole --skip-dcp > outputs/logs/rtts-blind.log 2>&1
fi

# --- 2. Heavier-β regime eval ----------------------------------------------
if [[ ! -f results/machine_learning/regime-full-val-20260929.json ]]; then
  log "stage 2: heavier-β regime eval"
  OMP_NUM_THREADS=4 uv run python scripts/machine_learning/evaluate_regime.py \
    --run-id regime-full-val-20260929 \
    --checkpoint gated=outputs/checkpoints/unet-gated-context-20260925/best.pt \
    --checkpoint blind=outputs/checkpoints/unet-blind-20260924/best.pt \
    --routed-classifier outputs/checkpoints/density-classifier-20260924/best.pt \
    --routed-branches \
      outputs/checkpoints/branch-light-20260924/best.pt \
      outputs/checkpoints/branch-medium-20260924/best.pt \
      outputs/checkpoints/branch-heavy-20260924/best.pt \
    > outputs/logs/regime-eval.log 2>&1
fi

# --- 3. Context-gated retrain, seed 1729 (crop-rotation fix) ---------------
if [[ ! -f results/machine_learning/unet-gated-context-full-val-20260929.json ]]; then
  log "stage 3: context-gated retrain seed 1729 (fixed crop rotation)"
  RUN=unet-gated-context-20260929 SCORE_RUN=unet-gated-context-full-val-20260929 \
    SEED=1729 WORKERS=6 bash scripts/machine_learning/run_context_method.sh \
    > outputs/logs/train-context-seed1729.log 2>&1
fi
# --- 4. Seed variance: 2 extra seeds ---------------------------------------
for SEED in 43 7; do
  if [[ ! -f "results/machine_learning/unet-gated-context-full-val-seed${SEED}.json" ]]; then
    log "stage 4: seed ${SEED} train + score"
    RUN="unet-gated-context-seed${SEED}-20260929" \
      SCORE_RUN="unet-gated-context-full-val-seed${SEED}" SEED="${SEED}" WORKERS=6 \
      bash scripts/machine_learning/run_context_method.sh \
      > "outputs/logs/train-context-seed${SEED}.log" 2>&1
  fi
done

# --- 5. Per-pixel gate ------------------------------------------------------
SPATIAL_RUN="unet-gated-spatial-20260929"
if [[ ! -f "results/machine_learning/unet-gated-spatial-full-val-20260929.json" ]]; then
  if [[ ! -f "outputs/checkpoints/${SPATIAL_RUN}/best.pt" ]]; then
    log "stage 5: per-pixel gate training"
    nice -n 10 uv run python scripts/machine_learning/train_unet.py \
      --run-id "$SPATIAL_RUN" --model gated --gate-spatial --gate-init 0.95 \
      --cache-dir data/derived/crops/train --size medium --epochs 50 --batch 16 \
      --workers 6 --seed 1729 --val-scenes-per-beta 10 --val-every 5 \
      > outputs/logs/train-spatial.log 2>&1
  fi
  # Score first: the probe is a diagnostic and must never cost the 3 h training
  # (it asserts against a collapsed mask, and `set -e` would abort the chain).
  log "stage 5: full-val score"
  nice -n 10 uv run python scripts/machine_learning/benchmark_model.py \
    --run-id unet-gated-spatial-full-val-20260929 --mode gated \
    --checkpoint "outputs/checkpoints/${SPATIAL_RUN}/best.pt" \
    --split val --workers 4 > outputs/logs/score-spatial.log 2>&1
  log "stage 5: mask-collapse probe"
  OMP_NUM_THREADS=4 uv run python scripts/machine_learning/probe_gate_mask.py \
    --checkpoint "outputs/checkpoints/${SPATIAL_RUN}/best.pt" \
    > outputs/logs/spatial-mask-probe.log 2>&1 \
    || log "WARNING: mask-collapse probe failed; read outputs/logs/spatial-mask-probe.log before trusting the score"
fi

log "chain complete"
