#!/usr/bin/env bash
# Method 2b (ours, fixed): context-gated U-Net, then its full-val score.
#
# The gate reads a whole downscaled scene instead of the restoration path's
# 256 crop. The plain gated run read pooled d 0.72 on full images but learned
# 0.80-0.88 on crops, so it kept ~40% hazy at eval time and lost ~3 dB; here
# the gate's input is the same 512x256 whole-image view in training and eval.
#
# Resumable: continues from the run's last checkpoint (safe after a power cut).
# Low impact: worker/thread caps + nice, safe to leave running on this box.
#
#   RESUME=1 bash scripts/machine_learning/run_context_method.sh
#   RUN=unet-gated-context-20260925 SCORE_RUN=unet-gated-context-full-val-20260927 \
#     bash scripts/machine_learning/run_context_method.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DATE="${DATE:-$(date +%Y%m%d)}"
RUN="${RUN:-unet-gated-context-${DATE}}"
SCORE_RUN="${SCORE_RUN:-unet-gated-context-full-val-${DATE}}"
CONTEXT_DIR="${CONTEXT_DIR:-data/derived/crops/resize-512x256}"
CACHE_DIR="${CACHE_DIR:-data/derived/crops/train}"
WORKERS="${WORKERS:-3}"
NICE_LEVEL="${NICE_LEVEL:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
extra=()
if [[ -d "runs/machine_learning/${RUN}" ]]; then
  extra=(--resume)
  echo "== resuming ${RUN} =="
else
  echo "== ${RUN} =="
fi
echo "  workers=${WORKERS} nice=${NICE_LEVEL} threads=${OMP_NUM_THREADS}"

nice -n "$NICE_LEVEL" "$UV" run python scripts/machine_learning/train_unet.py \
  --run-id "$RUN" --model gated --gate-context \
  --cache-dir "$CACHE_DIR" --context-dir "$CONTEXT_DIR" --gate-init 0.95 \
  --size medium --epochs 50 --batch 16 --workers "$WORKERS" \
  --val-scenes-per-beta 10 --val-every 5 "${extra[@]}"

if [[ -f "results/machine_learning/${SCORE_RUN}.json" ]]; then
  echo "== ${SCORE_RUN} already scored; skipping =="
else
  echo "== scoring ${RUN} on full val =="
  nice -n "$NICE_LEVEL" "$UV" run python scripts/machine_learning/benchmark_model.py \
    --run-id "$SCORE_RUN" --mode gated \
    --checkpoint "outputs/checkpoints/${RUN}/best.pt" \
    --split val --workers 4
fi
echo "== context-gated chain complete: results/machine_learning/${SCORE_RUN}.json =="
