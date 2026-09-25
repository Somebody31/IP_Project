#!/usr/bin/env bash
# Method 2 (ours): density-gated U-Net, then the d==1 ablation.
#
# Low-impact by default: this box is shared with interactive work, so the data
# loader gets few workers, numeric libraries are thread-capped, and the whole
# job runs under `nice`. Raise WORKERS for speed when the machine is free.
#
#   WORKERS=3 bash scripts/machine_learning/run_gated_method.sh          # low impact
#   WORKERS=12 nice -n 0 bash scripts/machine_learning/run_gated_method.sh  # full speed
#
# Resumable: RESUME=1 continues an existing run from its last checkpoint.
set -euo pipefail

cd "$(dirname "$0")/../.."
DATE="${DATE:-$(date +%Y%m%d)}"
GATED_RUN="${GATED_RUN:-unet-gated-${DATE}}"
FROZEN_RUN="${FROZEN_RUN:-unet-gated-frozen-${DATE}}"
WORKERS="${WORKERS:-3}"
NICE_LEVEL="${NICE_LEVEL:-15}"

# keep numpy/BLAS/skimage from saturating every core
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

run_step() {
  local run_id="$1"; shift
  local extra=()
  if [[ -d "runs/machine_learning/${run_id}" ]]; then
    if [[ "${RESUME:-0}" == "1" ]]; then
      extra=(--resume)
      echo "== resuming ${run_id} =="
    else
      echo "== skip ${run_id} (already exists; set RESUME=1 to continue it) =="
      return 0
    fi
  else
    echo "== ${run_id} =="
  fi
  echo "   workers=${WORKERS} nice=${NICE_LEVEL} threads=${OMP_NUM_THREADS}"
  nice -n "$NICE_LEVEL" uv run python "$@" "${extra[@]}"
}

run_step "$GATED_RUN" scripts/machine_learning/train_unet.py \
  --run-id "$GATED_RUN" --model gated --size medium --epochs 50 --batch 16 \
  --workers "$WORKERS" --val-scenes-per-beta 10 --val-every 5

run_step "$FROZEN_RUN" scripts/machine_learning/train_unet.py \
  --run-id "$FROZEN_RUN" --model gated --frozen-gate --size medium --epochs 50 \
  --batch 16 --workers "$WORKERS" --val-scenes-per-beta 10 --val-every 5

echo "== gated-method chain complete =="
