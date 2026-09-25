#!/usr/bin/env bash
# Method 2 (ours): density-gated U-Net, then the d==1 ablation that tests
# whether the learned gate itself earns the gain. Resumable: an existing run
# directory is skipped. Trains from PNGs on the fly (the 256px crop cache was
# deliberately not rebuilt — see docs/plan.md "Measured cache economics").
set -euo pipefail

cd "$(dirname "$0")/../.."
DATE="${DATE:-$(date +%Y%m%d)}"
GATED_RUN="${GATED_RUN:-unet-gated-${DATE}}"
FROZEN_RUN="${FROZEN_RUN:-unet-gated-frozen-${DATE}}"

run_step() {
  local run_id="$1"; shift
  if [[ -d "runs/machine_learning/${run_id}" ]]; then
    echo "== skip ${run_id} (already exists) =="
    return 0
  fi
  echo "== ${run_id} =="
  uv run python "$@"
}

run_step "$GATED_RUN" scripts/machine_learning/train_unet.py \
  --run-id "$GATED_RUN" --model gated --size medium --epochs 50 --batch 16 \
  --workers 12 --val-scenes-per-beta 10 --val-every 5

run_step "$FROZEN_RUN" scripts/machine_learning/train_unet.py \
  --run-id "$FROZEN_RUN" --model gated --frozen-gate --size medium --epochs 50 \
  --batch 16 --workers 12 --val-scenes-per-beta 10 --val-every 5

echo "== gated-method chain complete =="
