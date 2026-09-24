#!/usr/bin/env bash
# Run order (docs/plan.md P3.1): blind U-Net (loop validation) -> paper-style
# routed system (classifier + 3 per-class branches). Ours (gated) comes later.
# Each step is skipped if its run directory already exists, so the chain is
# resumable after an interruption.
set -euo pipefail

cd "$(dirname "$0")/../.."
DATE="${DATE:-$(date +%Y%m%d)}"
BLIND_RUN="${BLIND_RUN:-unet-blind-${DATE}}"
CLS_RUN="${CLS_RUN:-density-classifier-${DATE}}"
LIGHT_RUN="${LIGHT_RUN:-branch-light-${DATE}}"
MEDIUM_RUN="${MEDIUM_RUN:-branch-medium-${DATE}}"
HEAVY_RUN="${HEAVY_RUN:-branch-heavy-${DATE}}"

run_step() {
  local run_id="$1"; shift
  if [[ -d "runs/machine_learning/${run_id}" ]]; then
    echo "== skip ${run_id} (already exists) =="
    return 0
  fi
  echo "== ${run_id} =="
  uv run python "$@"
}

run_step "$BLIND_RUN" scripts/machine_learning/train_unet.py \
  --run-id "$BLIND_RUN" --model blind --size medium --epochs 50 --batch 16 --workers 6 \
  --val-scenes-per-beta 10 --val-every 5

run_step "$CLS_RUN" scripts/machine_learning/train_classifier.py \
  --run-id "$CLS_RUN" --epochs 12 --batch 32 --workers 6

run_step "$LIGHT_RUN" scripts/machine_learning/train_unet.py \
  --run-id "$LIGHT_RUN" --model blind --size light --beta-class light \
  --epochs 50 --batch 16 --workers 6 --val-scenes-per-beta 10

run_step "$MEDIUM_RUN" scripts/machine_learning/train_unet.py \
  --run-id "$MEDIUM_RUN" --model blind --size medium --beta-class medium \
  --epochs 50 --batch 16 --workers 6 --val-scenes-per-beta 10

run_step "$HEAVY_RUN" scripts/machine_learning/train_unet.py \
  --run-id "$HEAVY_RUN" --model blind --size heavy --beta-class heavy \
  --epochs 50 --batch 16 --workers 6 --val-scenes-per-beta 10

echo "== paper-method chain complete =="
