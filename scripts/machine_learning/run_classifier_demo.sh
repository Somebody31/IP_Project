#!/usr/bin/env bash
# Demonstration, not a new method: retrain the density classifier the way it
# *should* have been trained, to give the paper's routed design its fairest
# router, then re-evaluate the routed system with it.
#
# What changes vs the deployed classifier (density-classifier-20260924):
#   input  256px crops        -> whole image downscaled to 512x256
#   epochs 12                 -> 50
# The architecture is deliberately unchanged (DensityClassifier, 61,059 params),
# so any improvement is attributable to input format and training budget alone,
# and nothing in the evaluation path needs to change: load_classifier reads the
# input mode from the checkpoint's own `args`.
#
# Measured context: the deployed classifier never fit its own training set
# (train_acc 0.61 after 12 epochs on crops) and routes at 59.7% val accuracy,
# costing 0.77 dB against oracle routing. The ceiling this can reach is the
# oracle row (27.19 dB); the 2.27 dB gap to the single model is structural
# (each branch sees 1/3 of the data) and cannot improve here.
set -euo pipefail
cd "$(dirname "$0")/../.."
CLASSIFIER_RUN="${CLASSIFIER_RUN:-density-classifier-v2-20260927}"
ROUTED_RUN="${ROUTED_RUN:-routed-learned-v2-full-val-20260927}"
NICE_LEVEL="${NICE_LEVEL:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

# The context-gate score owns the GPU first; wait for that chain to finish.
while systemctl --user is-active --quiet dehaze-context-train.service; do
  echo "waiting for dehaze-context-train.service to finish the score"
  sleep 30
done

echo "== training ${CLASSIFIER_RUN}: whole images 512x256, 50 epochs =="
nice -n "$NICE_LEVEL" "$UV" run python scripts/machine_learning/train_classifier.py \
  --run-id "$CLASSIFIER_RUN" --resize 512x256 --epochs 50 --lr 3e-4 --workers 6

echo "== re-evaluating the routed system with the fair classifier =="
nice -n "$NICE_LEVEL" "$UV" run python scripts/machine_learning/benchmark_model.py \
  --run-id "$ROUTED_RUN" --mode routed --routing classifier \
  --classifier "outputs/checkpoints/${CLASSIFIER_RUN}/best.pt" \
  --branch-light outputs/checkpoints/branch-light-20260924/best.pt \
  --branch-medium outputs/checkpoints/branch-medium-20260924/best.pt \
  --branch-heavy outputs/checkpoints/branch-heavy-20260924/best.pt \
  --split val --workers 4

echo "== done: results/machine_learning/${ROUTED_RUN}.json =="
