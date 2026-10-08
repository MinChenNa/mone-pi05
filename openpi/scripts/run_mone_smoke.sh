#!/usr/bin/env bash
# One-GPU, two-step end-to-end validation before the 4-GPU layer sweep.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh

: "${MONE_CHECKPOINT:?Set MONE_CHECKPOINT to a converted pi05_libero checkpoint}"
: "${MONE_DATA_DIR:?Set MONE_DATA_DIR to a LIBERO LeRobot dataset}"
[[ -f "$MONE_CHECKPOINT/model.safetensors" ]] || { echo "missing model.safetensors" >&2; exit 2; }
[[ -d "$MONE_CHECKPOINT/assets/physical-intelligence/libero" ]] || { echo "missing LIBERO norm stats" >&2; exit 2; }
[[ -f "$MONE_DATA_DIR/meta/episodes.jsonl" ]] || { echo "missing episodes.jsonl" >&2; exit 2; }

mkdir -p ../logs ../outputs
LOG="../logs/mone-smoke-$(date +%Y%m%d-%H%M%S).log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u scripts/train_mone_layer.py \
  --checkpoint "$MONE_CHECKPOINT" --data-dir "$MONE_DATA_DIR" \
  --episodes "${EPISODES:-0,1,2,10}" --layer "${LAYER:-9}" \
  --steps "${STEPS:-2}" --output "${OUTPUT:-../outputs/mone_layer_smoke.pt}" \
  2>&1 | tee "$LOG"
