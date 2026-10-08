#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
RUN_TAG="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p ../logs ../outputs/heldout
# Two sequential evaluations use the first allocated GPU and preserve scheduler visibility.
for layer in 4 9; do
  adapter="../outputs/mone_layer_L${layer}_20260926-090110.pt"
  [[ -f "$adapter" ]] || { echo "missing adapter: $adapter" >&2; exit 2; }
  output="../outputs/heldout/L${layer}-${RUN_TAG}.json"
  log="../logs/mone-heldout-L${layer}-${RUN_TAG}.log"
  echo "[eval] layer=$layer log=$log report=$output"
  python -u scripts/eval_mone_heldout.py --checkpoint "$MONE_CHECKPOINT" \
    --data-dir "$MONE_DATA_DIR" --adapter "$adapter" --output "$output" \
    --tasks 8 --frames 3 --draws 3 >"$log" 2>&1 || {
      rc=$?; tail -40 "$log" >&2; exit "$rc";
    }
done
echo "[done] reports in ../outputs/heldout/*${RUN_TAG}.json"
