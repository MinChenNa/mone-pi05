#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

prior="../outputs/heldout/L4-20260926-093002-6435.json"
adapter="../outputs/mone_layer_L4_20260926-090110.pt"
[[ -f "$prior" ]] || { echo "missing prior report: $prior" >&2; exit 2; }
[[ -f "$adapter" ]] || { echo "missing adapter: $adapter" >&2; exit 2; }
run_tag="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p ../logs ../outputs/heldout
output="../outputs/heldout/L4-controls-${run_tag}.json"
log="../logs/mone-controls-L4-${run_tag}.log"
echo "[eval] layer=4 log=$log report=$output"
python -u scripts/eval_mone_heldout.py --checkpoint "$MONE_CHECKPOINT" \
  --data-dir "$MONE_DATA_DIR" --adapter "$adapter" --output "$output" \
  --exclude-report "$prior" --tasks 20 --frames 3 --draws 3 >"$log" 2>&1 || {
    rc=$?; tail -40 "$log" >&2; exit "$rc";
  }
echo "[done] $output"
