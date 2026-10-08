#!/usr/bin/env bash
# Isolated L4 pilot, then paired evaluation on the 12 previously unused tasks.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

initial="../outputs/mone_layer_L4_20260926-090110.pt"
first="../outputs/heldout/L4-20260926-093002-6435.json"
source_report="../outputs/heldout/L4-controls-20260926-094629-91520.json"
for required in "$initial" "$first" "$source_report"; do
  [[ -f "$required" ]] || { echo "missing required asset: $required" >&2; exit 2; }
done
python - <<'PY'
import torch
devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('[preflight] visible GPUs:', devices, flush=True)
if not devices:
    raise SystemExit('A CUDA GPU is required')
PY

run_tag="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p ../logs ../outputs/heldout
smoke="../outputs/mone_selective_smoke_L4_${run_tag}.pt"
smoke_log="../logs/mone-selective-smoke-L4-${run_tag}.log"
echo "[smoke] log=$smoke_log"
python -u scripts/train_mone_selective.py --checkpoint "$MONE_CHECKPOINT" \
  --data-dir "$MONE_DATA_DIR" --source-report "$source_report" \
  --initial-adapter "$initial" --output "$smoke" \
  --frames 1 --steps 2 --smoke >"$smoke_log" 2>&1 || {
    rc=$?; tail -40 "$smoke_log" >&2; exit "$rc";
  }

adapter="../outputs/mone_selective_L4_${run_tag}.pt"
train_log="../logs/mone-selective-L4-${run_tag}.log"
echo "[train] log=$train_log adapter=$adapter"
python -u scripts/train_mone_selective.py --checkpoint "$MONE_CHECKPOINT" \
  --data-dir "$MONE_DATA_DIR" --source-report "$source_report" \
  --initial-adapter "$initial" --output "$adapter" \
  --frames 3 --steps 360 >"$train_log" 2>&1 || {
    rc=$?; tail -40 "$train_log" >&2; exit "$rc";
  }

for label in old selective; do
  if [[ "$label" == old ]]; then
    selected_adapter="$initial"
  else
    selected_adapter="$adapter"
  fi
  output="../outputs/heldout/L4-${label}-unused12-${run_tag}.json"
  log="../logs/mone-${label}-unused12-${run_tag}.log"
  echo "[eval] label=$label log=$log report=$output"
  python -u scripts/eval_mone_heldout.py --checkpoint "$MONE_CHECKPOINT" \
    --data-dir "$MONE_DATA_DIR" --adapter "$selected_adapter" --output "$output" \
    --exclude-report "$first" --exclude-report "$source_report" \
    --tasks 12 --frames 3 --draws 3 >"$log" 2>&1 || {
      rc=$?; tail -40 "$log" >&2; exit "$rc";
    }
done
echo "[done] trained_adapter=$adapter"
echo "[done] reports=../outputs/heldout/L4-{old,selective}-unused12-${run_tag}.json"
