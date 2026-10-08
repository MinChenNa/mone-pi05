#!/usr/bin/env bash
# Frozen L4 history-content screen. No training and no writes outside Mone.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

stage="${1:-prepare}"
tag=20260926-124724-16706
adapter="../outputs/mone_selective_L4_${tag}.pt"
plan="../outputs/heldout/L4-history-plan-${tag}.json"
mkdir -p ../outputs/heldout ../logs

if [[ "$stage" == prepare ]]; then
  python scripts/eval_mone_history_screen.py prepare \
    --checkpoint "$MONE_CHECKPOINT" --data-dir "$MONE_DATA_DIR" \
    --adapter "$adapter" --plan "$plan"
  exit 0
fi
if [[ "$stage" != screen && "$stage" != confirm ]]; then
  echo "usage: bash scripts/run_mone_history_screen.sh {prepare|screen|confirm}" >&2
  exit 2
fi
[[ -f "$plan" ]] || { echo "missing plan; run prepare first: $plan" >&2; exit 2; }
if [[ "$stage" == confirm ]]; then
  screen_report="../outputs/heldout/L4-history-screen-${tag}.json"
  [[ -f "$screen_report" ]] || { echo "missing screen report: $screen_report" >&2; exit 2; }
  verdict="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$screen_report")"
  [[ "$verdict" == screen_positive ]] || {
    echo "confirmation skipped: screen verdict=$verdict" >&2; exit 2;
  }
fi

python - <<'PY'
import torch
devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('[preflight] visible GPUs:', devices, flush=True)
if len(devices) < 4:
    raise SystemExit('Four visible CUDA GPUs are required')
PY

smoke="../outputs/heldout/L4-history-${stage}-smoke-${tag}.json"
smoke_log="../logs/mone-history-${stage}-smoke-${tag}.log"
echo "[smoke] log=$smoke_log report=$smoke"
CUDA_VISIBLE_DEVICES=0 python -u scripts/eval_mone_history_screen.py evaluate \
  --checkpoint "$MONE_CHECKPOINT" --data-dir "$MONE_DATA_DIR" \
  --adapter "$adapter" --plan "$plan" --output "$smoke" \
  --split "$stage" --shards 4 --shard 0 --frames 1 --draws 1 --smoke >"$smoke_log" 2>&1 || {
    rc=$?; tail -50 "$smoke_log" >&2; exit "$rc";
  }

pids=()
outputs=()
cleanup() { for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
for gpu in 0 1 2 3; do
  output="../outputs/heldout/L4-history-${stage}-shard${gpu}-${tag}.json"
  log="../logs/mone-history-${stage}-shard${gpu}-${tag}.log"
  outputs+=("$output")
  echo "[eval] GPU=$gpu log=$log report=$output"
  CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/eval_mone_history_screen.py evaluate \
    --checkpoint "$MONE_CHECKPOINT" --data-dir "$MONE_DATA_DIR" \
    --adapter "$adapter" --plan "$plan" --output "$output" \
    --split "$stage" --shards 4 --shard "$gpu" >"$log" 2>&1 &
  pids+=("$!")
done
rc=0
for gpu in 0 1 2 3; do
  if ! wait "${pids[$gpu]}"; then
    rc=1
    tail -50 "../logs/mone-history-${stage}-shard${gpu}-${tag}.log" >&2
  fi
done
(( rc == 0 )) || exit "$rc"
report="../outputs/heldout/L4-history-${stage}-${tag}.json"
python scripts/check_mone_history_screen.py --plan "$plan" --split "$stage" \
  --inputs "${outputs[@]}" --output "$report"
echo "[done] $report"
