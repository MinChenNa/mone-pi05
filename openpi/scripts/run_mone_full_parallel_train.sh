#!/usr/bin/env bash
# Resume history training and start the matched current-frame control on the same four H200s.
# Stop the old sequential launcher first: it otherwise starts a second control job.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export MONE_CHECKPOINT="$MONE_SERVER_ROOT/checkpoints/pi05_libero_official_pytorch"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

baseline_summary="${1:?Usage: bash scripts/run_mone_full_parallel_train.sh BASELINE_SUMMARY_JSON}"
tag="${MONE_FULL_TAG:-full18_20260927}"
steps="${MONE_FULL_STEPS:-3000}"
accumulation="${MONE_FULL_ACCUMULATION:-8}"
plan="../outputs/mone_full_plan_stratified_20260927.json"
root="../outputs/full18/$tag"
memory="$root/memory"
control="$root/control"
attempt="$(date -u +%Y%m%d-%H%M%S)-$$"
mkdir -p "$memory" "$control" ../logs

[[ -s "$MONE_CHECKPOINT/model.safetensors" && -s "$baseline_summary" && -s "$plan" ]] || {
  echo 'Missing official checkpoint, baseline summary, or frozen data plan' >&2; exit 2;
}
[[ -f "$root/smoke/complete.json" ]] || {
  echo 'The original four-GPU smoke test must complete first' >&2; exit 2;
}
python - "$baseline_summary" "$MONE_CHECKPOINT" "$tag" <<'PY'
import json, os, pathlib, sys
import torch

baseline = json.load(open(sys.argv[1]))
if (baseline.get('status') != 'complete' or baseline.get('successes', 0) <= 0 or
        baseline.get('checkpoint') != str(pathlib.Path(sys.argv[2]).resolve())):
    raise SystemExit('Official pi0.5 baseline/checkpoint preflight failed')
tag = sys.argv[3]
conflicts = []
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit():
        continue
    try:
        command = (entry / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if ('run_mone_full_4xh200.sh' in command or
            ('train_mone_all_layers.py' in command and f'/full18/{tag}/' in command)):
        conflicts.append((entry.name, command[:200]))
if conflicts:
    for pid, command in conflicts:
        print(f'[preflight] conflicting process PID={pid}: {command}', file=sys.stderr)
    raise SystemExit('Stop the old launcher and training workers before parallel resume')
if torch.cuda.device_count() != 4 or any(
        'H200' not in torch.cuda.get_device_name(i).upper() for i in range(4)):
    raise SystemExit('Four visible H200 GPUs are required')
free_gib = [torch.cuda.mem_get_info(i)[0] / 1024**3 for i in range(4)]
if min(free_gib) < 25:
    raise SystemExit(f'Need at least 25 GiB free per GPU for the second model; free={free_gib}')
print(f'[preflight] baseline={baseline["successes"]}/{baseline["trials"]} '
      f'free_gpu_gib={[round(value, 1) for value in free_gib]}', flush=True)
PY

launch_one() {
  local mode="$1" run_dir="$2" log="$3"
  if [[ -f "$run_dir/complete.json" ]]; then
    echo "[skip] already complete: $run_dir"
    return
  fi
  local latest resume=()
  latest="$(find "$run_dir" -maxdepth 1 -name 'step*.pt' -type f | sort | tail -1)"
  if [[ -n "$latest" ]]; then
    resume=(--resume "$latest")
    echo "[resume] mode=$mode checkpoint=$latest"
  elif compgen -G "$run_dir/rank*.jsonl" >/dev/null; then
    echo "Partial metrics without checkpoint in $run_dir; inspect before starting" >&2
    exit 2
  fi
  echo "[launch] mode=$mode steps=$steps accumulation=$accumulation log=$log"
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/train_mone_all_layers.py --checkpoint "$MONE_CHECKPOINT" \
    --data-dir "$MONE_DATA_DIR" --plan "$plan" --run-dir "$run_dir" \
    --history-mode "$mode" --steps "$steps" --accumulation "$accumulation" \
    --save-every 250 "${resume[@]}" >"$log" 2>&1 &
  pids+=("$!")
  labels+=("$mode")
  logs+=("$log")
}

pids=()
labels=()
logs=()
stop_children() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap stop_children INT TERM
launch_one true_history "$memory" "../logs/mone-full-${tag}-memory-parallel-${attempt}.log"
launch_one current_frame "$control" "../logs/mone-full-${tag}-control-parallel-${attempt}.log"
failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failed=1
    echo "[failed] ${labels[$index]} log tail:" >&2
    tail -60 "${logs[$index]}" >&2 || true
  else
    echo "[done] ${labels[$index]} training completed"
  fi
done
(( failed == 0 )) || exit 1

echo '[eval] both matched adapters complete; starting held-out evaluations'
MONE_FULL_STAGE=eval bash scripts/run_mone_full_4xh200.sh "$baseline_summary"
MONE_FULL_STAGE=rollout bash scripts/run_mone_full_4xh200.sh "$baseline_summary"
