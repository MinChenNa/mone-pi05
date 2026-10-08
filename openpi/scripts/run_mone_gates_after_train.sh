#!/usr/bin/env bash
# Run offline gate and a paired LIBERO rollout screen after a selective run completes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LIBERO_CONFIG_PATH="$MONE_SERVER_ROOT/assets/libero_config"
export PYTHONPATH="$MONE_SERVER_ROOT/vendor/libero:$PYTHONPATH"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
# The H200 image omits GLVND sonames. Append private copies so existing system
# libraries keep precedence while the missing names become loadable.
gl_lib="$MONE_SERVER_ROOT/assets/glstub/lib"
[[ -f "$gl_lib/libEGL.so.1" ]] || { echo "missing Mone EGL library: $gl_lib" >&2; exit 2; }
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$gl_lib"

run_tag="${1:?Usage: bash scripts/run_mone_gates_after_train.sh TRAIN_RUN_TAG}"
old="../outputs/heldout/L4-old-unused12-${run_tag}.json"
new="../outputs/heldout/L4-selective-unused12-${run_tag}.json"
train="../outputs/mone_selective_L4_${run_tag}.json"
adapter="../outputs/mone_selective_L4_${run_tag}.pt"
source_report="../outputs/heldout/L4-controls-20260926-094629-91520.json"
gate1="../outputs/heldout/L4-gate1-${run_tag}.json"
for attempt in $(seq 1 720); do
  if [[ -f "$old" && -f "$new" && -f "$train" && -f "$adapter" ]]; then
    break
  fi
  if (( attempt % 6 == 1 )); then
    echo "[wait] selective training/evaluation is not complete; checking again in 10s"
  fi
  sleep 10
done
for required in "$old" "$new" "$train" "$adapter" "$source_report"; do
  [[ -f "$required" ]] || { echo "missing completed asset: $required" >&2; exit 2; }
done
if [[ ! -f "$gate1" ]]; then
  python scripts/check_mone_gate1.py --old "$old" --new "$new" \
    --train "$train" --output "$gate1"
fi
verdict="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$gate1")"
echo "[gate1] verdict=$verdict report=$gate1"
if [[ "$verdict" == fail_current_adapter ]]; then
  echo "[gate2] skipped because the new adapter failed the paired offline screen"
  exit 0
fi

python - <<'PY'
import torch
devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('[preflight] visible GPUs:', devices, flush=True)
if len(devices) < 3:
    raise SystemExit('Three visible CUDA GPUs are needed for the parallel rollout pilot')
PY

python - <<'PY'
from mujoco.egl import egl_ext as egl
from robosuite.utils.binding_utils import GLContext
devices = egl.eglQueryDevicesEXT()
print(f'[preflight] EGL devices: {len(devices)}', flush=True)
context = GLContext(max_width=64, max_height=64, device_id=-1)
context.make_current()
context.free()
print('[preflight] EGL context: ready', flush=True)
PY

smoke="../outputs/heldout/L4-rollout-smoke-${run_tag}.json"
smoke_log="../logs/mone-rollout-smoke-${run_tag}.log"
echo "[gate2-smoke] log=$smoke_log"
python -u scripts/eval_mone_rollout.py --checkpoint "$MONE_CHECKPOINT" \
  --data-dir "$MONE_DATA_DIR" --adapter "$adapter" --task-report "$new" \
  --source-report "$source_report" --output "$smoke" \
  --suites libero_spatial --max-tasks 1 --trials 1 >"$smoke_log" 2>&1 || {
    rc=$?; tail -50 "$smoke_log" >&2; exit "$rc";
  }

pids=()
outputs=()
cleanup() { for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
suites=(libero_10 libero_goal libero_spatial)
for gpu in 0 1 2; do
  suite="${suites[$gpu]}"
  output="../outputs/heldout/L4-rollout-${suite}-${run_tag}.json"
  log="../logs/mone-rollout-${suite}-${run_tag}.log"
  outputs+=("$output")
  echo "[gate2] GPU=$gpu suite=$suite log=$log report=$output"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" python -u scripts/eval_mone_rollout.py \
    --checkpoint "$MONE_CHECKPOINT" --data-dir "$MONE_DATA_DIR" \
    --adapter "$adapter" --task-report "$new" --source-report "$source_report" \
    --output "$output" --suites "$suite" --max-tasks 1 --trials 5 >"$log" 2>&1 &
  pids+=("$!")
done
rc=0
for index in 0 1 2; do
  if ! wait "${pids[$index]}"; then
    rc=1
    tail -50 "../logs/mone-rollout-${suites[$index]}-${run_tag}.log" >&2
  fi
done
(( rc == 0 )) || exit "$rc"
merged="../outputs/heldout/L4-rollout-paired-${run_tag}.json"
python scripts/merge_mone_rollouts.py --inputs "${outputs[@]}" --output "$merged"
echo "[done] gate1=$gate1 gate2=$merged"
