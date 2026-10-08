#!/usr/bin/env bash
# Targeted, post-hoc diagnostic of the three full18 closed-loop regressions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export MONE_CHECKPOINT="$MONE_SERVER_ROOT/checkpoints/pi05_libero_official_pytorch"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LIBERO_CONFIG_PATH="$MONE_SERVER_ROOT/assets/libero_config"
export PYTHONPATH="$MONE_SERVER_ROOT/vendor/libero:$PYTHONPATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
gl_lib="$MONE_SERVER_ROOT/assets/glstub/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$gl_lib"

tag="${MONE_FULL_TAG:-full18_20260927}"
root="../outputs/full18/$tag"
plan="../outputs/mone_full_plan_stratified_20260927.json"
adapter="$root/memory/step003000.pt"
paired="$root/rollout-paired.json"
attempt="$(date -u +%Y%m%d-%H%M%S)-$$"
output_root="../outputs/diagnostics/failure-replay-$attempt"
mkdir -p "$output_root" ../logs
[[ -s "$MONE_CHECKPOINT/model.safetensors" && -f "$adapter" && -f "$paired" && -f "$plan" ]] || {
  echo 'Missing official checkpoint, completed adapter, plan, or original rollout report' >&2; exit 2;
}
python - <<'PY'
import torch
from mujoco.egl import egl_ext as egl
devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('[preflight] GPUs:', devices, 'EGL:', len(egl.eglQueryDevicesEXT()), flush=True)
if len(devices) < 3 or any('H200' not in name.upper() for name in devices[:3]):
    raise SystemExit('At least three visible H200 GPUs are required')
PY

pids=()
outputs=()
suites=(libero_goal libero_spatial libero_object)
for gpu in 0 1 2; do
  suite="${suites[$gpu]}"
  output="$output_root/$suite.json"
  log="../logs/mone-failure-replay-$suite-$attempt.log"
  outputs+=("$output")
  echo "[replay] GPU=$gpu suite=$suite log=$log report=$output"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    python -u scripts/eval_mone_failure_replay.py \
    --checkpoint "$MONE_CHECKPOINT" --plan "$plan" \
    --memory-adapter "$adapter" --paired-report "$paired" \
    --suite "$suite" --output "$output" >"$log" 2>&1 &
  pids+=("$!")
done
failed=0
for gpu in 0 1 2; do
  if ! wait "${pids[$gpu]}"; then
    failed=1
    echo "[failed] ${suites[$gpu]}" >&2
    tail -60 "../logs/mone-failure-replay-${suites[$gpu]}-$attempt.log" >&2
  fi
done
(( failed == 0 )) || exit 1
python scripts/check_mone_failure_replay.py --inputs "${outputs[@]}" \
  --output "$output_root/summary.json"
echo "[done] $output_root/summary.json"
