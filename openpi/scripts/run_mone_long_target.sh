#!/usr/bin/env bash
# One-GPU LIBERO-10 targeted rollout for the frozen full18 Mone adapter.
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
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$MONE_SERVER_ROOT/assets/glstub/lib"

task_id="${1:?usage: run_mone_long_target.sh TASK_ID TRIALS TAG [MODE]}"
trials="${2:?usage: run_mone_long_target.sh TASK_ID TRIALS TAG [MODE]}"
tag="${3:?usage: run_mone_long_target.sh TASK_ID TRIALS TAG [MODE]}"
mode="${4:-relevant}"
[[ "$task_id" == 4 || "$task_id" == 8 || "$task_id" == 9 ]] || {
  echo 'Only preselected LIBERO-10 tasks 4, 8 and 9 are allowed' >&2; exit 2;
}
[[ "$trials" =~ ^[1-9][0-9]*$ && "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || {
  echo 'Invalid trial count or tag' >&2; exit 2;
}
[[ "$mode" == relevant || "$mode" == static_initial || "$mode" == current_frame || "$mode" == same_task ]] || {
  echo 'Invalid memory input mode' >&2; exit 2;
}
root="../outputs/full18/${MONE_FULL_TAG:-full18_20260927}"
plan="../outputs/mone_full_plan_stratified_20260927.json"
adapter="$root/memory/step003000.pt"
output="../outputs/long-targets/$tag/task${task_id}-${mode}-trials${trials}.json"
[[ -s "$MONE_CHECKPOINT/model.safetensors" && -f "$adapter" && -f "$plan" ]] || {
  echo 'Missing official checkpoint, Mone adapter, or plan' >&2; exit 2;
}
python - <<'PY'
import torch
from mujoco.egl import egl_ext as egl
devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('[preflight] GPUs:', devices, 'EGL:', len(egl.eglQueryDevicesEXT()), flush=True)
if len(devices) != 1 or not any(name in devices[0].upper() for name in ('H100', 'H200')):
    raise SystemExit('Exactly one H100 or H200 GPU must be visible')
PY
python -u scripts/eval_mone_long_targets.py \
  --checkpoint "$MONE_CHECKPOINT" --plan "$plan" --memory-adapter "$adapter" \
  --data-dir "$MONE_DATA_DIR" --task-id "$task_id" --mode "$mode" \
  --trials "$trials" --output "$output"
