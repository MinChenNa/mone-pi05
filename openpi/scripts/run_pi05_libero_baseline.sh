#!/usr/bin/env bash
# Test official pi0.5 LIBERO competence before training a new Mone adapter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export MONE_CHECKPOINT="${MONE_OFFICIAL_CHECKPOINT:-$MONE_SERVER_ROOT/checkpoints/pi05_libero_official_pytorch}"
echo "[baseline] checkpoint=$MONE_CHECKPOINT"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LIBERO_CONFIG_PATH="$MONE_SERVER_ROOT/assets/libero_config"
export PYTHONPATH="$MONE_SERVER_ROOT/vendor/libero:$PYTHONPATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
gl_lib="$MONE_SERVER_ROOT/assets/glstub/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$gl_lib"
[[ -s "$MONE_CHECKPOINT/model.safetensors" ]] || {
  echo "missing official converted checkpoint; run prepare_mone_official_libero.sh all" >&2; exit 2;
}
[[ -s "$MONE_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json" ]] || {
  echo "missing official LIBERO normalization stats" >&2; exit 2;
}
python - <<'PY'
import torch
devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('[preflight] visible GPUs:', devices, flush=True)
if len(devices) < 3:
    raise SystemExit('Three visible CUDA GPUs are required')
from mujoco.egl import egl_ext as egl
from robosuite.utils.binding_utils import GLContext
print('[preflight] EGL devices:', len(egl.eglQueryDevicesEXT()), flush=True)
context = GLContext(max_width=64, max_height=64, device_id=-1)
context.make_current()
context.free()
print('[preflight] EGL context: ready', flush=True)
PY

run_tag="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p ../logs ../outputs/baseline
smoke="../outputs/baseline/pi05-official-smoke-${run_tag}.json"
smoke_log="../logs/pi05-official-baseline-smoke-${run_tag}.log"
echo "[smoke] log=$smoke_log"
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -u scripts/eval_pi05_libero_baseline.py \
  --checkpoint "$MONE_CHECKPOINT" --suite libero_spatial --task-id 2 --trials 1 \
  --output "$smoke" >"$smoke_log" 2>&1 || {
    rc=$?; tail -50 "$smoke_log" >&2; exit "$rc";
  }

suites=(libero_spatial libero_goal libero_10)
task_ids=(2 4 1)
pids=()
outputs=()
cleanup() { for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
for gpu in 0 1 2; do
  suite="${suites[$gpu]}"
  output="../outputs/baseline/pi05-official-${suite}-${run_tag}.json"
  log="../logs/pi05-official-${suite}-${run_tag}.log"
  outputs+=("$output")
  echo "[baseline] GPU=$gpu suite=$suite task=${task_ids[$gpu]} log=$log report=$output"
  CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
    python -u scripts/eval_pi05_libero_baseline.py --checkpoint "$MONE_CHECKPOINT" \
    --suite "$suite" --task-id "${task_ids[$gpu]}" --trials 5 --output "$output" \
    >"$log" 2>&1 &
  pids+=("$!")
done
rc=0
for gpu in 0 1 2; do
  if ! wait "${pids[$gpu]}"; then
    rc=1
    tail -50 "../logs/pi05-official-${suites[$gpu]}-${run_tag}.log" >&2
  fi
done
(( rc == 0 )) || exit "$rc"
summary="../outputs/baseline/pi05-official-summary-${run_tag}.json"
python - "$summary" "${outputs[@]}" <<'PY'
import json, sys
from pathlib import Path
summary = Path(sys.argv[1])
reports = []
for path in sys.argv[2:]:
    result = json.load(open(path))
    reports.append(result)
    print(f"[done] {result['suite']}:{result['task_id']} "
          f"success={result['successes']}/{result['trials']} report={path}")
summary.write_text(json.dumps(dict(status='complete', checkpoint=reports[0]['checkpoint'],
                                   successes=sum(r['successes'] for r in reports),
                                   trials=sum(r['trials'] for r in reports),
                                   reports=sys.argv[2:]), indent=2))
print(f'[done] baseline_summary={summary}')
PY
