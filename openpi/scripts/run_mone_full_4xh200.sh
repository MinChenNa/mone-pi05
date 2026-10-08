#!/usr/bin/env bash
# End-to-end 18-context-layer Mone experiment on four H200 GPUs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export MONE_CHECKPOINT="${MONE_OFFICIAL_CHECKPOINT:-$MONE_SERVER_ROOT/checkpoints/pi05_libero_official_pytorch}"
echo "[mone_full] official_checkpoint=$MONE_CHECKPOINT"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LIBERO_CONFIG_PATH="$MONE_SERVER_ROOT/assets/libero_config"
export PYTHONPATH="$MONE_SERVER_ROOT/vendor/libero:$PYTHONPATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
gl_lib="$MONE_SERVER_ROOT/assets/glstub/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$gl_lib"

baseline_summary="${1:?Usage: bash scripts/run_mone_full_4xh200.sh BASELINE_SUMMARY_JSON}"
stage="${MONE_FULL_STAGE:-all}"
[[ "$stage" == all || "$stage" == train || "$stage" == eval || "$stage" == rollout ]] || {
  echo "MONE_FULL_STAGE must be all, train, eval, or rollout" >&2; exit 2;
}
tag="${MONE_FULL_TAG:-full18_20260927}"
steps="${MONE_FULL_STEPS:-3000}"
accumulation="${MONE_FULL_ACCUMULATION:-8}"
trials="${MONE_FULL_ROLLOUT_TRIALS:-10}"
plan="${MONE_PLAN:-../outputs/mone_full_plan_stratified_20260927.json}"
root="../outputs/full18/$tag"
mkdir -p "$root" ../logs
[[ -s "$MONE_CHECKPOINT/model.safetensors" ]] || {
  echo "missing official converted pi0.5 LIBERO checkpoint: $MONE_CHECKPOINT" >&2; exit 2;
}
[[ -s "$MONE_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json" ]] || {
  echo "missing official LIBERO normalization stats" >&2; exit 2;
}
[[ -f "$plan" && -f "$baseline_summary" ]] || {
  echo "missing frozen plan or baseline summary" >&2; exit 2;
}
python - "$baseline_summary" "$MONE_CHECKPOINT" <<'PY'
import json, pathlib, sys
report = json.load(open(sys.argv[1]))
checkpoint = str(pathlib.Path(sys.argv[2]).resolve())
if report.get('status') != 'complete' or report.get('successes', 0) <= 0:
    raise SystemExit('Official pi0.5 baseline has not succeeded; inspect policy before Mone training')
if report.get('checkpoint') != checkpoint:
    raise SystemExit('Baseline used a different checkpoint')
print(f"[preflight] official baseline success={report['successes']}/{report['trials']}")
PY
python - <<'PY'
import torch
devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('[preflight] visible GPUs:', devices, flush=True)
if len(devices) != 4 or any('H200' not in item.upper() for item in devices):
    raise SystemExit('Four H200 GPUs are required')
PY

train_one() {
  local history_mode="$1" run_dir="$2" log="$3" training_steps="$4" gradient_accumulation="$5"
  mkdir -p "$run_dir"
  if [[ -f "$run_dir/complete.json" ]]; then
    echo "[skip] completed training: $run_dir"
    return
  fi
  local resume=()
  local latest
  latest="$(find "$run_dir" -maxdepth 1 -name 'step*.pt' -type f | sort | tail -1)"
  if [[ -n "$latest" ]]; then
    resume=(--resume "$latest")
    echo "[resume] $history_mode from $latest"
  elif compgen -G "$run_dir/rank*.jsonl" >/dev/null; then
    echo "partial metrics without checkpoint in $run_dir; choose a new MONE_FULL_TAG" >&2
    exit 2
  fi
  echo "[train] mode=$history_mode steps=$training_steps effective_batch=$((4*gradient_accumulation)) log=$log"
  python -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/train_mone_all_layers.py --checkpoint "$MONE_CHECKPOINT" \
    --data-dir "$MONE_DATA_DIR" --plan "$plan" --run-dir "$run_dir" \
    --history-mode "$history_mode" --steps "$training_steps" \
    --accumulation "$gradient_accumulation" --save-every 250 \
    "${resume[@]}" >"$log" 2>&1 || {
      rc=$?; tail -80 "$log" >&2; exit "$rc";
    }
}

if [[ "$stage" == all || "$stage" == train ]]; then
  train_one true_history "$root/smoke" "../logs/mone-full-${tag}-smoke.log" 2 1
  train_one true_history "$root/memory" "../logs/mone-full-${tag}-memory.log" "$steps" "$accumulation"
  train_one current_frame "$root/control" "../logs/mone-full-${tag}-control.log" "$steps" "$accumulation"
fi
memory="$root/memory/step$(printf '%06d' "$steps").pt"
control="$root/control/step$(printf '%06d' "$steps").pt"
[[ -f "$memory" && -f "$control" ]] || {
  echo "missing matched trained adapters: $memory and $control" >&2; exit 2;
}

if [[ "$stage" == all || "$stage" == eval ]]; then
  for split in val test; do
    pids=()
    outputs=()
    for gpu in 0 1 2 3; do
      output="$root/${split}-shard${gpu}.json"
      log="../logs/mone-full-${tag}-${split}-shard${gpu}.log"
      outputs+=("$output")
      if [[ -f "$output" ]]; then
        echo "[skip] completed evaluation shard: $output"
        continue
      fi
      [[ ! -f "${output%.json}.jsonl" ]] || {
        echo "partial evaluation shard exists: ${output%.json}.jsonl" >&2; exit 2;
      }
      echo "[eval] split=$split GPU=$gpu log=$log"
      CUDA_VISIBLE_DEVICES="$gpu" python -u scripts/eval_mone_all_layers.py \
        --checkpoint "$MONE_CHECKPOINT" --data-dir "$MONE_DATA_DIR" --plan "$plan" \
        --memory-adapter "$memory" --control-adapter "$control" \
        --split "$split" --shard "$gpu" --shards 4 --output "$output" >"$log" 2>&1 &
      pids+=("$!")
    done
    rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    (( rc == 0 )) || { echo "evaluation failed; inspect ../logs/mone-full-${tag}-${split}-shard*.log" >&2; exit "$rc"; }
    report="$root/${split}-paired.json"
    if [[ ! -f "$report" ]]; then
      python scripts/check_mone_full_eval.py --plan "$plan" --split "$split" \
        --inputs "${outputs[@]}" --output "$report"
    fi
  done
fi

if [[ "$stage" == all || "$stage" == rollout ]]; then
  [[ -f "$root/test-paired.json" ]] || {
    echo "test offline report must complete before rollout" >&2; exit 2;
  }
  python - <<'PY'
from mujoco.egl import egl_ext as egl
from robosuite.utils.binding_utils import GLContext
print('[preflight] EGL devices:', len(egl.eglQueryDevicesEXT()), flush=True)
context = GLContext(max_width=64, max_height=64, device_id=-1)
context.make_current()
context.free()
print('[preflight] EGL context: ready', flush=True)
PY
  pids=()
  outputs=()
  suites=(libero_spatial libero_object libero_goal libero_10)
  for gpu in 0 1 2 3; do
    suite="${suites[$gpu]}"
    output="$root/rollout-${suite}.json"
    log="../logs/mone-full-${tag}-rollout-${suite}.log"
    outputs+=("$output")
    if [[ -f "$output" ]]; then
      echo "[skip] completed rollout suite: $output"
      continue
    fi
    [[ ! -f "${output%.json}.jsonl" ]] || {
      echo "partial rollout suite exists: ${output%.json}.jsonl" >&2; exit 2;
    }
    echo "[rollout] GPU=$gpu suite=$suite trials=$trials log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" \
      python -u scripts/eval_mone_full_rollout.py \
      --checkpoint "$MONE_CHECKPOINT" --data-dir "$MONE_DATA_DIR" --plan "$plan" \
      --memory-adapter "$memory" --control-adapter "$control" \
      --suite "$suite" --trials "$trials" --output "$output" >"$log" 2>&1 &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  (( rc == 0 )) || { echo "rollout failed; inspect ../logs/mone-full-${tag}-rollout-*.log" >&2; exit "$rc"; }
  report="$root/rollout-paired.json"
  if [[ ! -f "$report" ]]; then
    python scripts/check_mone_full_rollout.py --plan "$plan" \
      --inputs "${outputs[@]}" --output "$report"
  fi
fi
if [[ -f "$root/test-paired.json" && -f "$root/rollout-paired.json" ]]; then
  conclusion="$root/conclusion.json"
  if [[ ! -f "$conclusion" ]]; then
    python scripts/check_mone_full_conclusion.py \
      --baseline "$baseline_summary" --offline "$root/test-paired.json" \
      --rollout "$root/rollout-paired.json" --output "$conclusion"
  fi
  echo "[done] conclusion=$conclusion"
fi
echo "[done] experiment=$root stage=$stage"
