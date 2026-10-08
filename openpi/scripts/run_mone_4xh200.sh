#!/usr/bin/env bash
# Run four unchanged single-layer mechanism experiments concurrently, one/H200.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh

: "${MONE_CHECKPOINT:?Set MONE_CHECKPOINT to a converted pi05_libero checkpoint}"
: "${MONE_DATA_DIR:?Set MONE_DATA_DIR to a LIBERO LeRobot dataset}"

IFS=' ' read -r -a LAYER_LIST <<< "${LAYERS:-4 9 14 17}"
[[ "${#LAYER_LIST[@]}" -eq 4 ]] || { echo "LAYERS must contain exactly four layer indices" >&2; exit 2; }

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -f "$MONE_CHECKPOINT/model.safetensors" ]] || { echo "[preflight] missing model.safetensors" >&2; exit 2; }
  [[ -d "$MONE_CHECKPOINT/assets/physical-intelligence/libero" ]] || { echo "[preflight] missing LIBERO norm stats" >&2; exit 2; }
  [[ -f "$MONE_DATA_DIR/meta/episodes.jsonl" ]] || { echo "[preflight] missing episodes.jsonl" >&2; exit 2; }
  python - <<'PY'
import torch
devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("[preflight] GPUs:", devices)
if len(devices) != 4:
    raise SystemExit(f"Expected exactly 4 visible GPUs, found {len(devices)}")
if any("H200" not in name.upper() for name in devices):
    raise SystemExit("All four visible GPUs must be H200")
PY
fi

mkdir -p ../logs ../outputs
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM

for gpu in 0 1 2 3; do
  layer="${LAYER_LIST[$gpu]}"
  output="../outputs/mone_layer_L${layer}_${RUN_TAG}.pt"
  log="../logs/mone-layer-L${layer}-${RUN_TAG}.log"
  cmd=(python -u scripts/train_mone_layer.py --checkpoint "$MONE_CHECKPOINT"
       --data-dir "$MONE_DATA_DIR" --episodes "${EPISODES:-0,1,2,10}"
       --layer "$layer" --steps "${STEPS:-100}" --history-tokens "${HISTORY_TOKENS:-64}"
       --learning-rate "${LEARNING_RATE:-1e-4}" --output "$output")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '[dry-run] GPU %d:' "$gpu"; printf ' %q' "${cmd[@]}"; printf '\n'
  else
    echo "[launch] GPU=$gpu layer=$layer log=$log"
    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log" 2>&1 &
    pids+=("$!")
  fi
done

[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
