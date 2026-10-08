#!/usr/bin/env bash
# Download the upstream LIBERO-finetuned pi0.5 into Mone, then convert privately.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

stage="${1:-all}"
if [[ "$stage" != all && "$stage" != download && "$stage" != convert ]]; then
  echo "usage: bash scripts/prepare_mone_official_libero.sh {all|download|convert}" >&2
  exit 2
fi

source_root="$MONE_SERVER_ROOT/checkpoints/pi05_libero_official_jax"
target="$MONE_SERVER_ROOT/checkpoints/pi05_libero_official_pytorch"
working="$MONE_SERVER_ROOT/checkpoints/.pi05_libero_official_pytorch_converting"
prefix=checkpoints/pi05_libero/
bucket=https://storage.googleapis.com/openpi-assets

if [[ "$stage" == all || "$stage" == download ]]; then
  listing="$(curl -fsS --retry 3 --connect-timeout 20 --max-time 60 \
    "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=${prefix}&maxResults=1000")"
  count=0
  while read -r name size; do
    [[ -n "$name" ]] || continue
    rel="${name#${prefix}}"
    [[ "$rel" != "$name" && "$rel" != /* && "$rel" != *..* ]] || {
      echo "unsafe object name: $name" >&2; exit 2;
    }
    destination="$source_root/$rel"
    partial="$destination.download"
    mkdir -p "$(dirname "$destination")"
    if [[ -f "$destination" ]]; then
      [[ "$(stat -c %s "$destination")" == "$size" ]] || {
        echo "existing file has wrong size: $destination" >&2; exit 2;
      }
      echo "[skip] $rel"
    else
      echo "[get] $rel size=$size"
      curl -fsSL --retry 6 --retry-delay 5 --retry-all-errors \
        --connect-timeout 20 -C - -o "$partial" "$bucket/$name"
      [[ "$(stat -c %s "$partial")" == "$size" ]] || {
        echo "download size mismatch: $partial" >&2; exit 2;
      }
      mv "$partial" "$destination"
    fi
    count=$((count + 1))
  done < <(printf '%s' "$listing" | python3 -c '
import json, sys
data = json.load(sys.stdin)
if data.get("nextPageToken"):
    raise SystemExit("GCS listing needs pagination")
for item in data.get("items", []):
    print(item["name"], item["size"])
')
  (( count >= 10 )) || { echo "incomplete object listing: $count" >&2; exit 2; }
  echo "[downloaded] files=$count source=$source_root"
fi

if [[ "$stage" == all || "$stage" == convert ]]; then
  [[ -f "$source_root/params/_METADATA" ]] || { echo "missing JAX metadata" >&2; exit 2; }
  [[ -f "$source_root/assets/physical-intelligence/libero/norm_stats.json" ]] || {
    echo "missing official LIBERO norm stats" >&2; exit 2;
  }
  if [[ -f "$target/model.safetensors" ]]; then
    echo "[skip] converted checkpoint already exists: $target"
  else
    # Orbax restore, NumPy arrays and PyTorch state dict overlap in host RAM.
    # The 16 GiB login container was OOM-killed during this exact conversion.
    memory_limit=''
    if [[ -f /sys/fs/cgroup/memory.max ]]; then
      memory_limit="$(< /sys/fs/cgroup/memory.max)"
    elif [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
      memory_limit="$(< /sys/fs/cgroup/memory/memory.limit_in_bytes)"
    fi
    min_ram_gib="${MONE_MIN_CONVERT_RAM_GIB:-32}"
    [[ "$min_ram_gib" =~ ^[0-9]+$ ]] || { echo "invalid MONE_MIN_CONVERT_RAM_GIB" >&2; exit 2; }
    if [[ "$memory_limit" =~ ^[0-9]+$ ]] && \
       (( memory_limit < min_ram_gib * 1024 * 1024 * 1024 )); then
      echo "conversion needs a larger-memory node: cgroup limit=$((memory_limit/1024/1024/1024)) GiB, required >=${min_ram_gib} GiB" >&2
      echo "run this same script on the H200 node; downloaded JAX assets are already on shared storage" >&2
      exit 2
    fi
    [[ ! -e "$target" && ! -e "$working" ]] || {
      echo "partial conversion exists; inspect before retrying: $target or $working" >&2; exit 2;
    }
    echo "[convert] official LIBERO pi0.5 JAX -> PyTorch; requires substantial host RAM"
    python -u examples/convert_jax_model_to_pytorch.py \
      --checkpoint-dir "$source_root" --config-name pi05_libero \
      --output-path "$working" --precision bfloat16
    [[ -s "$working/model.safetensors" ]] || { echo "conversion produced no model" >&2; exit 2; }
    mv "$working" "$target"
  fi
  mkdir -p "$target/assets/physical-intelligence/libero"
  cp "$source_root/assets/physical-intelligence/libero/norm_stats.json" \
    "$target/assets/physical-intelligence/libero/norm_stats.json"
  echo "[ready] MONE_CHECKPOINT=$target"
fi
