#!/usr/bin/env bash
# Convert the shared official pi0.5 base checkpoint into Mone's private
# PyTorch checkpoint. Sources are read-only; all output stays under Mone.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/mone_env.sh

RAVENKING_ROOT="$(cd ../.. && pwd)"
MONE_SERVER_ROOT="$(cd .. && pwd)"
PI05_JAX_CHECKPOINT="${PI05_JAX_CHECKPOINT:-${PI05_JAX_PARAMS:-$RAVENKING_ROOT/acot-vla/checkpoints/pi05_base}}"
# The converter accepts the checkpoint root and appends /params internally.
# Also tolerate the old PI05_JAX_PARAMS=/.../params override.
if [[ "$(basename "$PI05_JAX_CHECKPOINT")" == "params" && -f "$PI05_JAX_CHECKPOINT/_METADATA" ]]; then
  PI05_JAX_CHECKPOINT="$(dirname "$PI05_JAX_CHECKPOINT")"
fi
LIBERO_STATS="${LIBERO_STATS:-$RAVENKING_ROOT/acot-vla/assets/acot_libero_action_cot_explicit_implicit_co_fusion/libero/norm_stats.json}"
SHARED_TOKENIZER="${SHARED_TOKENIZER:-$RAVENKING_ROOT/acot-vla/assets/_openpi_cache/big_vision/paligemma_tokenizer.model}"

[[ -f "$PI05_JAX_CHECKPOINT/params/_METADATA" ]] || { echo "missing shared pi0.5 params: $PI05_JAX_CHECKPOINT/params" >&2; exit 2; }
[[ -f "$LIBERO_STATS" ]] || { echo "missing shared LIBERO norm stats: $LIBERO_STATS" >&2; exit 2; }
[[ -f "$SHARED_TOKENIZER" ]] || { echo "missing shared tokenizer: $SHARED_TOKENIZER" >&2; exit 2; }
[[ -f "$MONE_DATA_DIR/meta/episodes.jsonl" ]] || { echo "missing shared LIBERO data: $MONE_DATA_DIR" >&2; exit 2; }

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "[dry-run] source checkpoint=$PI05_JAX_CHECKPOINT"
  echo "[dry-run] source dataset=$MONE_DATA_DIR"
  echo "[dry-run] output checkpoint=$MONE_CHECKPOINT"
  exit 0
fi

mkdir -p "$MONE_SERVER_ROOT/assets/_openpi_cache/big_vision"
cp -f "$SHARED_TOKENIZER" "$MONE_SERVER_ROOT/assets/_openpi_cache/big_vision/paligemma_tokenizer.model"

if [[ ! -f "$MONE_CHECKPOINT/model.safetensors" ]]; then
  echo "[prepare] converting pi0.5 JAX checkpoint; this needs substantial host RAM"
  python examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir "$PI05_JAX_CHECKPOINT" \
    --config-name pi05_libero \
    --output-path "$MONE_CHECKPOINT" \
    --precision bfloat16
else
  echo "[prepare] converted checkpoint already exists; skipping conversion"
fi

mkdir -p "$MONE_CHECKPOINT/assets/physical-intelligence/libero"
cp -f "$LIBERO_STATS" "$MONE_CHECKPOINT/assets/physical-intelligence/libero/norm_stats.json"
echo "[prepare] Mone assets ready: $MONE_CHECKPOINT"
