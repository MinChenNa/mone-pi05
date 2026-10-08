#!/usr/bin/env bash
# Source this file before running MoNe experiments.

MONE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MONE_SERVER_ROOT="$(cd "$MONE_ROOT/.." && pwd)"
[[ -x "$MONE_ROOT/.venv/bin/python" ]] || {
  echo "[mone_env] missing environment; run: bash scripts/setup_mone_env.sh" >&2
  return 1 2>/dev/null || exit 1
}

source "$MONE_ROOT/.venv/bin/activate"
export PYTHONPATH="$MONE_ROOT/src:$MONE_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
# Ignore the literal placeholders from the early quick-start instructions when
# they remain exported in a long-lived shell. Real custom paths are preserved.
if [[ -z "${MONE_CHECKPOINT:-}" || "${MONE_CHECKPOINT:-}" == "/path/to/pi05_libero_pytorch" ]]; then
  export MONE_CHECKPOINT="$MONE_SERVER_ROOT/checkpoints/pi05_libero_pytorch"
fi
if [[ -z "${MONE_DATA_DIR:-}" || "${MONE_DATA_DIR:-}" == "/path/to/libero" ]]; then
  export MONE_DATA_DIR="$MONE_SERVER_ROOT/data/libero"
fi
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$MONE_SERVER_ROOT/assets/_openpi_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "[mone_env] project=$MONE_ROOT"
echo "[mone_env] python=$(command -v python)"
echo "[mone_env] checkpoint=$MONE_CHECKPOINT"
echo "[mone_env] dataset=$MONE_DATA_DIR"
echo "[mone_env] openpi_cache=$OPENPI_DATA_HOME"
