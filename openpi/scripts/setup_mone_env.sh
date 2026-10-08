#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${MONE_UV:-$(command -v uv || true)}"
PYTHON_DIR="$ROOT/../.tools/python"
cd "$ROOT"

[[ -x "$UV" ]] || { echo "uv not found: $UV" >&2; exit 2; }
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
# Keep the interpreter on shared storage too. A venv pointing into /root is
# unusable after moving from a login node to a GPU node.
"$UV" python install 3.11 --install-dir "$PYTHON_DIR" --no-bin
PYTHON_BIN="$(find "$PYTHON_DIR" -type f -path '*/bin/python3.11' -print -quit)"
[[ -x "$PYTHON_BIN" ]] || { echo "shared Python 3.11 installation failed" >&2; exit 2; }
if [[ ! -x .venv/bin/python ]] || [[ "$(readlink -f .venv/bin/python)" != "$PYTHON_BIN" ]]; then
  if [[ -e .venv ]]; then
    echo "Existing .venv uses a different interpreter; move it aside before setup." >&2
    exit 2
  fi
  "$UV" venv --python "$PYTHON_BIN" .venv
fi
GIT_LFS_SKIP_SMUDGE=1 "$UV" sync --python "$PYTHON_BIN" --frozen --link-mode copy
"$UV" pip install pyarrow

# The bundled replacement is required by this OpenPI revision. The environment
# uses link-mode=copy so this never modifies uv's shared package cache.
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
import transformers

source = Path("src/openpi/models_pytorch/transformers_replace")
destination = Path(transformers.__file__).parent
shutil.copytree(source, destination, dirs_exist_ok=True)
print(f"Patched Transformers in {destination}")
PY

.venv/bin/python - <<'PY'
import jax, pyarrow, torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("jax", jax.__version__)
print("transformers", transformers.__version__)
print("pyarrow", pyarrow.__version__)
PY
