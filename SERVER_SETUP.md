# 服务器运行说明（Linux / NVIDIA GPU）

压缩包包含当前 openpi 源码、本次记忆模块及训练入口、依赖锁文件和实验说明。
不含 Windows 虚拟环境、模型权重、数据集和 Git 历史。无需覆盖服务器已有仓库，
解压后在独立目录运行即可。完整训练效果尚未验证。

## 1. 安装环境

服务器需已安装 uv，能够访问依赖源，并有可用的 NVIDIA 驱动。

```bash
cd mone_pi05_server/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --frozen --link-mode copy
uv pip install pyarrow
```

应用仓库随附的 Transformers 补丁（必须执行）：

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
import transformers
source = Path('src/openpi/models_pytorch/transformers_replace')
destination = Path(transformers.__file__).parent
shutil.copytree(source, destination, dirs_exist_ok=True)
print('Patched:', destination)
PY
```

环境安装使用 copy 模式，避免补丁修改 uv 的共享缓存。
若之后重新同步或重装 Transformers，请重新应用补丁。

## 2. 检查

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -m pytest --confcutdir=src/openpi/models_pytorch src/openpi/models_pytorch/mone_layer_test.py src/openpi/models_pytorch/mone_pi05_test.py -q
.venv/bin/python scripts/train_mone_layer.py --help
```

本地已通过上述 6 项模块测试；服务器安装流程尚未在 Linux 上实测。

## 3. 最小训练

将下面的两个路径改为服务器上的实际路径。先运行 2 步确认显存和完整链路，
再将 steps 改为 100。默认使用第 10 层（索引 9）。

```bash
.venv/bin/python scripts/train_mone_layer.py \
  --checkpoint /path/to/pi05_libero \
  --data-dir /path/to/libero \
  --episodes 0,1,2,10 \
  --layer 9 \
  --steps 2 \
  --output ../outputs/mone_layer_smoke.pt
```

checkpoint 需包含 `model.safetensors` 及
`assets/physical-intelligence/libero` 下的归一化统计量。
数据需包含 `meta/episodes.jsonl` 以及
`data/chunk-000/episode_000000.parquet` 等文件，具体见 MONE_QUICKSTART.md。
这是离线训练检查，不需要 LIBERO 仿真环境；第三方仿真子模块未包含在包中。

输出 `.pt` 为 adapter 权重，旁边 `.json` 为 loss 和对照诊断。
先确认 loss 有限、梯度非零、100 步训练可降低小数据 loss，
再判断是否值得开展独立轨迹评估。报告不是机器人任务成功率。
