# Mone × π0.5：4×H200 首轮试验

该目录是独立 Mone 工程，不导入或修改 ACoT 源码。环境部署只复用了此前实践中的
独立 venv、锁文件、离线变量、硬件预检和日志管理方式。

当前 `train_mone_layer.py` 是单层、小样本机制验证，并非完整规模训练器。首轮不改变
其优化目标或同步语义：四张 H200 分别训练第 5、10、15、18 层（索引 4/9/14/17），
比较插入层位；这比将 16 个样本强行做 DDP 更适合作为第一轮筛选。

```bash
cd mone_pi05_server/openpi
bash scripts/setup_mone_env.sh

# 首次执行：只读共享 pi0.5/LIBERO 资产，转换结果写入 Mone 自己的目录。
bash scripts/prepare_mone_assets.sh

# 必须先通过单卡两步检查。
bash scripts/run_mone_smoke.sh

# 查看四卡命令，不占 GPU。
DRY_RUN=1 bash scripts/run_mone_4xh200.sh

# 四卡并行层位筛选，默认每个实验 100 步。
bash scripts/run_mone_4xh200.sh
```

可通过 `LAYERS="..."`、`STEPS`、`EPISODES`、`HISTORY_TOKENS` 和
`LEARNING_RATE` 覆盖默认值。每个实验独立写入 `../outputs`，日志写入 `../logs`。

运行前 checkpoint 必须包含 `model.safetensors` 和
`assets/physical-intelligence/libero`；数据必须是包内说明要求的 LeRobot 布局。
登录节点没有 GPU，因此环境测试不能代替 H200 节点上的 smoke。

## 本机共享盘约定

`/path/to/workspace` 会同时挂载到登录节点和
GPU 计算节点。计算节点不能联网，因此 Python、tokenizer、数据和权重必须预先放在该共享盘。
脚本默认只读复用已经验证的共享 LIBERO 数据和官方 pi0.5 JAX base；不会修改 ACoT 代码、
环境或 checkpoint。转换出的 PyTorch 权重、tokenizer 副本和训练结果全部写在
`mone_pi05_server` 内。

默认路径由 `scripts/mone_env.sh` 自动设置，无需手写 `/path/to/...`：

```text
MONE_DATA_DIR   = ravenking/acot-vla/datasets/your_hf_username/libero
MONE_CHECKPOINT = ravenking/mone_pi05_server/checkpoints/pi05_libero_pytorch
OPENPI_DATA_HOME= ravenking/mone_pi05_server/assets/_openpi_cache
```

首次需在内存充足的计算节点转换权重：

```bash
cd /path/to/workspace/mone_pi05_server/openpi
bash scripts/prepare_mone_assets.sh
bash scripts/run_mone_smoke.sh
```
