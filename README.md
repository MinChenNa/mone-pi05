# MoNe × π0.5

为冻结的 π0.5 添加可读写的快速权重记忆，研究正确历史是否能改善机器人动作预测与闭环任务表现。

本仓库包含单层原型、18 层训练主线以及 LIBERO 离线和闭环对照评估。当前实现采用 delta-rule 快速权重与门控残差接入；是 MoNe 风格研究原型。full18 已完成官方 LIBERO 主干上的训练与闭环评测；最新总结还包含后续 stateful_v1 持续记忆 pilot。两代结果尚未证明成功率优于原始 π0.5。仓库包含结果总结，不包含原始实验报告、数据与模型权重；stateful_v1 代码尚未整合。

## 方法

```text
历史观察 t−10 → 冻结 PaliGemma → 各层历史特征 → 各层独立快速权重 W
当前观察     → PaliGemma 18 层 → 各层从 W 读取并注入残差 → action expert → 动作
```

主干参数冻结，训练 Q/K/V/输出投影、归一化与 gate。每层的记忆状态是固定大小矩阵；不增加当前 prefix 的 token 数。默认历史是 t−10 的一帧，不是连续十帧，状态按样本重新构建。18 层模型还包含同规模的当前帧写入对照，检验额外参数与历史内容各自的作用。

## 环境

完整训练使用 Linux、Python 3.11、NVIDIA GPU 和 uv。默认 18 层运行器针对 4×H200；CPU 测试无需权重和数据。OpenPI 依赖版本由 `openpi/uv.lock` 固定，Transformers 补丁必须安装。

```bash
cd openpi
bash scripts/setup_mone_env.sh
export MONE_DATA_DIR=/absolute/path/to/libero
export MONE_CHECKPOINT=/absolute/path/to/pi05_libero_pytorch
source scripts/mone_env.sh
```

setup 使用系统 PATH 中的 uv，可通过 `MONE_UV` 指定。首次依赖安装需要联网；实验运行默认离线。环境和解释器放在项目目录，可用于共享文件系统。已存在且使用不同解释器的虚拟环境需先自行移走。

checkpoint 需包含 `model.safetensors` 与 `assets/physical-intelligence/libero/norm_stats.json`。数据使用 LeRobot 布局，含 `meta/episodes.jsonl` 及 `data/chunk-000/episode_*.parquet`。请区分 π0.5 base 与官方 LIBERO 微调权重，并使用匹配的归一化统计量。

## 快速检查与单层实验

```bash
export PYTHONPATH="$PWD/src:$PWD"
.venv/bin/python -m pytest --confcutdir=src/openpi/models_pytorch \
  src/openpi/models_pytorch/mone_layer_test.py \
  scripts/mone_all_layers_test.py scripts/mone_guard_test.py \
  scripts/mone_selective_objective_test.py -q
.venv/bin/python scripts/train_mone_layer.py \
  --checkpoint "$MONE_CHECKPOINT" --data-dir "$MONE_DATA_DIR" \
  --layer 9 --steps 2 --output ../outputs/single-layer-smoke.pt
```

真实模型 smoke 检查 loss 有限、梯度非零和空记忆一致性。模块测试通过不等于真实训练或任务成功验证。

## 18 层实验

先按 [闭环环境说明](docs/ROLLOUT_SETUP.md) 补齐仿真依赖、资产和配置，再完成官方 π0.5 LIBERO 无记忆闭环基线。详细前置资产与步骤见 [实验说明](FULL18_EXPERIMENT.md)。数据计划必须在训练前冻结：

```bash
python scripts/plan_mone_full.py --data-dir "$MONE_DATA_DIR" \
  --output ../outputs/mone_full_plan_stratified_20260927.json
bash scripts/run_mone_full_4xh200.sh /absolute/path/to/baseline-summary.json
```

全流程包括两步 DDP smoke、历史模型 A 与当前帧对照 B、验证/测试离线评估、闭环 rollout 和结论汇总。任务按四个 suite 分层划分为 24/8/8，记录计划和主干哈希。默认每卡 microbatch 1、梯度累积 8、有效 batch 32，A/B 各 3,000 步。阶段、标签和步数见运行器顶部配置。断点续训必须保持 seed、累积次数、主干和数据计划一致。

## 最新实验结果（2026-10-08）

**目前没有证据表明记忆 adapter 的闭环成功率优于原始 π0.5。** 完整范围、训练配置、消融和限制见 [实验结果总结](docs/MONE_PI05_RESULTS.md)。数字来自用户提供的实验总结，本次未复跑或重新计算原始报告。

| 实验 | π0.5 | 记忆模型 | 范围 |
|---|---:|---:|---|
| full18 | 80/80（100%） | 77/80（96.25%） | adapter 未见的 8 个测试任务，3000 步训练 |
| stateful_v1 Online | 74/80（92.5%） | 74/80（92.5%） | 4 个 Long 任务：2 个验证、2 个测试，250 步 pilot |

full18 的正确历史相对基线改善离线 flow loss 约 0.00163，但相对同任务错误历史仅约 0.00006，区间跨零；满分基线也限制了正向成功率增益的观察。stateful_v1 在 Long task 8 上 Online 为 15/20，基线 14/20，No-write 10/20；这些结果提示动态写入可能有作用，但尚未证明持续累积记忆或正确历史关联具有稳定收益。两代任务划分、训练预算和协议不同，不可直接比较模型优劣。

stateful_v1 的 task 7 No-write 未完成，不计入总成功率，也不把缺失试次算作失败。四任务合计同时包括验证和测试任务；“未见”仅指 adapter 训练划分。实验均为单训练 seed，结论范围为 LIBERO 仿真。

持续记忆实验由用户提供结果记录，其代码与原始 JSON 尚未提供；仓库现有 full18 每次决策重新构造状态，不代表持续记忆实现。早期 L4/base 主干实验记录保留在 [历史评估说明](HELDOUT_VALIDATION.md)。下一阶段应在预先固定的非天花板任务上，用更多配对初始状态及至少 3 个训练 seed 比较基线、Online、同 adapter 重置/No-write、Short 和 Shuffled。

## 代码导航

| 文件 | 职责 |
|---|---|
| `openpi/src/openpi/models_pytorch/mone_pi05.py` | delta-rule 写入、读取及早期 prefix 原型 |
| `openpi/src/openpi/models_pytorch/mone_layer.py` | 单层门控残差和历史特征采集 |
| `openpi/src/openpi/models_pytorch/mone_all_layers.py` | 18 层独立记忆与配对训练目标 |
| `openpi/scripts/train_mone_all_layers.py` | 四卡 DDP 训练与断点保存 |
| `openpi/scripts/eval_mone_all_layers.py` | 配对离线对照 |
| `openpi/scripts/eval_mone_full_rollout.py` | 配对 LIBERO 闭环评估 |
| `openpi/scripts/check_mone_full_conclusion.py` | 证据条件检查与最终判定 |

早期 prefix token 接入与当前 layer hook 接入属于不同结构，权重及推理路径需匹配。环境特定的历史运维脚本保留在 `openpi/scripts`，先读参数和前置条件再使用。运行输出保存在根目录 `outputs/`、`logs/`，均由 Git 忽略。

仅跑 CPU 检查可安装 CPU 版 PyTorch、pytest 和 numpy，然后按上面测试命令执行；无需安装完整 OpenPI 环境。闭环和真实模型训练需另行验证。服务器包没有包含全套 LIBERO 仿真资产，本仓库中的 vendor 源码用于固定代码版本，闭环资产需要单独准备。

## 来源与许可

项目基于 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)。OpenPI 代码使用 [Apache-2.0](openpi/LICENSE)，Gemma 相关条件见 [LICENSE_GEMMA](openpi/LICENSE_GEMMA.txt)。随附 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 源码使用 [MIT](vendor/libero/LICENSE)，归属和条件见 [第三方说明](THIRD_PARTY_NOTICES.md)。数据集及模型权重须按各自条款单独获取。
