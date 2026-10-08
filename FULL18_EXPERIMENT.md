# Mone π0.5：18 层上下文记忆完整试验

> 历史记录：本页保留当时的计划与实验进展，后续 full18 与持续记忆结果见 [2026-10-08 实验总结](docs/MONE_PI05_RESULTS.md)。后续结果状态以该总结为准。

范围固定为 PaliGemma 的 18 个视觉语言上下文 Transformer 层。每层一个独立
`LayerMemory`，从同一历史观测的一次冻结前向中提取该层特征，写入该层的
快速权重；当前观测只在对应层读取。π0.5 主干冻结，动作 expert 和视觉
编码器不插入新模块。旧 L4 adapter 不参与本试验。

## 实验与对照

- 数据计划：`outputs/mone_full_plan_stratified_20260927.json`，四个 LIBERO
  suite 各有 6/2/2 个 train/val/test 任务，计划内容带 SHA256 校验。
- 模型 A：18 层记忆写入同轨迹 10 步前观测。
- 模型 B：同样 18 层、参数量、训练步数、数据抽样与种子，写入当前帧；
  作为非历史参数对照。
- 两组均采用 7 维真实动作 flow loss 与有界的同任务错误历史偏好项。
  冻结主干无记忆 loss 作为错误历史损失上限，避免故意破坏错误输入。
- 默认 4×H200、每卡 microbatch 1、累积 8 次，有效 batch 32；每组
  3,000 次优化步骤，定期保存断点。训练任务与最终测试任务完全分离。
- 验证和最终测试在同样的动作、噪声、时间下比较无记忆、正确历史、
  同任务错误历史、固定历史、当前帧，以及模型 B。按任务汇总配对
  差值；重复噪声不计为独立样本。
- 闭环在测试集 8 个任务各 10 个官方初始状态上，用相同动作采样种子
  比较上述五种可运行条件，共 80 个配对初始状态、最多 400 条 rollout。

## 启动顺序

在内存额度至少 32 GiB 的 H200 节点，先完成官方 π0.5 LIBERO 转换：

```bash
cd /path/to/workspace/mone_pi05_server/openpi
bash scripts/prepare_mone_official_libero.sh convert
bash scripts/run_pi05_libero_baseline.sh
```

基线脚本末尾打印 `baseline_summary=...json`。确认它至少有一个成功案例后，
将该路径传给完整试验：

```bash
bash scripts/run_mone_full_4xh200.sh ../outputs/baseline/pi05-official-summary-实际运行标签.json
```

全流程先做 2 步 DDP smoke，接着顺序训练 A/B、做 val/test 离线配对，最后
并行闭环。运行名默认 `full18_20260927`，可通过 `MONE_FULL_TAG` 改名；
中断后同名重跑会从最近的训练断点恢复。`MONE_FULL_STAGE` 可设为
`train`、`eval`、`rollout` 或 `all`。训练步数、累计次数、闭环次数可用
`MONE_FULL_STEPS`、`MONE_FULL_ACCUMULATION`、`MONE_FULL_ROLLOUT_TRIALS`
设置，但用于最终比较时两组必须保持一致。

## 结论口径

离线 `test-paired.json` 预设要求正确历史对无记忆、错误历史、固定历史、
当前帧和同参数模型 B 的任务级配对区间下界都大于 0，且对非空对照的
最小平均收益至少 0.001。闭环 `rollout-paired.json` 要求官方基线能成功、
正确历史对每个对照的任务级区间下界大于 0，最小成功率差至少 5 个百分点。
满足两者可称为支持本实现有历史特异性与任务成功收益。单次训练种子、
8 个测试任务和可能接近成功率上限的官方模型仍限制结论外推；若区间跨零，
报告 `inconclusive`，不把无显著差异解释成思想本身无效。
最终自动写入 `outputs/full18/<运行名>/conclusion.json`，并校验三道关使用的
checkpoint、训练 adapter 和测试集一致。

此前登录容器只有 16 GiB 且无 GPU，官方 JAX 资产已经下载但 PyTorch
转换与全层 GPU smoke 尚未在该容器完成。ACoT 目录只读复用 LIBERO 数据。
