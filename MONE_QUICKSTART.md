# π0.5 单层记忆最小实验

新增实现沿用现有 delta-rule fast weights，默认在 PaliGemma 第 10 层
（从 0 计数为 9）的 input layer norm 输出注入有界记忆残差。
历史和当前 query 使用同一层特征。没有新增 token 或改变 RoPE 位置。
训练和推理使用同一挂接点；这是一版残差 adapter 实验，**不是论文完整复现，
也不是讨论中显式追加 K_mem/V_mem 的实现**。

原 π0.5 参数冻结；只训练记忆读写投影、归一化和 gate。历史来自同一轨迹
当前帧之前 10 帧，缓存冻结特征，再通过原生 flow-matching action loss
反传到 adapter。短历史的 delta 更新直接求导，不使用二阶梯度或长序列元学习。
每个样本重新初始化记忆，避免跨 episode 污染。

## 运行

在 `F:\VLA\MoNe\openpi` 中执行，替换两个实际目录：

```powershell
.\.venv\Scripts\python.exe scripts/train_mone_layer.py --checkpoint <pi05_libero目录> --data-dir <LIBERO数据目录> --episodes 0,1,2,10 --steps 100 --layer 9 --output ../checkpoints/mone_layer.pt
```

checkpoint 需有 `model.safetensors` 和
`assets/physical-intelligence/libero` 的归一化统计量；数据需有
`meta/episodes.jsonl` 和 `data/chunk-000/episode_000000.parquet` 等文件。
沿用本项目已有 LIBERO 图像、状态、动作列及预处理流程。仅支持该数据布局。
截至本次检查，工作区数据目录为空，checkpoint 目录只有旧 adapter 文件。

默认 4 条轨迹、每条最多 4 个样本，batch size 1，256 维记忆。
固定 noise 和 time，禁用训练图像增强，方便确认能否过拟合。
这不是完整训练分布。输出 adapter 权重及同名 JSON 诊断报告。
报告比较 baseline / relevant / empty / shuffled（历史 token 顺序置换），
全部属于训练集诊断；后续应另做独立轨迹上的正确/错误历史对照。
报告中的 loss 是 flow-matching loss，不是执行动作 MSE 或任务成功率。

## 推理接入

在加载原始模型后创建 `LayerMemory` 并加载保存的 `adapter` state dict。
使用 `capture_history(model, historical_observation, layer=9)` 提取历史，
通过 `adapter.build_state(features, mask)` 构建状态，然后：

```python
with adapter.activate(model, memory_state, layer=9):
    actions = model.sample_actions(device, current_observation)
```

历史必须在当前帧之前；每条新轨迹重建状态。不要与旧 prefix adapter 同时启用。
训练时上下文必须覆盖 backward，以兼容梯度检查点重算。挂接上下文不支持并发请求。

## 已验证与下一步

模块测试覆盖空记忆恒等、历史影响、读写投影梯度及 padding 不写入。
运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest --confcutdir=src/openpi/models_pytorch src/openpi/models_pytorch/mone_layer_test.py -q
```

首先检查真实 checkpoint 的有限 loss、非零 adapter 梯度以及 tiny-set loss
下降；之后才扩大数据或接入多层。未完成真实 checkpoint 训练前，不能据此认定方向有效。
