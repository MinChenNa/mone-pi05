# MoNe + π0.5 实验结果
> 资料说明：本页整合用户于 2026-10-08 提供的实验总结。原始 JSON/JSONL、训练权重及 stateful_v1 代码尚未随本次更新提供；下面的服务器报告路径用于溯源，不能在 GitHub 中直接打开。数字按提供的总结记录，本次未重新计算统计量或复跑实验。当前仓库公开的实现范围仍为早期 full18 及单层原型。

更新：2026-10-08。本文只汇总本地已经生成的实验报告；未完成试次不计入成功率。

## 一句话结论

**目前没有证据表明 MoNe + π0.5 的闭环成功率优于原始 π0.5。** 全 18 层早期版本在 8 个未见任务上是 77/80，对照 π0.5 为 80/80；后续有状态版本在 4 个未见 LIBERO-Long 任务上与 π0.5 同为 **74/80**。有状态版本在 Long task 8 上表现出一些与动态记忆写入有关的信号，但尚不足以证明持续在线记忆带来稳定收益。

这里的“MoNe”指本项目的记忆 adapter 原型，并非对 MoNe 论文中内层损失和 Memory-KV 接口的忠实复现。两代实现、训练预算和任务划分不同，不能直接当作同一模型的前后性能对比。

## 模型与评测范围

- 主干：官方 LIBERO π0.5 检查点转换得到的 PyTorch 模型；主干冻结，仅训练附加的记忆 adapter。两代模型都使用同一主干哈希 `5365dd93538e9fb2e574ed793a08316e1a9b517970f364190e937a82a6c12e69`。
- 早期 **full18**：在 π0.5 的 18 个视觉语言上下文 Transformer 层加入记忆模块；每次决策从近期历史重新构造状态，**不跨决策持续保存**。训练 3000 步，有效 batch 32，单个训练 seed。完成记录：full18 训练摘要（服务器报告路径：`outputs/full18/full18_20260927/memory/complete.json`）。
- 后续 **stateful_v1**：同样覆盖 18 个上下文层，在一个机器人回合内持续更新记忆。Online 与单独训练的 Short 对照各训练 250 步、4 GPU、单个 seed；4 步序列、每 5 环境步重新决策、截断反传长度 2。完成记录：Online（服务器报告路径：`outputs/stateful/stateful_task8val_20260929/online/complete.json`）、Short（服务器报告路径：`outputs/stateful/stateful_task8val_20260929/short/complete.json`）。
- 下表中的“未见任务”是指 **adapter 训练任务划分** 中未见；π0.5 主干本身是官方 LIBERO 检查点，不能据此声称主干从未接触 LIBERO。

## 结果一：full18 早期版本

在冻结划分的 8 个测试任务、每任务 10 个配对初始状态上，闭环成功率为：

| 模式 | 成功次数 |
| --- | ---: |
| 原始 π0.5 | **80/80** |
| full18 + 相关历史 | 77/80 |
| full18 + 同任务其他历史 | 77/80 |
| full18 + 固定历史 | 78/80 |
| 参数量对照 | 77/80 |

来源：full18 闭环配对报告（服务器报告路径：`outputs/full18/full18_20260927/rollout-paired.json`）。该测试集的 π0.5 基线已满分，因此不能测出正向成功率增益；观察到的 3 次额外失败也不能被离线 loss 的下降抵消。

离线 7 维动作 flow loss 上，相关历史相对原始 π0.5 的平均改善为约 `0.00163`，但相对“同任务其他历史”的平均改善仅约 `0.00006`，其区间跨过 0，报告判定为 `inconclusive`。来源：full18 离线测试报告（服务器报告路径：`outputs/full18/full18_20260927/test-paired.json`）。**离线 loss 变好不等于闭环任务成功率变好。**

另有 Long task 8 的定向评测达到 33/50，但该任务属于这版 adapter 的**训练划分**，且该报告没有同协议的 π0.5 基线，因此不能作为未见任务优于 π0.5 的证据。来源：full18 task 8 报告（服务器报告路径：`outputs/long-targets/longhard20260928/task8-trials50.json`）。

## 结果二：stateful_v1 持续记忆版本

重新划分任务后，LIBERO-Long task 8、1 属于 adapter 验证集，task 2、7 属于当轮测试集；其余 6 个 Long 任务参与训练。以下均使用相同任务、相同前 20 个初始状态及配对评测协议。

| LIBERO-Long task | π0.5 | Online | Short | No-write |
| --- | ---: | ---: | ---: | ---: |
| 8（验证） | 14/20 | **15/20** | 14/20 | 10/20 |
| 1（验证） | 20/20 | 20/20 | 19/20 | 20/20 |
| 2（测试） | **20/20** | 19/20 | 20/20 | 20/20 |
| 7（测试） | 20/20 | 20/20 | 20/20 | **未完成** |
| **四任务合计** | **74/80** | **74/80** | 73/80 | 不计算 |

来源：task 8 全模式报告（服务器报告路径：`outputs/stateful/stateful_task8val_20260929/eval/task8-paired-20-all-modes.json`）、task 1 报告（服务器报告路径：`outputs/stateful/stateful_task8val_20260929_long_broad/eval/task1-paired-20.json`）、task 2 报告（服务器报告路径：`outputs/stateful/stateful_task8val_20260929_long_broad/eval/task2-paired-20.json`）、task 7 部分模式报告（服务器报告路径：`outputs/stateful/stateful_task8val_20260929_task7_fresh/eval/task7-paired-20-partial-modes.json`）。task 7 的 No-write 多次在 MuJoCo/EGL 渲染中断，只有部分试次完成；**不能把缺失的 19 次算成失败**。

消融的含义：Online 在整个回合中保留并更新记忆；No-write 使用**同一 Online adapter**，保留学到的初始记忆状态但推理时不写入；Short 是**另行训练**的相同结构 adapter，每次决策先重置、再写入最近历史。因此 Online 对 No-write 比较的是动态写入的作用；Online 对 Short 比较还混入了训练差异。使用同一 Online adapter、每次决策重置的 `reset_each_step` 是更直接的持续累积对照。

唯一没有明显基线天花板效应的 task 8，20 次试次的更多模式结果为：Online 15、π0.5 14、Short 14、No-write 10、Shuffled 13、Reset-each-step 13、Mid-reset 12。配对比较中，Online 相对 π0.5 只多 1 次（exact McNemar `p=1.0`），相对 No-write 多 5 次（`p=0.0625`），相对 Shuffled 多 2 次（`p=0.625`）。这些数字**提示动态写入可能有作用，但既未证明优于 π0.5，也未证明跨决策累积记忆或正确历史关联具有稳定优势**。

## 解释边界

1. stateful_v1 是 **250 步、单训练 seed 的 pilot**，不是充分调参或多 seed 复现。20 次试次对小幅差异的判别力有限。
2. 四个未见 Long 任务中，task 1、2、7 的 π0.5 基线均为 20/20，主要用于检查是否退化，几乎没有改善空间。task 8 曾因旧实验的难度被选入新验证划分，因此其结果带有事后选任务的探索性质。
3. task 2、7 已用于本轮报告；如果据此继续选择模型或调整方法，后续不能再把它们称为完全未触碰的最终测试集。
4. No-write 的分布与 Online 训练时所见不完全一致；其下降不能单独证明“长期记忆”机制。Short 和 Shuffled 的结果也尚未支持该机制的明确优势。
5. 本文只报告 LIBERO 仿真与离线 flow loss；没有真实机器人、其他基准或独立多 seed 的性能结论。

## 当前判断与下一步

当前最稳妥的判断是：**记忆写入存在值得追查的机制线索，但“MoNe + π0.5 提升长期任务成功率”尚未得到支持。** 如果继续投入评测，应预先固定至少一个非天花板、未参与 adapter 训练的任务集合，用更多配对初始状态和至少 3 个训练 seed 同时比较 π0.5、Online、同 adapter 的重置/No-write、Short 与 Shuffled；重点检验 Online 是否稳定胜过 π0.5、短历史及错误历史。重复测已接近满分的任务，信息量很低。
