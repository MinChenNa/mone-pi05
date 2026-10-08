# Mone 工作历史与固定运行约束

## 共享盘与节点

- 持久根目录是 `/path/to/workspace`。
- 登录节点和 H200 计算节点看到同一份 `global_user` 内容；节点本地的 `/root` 不共享。
- GPU 节点不能联网。所有 Python、依赖、tokenizer、数据与权重必须先落到共享盘。
- 不把 `/inspire/hdd/project/project-public/...` 当作本项目持久运行目录，也不把环境解释器放在 `/root`。
- `.venv/bin/python` 必须解析到 `mone_pi05_server/.tools/python/...`，不能解析到 `/root/.local/...`。

## 可只读复用的共享资产

- LIBERO LeRobot 数据：`ravenking/acot-vla/datasets/your_hf_username/libero`。
- 官方 pi0.5 JAX base：`ravenking/acot-vla/checkpoints/pi05_base/params`。
- 已验证的 LIBERO norm stats：
  `ravenking/acot-vla/assets/acot_libero_action_cot_explicit_implicit_co_fusion/libero/norm_stats.json`。
- PaliGemma tokenizer：`ravenking/acot-vla/assets/_openpi_cache/big_vision/paligemma_tokenizer.model`。

复用仅指读取公共数据/官方基础权重。不得修改 ACoT 源码、虚拟环境、训练配置、checkpoint
或实验结果。Mone 的转换产物与输出全部写入 `mone_pi05_server`。

## Mone 固定路径

- 独立环境：`mone_pi05_server/openpi/.venv`。
- 共享 Python：`mone_pi05_server/.tools/python`。
- PyTorch π0.5：`mone_pi05_server/checkpoints/pi05_libero_pytorch`。
- tokenizer 缓存：`mone_pi05_server/assets/_openpi_cache`。
- 训练日志/输出：`mone_pi05_server/logs`、`mone_pi05_server/outputs`。

## 可复用的运行经验

- GPU 型号和数量必须在正式运行前检查；登录节点只做静态验证。
- 离线变量应默认开启，使意外联网立即失败，而不是在 GPU 节点无输出卡住。
- 训练日志直接重定向到文件，避免以 `python | tee` 作为唯一日志链路。
- 正式训练前先做两步 smoke；不要把日志暂时不增长直接判断成训练死亡。
- 启动前检查重复训练进程。中断后先确认 GPU 无残留进程，再重新启动。

本记录来自 `docs/acot-vla-setup.md`、`交付说明.md`、ACoT 启动脚本和实际日志；只抽取
部署与运维经验，不继承 ACoT 的模型结构或训练语义。

## 2026-09-26：100 步试验后的验证

四个 adapter 已完成 100 步训练。旧 shuffled 是 token roll(1)；尾部 padding
被移到开头时，有效写入顺序不变，不能把相同 loss 解读为不利用历史内容。
新增 `scripts/eval_mone_heldout.py` 与 `scripts/run_mone_heldout.sh`，冻结现有权重，
评估层位 4、9 的独立轨迹和错误历史对照。详见 `HELDOUT_VALIDATION.md`。
登录节点无 GPU；CPU 数据/mask/统计检查与真实 GPU held-out loss 结果必须区分。

首次 H200 验证已完成：L4 在 16 条新轨迹上，7 维真实动作 loss 为 baseline
0.332867、真实历史 0.325814、跨任务历史 0.327178、同任务其他轨迹 0.326131。
这支持继续查证历史内容效应；尚未证明同轨迹历史的独特收益。后续脚本
`scripts/run_mone_controls.sh` 只评估既有 L4，排除首次验证的 8 个任务，
加入当前帧与固定历史对照，并给出任务级汇总。

第二轮 20 任务/40 轨迹 H200 验证完成：L4 七维动作 loss 的 baseline 0.347526，
真实历史 0.342936，跨任务历史 0.342960，同任务历史 0.342859，当前帧
0.342806，固定记忆 0.342947。真实历史对任何非空对照都无稳定优势。
旧目标仅优化真实历史下动作 loss，且原训练只覆盖 4 条轨迹、3 个相近碗任务。
因此新增 `scripts/train_mone_selective.py` 作为独立试验，不覆盖旧脚本与权重；
`scripts/run_mone_selective.sh` 在20个已用于第二轮分析的任务上训练，
并让新旧 L4 在其余12个未用任务上配对验证。

该试验完成，tag `20260926-124724-16706`：12任务七维动作 loss 的旧 L4
真实历史 0.334673，新 L4 真实历史 0.326581，base 0.338321。
离线第一关判为 inconclusive：新 L4 优于固定历史和当前帧，但与跨任务、同任务
错误历史的任务级区间仍跨零。新增 `scripts/check_mone_gate1.py`、
`scripts/eval_mone_rollout.py` 和 `scripts/run_mone_gates_after_train.sh`，
闭环采样使用与训练一致的 LayerMemory L4 hook。LIBERO 源码与模拟依赖只装入
Mone 目录/环境；LIBERO 初始状态和动作反归一化已作兼容处理。
首轮 GPU 闭环 smoke 在 EGL 导入时遇到 `NoneType.eglQueryString`，与历史
ACOT 部署记录所述的 GPU 镜像缺 `libEGL.so.1` 同因。已只读复制 GLVND
兼容库到 Mone 自己的 `assets/glstub/lib`，runner 将其追加到库搜索路径并
预先建立 EGL 上下文检查；没有改动 ACOT 文件。

第二次 H200 闭环运行完成，tag 同上。EGL 上下文已正常创建，三个 suite 的
15 个配对初始状态各跑 baseline、真实历史、固定历史一次，共 45 次 rollout。
三组分别为 0/15 成功，所有轨迹都跑到各自任务的步数上限（LIBERO-10 520、
Goal 300、Spatial 220），无评估器异常。详见 `HELDOUT_VALIDATION.md` 和
`outputs/heldout/L4-rollout-paired-20260926-124724-16706.json`。
由于当前 PyTorch 权重由官方 pi0.5 **base** 转换，而不是 LIBERO 微调权重，
零成功率不能用来判定 Mone 记忆机制无效；但当前配置没有可用的闭环成功率
信号，不能继续以此宣称 idea 有效，也不宜盲目扩大训练。若切换为有 LIBERO
基线能力的 π0.5，须在新 backbone 上重新训练 adapter 并重复配对对照；
不能把基于旧 base 特征训练的 adapter 直接跨 backbone 套用。

已按冻结模型的历史内容筛查方案写好独立脚本，生成哈希锁定的样本清单：
`outputs/heldout/L4-history-plan-20260926-124724-16706.json`。12任务各有
2条筛查轨迹、2条确认轨迹、1条历史供体；固定历史另取独立供体。每条轨迹
预定5个时刻，比较无遮挡、腕部图像缺失、双相机中心遮挡下的无记忆/正确
历史/同任务错误历史/固定历史/当前帧五组动作损失。运行入口为
`openpi/scripts/run_mone_history_screen.sh` 的 `screen` 和 `confirm` 阶段。
登录节点仅完成清单、静态与单元检查；GPU smoke 和评估尚待 H200 节点执行。

H200 筛查已完成，12任务/24条新轨迹/1080个配对条件与噪声样本，四个 shard
均成功。结论为 `inconclusive`。中心遮挡下正确历史相对同任务错误历史的
动作7维 loss 收益仅 0.000219，低于预设 0.001；相对固定历史的任务级
区间跨零。更关键的是中心遮挡并未提高无记忆 baseline 的 loss（无遮挡
0.345341，遮挡 0.343582），所以该压力条件未达到预设有效性要求。
预设停止规则因此阻止 `confirm`，封存的确认轨迹没有运行。详见
`outputs/heldout/L4-history-screen-20260926-124724-16706.json`。

2026-09-26 官方 π0.5 LIBERO 基线准备：查阅 ACoT 本地资产后确认，其
`pi05_base` 是基础权重；`ravenking/checkpoints/LIBERO-{Goal,Spatial,...}` 是
OpenVLA 架构，不能当作 π0.5 微调权重。官方公开源
`gs://openpi-assets/checkpoints/pi05_libero/` 的 16 个对象已完整下载到 Mone
自己的 `checkpoints/pi05_libero_official_jax`（约12GB）；官方 norm stats 与
先前复用的 ACoT stats 哈希不同。Orbax 元数据读取成功。登录容器 cgroup
上限16GiB，JAX→PyTorch 转换时被 SIGKILL；脚本现有内存预检，需在
>=32GiB 的计算节点运行 `openpi/scripts/prepare_mone_official_libero.sh convert`。
转换输出路径是独立的 `checkpoints/pi05_libero_official_pytorch`，不会覆盖
原 base 转换权重。另有无 Mone 的闭环基线入口
`openpi/scripts/run_pi05_libero_baseline.sh`；转换后运行，先确认三任务
各5条初始状态能有非零成功，再决定是否在该 backbone 上重训 Mone。
