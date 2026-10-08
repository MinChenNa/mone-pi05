# LIBERO 闭环环境

训练只依赖 LeRobot 演示数据；闭环评估还需 MuJoCo/robosuite、LIBERO 原始资产和官方初始状态。发布包不包含完整的 `init_files` 和 `assets`。

1. 按官方 [LIBERO 安装说明](https://github.com/Lifelong-Robot-Learning/LIBERO#installation) 准备完整 checkout，使用实验记录中的版本 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。可放在项目 `assets/libero-src/`（Git 忽略）。确认其中 `libero/libero/assets`、`init_files`、`bddl_files` 均存在。
2. 在 OpenPI 环境内准备仿真依赖。LIBERO 原始 requirements 与 OpenPI 的 torch、numpy 和 transformers 固定版本有冲突，不要直接覆盖完整 OpenPI 环境。历史运行使用项目自己的仿真环境；新服务器应根据实际兼容性安装并做 EGL smoke。保留 OpenPI 的 torch 2.7.1 与 transformers 4.53.2 及补丁。
3. 在 `openpi/` 下运行：

```bash
python scripts/prepare_libero_config.py \
  --libero-root ../assets/libero-src \
  --data-dir "$MONE_DATA_DIR"
export LIBERO_CONFIG_PATH="$PWD/../assets/libero_config"
export PYTHONPATH="$PWD/../vendor/libero:$PYTHONPATH"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python -c 'from robosuite.utils.binding_utils import GLContext; c=GLContext(64,64); c.make_current(); c.free()'
```

Linux 容器需提供 EGL 驱动库；配置好的系统可直接使用系统库。历史环境中放在 `assets/glstub/lib` 的私有库不是必需路径，运行器优先通过真实 EGL context 检查可用性。相关库不随本仓库发布。

4. 配置官方权重：`export MONE_OFFICIAL_CHECKPOINT=/absolute/path/to/pi05_libero_pytorch`，再运行 `bash scripts/run_pi05_libero_baseline.sh`。该脚本需要至少 3 张可见 GPU。记录成功率和动作范围后再进行记忆训练。

本次发布在 Windows 环境完成 CPU 检查；Linux 仿真依赖安装、EGL 与真实模型闭环需要在目标服务器检验。旧运维说明中 `/path/to/workspace` 是示例路径，不是本机可直接执行的路径。
