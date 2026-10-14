# GPU 前四项验收报告（2026-10-13）

## 结论与范围

四项工作均已执行并留存证据，但不是四项全部通过。当前可以确认三模态 GPU 短训可运行、AMP 跳步可解释、checkpoint 保存状态可以精确恢复；不能确认发布级环境已对齐、跨进程训练轨迹可复现，也不能确认科学指标达到发表级。

| 项目 | 状态 | 证据与未完成项 |
| --- | --- | --- |
| 1. CUDA/PyTorch 环境固定与记录 | 已记录并限制 PyTorch 版本；版本对齐未完成 | PyTorch 2.7.1+cu118，编译 toolkit/nvcc 11.7；未建立可重建的容器/完整安装锁 |
| 2. optimizer skipped steps 解释 | 通过当前短训观测验收 | 逐通道累计 `amp_scale_reduced`；没有把跳步计为成功更新 |
| 3. 3–5 epoch 科学短基线 | 运行验收通过；科学性能未达标 | 3 epoch × 每通道 100 steps，5 通道，CUDA 确认；astigmatism/DH 的 held-out 检测仍为零 |
| 4. 连续与中断恢复等价性 | 精确恢复通过；端到端等价性未通过 | 三模态立即恢复各比较 733 tensors，零差异；最终跨运行对照失败，且部分轨迹在中断前已经分叉 |

没有修改物理方程、loss、坐标/单位/scale contract、数值 kernel 或 checkpoint schema 来制造通过；没有放宽 immutable resume 配置检查。

## 实现与可追溯性

- `training/optimizer_step.py` 返回跳步原因；`modality_epoch.py`、`modality_progress.py`、worker 将其累计到日志、摘要与训练状态。
- `ModalityEpochResult` 新字段位于末尾并提供默认值，保留旧构造兼容性；旧进度没有原因字段时仍可恢复。
- 两个 smoke sbatch 使用训练同一 Python 解释器采集环境。最终修订将环境文件命名为 `gpu_environment-<job_id>.json`，避免 resume 覆盖旧环境证据。
- 新 `scripts/train/unitypsf_gpu_qualification.sbatch` 和 `qualify_gpu_baseline.py` 运行真实 canonical expert-parallel worker；三个 rank 在 epoch 1 保存完成后受控退出，再以不变的三 epoch 配置恢复。
- 环境记录包含 Python、PyTorch、CUDA build、nvcc、cuDNN、GPU/driver、源文件 SHA256；另存 pip freeze 与 extension binary SHA256。这是运行清单，不等于可重建环境锁。
- qualification 只在子进程使用保存/恢复观测包装，不改 checkpoint 内容。独立运行具有不同 run-id/路径，因此 resolved contract hash 不同；模型/data seed 与训练预算相同。

## 环境

正式证据以 job 5721 为准，使用三张可见 RTX 3090（`CUDA_VISIBLE_DEVICES=0,1,2`），无 CPU fallback：

| 项目 | 观测值 |
| --- | --- |
| Python | 3.11.7，`/home/guest/anaconda3/bin/python` |
| PyTorch | 2.7.1+cu118 |
| PyTorch CUDA build | 11.8 |
| extension 编译 toolkit/nvcc | `/usr/local/cuda-11.7`，11.7.99 |
| Driver | 570.133.07 |
| cuDNN 返回版本 | 90100 |

警告准确含义是 **extension 编译 toolkit 11.7 与 PyTorch CUDA build 11.8 不一致**，不能据此宣称已加载的 CUDA runtime 是 11.7。未安装/替换系统 CUDA，没有消除该不一致。环境记录的 nvidia-smi 查询列出宿主四张卡，不表示作业使用了四张卡。

job 5721 的 qualification worker 关闭 cuDNN benchmark/TF32，开启 cuDNN deterministic 和 PyTorch deterministic algorithms，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。这些设置不保证自定义 CUDA kernel 确定性。

## 短训结果

连续基线每个通道尝试 300 步，共 1500 步；成功更新 1493 次，跳过 7 次，所有已观测原因都是 `amp_scale_reduced`。

| 模态/通道 | 首步 → 最后一步训练 loss | 成功更新 | 跳步 |
| --- | --- | --- | --- |
| emitter_2d / left | 1146.0209 → 69.8288 | 299 | 1 |
| emitter_2d / right | 1007.6252 → 45.3320 | 298 | 2 |
| astigmatism / left | 1431.4104 → 228.2445 | 298 | 2 |
| astigmatism / right | 1138.8407 → 203.7340 | 299 | 1 |
| double_helix / main | 1011.7573 → 183.9521 | 299 | 1 |

上述首末 loss 不是跨 seed 收敛统计，也不意味着每步单调下降。跳步原因来自 scaler 缩小的直接观测，尚未定位具体异常梯度 tensor；非有限 loss 仍直接报错，不被归为正常跳步。

held-out 随 epoch 的变化：

- emitter_2d：eval loss 约 166.79 → 105.44 → 72.49；每通道预测 emitter 数 0 → 569 → 1303；recall 0 → 0.06403 → 0.14664。
- astigmatism：eval loss 约 342.17 → 261.42 → 230.58；左右通道 predicted emitters / recall 均保持 0。
- double_helix：eval loss 283.65 → 233.34 → 196.50；predicted emitters / recall 保持 0。

因此 loss 下降只能支持短训优化发生，不能代替定位精度与检测能力验收；零匹配下 RMSE 为 null，不应解释为零误差。当前短训不足以判定低召回是预算不足还是 pipeline 问题。

## 恢复验收与对照限制

三路立即恢复分别验证 model、optimizer、scheduler、AMP scaler、RNG、channel progress，每路 733 tensors，零 tensor/scalar 差异，最大绝对差 0。这支持“保存的训练状态正确恢复”，不等于整个 resumed trajectory 与另一运行一致。

最终对照使用 tensor `rtol=1e-5, atol=1e-6`，整数/RNG 保持精确比较。5721 中 emitter_2d 与 astigmatism 最终 tensor 不一致；DH 的 733 tensors 与完整 loss history 相同，但 Python/NumPy RNG 路径仍有 1246 项数值 mismatch。整个 `resume-equivalence.json` 状态为 failed，SLURM 外层退出 1 是门禁失败，不是训练 worker 崩溃。

关键限制：两组是独立新进程训练，未证明共同起始状态和共同 epoch-1 状态。astigmatism loss 从零基索引 8、2D left 从 13、right 从 12 就已不同，均早于 epoch-1 中断。因此不能把最终差异归因于 checkpoint/resume 缺陷。

需要隔离而尚未证实的因素：

1. `localization/cuda_extension/kernels.cu:151` 使用浮点 `atomicAdd` 聚合 frame。并行加法顺序是确定性风险候选；本轮未隔离证明它是全部差异根因，未修改 kernel。
2. canonical runtime 的 model 初始化使用 `torch.manual_seed`，没有统一全局 Python/NumPy 初始 seed。独立进程的 RNG 身份对照不具备完整共同初始条件；立即恢复 RNG 仍是准确的。

最终审查修订了未来 harness 的判定：精确值相等与容差等价分别报告，立即恢复同时检查 scalar 差异；新环境记录显式写入 deterministic 设置和 harness SHA256。**这些修订晚于 5721，未再次运行 GPU；现有 5721 证据不被改写，也不冒充最终脚本的新运行结果。** 5721 实际零差异恢复和超容差终态失败结论不受修订影响。

## 作业证据与排除项

主要证据：

- `output/unitypsf/gpu-qualification-5721/environment.json`、`pip-freeze.txt`
- 同目录 `continuous-gate.json`、`interrupted-gate.json`：均通过五通道运行门禁。
- 同目录 `restore-emitter_2d.json`、`restore-astigmatism.json`、`restore-double_helix.json`：均通过立即恢复。
- 同目录 `resume-equivalence.json`：最终等价门禁 failed。
- 同目录 `continuous.log`、`interrupted.log`、`resume.log` 与 `logs/slurm/unitypsf_gpu_qualification-5721.out/.err`。
- `output/unitypsf/gpu-qualification-5721-continuous/metrics/joint_training_summary.json` 与对应 `-interrupted` 目录：完整 loss、held-out、步数及跳步原因。

连续 checkpoint SHA256：`ab7dfe84fc06d08dce018133550acc9068d5b07443910096e6cf26c17adafd9a`。
恢复运行 checkpoint SHA256：`1ffacfbf9cbb5c9a8c875b8ce289ffccd1667dd31ce49011ea13666246c9fa8d`。

不能混入正式等价证据的历史作业：

| 作业 | 解释 |
| --- | --- |
| 5713 | smoke action 将参数限制为 1 epoch / 8 steps，不是三 epoch 科学基线 |
| 5714 | 有效 3 epoch / 32 steps 基线，但非中断恢复对照 |
| 5715 | 对已完成 5714 执行幂等 resume，不是中途恢复 |
| 5716/5717 | 1 epoch 后改成 3 epoch 的 resume 被 immutable contract 拒绝；是正确保护，不应放松 |
| 5718 | 普通连续训练，没有真实中断 |
| 5719 | qualification 启动模块配置错误，无有效科学结果；已修复 |
| 5720 | 完成实际中断/恢复但独立轨迹分叉；随后增加确定性设置进行 5721 |

## 软件验证与工作区

- 核心观测改动后全套 pytest：436 passed、1 skipped；最终兼容性修订后相关回归：62 passed。
- 配置范围 mypy（六个模块）、修改范围 Ruff、三份 sbatch 的 bash syntax 检查通过。
- 全仓 Ruff 仍有 `training/localizer_eval.py` 两个既有 F841，不在本轮修改范围。
- 项目在父 Git 下整体是未跟踪目录（`?? ./`），不能宣称工作区干净或以 git diff 检查覆盖本轮改动。未提交、未删除历史实验。
- 新源码/测试、脚本、报告分别位于 src/tests、scripts/train、docs/reviews；运行产物在 output、logs 和 .local。根目录既有 `.mypy_cache`、`.pytest_cache`、`.ruff_cache` 仍保留，未作为本轮无关清理删除。

## 下一轮仅做发布阻塞项

1. 对齐并验证 extension 编译 toolkit 与 PyTorch CUDA build，生成可重建环境锁；不要静默全局升级。
2. 建立共同初始状态/中断点校验，统一 Python、NumPy、PyTorch seed，先验证同配置重复运行，再判断连续/恢复轨迹等价性。
3. 隔离 CUDA frame 聚合确定性风险；若需要修改数值实现，单独提出方案并通过科学回归，不在本轮强行重写。
4. 复现性门禁通过后，再扩展科学训练预算与多 seed 统计，追查 astigmatism/DH 零检测，最终验证 recall、precision、XY/Z RMSE 与 photon error。

暂停无关拆文件、facade 和私有依赖清理。先把上述真实验收阻塞项收敛，不再用结构优化代替科学证据。

## 对齐回归：job 5726

为验证前述零检测是否只是训练配置差异，新增了可参数化的对齐 qualification：输入尺寸统一为 `64×64`，batch size 为 `64`，每 epoch 为 `157` steps；astigmatism left/right 使用 5704 可复用的 coefficient maps。对齐参数由 `scripts/train/unitypsf_gpu_aligned_qualification.sbatch` 固定传入，maps、尺寸、预算均写入 resolved contract。

job 5725 仅因 map 参数解析错误在训练前退出，排除，不计入科学结果。有效 job 5726 的连续与中断运行门禁均通过，三路立即恢复均通过。连续训练第 3 epoch 的 held-out 结果为：emitter_2d left/right recall `0.773`；astigmatism left/right recall `0.6905 / 0.7192`；double_helix recall `0.7476`。因此此前三 epoch 的零检测不是当前解码链路必然为零，主要由原始短预算与配置差异共同造成。

但 job 5726 的独立连续/恢复终态比较仍为 `failed`：astigmatism 与 emitter_2d 在中断前已有分叉，double_helix 模型 tensor 一致但仍有 RNG 路径差异。这是 qualification 对照条件/确定性问题，不能宣称 resume 端到端等价已解决。有效对齐训练结果位于 `output/unitypsf/gpu-aligned-qualification-1791896445/`。
