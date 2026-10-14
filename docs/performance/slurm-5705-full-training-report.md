# UnityPSF v0.3：SLURM Job 5705 完整训练报告

## 结论

Job 5705 完成了单 GPU、300 epoch、47,100 step 的 CUDA-GEMM 训练。从运行目录创建到 `stage_status.json` 写入，历时约 **3 小时 37 分 49 秒**，与“约 3 小时 40 分钟”的记忆相符。最终在线模拟评估的 Jaccard 为 **0.9486**，横向和轴向 RMSE 分别为 **13.10 nm** 和 **35.10 nm**。

本次使用 9,984 个中心帧的固定 simulation cache，跨 epoch 循环复用。ROI gamma 优化执行了 28 个周期、左右 domain 共 56 次；但本次配置没有连接 gamma 到训练物理条件的 coefficient map feedback，所有更新均记录 `feedback_skipped=true`，因此这些更新没有触发训练 cache 刷新。

## 1. 任务身份与配置

| 项目 | Job 5705 的实际配置 |
| --- | --- |
| SLURM 脚本 | [`scripts/train/unitypsf_cuda_gemm_roi_gamma.sbatch`](../../scripts/train/unitypsf_cuda_gemm_roi_gamma.sbatch) |
| 解析后的配置 | [`.local/tmp/unitypsf_roi_gamma_resolved_5705.yaml`](../../.local/tmp/unitypsf_roi_gamma_resolved_5705.yaml) |
| 运行目录 | [`output/unitypsf_cuda_gemm_roi_gamma_5705/`](../../output/unitypsf_cuda_gemm_roi_gamma_5705/) |
| GPU | 1 张 |
| 模型 | `active_smlm_soft_moe_double_unet` |
| Simulation backend | `cuda_gemm` |
| 训练预算 | 300 epoch × 157 step/epoch = 47,100 step |
| Batch size | 64 |
| 图像与时间窗口 | 64 × 64 像素，连续 3 帧，监督中间帧 |
| Optimizer / AMP | AdamW；FP16 AMP |
| ROI gamma | Epoch 30 开始，此后每 10 epoch 更新；左右 domain 各 100 个 Adam step |
| Localizer evaluation | 每 5 epoch 一次；固定的在线模拟评估 batch，共 4 batch × 4 样本 |

任务的 [run manifest](../../output/unitypsf_cuda_gemm_roi_gamma_5705/metadata/run_manifest.json) 记录了模型、训练 runtime、评估来源和 ROI gamma 路线。[stage status](../../output/unitypsf_cuda_gemm_roi_gamma_5705/metadata/stage_status.json) 为 `completed`，记录了全部 300 个 epoch、`global_step=47100` 和 56 次 gamma 更新。训练、评估、gamma 指标分别见 [`training_metrics.jsonl`](../../output/unitypsf_cuda_gemm_roi_gamma_5705/metrics/training_metrics.jsonl)、[`eval_metrics.jsonl`](../../output/unitypsf_cuda_gemm_roi_gamma_5705/metrics/eval_metrics.jsonl) 和 [`gamma_update_metrics.jsonl`](../../output/unitypsf_cuda_gemm_roi_gamma_5705/metrics/gamma_update_metrics.jsonl)；评估文件共 60 条记录。

## 2. 完整训练周期

1. **启动与配置解析。** SLURM 脚本检查 raw TIFF，合并 `configs/base/microtube.yaml` 和 `configs/overrides/standard_roi_gamma.yaml`，生成本次 resolved YAML，然后启动 high-fidelity localization trainer。
2. **训练 warmup。** Epoch 1–29 使用 CUDA-GEMM simulation 生成训练数据并优化 localizer。`peak_zmap_bootstrap.enabled=false`，本次没有执行该 bootstrap。
3. **首次 ROI gamma 更新。** Epoch 30 从 raw TIFF 自动构建 ROI library，按左右 domain 执行 gamma 优化，并记录 held-out ROI 重建监控值。
4. **交替周期。** Epoch 30、40、…、300 共 28 个触发点；每次左右 domain 各写一条 gamma metric，因此共 56 条。每 5 epoch 运行一次 localizer evaluation，共 60 次。
5. **Checkpoint 与收尾。** 每个 epoch 写入最新 checkpoint；评估 loss 创新低时写入 best checkpoint。Epoch 300 的训练和评估之后，执行最后一次左右 domain 的 gamma 更新与诊断，最终写入 `stage_status.json`。

训练循环的顺序是“训练及最新 checkpoint → evaluation 及 best checkpoint → epoch-end gamma hook”，见 [`training/loop.py`](../../src/unity_psf/training/loop.py)。因此最终的 localizer metrics 和 best checkpoint **不包含** Epoch 300 之后执行的最后一次 gamma 更新；本次 gamma 更新也没有反馈到训练物理条件。

## 3. Simulation cache：生成、复用和失效

### 本次实际策略

| 参数 | 值 | 作用 |
| --- | ---: | --- |
| `batch_strategy` | `cached_window` | 使用序列窗口缓存 |
| `cached_window_cache_frames` | 9,984 | 缓存中心帧数量 |
| `sequence_count` | 64 | 把中心帧组织成序列 |
| `cached_window_order` | `temporal` | 按时间顺序读取 |
| `cached_window_cache_storage` | `cpu` | 序列数据存于 CPU pinned memory |
| `cached_window_prefetch_batches` | 1 | 预取下一个 batch 到 GPU |
| `vector_batch_size` | 4,096 | CUDA-GEMM renderer 的向量批量 |

第一次请求训练 batch 时，provider 创建固定 cache 并调用 `materialize_all()`。CUDA-GEMM renderer 生成序列；每个中心帧带有相邻两帧的上下文。9,984 个中心帧按 batch size 64 形成 **156 个完整 cache batch**。每个 epoch 训练 157 step，因此会在同一个 cache 上循环一次；`set_epoch()` 将起始 batch 按 `((epoch-1) × 157) mod 156` 平移。其结果是 **跨 epoch 复用模拟帧，改变读取起点，而非每个 epoch 重新仿真**。

CPU storage 时，序列缓存不受 `cached_window_max_gpu_sequences=4` 的 GPU 序列上限约束。CPU tensor 使用 pinned memory，训练 adapter 在单独 CUDA stream 上提前传输一个 batch。实现见 [`localization/data/online.py`](../../src/unity_psf/localization/data/online.py) 和 [`localization/training_adapter.py`](../../src/unity_psf/localization/training_adapter.py)。

### 物理条件变化时的代码路径与本次实际行为

代码以 `(condition_store_version, cache_generation)` 作为 cache signature；版本变化时清空 pool，并在下一次请求时重新生成。这个机制支持物理参数更新后的 cache 失效。

**Job 5705 没有走到这条 feedback 路径。** 解析后的配置为 `dual_domain_coeff_maps: []`，运行时没有构建 `ConditioningProviderStore`；gamma feedback function 因而返回 `None`。56 条 gamma metrics 全部记录 `gamma_update_accepted=true` 和 `feedback_skipped=true`，均没有 `condition_store_version` 或 `feedback_deferred_commit` 字段。这里的 `accepted` 对应 `heldout_accept_policy=monitor`，并不表示 gamma 已写回训练 simulation。

相关实现见 [`training/high_fidelity/engine.py`](../../src/unity_psf/training/high_fidelity/engine.py) 中的 condition store 与 gamma feedback 构造，以及 [`gamma_update/hook.py`](../../src/unity_psf/gamma_update/hook.py) 中的接受策略和 `feedback_skipped` 记录。

## 4. 时间开销

以下时间为运行产物的本地文件时间戳，时区为 Asia/Shanghai。

| 事件 | 时间 |
| --- | --- |
| 运行目录创建 | 2026-10-09 19:41:48 |
| 最后一个训练 metric 写入 | 2026-10-09 23:17:29 |
| 最终 eval metric 和 best checkpoint 写入 | 2026-10-09 23:17:31 |
| 最终 gamma metric、stage status 写入 | 2026-10-09 23:19:37 |

- **运行目录创建至完成：3 小时 37 分 48.97 秒。**
- **创建至最后训练 metric：约 3 小时 35 分 41 秒。**
- 最后训练 metric 至 stage status 写入：约 2 分 08 秒，包含最终评估、checkpoint、gamma 监控和收尾。
- 将完整运行时间除以 47,100 step，得到约 **277.5 ms/step** 的整体摊销值；此值包含 cache 生成、周期评估、gamma 优化、诊断及写盘，不能当作纯训练 step latency。

SLURM accounting 在当前环境返回 `Slurm accounting storage is disabled`，因此这里是**产物时间戳估计**，不是 `sacct Elapsed`。Job 5705 未开启分阶段 timing profile，无法从本次产物准确拆分 cache 构建、模型前向、反向、评估和 ROI gamma 的各自耗时。另一个 30 epoch 性能基准的 85.28 ms/update 及分段耗时见 [LiteLoc 速度对比](unitypsf-v03-liteloc-training.md)，不能代入本任务作为实测值。

## 5. 最终 localizer metrics

最终评估对应 Epoch 300、Global step 47,100。评估数据源在 run manifest 中标记为 `online_generation`：它是**在线模拟 held-out 数据**，不是 raw TIFF 的真实定位精度。

| 指标 | 最终值 |
| --- | ---: |
| Eval loss | -201.311466 |
| Precision | 0.990244 |
| Recall | 0.957547 |
| Jaccard | 0.948598 |
| 横向 RMSE | 13.100 nm |
| 轴向 RMSE | 35.096 nm |
| Predicted emitters | 615 |
| Target emitters | 636 |

最后一个训练 batch 的 loss 为 `-186.6266`，最后一个 epoch 的 157 个 batch 平均 loss 约为 `-194.4099`。`checkpoint_best.pt` 依据**最低 eval loss** 选择，而最终评估的 `-201.311466` 是本次最低 eval loss，因此最终 checkpoint 也是按该规则选出的 best checkpoint。最高 Jaccard 为 `0.951713`，出现于 Epoch 75；它不是 best checkpoint 的选择准则。

## 6. ROI gamma 最终监控值

Epoch 300 的最后一次更新中，左右 domain 各执行 100 个 Adam step，学习率为 0.025。监控报告：[left](../../output/unitypsf_cuda_gemm_roi_gamma_5705/artifacts/roi_bank_gamma/step_00047100/source_auto_built/domain_left/gamma_update_monitor.md) · [right](../../output/unitypsf_cuda_gemm_roi_gamma_5705/artifacts/roi_bank_gamma/step_00047100/source_auto_built/domain_right/gamma_update_monitor.md)。

| Domain | Selected Poisson NLL | Held-out loss delta | 更新记录 |
| --- | ---: | ---: | --- |
| left | 713,992.31 | +7.125 | accepted；feedback skipped |
| right | 717,410.38 | +2.000 | accepted；feedback skipped |

56 条 gamma 更新中有 29 条 held-out loss delta 为负，即监控指标改善。最后一次左右 domain 的 delta 均为正，表示各自 held-out loss 小幅上升。`monitor` 策略仍记录 `accepted=true`；此值不是严格的 held-out 改善门禁。

## 汇报用摘要

> Job 5705 完成 300 epoch、47,100 step 的单 GPU CUDA-GEMM 训练，运行产物时间戳对应约 3 小时 38 分钟。训练采用 9,984 帧 CPU pinned cached-window simulation cache，跨 epoch 复用并循环读取，GPU 端提前预取一个 batch。最终在线模拟评估达到 precision 0.9902、recall 0.9575、Jaccard 0.9486、横向 RMSE 13.10 nm、轴向 RMSE 35.10 nm。ROI gamma 优化共执行 28 个周期、左右 domain 共 56 次；本次配置未启用 coefficient map feedback，因此 gamma 更新没有触发训练物理条件或 simulation cache 刷新。上述指标反映在线模拟评估结果，真实 raw TIFF 定位质量仍需单独验证。
