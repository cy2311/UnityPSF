# UnityPSF 双通道训练与 `microtube_dual` 对比总结

更新时间：2026-09-29

## 结论先说

当前 UnityPSF 的普通训练可以正常学习，但正式训练在多轮 physical update 之后发生明显退化。问题不在 CUDA 仿真图像塌缩，也不在模型完全学不动，主要风险在于 physical update 导出的物理 map 逐轮偏离了训练模型实际需要的 PSF。

当前正式 run 在大约第 25 个 epoch 附近达到最好轴向 RMSE，之后随着 physical update 增多而恶化：

- 最好 axial RMSE：`54.83 nm`
- 最后 axial RMSE：`128.31 nm`
- 最好 lateral RMSE：`19.38 nm`
- 最后 lateral RMSE：`52.75 nm`

因此目前不能把 300 epoch 最终 checkpoint 当作有效结果；应优先保存和评估 physical update 之前或退化开始之前的 checkpoint。

## 对比对象

### 当前 UnityPSF 正式训练

目录：

`unity_v0.3/output/unitypsf_cuda_gemm_roi100_update20_every10_batch64_from_scratch_20260929`

配置特点：

- direct CUDA GEMM PSF simulation
- 双通道 left/right
- batch size `64`
- sparse routing
- 100 个 ROI library
- 每 10 个 epoch 做一次 physical update
- 每 5 个 epoch 做一次 eval
- 从头训练，没有加载旧 checkpoint

### `microtube_dual` / 旧 Neptune baseline

主要训练 metrics 来自：

`results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat/training_dual_anchor99_gamma10_conditionfix/run/metrics/eval_metrics.jsonl`

这个 baseline 使用的是旧的 physical-update contract：固定 astigmatism baseline 与可更新 residual gamma 分开处理，并且最终 map 约为 `(2,2)=-70 nm`。

## 训练 metrics 对比

### 300 epoch 最终结果

| Run | Axial RMSE | Lateral RMSE | Jaccard | Recall |
|---|---:|---:|---:|---:|
| `microtube_dual` FD + physical update | `41.15 nm` | `25.76 nm` | `0.874` | `0.932` |
| `microtube_dual` uniform ablation | `33.30 nm` | `22.77 nm` | `0.909` | `0.946` |
| UnityPSF 当前正式 run | `128.31 nm` | `52.75 nm` | `0.814` | `0.837` |

UnityPSF 当前正式 run 的最终 axial RMSE 比：

- FD baseline 高约 `87.2 nm`
- uniform baseline 高约 `95.0 nm`

### 最佳训练阶段

| Run | 最佳 Axial RMSE | 对应训练位置 |
|---|---:|---:|
| `microtube_dual` FD + physical update | `31.27 nm` | global step `53793` |
| `microtube_dual` uniform ablation | `24.64 nm` | global step `77562` |
| UnityPSF 当前正式 run | `54.83 nm` | global step `3925` |

UnityPSF 在早期已经能达到约 `55 nm`，说明网络和 CUDA simulator 能学习；退化发生在后续 physical update 阶段。

## UnityPSF 当前 run 的退化轨迹

| Global step | Axial RMSE | Lateral RMSE | Jaccard | Recall |
|---:|---:|---:|---:|---:|
| `785` | `80.47 nm` | `31.75 nm` | `0.430` | `0.430` |
| `1570` | `64.59 nm` | `25.14 nm` | `0.545` | `0.545` |
| `2355` | `58.87 nm` | `21.56 nm` | `0.673` | `0.673` |
| `3925` | `54.83 nm` | `20.96 nm` | `0.771` | `0.777` |
| `4710` | `54.97 nm` | `19.38 nm` | `0.798` | `0.802` |
| `9420` | `57.05 nm` | `21.46 nm` | `0.880` | `0.917` |
| `20410` | `61.16 nm` | `24.47 nm` | `0.904` | `0.930` |
| `28260` | `87.06 nm` | `32.85 nm` | `0.882` | `0.904` |
| `36110` | `109.51 nm` | `41.74 nm` | `0.862` | `0.880` |
| `43960` | `120.94 nm` | `50.89 nm` | `0.815` | `0.835` |
| `47100` | `128.31 nm` | `52.75 nm` | `0.814` | `0.837` |

Jaccard 先升高、后下降，说明模型早期学会检出 emitter，但后期物理模型偏移后，定位和检出一起变差。

## Anchor 对照实验

任务：`5686`，相同 seed、batch、模型、sparse routing、CUDA GEMM，训练 10 epoch。

| 设置 | 第 5 epoch axial | 第 10 epoch axial | 第 10 epoch lateral | 第 10 epoch Jaccard |
|---|---:|---:|---:|---:|
| map `(2,2)=+99 nm` | `82.41 nm` | `66.90 nm` | `22.26 nm` | `0.544` |
| residual map `0` + fixed anchor `+99 nm` | `89.64 nm` | `71.28 nm` | `27.24 nm` | `0.555` |
| map `(2,2)=-70 nm` | `47.17 nm` | `56.54 nm` | `25.30 nm` | `0.490` |

解释：

- `map99` 与 `map0 + fixed99` 的实际模拟 PSF 几乎相同；相同 seed 下最大像素差约 `3.42 ADU`，平均绝对差约 `0.07 ADU`。
- 但这两个设置给 FiLM 网络的 conditioning 不完全相同，因此训练结果不能简单解释为“固定 anchor 一定更差”。
- `map-70` 的初始 axial 表现明显更好，说明旧 baseline 的有效物理 map 与当前 `+99` 初始化存在明显差异。

对照 metrics：

- [map99 metrics](../output/unitypsf_anchor_map99_10ep_5686/metrics/eval_metrics.jsonl)
- [map0 + fixed99 metrics](../output/unitypsf_anchor_map0_fixed99_10ep_5686/metrics/eval_metrics.jsonl)
- [map-70 metrics](../output/unitypsf_anchor_map_minus70_10ep_5686/metrics/eval_metrics.jsonl)

## Simulation PNG 证据

总对比图：

[anchor_ablation_simulation_comparison.png](../output/anchor_ablation_simulation_5689/anchor_ablation_simulation_comparison.png)

逐组 triplet：

- [map99 left triplets](../output/unitypsf_anchor_map99_10ep_5686/diagnostics/simulation_frames/left_triplets.png)
- [map0 + fixed99 left triplets](../output/unitypsf_anchor_map0_fixed99_10ep_5686/diagnostics/simulation_frames/left_triplets.png)
- [map-70 left triplets](../output/unitypsf_anchor_map_minus70_10ep_5686/diagnostics/simulation_frames/left_triplets.png)

观察结果：

- 三组都能生成正常的非黑、非 NaN、非单点塌缩 PSF。
- `map99` 与 `map0 + fixed99` 的视觉形态基本一致。
- `map-70` 的 PSF 形态明显不同，但它对应的训练初始 axial RMSE 更低。
- 因此“仿真图像坏了”不是当前主要问题。

## 与 `microtube_dual` 的定位融合结果对比

这些结果是已经训练好的 checkpoint 在 64x64 simulated ROI 上的后处理评估，不等同于训练过程 RMSE。

### Density = 1 emitter/um²

来自：

`results/microtube/microtube_dual/dual_fd_full_roi_map_eval_64px_results.json`

| 方法 | Lateral radial RMSE | Axial RMSE | 3D RMSE |
|---|---:|---:|---:|
| left checkpoint alone | `26.11 nm` | `21.95 nm` | `34.11 nm` |
| inverse-variance raw fusion | `25.68 nm` | `21.68 nm` | `33.61 nm` |
| calibrated fusion | `25.66 nm` | `21.60 nm` | `33.54 nm` |
| full-ROI joint Poisson MAP | `26.07 nm` | `21.85 nm` | `34.01 nm` |

### Density = 2 emitters/um²

| 方法 | Lateral radial RMSE | Axial RMSE | 3D RMSE |
|---|---:|---:|---:|
| left checkpoint alone | `25.41 nm` | `28.05 nm` | `37.84 nm` |
| inverse-variance raw fusion | `25.01 nm` | `27.94 nm` | `37.50 nm` |
| calibrated fusion | `24.80 nm` | `27.66 nm` | `37.15 nm` |
| full-ROI joint Poisson MAP | `25.37 nm` | `27.98 nm` | `37.77 nm` |

### Density = 5 emitters/um²

| 方法 | Lateral radial RMSE | Axial RMSE | 3D RMSE |
|---|---:|---:|---:|
| left checkpoint alone | `36.22 nm` | `54.41 nm` | `65.36 nm` |
| inverse-variance raw fusion | `35.17 nm` | `53.86 nm` | `64.32 nm` |
| calibrated fusion | `34.98 nm` | `53.91 nm` | `64.26 nm` |
| full-ROI joint Poisson MAP | `36.19 nm` | `54.37 nm` | `65.31 nm` |

这些结果说明当前已实现的 checkpoint-output fusion 和 joint Poisson MAP 只带来小幅改善，不能解释 UnityPSF 正式训练后期的退化。真正有价值的上限来自物理模型和训练模型保持一致，而不是单纯在错误的后处理输出上加权。

## CRLB / Fisher 结果

`microtube_dual` 中的 zmap 联合 Fisher/CRLB 计算显示，双通道共享 XYZ 时理论局部精度优于单通道：

- left median CRLB：`[14.13, 8.32, 10.73] nm`
- right median CRLB：`[13.78, 16.04, 28.67] nm`
- joint median CRLB：`[8.84, 6.14, 9.17] nm`

相对 left 的中位数信息增益约为：

- x：`1.60x`
- y：`1.42x`
- z：`1.18x`

这证明联合物理模型在理论上有价值，但 CRLB 是理想局部下界，不能代替实际网络 RMSE。当前 UnityPSF 的实际瓶颈是模型/物理更新不一致，不是 Fisher 信息不存在。

## 当前已发现的实现风险

### 1. Physical update 可能把 map 推偏

正式 run 的 gamma update 在后期仍被接受，即使 eval RMSE 已持续恶化。当前 accept policy 主要监控 ROI Poisson loss，没有把固定 eval set 的定位 RMSE 作为回滚条件。

### 2. Anchor 语义尚未完全统一

当前已加入独立 fixed anchor 支持，但仍需完成以下 contract：

- residual map 与 fixed anchor 的语义必须写入 resolved config 和 physical state；
- conditioning 输入必须与实际 renderer 的 effective PSF 一致；
- 导出 map 时必须明确是 residual map 还是 effective map；
- 禁止 map 已含 `+99 nm` 时再叠加 `fixed_astig_anchor_nm=99` 的 double count。

### 3. Anchor 对照实验尚未覆盖 physical update

`5686` 只关闭了 ROI physical update，验证的是初始 simulator 与网络训练，不是多轮 physical update 的完整链路。因此它能说明 anchor 初始化影响，但不能单独证明 update contract 已修好。

## 当前判断

当前结果更支持下面这个诊断：

> UnityPSF 的 direct CUDA simulation 是正常的，网络早期也能学习；主要退化来自 physical update 对物理 map 的累计修改，以及 residual map、fixed anchor、FiLM conditioning、导出 map 之间的语义没有完全统一。

这条路线不是走不通。`microtube_dual` 的结果说明双通道物理联合和 Fisher 信息确实有价值，但必须先让 physical update 在固定 eval set 上稳定，否则后续联合定位和 CRLB 都建立在漂移的 PSF 模型上。

## 建议的下一步

1. 固定 100-frame eval set，physical update 前后都评估同一批数据。
2. 每次 physical update 同时记录：axial/lateral RMSE、Jaccard、map `(2,2)`、全 map RMS、ROI held-out Poisson loss。
3. 一旦 fixed eval axial RMSE 连续恶化，拒绝该次 map update，并恢复上一个 physical state。
4. 完成 `residual map + fixed anchor` 的统一 conditioning 和 physical-state/export 语义。
5. 只在 physical update 稳定后，再继续比较 Bayesian joint MAP、GlobLoc-style fit 和 Fisher/CRLB。

## 原始证据路径

- UnityPSF 正式 eval：`output/unitypsf_cuda_gemm_roi100_update20_every10_batch64_from_scratch_20260929/metrics/eval_metrics.jsonl`
- UnityPSF physical update metrics：`output/unitypsf_cuda_gemm_roi100_update20_every10_batch64_from_scratch_20260929/metrics/gamma_update_metrics.jsonl`
- `microtube_dual` FD baseline eval：`results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat/training_dual_anchor99_gamma10_conditionfix/run/metrics/eval_metrics.jsonl`
- `microtube_dual` uniform baseline eval：`results/microtube_right_benchmark_20260711/runs/neptune_v03_uniform_ablation/training_peak_anchor99/run/metrics/eval_metrics.jsonl`
- `microtube_dual` joint Poisson evaluation：`results/microtube/microtube_dual/dual_fd_joint_map_eval_64px_results.json`
- `microtube_dual` full-ROI joint evaluation：`results/microtube/microtube_dual/dual_fd_full_roi_map_eval_64px_results.json`
- `microtube_dual` zmap Bayesian evaluation：`results/microtube/microtube_dual/zmap_bayesian_eval_64px_results.json`
- `microtube_dual` joint CRLB summary：`results/microtube/microtube_dual/unitypsf_zmap_joint_crlb_200frame_summary.json`
