# UnityPSF 可视化训练与验收协议

- 状态：任务 11 起强制执行
- 日期：2026-08-04
- 目标：让每一次开发、训练和推理都产生人能直接判断的科学结果，而不只产生日志和测试数字

## 1. 核心原则

UnityPSF 的开发必须同时满足三种证据：

1. **工程证据**：测试、schema、hash、checkpoint round trip 和退出码正确。
2. **数值证据**：loss、precision/recall、误差、训练稳定性和资源开销可量化。
3. **视觉证据**：输入、PSF 形态、路由、定位叠加、重建和物理状态能被直接检查。

任何正式任务缺少其中一类证据，都不能标记为完成。CPU 只允许执行 contract
测试和短 smoke；正式训练、评估和论文图数据必须来自日志明确确认 CUDA 的 SLURM
GPU 作业。

## 2. 第一个可见里程碑

第一阶段不等待 Double Helix 数据，先完成：

```text
一个 unitypsf_joint.ckpt
    +-- Emitter2DExpert(channel=main)      <- 真实 Origami 2D
    +-- AstigmatismExpert(channel=left)    <- 独立网络和物理状态
    +-- AstigmatismExpert(channel=right)   <- 独立网络和物理状态
```

这就是“**双模态 + 多通道**”的第一版正式含义：

- 双模态：`emitter_2d` 与 `astigmatism`；
- 多通道：统一 channel contract 支持 `main/left/right`，其中像散 left/right 必须在真实流程中独立验收；
- 一个模型：统一 `UnityPSF` 对象和 `localize(...)` API；
- 一个 checkpoint：所有已验证实例位于同一个 joint checkpoint；
- 一次训练：一个父 run 协调多个模态/通道，可按单卡轮转或 Expert Parallel 执行；
- 分开看结果：所有图和指标必须保留 modality/channel 维度，禁止只报平均值。

Origami 数据入口使用工作区现有的
`datasets/training_sets/origami_2d`。该路径当前指向共享数据归档，训练配置只能记录
数据 manifest 和 hash，不能将大型 OME-TIFF 复制进 Unity 仓库或 checkpoint。

## 3. 每个 run 的固定产物

所有运行产物写入：

```text
output/unitypsf/<run_id>/
    config/
        resolved.yaml
        data_manifest.json
        route_plan.json
    checkpoints/
        latest/unitypsf_joint.ckpt
        milestones/epoch_<NNNN>.ckpt
        release/unitypsf_joint.ckpt
    metrics/
        train_metrics.jsonl
        eval_metrics.jsonl
        route_metrics.jsonl
        resource_metrics.json
        summary.json
    figures/
        00_input_audit.png
        01_psf_patch_montage.png
        02_route_and_step_balance.png
        03_training_curves.png
        04_prediction_overlay_<modality>_<channel>.png
        05_error_by_z_<modality>_<channel>.png
        06_reconstruction_<modality>_<channel>.png
        07_physical_state_<modality>_<channel>.png
        08_cross_modality_scorecard.png
    report/
        report.html
        figure_index.json
    provenance.json
```

SLURM 日志统一写入 `logs/slurm/unitypsf/<run_id>/`。Matplotlib 缓存写入
`.local/cache/matplotlib/`。不得在项目根目录生成 PNG、JSON、checkpoint 或
`slurm-*.out`。

## 4. 强制图组

### 4.1 输入数据审计

`00_input_audit.png` 必须包含：

- 每个模态和通道的真实原始帧；
- 实际训练 crop 边界；
- ADU 强度直方图与饱和像素比例；
- 帧数、尺寸、dtype、相机 baseline 和 gain；
- train/validation/test acquisition 分组数量。

`01_psf_patch_montage.png` 必须包含每个模态的代表性 patch。像散按 z-bin 展示，
Double Helix 到位后按 z 或 lobe angle 展示。Origami 没有 z 标签时按强度和局部密度
分层抽样，禁止只挑最好看的 patch。

### 4.2 路由与训练平衡

`02_route_and_step_balance.png` 必须展示：

- `(modality, channel_id)` 路由计数；
- 每个 expert 的 forward 次数和 optimizer step 数；
- 每个 expert 看到的样本数；
- 拒识和路由冲突数量；
- Expert Parallel 时每个 rank 对应的 expert instance。

任何 expert 长时间没有 step、路由计数与数据 manifest 不符、或 left/right 被合并，
都必须停止训练排查。

### 4.3 训练曲线

`03_training_curves.png` 至少分面展示：

- 每个模态/通道的 train 与 validation total loss；
- detection、localization、photon、background 和模态专属 loss；
- learning rate、gradient norm 和 AMP scale；
- Astigmatism 的 gamma update 指标；
- 后续 DH 的 lobe-angle、lobe-separation 和 calibration loss。

不同 expert 的 loss 尺度不得直接求平均后只画一条线。图中必须标记 resume、gamma
更新、checkpoint commit 和异常 step。

### 4.4 定位叠加与重建

每个模态/通道都必须生成 `04_prediction_overlay_*` 和 `06_reconstruction_*`：

- 原始图像；
- detection probability 或候选点；
- 过滤前定位；
- 过滤后定位；
- 定位重建；
- 原始图和重建图的并排或叠加比较。

有 ground truth 的模拟数据必须同时画匹配/漏检/误检，并输出 precision、recall、
Jaccard、x/y/z bias 和 RMSE。真实 Origami 没有 ground truth 时，重点检查结构连续性、
重复伪影、网格伪影、背景假阳性、局部密度和 photon 分布，不能伪造 GT 指标。

### 4.5 物理状态

Astigmatism 的 `07_physical_state_*` 必须包含：

- 当前 peak zmap；
- 初始与当前 gamma/系数差值；
- left/right 分开显示且使用相同色标；
- raw ROI 与当前物理模型重建对比；
- 物理状态版本和 checkpoint epoch。

Double Helix 数据到位后增加 lobe angle-to-z 曲线、有效范围、残差和外推区域。

### 4.6 跨模态总览

`08_cross_modality_scorecard.png` 只做并排对照，不隐藏单项失败。至少显示：

- 每个模态/通道是否完成；
- 最佳与最新 epoch；
- 主要 localization 指标；
- 样本数、参数量、峰值显存和吞吐；
- checkpoint 中对应实例的 hash；
- failed、unsupported 和 not-evaluated 状态。

## 5. 更新频率

- 训练开始前：必须生成输入审计和 PSF montage。
- 每个 epoch：写机器可读 metrics，不要求每次重画大图。
- 默认每 5 个 epoch：刷新轻量训练曲线和固定样本 quicklook。
- 每个 milestone checkpoint：运行完整 validation 并生成全部图组。
- 训练结束：生成不可变 `report.html`、`summary.json`、figure hash 和 release checkpoint hash。

固定可视化样本由数据 manifest 指定，训练过程中不得根据结果换成更好看的样本。

## 6. 人工验收门

每个任务完成后必须汇报：

1. SLURM job ID 和日志中的 CUDA 证据；
2. `report.html` 的绝对路径；
3. 至少一张输入审计图、一张训练曲线和每个模态/通道的一张结果图；
4. `summary.json` 的关键指标；
5. release checkpoint 的 SHA-256；
6. 已知失败、未支持模态和未完成通道。

以下情况不能通过人工验收：

- 总 loss 下降但某个模态或通道持续恶化；
- 路由计数或 optimizer step 不平衡且无配置依据；
- 真实图像出现明显重复、网格、背景点或结构断裂；
- left/right 的 peak zmap、gamma 或预测结果意外完全相同；
- checkpoint 能加载但图像结果为空、全亮、全暗或单位错误；
- CPU fallback 的结果被混入正式报告。

## 7. 图形规范

- PNG 至少 300 DPI；曲线和统计图同时输出 PDF 或 SVG。
- 使用色盲友好配色，并用线型/marker 冗余编码模态。
- 热图使用感知均匀 colormap，差值图使用以 0 为中心的发散 colormap。
- 所有轴标注单位；误差条必须说明 SD、SEM 或置信区间。
- 图中显示真实样本数量，不只显示均值。
- 同一类 left/right 对比必须使用一致坐标范围和色标。
