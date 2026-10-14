# UnityPSF v0.3：CLIMELoc 风格 Physical Update 融合训练方案

- 状态：阶段 A/B 接口与 epoch-end 接入已完成；尚未开始重新训练
- 目标：在 UnityPSF 中加入连续通道变换、双通道原始光斑联合拟合和物理模型驱动训练
- 当前依据：真实 Sample TIFF 上的 6 个 ROI 联合 CRLB 均低于左通道；中位理论改善为 X 1.60x、Y 1.42x、Z 1.18x

## 1. 总体架构

```text
left/right raw ROI
        |
        v
channel-specific zmap PSF renderer
        |
        v
continuous channel transform T_left/T_right
        |
        v
shared XYZ + channel-specific photon/background
        |
        v
joint Poisson MAP / Physical Update
        |
        v
updated physical state -> dual-channel training simulator -> UnityPSF checkpoint
```

左通道作为坐标参考系。右通道使用相对于左通道的变换。网络结构和输出 contract
第一阶段保持不变，只替换物理模型、模拟器和训练目标。

## 2. 数据和坐标 contract

### 2.1 通道数据

- 每个 emitter 必须有同一个 GT XYZ。
- 左右通道允许使用不同 photon、background 和 camera noise。
- 左右通道使用各自的 zmap coefficient map。
- 所有坐标统一为 nm；ROI 内坐标和全场坐标必须显式记录 origin。
- 右通道 crop 偏移不能混入 emitter 的物理坐标。

### 2.2 Transformation map

第一版使用二维 tensor-product Legendre map：

\[
T_x(x,y)=\sum_{i,j}a_{ij}L_i(x)L_j(y),\qquad
T_y(x,y)=\sum_{i,j}b_{ij}L_i(x)L_j(y)
\]

- 二阶 9 项作为默认模型；三阶作为消融对照。
- z 先使用 identity，后续只有 beads 证据支持时才加入 z-dependent 项。
- left map 固定 identity；right map 优化相对变换。
- map 系数加零阶/一阶锚定和空间平滑正则，防止整体漂移和局部发散。
- 每次 Physical Update 保存系数、版本号、训练 round、数据 hash 和残差统计。

## 3. 实施阶段

### 阶段 A：接口冻结和单 ROI 基线

目标：固定所有模块的输入输出，不改网络。

任务：

1. 冻结 `ZmapCoefficientRenderer` 的 field origin、ROI origin 和单位 contract。
2. 冻结 `PixelLegendreTransform` 的系数顺序、归一化范围和参考通道。
3. 冻结联合 fitter 的参数顺序：shared XYZ、per-channel photon/background。
4. 增加 fit manifest：zmap hash、map hash、ROI origin、初值、loss、收敛状态。

验收：已有单 ROI raw TIFF 联合拟合可复现；左右 map 不串写；现有单通道测试保持通过。

### 阶段 B：双通道 Physical Update

目标：让 map 和 PSF 参数从原位光斑中更新。

每轮执行：

1. 使用当前 ckpt 检出左右 emitter。
2. 用当前 map 做跨通道软配对。
3. 取左右 raw ROI，使用各自 zmap renderer。
4. 对每个配对 ROI 做共享 XYZ 的 Poisson MAP 拟合。
5. 汇总所有 ROI 的残差和梯度，更新 right Legendre map 系数。
6. 更新需要更新的 PSF/physical 参数；网络参数暂时冻结。
7. 在 held-out ROI 上计算 Poisson NLL、配准残差和 CRLB。

建议默认配置：

- 先低密度 emitter、单 emitter ROI；
- 每轮 map 更新使用 robust loss，并剔除明显错误匹配；
- map 更新最多 3--5 轮；
- 每轮保存独立 physical state，不覆盖上一轮。

验收：held-out ROI 的 NLL 不恶化，map 残差下降，联合 CRLB 仍低于单通道。

### 阶段 C：有 GT 双通道训练模拟器

目标：用同一组 GT XYZ 生成左右训练光斑。

模拟器必须支持：

- left/right 独立 zmap；
- Legendre channel transform；
- 独立 photon/background；
- Poisson、read noise、camera offset；
- emitter density 和 overlap；
- channel dropout/低 SNR；
- 输出 GT XYZ、mask、每通道 photometry 和 transform version。

第一版只做单 emitter 和 2-emitter overlap，确认 simulator 与 renderer 完全一致后，
再加入高密度场景。

### 阶段 D：重新训练 UnityPSF

目标：让网络学习经过通道变换和物理模型更新后的数据分布。

第一轮训练策略：

1. 从当前 checkpoint 继续训练；
2. 网络结构不变；
3. 使用双通道有 GT simulator；
4. 保留原有 localization/GMM loss；
5. 增加轻量 channel consistency loss；
6. 物理 Poisson loss 只作为辅助项，初始权重很小；
7. map 系数先冻结，确认网络稳定后再允许交替更新。

推荐 loss：

```text
L = L_localization
  + lambda_consistency * L_channel_consistency
  + lambda_poisson * L_joint_poisson
  + lambda_map * L_map_smooth
```

初始建议：`lambda_consistency` 和 `lambda_poisson` 从很小值开始，避免网络为了降低
物理 loss 破坏现有 detector。

### 阶段 E：高密度和多 emitter linking

目标：把 CLIMELoc 的高密度优势接入，而不是只做单 emitter 平均。

顺序：

1. 2-emitter overlap 联合 PSF；
2. soft association/linking；
3. 3-emitter overlap；
4. 高密度 emitter 全局/局部联合拟合。

这一阶段不与第一轮训练同时引入，避免无法判断性能变化来自 map、PSF 还是 linking。

## 4. 评估协议

### 有 GT 数据

必须报告：

- X/Y/Z RMSE；
- median、P90 和失败率；
- 单通道、简单加权融合、联合 MAP、联合 MAP+map update 四组对照；
- density 0.2、1、2、5 emitter/um²；
- photon/background 分层结果；
- overlap 分层结果。

### 无 GT Sample TIFF

只报告：

- joint Poisson NLL；
- channel residual；
- CRLB；
- map smoothness 和漂移；
- linking 稳定性。

不能把 Sample TIFF 的 CRLB 写成真实 RMSE。

## 5. 训练和发布门槛

进入正式重训前：

- 有 GT 双通道模拟中，联合 MAP 的 XYZ RMSE 优于左右任一单通道；
- map update 不造成 held-out NLL 恶化；
- 变换残差没有明显空间热点；
- 低 SNR 通道不会拖坏高 SNR 通道；
- 高密度测试没有明显增加 false merge 和 false split。

正式发布前：

- 新旧 checkpoint 在同一 GT 测试集上配对比较；
- 保留旧 checkpoint 作为回滚基线；
- physical state、map、zmap 和训练配置全部写入 manifest；
- 训练、推理和报告脚本可使用同一套版本 contract。

## 6. 明确不做的事情

- 在没有 GT RMSE 前，不宣称已经突破 CRLB。
- 不把两个单通道输出简单平均当作最终联合模型。
- 不在第一轮同时加入高密度 linking、map 高阶项和网络结构大改。
- 不使用不可解释的 emitter-specific 自由变换替代连续 channel map。
- 不在验证联合 Physical Update 前重新训练网络。

## 7. 推荐执行顺序

```text
A. 冻结 contract
  -> B. 双通道 Physical Update
  -> C. 有 GT simulator
  -> D. 小规模继续训练
  -> E. held-out RMSE 验收
  -> F. 高密度 linking
  -> G. 正式训练和发布
```

当前应从 **阶段 A 和阶段 B** 开始；现有 CRLB 结果作为联合物理模型的初步 go/no-go
证据，但不替代有 GT 的训练验收。

## 8. 当前实现进度

已完成：

- `run_climeloc_physical_update`：多轮 right-to-left Legendre map + shared XYZ 联合物理拟合。
- `LegendreTransform.to_dict`：保存 basis、归一化、系数和 residual covariance。
- `persist_climeloc_physical_update`：原子写入
  `metadata/climeloc_physical_update.json`，并登记 artifact SHA256。
- high-fidelity epoch-end hook：gamma hook 之后按间隔调用 CLIMELoc hook；更新结果写入
  `metrics/climeloc_physical_update_metrics.jsonl`。
- `build_raw_tiff_climeloc_updater`：消费现有 `build_roi_bank_from_loc_harvest` 产出的左右
  ROI bank，按严格相同的 canonical frame window 对齐，使用检测 uncertainty 做跨通道配对，再调用 CUDA
  编译的 direct simulator renderer 进行多轮 Legendre + shared-XYZ Poisson/MAP 更新；
  `microtube` 基础配置的 simulation backend 默认已经切到 `native`。
- `build_raw_tiff_bank_provider`：每次 Physical Update 用当前内存模型分别执行左右
  `build_roi_bank_from_loc_harvest`，并把同一张 canonical frame-window 表注入两端，生成配对所需的真实 raw-TIFF ROI bank。
- 默认配置 `enabled: false`，因此旧训练路径不变。
- 正式任务使用与成熟 `microtube_dual` 对齐的 `417 steps/epoch`，以
  `global_step=12510` 作为 detector warm-up 完成门槛；只按 epoch 计数会把 2-step
  smoke 配置误当成正式预热。从该步数后按间隔启用 raw-TIFF CLIMELoc update。
- native/direct 训练将优化步数和 sequence 数分开：每 epoch 保持 417 个 optimizer
  steps，但使用默认 64 个 cached sequences，让每个 sequence 提供多个中心样本并复用
  一次物理模拟；禁止把 `sequence_count` 设成 `steps_per_epoch`。

尚未完成：

- raw TIFF 到左右 ROI、ckpt 检出、GT/软匹配的生产级 updater；当前 hook 通过依赖注入接收
  `PhysicalUpdateResult`，不会在 engine 内隐式构造数据。现在 raw-TIFF bank provider 和
  updater 适配器已经完成；engine 改为接收运行时 factory，并把**当前内存中的训练模型**
  传给 factory，不再要求加载左右 ckpt。
- 有 GT 的双通道 RMSE 验收和正式重训。

启用前，运行器必须注入 runtime updater factory，并保证它使用当前 epoch 的 raw ROI、左右
zmap 和 direct CUDA renderer；仅在配置中把 `enabled` 改为 `true` 而没有 factory 会直接
报错，避免训练悄悄退化成普通流程。

## 9. 正式训练任务

已提交从零训练任务：

- Slurm job：`5655`
- 脚本：`scripts/train/climeloc_native_from_scratch.sbatch`
- 训练：300 epochs，`simulation_backend: native`
- detector warm-up：epochs 1--30
- CLIMELoc：epoch 31 开始，每 10 epochs 更新
- renderer：`direct_cuda`
- raw TIFF：`/share/home/guest/race/datasets/samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif`
- zmap：左右真实 `coeff_maps.npz`

提交命令已返回 `Submitted batch job 5655`。当前 shell 无法读取 Slurm controller 的队列和
日志流，因此暂时不能在这里确认运行节点、GPU 名称或首个 epoch 输出；这些必须以
`logs/slurm/unity_climeloc_native-5655.out` 中的实际记录为准。

任务后来修正并重新提交为 job `5657`：`5655` 因 Slurm spool 路径解析失败，`5656` 因
zmap artifact 默认路径错误，均在训练启动前失败。`5657` 已进入 `RUNNING`，输入和配置
检查通过，左右 peak zmap bootstrap 已完成并生成 `checkpoint_latest.pt`。最近指标达到
`global_step=62`，约为第一次 CLIMELoc 更新点（epoch 31）；截至最近检查尚未生成
`climeloc_physical_update.json`，第一次 raw-TIFF/direct-CUDA 更新仍在执行中，stderr
没有运行时错误。

随后发现首次更新失败的实际原因是左右 ROI 独立筛选后没有共同窗口。现已撤回重叠窗口 fallback，改为
左右 detector 逐个处理同一张 canonical frame-window 表；updater 只接受完全相同的 `frame_window`，并通过严格同步测试。训练
脚本现在支持 `CLIMELOC_RESUME_CHECKPOINT`，可从 `5657` 的 checkpoint 继续而无需重复
前 30 个 warm-up epoch。

raw-TIFF detector 使用和训练一致的左右 Zernike/FiLM condition context，并记录每个窗口的
model output、概率、几何、光子、sigma 和最终 accepted 数量。如果某一更新周期 detector 没有留下任何 ROI，hook 会记录
`no_common_synchronous_roi` 并跳过该周期，在下一个间隔重试；这不会放宽窗口匹配规则。

修复后恢复任务为 job `5659`，当前已进入 `RUNNING`，配置检查通过。

### updater 的实际调用方式

```python
updater = build_raw_tiff_climeloc_updater(
    bank_provider=build_left_right_bank_provider(...),
    renderers_by_channel={"left": left_direct_cuda_renderer, "right": right_direct_cuda_renderer},
)
epoch_hook = build_climeloc_epoch_hook(
    layout=layout,
    config=config["train"]["climeloc_physical_update"],
    updater=updater,
)
```

updater 要求 `renderer_backend: direct_cuda`，不接受 LUT renderer；左右 bank 必须有共同的
完整 `frame_window`（grid cell 可不同）。一阶 map 至少需要 4 个
有效配对点；点数不足时会明确失败，不会用欠定 map 写入 physical state。

正式 GPU 验证的最小调用链为：

```text
live training model + raw TIFF
  -> build_raw_tiff_bank_provider
  -> build_raw_tiff_climeloc_updater
  -> build_climeloc_epoch_hook
  -> direct CUDA renderer + joint MAP
```
