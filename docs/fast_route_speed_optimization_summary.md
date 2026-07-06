# Neptune v0.3 Fast Route 速度优化总结

本文档整理 Neptune v0.3 fast route 相对 3052 基线所做的训练速度优化工作，目标是作为 PPT 汇报材料的文字底稿。重点说明：我们不是降低物理模型复杂度来换速度，而是在尽量保持 3052 训练语义和物理一致性的前提下，减少重复计算、缩短 renderer/LUT/sequence 的工程路径，并修复影响训练质量的 scheduler 配置问题。

## 1. 对标基线：3052

当前对标对象是 `neptune_iwae` 的 3052 run。它是现阶段最可信的训练质量基线。

3052 的关键训练设置如下：

```text
ROI size: 128 x 128
PSF patch: 51 x 51
batch_size: 24
steps_per_epoch: 417
center_samples_per_epoch: 24 x 417 = 10008
scheduler_step_unit: epoch
physical update start_epoch: 30
physical update interval: 5 epochs
target_projected_emitters: 5000
gamma_steps: 100
```

3052 的质量目标大致是：

```text
early stage:
  step 3336 / epoch 8: Jaccard ≈ 0.4066
  step 12927 / epoch 31: Jaccard ≈ 0.7577

final stage:
  Jaccard ≈ 0.878
  RMSE_lat ≈ 19-20 nm
  RMSE_ax ≈ 26 nm
```

因此，v0.3 fast route 的目标不是简单地追求更快，而是：

```text
1. 保持和 3052 一致的 epoch-based 训练语义。
2. 保留 field-dependent zmap、vector PSF、dual-domain physical update。
3. 减少 online simulation 和 physical update 中的重复工程开销。
4. 尽量把计算留在 GPU 上，减少 Python 调度和 CPU-GPU 往返。
5. 保证 early Jaccard / Recall 恢复到 3052 同步水平。
```

## 2. 原始 v0.3 慢在哪里

v0.3 相比 3052 增加了更完整的 online simulation、LUT、ROI library、physical update、diagnostic、profile 等系统模块。如果不做工程优化，额外开销主要来自以下几个方面：

```text
1. renderer / vector PSF context 生命周期过短。
2. vector PSF precompute 没有稳定跨 batch / epoch / physical version 复用。
3. field-dependent zmap 下，LUT 的生命周期和 physical model update 没有明确绑定。
4. 每个 step 重复组织 sequence/window/condition。
5. patch projection / placement 存在 Python 循环和 tensor 拼装开销。
6. physical update 中 posterior inference、projection、gamma objective 计算较重。
7. scheduler 曾经错误地按 optimizer step 衰减，导致训练质量明显退化。
```

这些问题可以分为两类：

```text
速度问题:
  重复构建 renderer、重复 build LUT、重复组织 cached window、projection 低效。

质量问题:
  scheduler_step_unit 错误使用 optimizer_step，导致 LR 过快衰减。
```

## 3. Fast Route 的总体架构

Fast route 的核心思路是把重物理计算从训练 step 中移出去，改成按 physical version 复用：

```text
physical model update
        ↓
生成新的 physical version
        ↓
build / prewarm 当前 physical version 对应的 LUT
        ↓
多个 epoch 复用同一个 physical version + LUT
        ↓
training step 只做 LUT lookup、subpixel shift、projection、loss
```

也就是说：

```text
physical model 不变时，PSF basis / LUT 不应该每个 step 重新构建。
```

Fast route 的训练 step 主要变成：

```text
sample emitters
        ↓
根据 xy / z / field origin lookup LUT patch
        ↓
subpixel shift
        ↓
Triton fused projection / placement
        ↓
Poisson camera noise
        ↓
GMM posterior loss
```

这条路线的原则是：

```text
物理语义不变，工程执行路径更快。
```

## 4. 优化一：Renderer / Vector PSF Context 复用

### 问题

早期 v0.3 online simulation 路径中，renderer 和 vector PSF context 生命周期偏短，容易在 batch / sequence 级别重复构建。

这会带来：

```text
1. pupil / wavevector / CZT helper 重复 precompute。
2. GPU tensor / buffer 重复分配。
3. Python object 初始化成本反复出现。
4. 每个 batch 都有隐藏的 renderer setup 开销。
```

### 优化

Fast route 引入 renderer cache 和 physical-version-aware context 复用：

```text
1. renderer / vector PSF context 在 physical model 生命周期内复用。
2. physical model 不变时，不重复初始化 vector PSF context。
3. physical update 后才刷新对应 renderer/LUT。
```

### 对齐 LUNAR 的设计

LUNAR 的 `Simulator` 和 `VectorPSFTorch` 是模型生命周期内复用的，`VectorPSFTorch.__init__()` 里的 `_pre_compute()` 会缓存 pupil、wavevector、CZT 辅助量。

v0.3 fast route 的方向与 LUNAR 对齐：

```text
把一次性 precompute 从 batch 级别提升到 renderer / physical version 生命周期。
```

## 5. 优化二：Physical-Versioned LUT Lifecycle

### 问题

如果 LUT 生命周期没有和 physical model update 绑定，就容易出现两类问题：

```text
1. 重复 build LUT，训练 step 或 epoch 内出现不必要开销。
2. LUT 和当前 physical model version 不一致，影响物理一致性。
```

### 优化

我们补上了 explicit physical-versioned LUT lifecycle：

```text
physical_version = k
        ↓
build / prewarm LUT once
        ↓
epoch k ... k + interval 内复用
        ↓
physical model update
        ↓
physical_version = k + 1
        ↓
重新 build / prewarm LUT
```

当前 fast route 中启用：

```text
NEPTUNE_V03_LUT_EPOCH_PREWARM=1
```

含义是：

```text
physical update 后预热 LUT；
同一个 physical version 内多个 epoch 复用；
training step 内不再 build LUT。
```

### 收益

```text
1. 避免 step 内重复 build LUT。
2. 避免 epoch 内重复构建 PSF basis。
3. 保证 LUT 始终对应当前 physical version。
4. 把重计算集中到 physical update 后的一次性流程。
```

## 6. 优化三：Global-Field LUT + FP16 Storage

### 问题

field-dependent zmap 意味着不同 FOV 位置的 PSF 不一样。如果每个 ROI origin 都单独 build LUT，会导致巨大重复开销；如果只用固定少数 ROI LUT，又会限制 zmap 覆盖。

### 优化

我们改成：

```text
lut_simulation.field_mode = global_field
lut_simulation.storage_dtype = fp16
lut_simulation.field_stride = 16
lut_simulation.subpixel_bins = 1
```

核心思想：

```text
1. LUT 覆盖整个 FOV 的 field-dependent zmap。
2. 每个 step 可以选择不同 ROI origin。
3. 不需要每个 ROI 单独 build LUT。
4. FP16 降低全场 LUT 显存占用。
5. 仍然保留 field-dependent PSF 变化。
```

### 解决的矛盾

这个优化解决了两个需求之间的冲突：

```text
需求 A: 每个 step / sequence 可以覆盖不同 zmap ROI。
需求 B: 不希望每个 step 重新构建 PSF LUT。
```

现在的方案是：

```text
全场提前建好 LUT；
step 内只根据 ROI origin 做 lookup / crop。
```

## 7. 优化四：Cached Window / Sequence Precompute

### 问题

3052 的实际训练语义是 epoch-based：

```text
batch_size = 24
steps_per_epoch = 417
samples_per_epoch = 10008
```

v0.3 早期从 batch-budget 切回 epoch 训练时，sequence/window 语义曾经混乱，并导致训练质量和速度问题。

### 优化

Fast route 启用了 cached-window 相关预计算：

```text
NEPTUNE_V03_CACHED_WINDOW_PRECOMPUTE=1
batch_strategy = cached_window
cached_window_order = auto
cached_window_max_gpu_sequences = 2
```

作用：

```text
1. sequence window 不在每个 step 重复组织。
2. zmap / film condition / onehot condition 提前准备。
3. step 内只取已经组织好的 batch。
4. 减少 Python 层切片、stack、pad、condition attach 的重复开销。
```

当前 profile 中这些开销已经相对较小：

```text
cached_window_attach_film_conditioning ≈ 0.6 ms
cached_window_condition_stack_onehot ≈ 0.5 ms
cached_window_slice_loop ≈ 1.2 ms
cached_window_stack_tensors ≈ 0.5 ms
```

## 8. 优化五：Triton Fused Projection / Placement

### 问题

patch projection / placement 如果通过 Python 循环、分散 tensor 操作、scatter/index 组合完成，会有明显调度和中间 tensor 开销。

### 优化

Fast route 启用：

```text
NEPTUNE_V03_PROJECTION_BACKEND=triton_fused
```

即使用 Triton fused projection / placement。

### 实测结果

当前 3362 profile 中：

```text
profile_triton_project_patches_to_frames_s ≈ 0.18 ms / step
```

这说明：

```text
patch placement / projection 已经基本不再是训练速度主瓶颈。
```

这是 fast route 里最明确的 GPU fused 优化之一。

## 9. 优化六：LUT Lookup + Shift Pipeline

当前 online simulation 的主要耗时大致为：

```text
online simulation total ≈ 36 ms / step
LUT lookup ≈ 5.7 ms / step
Fourier shift patches ≈ 4.6 ms / step
Triton project patches to frames ≈ 0.18 ms / step
Poisson camera noise ≈ 0.48 ms / step
GMM posterior loss ≈ 21.5 ms / step
```

这说明经过 fast route 后：

```text
1. projection / placement 已经非常快。
2. LUT lookup 和 shift 成本可控。
3. GMM posterior loss 成为 step 内较大的固定成本。
```

我们曾经测试过 bilinear / Triton shift 等替代路线，但对于 formal-equivalent 的 PSF 形态一致性，当前仍以 Fourier shift 作为更稳妥方案。

因此当前结论是：

```text
在保持 PSF 形态一致的前提下，projection 已经基本优化到位；
后续主要瓶颈转移到 GMM loss、LUT lookup/shift 和 physical update。
```

## 10. 优化七：Physical Update 工程整理

Physical update 仍然是重环节，因为它包含：

```text
1. ROI library 构建。
2. localizer inference。
3. posterior sample。
4. gamma update objective。
5. PSF projection / reconstruction。
6. physical model 更新。
7. LUT rebuild / prewarm。
```

我们已经做过的工程优化包括：

```text
1. renderer_batch_size benchmark，提高 renderer batch size。
2. selected/train ROI bank 和 heldout monitor 解耦。
3. heldout 不足时不 hard fail，只 monitor 或跳过。
4. observed loss / heldout monitor 降低重复计算。
5. diagnostic PNG / recon 生成降频。
6. posterior / projection batch size benchmark。
7. physical version 和 LUT 一致性管理。
```

当前为了对标 3052，3362 使用的是偏强配置：

```text
target_projected_emitters = 5000
gamma_steps = 100
num_posterior_samples = 25
start_epoch = 30
update_interval_epochs = 5
```

因此 3362 不是最省时间的 physical update 配置，而是更偏质量对齐和 3052 parity 的配置。

## 11. 关键质量修复：LR Scheduler 对齐 3052

这不是速度优化，但它是 fast route 能否对标 3052 的关键质量修复。

### 问题

v0.3 fast route 曾经沿用了 base config：

```text
lr_scheduler = StepLR
lr_step_size = 1000
lr_gamma = 0.9
lr_step_unit = optimizer_step
```

在当前训练配置下：

```text
steps_per_epoch = 417
```

如果按 optimizer step 衰减，相当于：

```text
每 1000 optimizer steps 衰减一次
约每 2.4 epoch 衰减一次
```

这会导致 early training 阶段 learning rate 过快降低，localizer 的 detection / recall 学不起来。

### 修复

我们将 fast route 改回和 3052 一致：

```text
lr_step_unit = epoch
```

即：

```text
StepLR(step_size=1000, gamma=0.9)
按 epoch step
```

### 修复效果

对照：

```text
3359: fast route，但 lr_step_unit = optimizer_step
3362: fast route，但 lr_step_unit = epoch
```

同步 early Jaccard：

```text
step 834:
  3359 J = 0.0004
  3362 J = 0.0309

step 1251:
  3359 J = 0.0008
  3362 J = 0.1249

step 1668:
  3359 J = 0.0004
  3362 J = 0.1784

step 2085:
  3359 J = 0.0000
  3362 J = 0.2251
```

结论：

```text
LR scheduler 修复后，fast route 的 early Jaccard / Recall 恢复正常；
之前 early Jaccard 起不来，最大根因是 LR 按 optimizer_step 过快衰减。
```

## 12. 当前 Fast Route 实测速度

当前 3362 配置：

```text
ROI = 128 x 128
PSF patch = 51 x 51
batch_size = 24
steps_per_epoch = 417
epochs = 300
old 3052 zmap init
lr_step_unit = epoch
physical update start_epoch = 30
physical update interval = 5
target_projected_emitters = 5000
```

当前实测：

```text
约 2792 steps 用时约 14 min
2792 / 417 ≈ 6.7 epoch
约 2.1 min / epoch
```

估算：

```text
30 epoch ≈ 1 hour
300 epoch 纯训练阶段 ≈ 10.5 hours
```

需要注意：

```text
epoch 30 后会开始 physical update；
physical update 会增加额外时间；
因此完整 300 epoch 总时间会高于纯训练阶段估算。
```

## 13. 当前 Fast Route 质量进展

3362 early metrics：

```text
step 2085 / epoch 5:
  Jaccard = 0.2251
  Precision = 1.0000
  Recall = 0.2251

step 2502 / epoch 6:
  Jaccard = 0.3139
  Precision = 1.0000
  Recall = 0.3139
  RMSE_lat = 14.30 nm
  RMSE_ax = 28.77 nm
```

和 3052 同步对比：

```text
3052 step 2085:
  Jaccard ≈ 0.2023

3362 step 2085:
  Jaccard ≈ 0.2251
```

说明：

```text
修正 scheduler 后，fast route 的 early Jaccard / Recall 已经恢复到 3052 同步水平附近。
```

仍需继续观察：

```text
1. step 3336 / epoch 8 是否接近 3052 Jaccard ≈ 0.4066。
2. step 12927 / epoch 31 是否接近 3052 Jaccard ≈ 0.7577。
3. physical update 后 RMSE_ax 是否进一步收敛。
```

## 14. Fast Route 相对 3052 的核心贡献

可以概括为四类：

### 14.1 减少 PSF 重复计算

```text
1. renderer / vector PSF context cache。
2. physical-versioned LUT lifecycle。
3. LUT prewarm。
4. global-field LUT。
5. fp16 LUT storage。
```

### 14.2 减少 step 内数据组织开销

```text
1. cached-window precompute。
2. condition cache。
3. sequence / window precompute。
4. step 内只做 lightweight batch assembly。
```

### 14.3 GPU fused 计算

```text
1. Triton fused projection / placement。
2. 减少 Python loop 和中间 tensor 操作。
3. projection 降到约 0.18 ms / step。
```

### 14.4 Physical update 工程整理

```text
1. selected/train ROI bank 和 heldout monitor 解耦。
2. monitor 降频。
3. diagnostic PNG 降频。
4. batch size / renderer batch size benchmark。
5. physical version 和 LUT 一致性管理。
```

## 15. 为什么这些优化不应该破坏仿真质量

Fast route 没有删除关键物理模块。

仍然保留：

```text
1. vector PSF 作为 LUT 来源。
2. field-dependent zmap。
3. left/right channel 独立 zmap / gamma。
4. physical model update。
5. ROI library / posterior / gamma objective。
6. Poisson camera noise。
7. GMM posterior loss。
```

变化主要是：

```text
把重复计算变成 cache / LUT / fused GPU kernel。
```

因此设计原则是：

```text
物理语义不变，工程执行路径更快。
```

## 16. 当前仍然存在的瓶颈

从 3362 profile 看，训练 step 里还比较重的部分包括：

```text
GMM posterior loss ≈ 21.5 ms / step
online simulation total ≈ 36 ms / step
LUT lookup ≈ 5.7 ms / step
Fourier shift patches ≈ 4.6 ms / step
cached sequence precompute windows ≈ 11-12 ms / step
```

而 projection 已经很小：

```text
Triton projection ≈ 0.18 ms / step
```

后续优化重点：

```text
1. GMM posterior loss 是否可以进一步 vectorize / fuse。
2. cached window precompute 是否能进一步提升到 epoch / sequence 级 cache。
3. Fourier shift 是否能在保证 PSF 形态一致的前提下降低成本。
4. physical update 的 posterior / projection / gamma objective 是否还能减少重复计算。
```

## 17. PPT 可用总结表

| 模块 | 3052 / 原始 v0.3 问题 | v0.3 fast route 优化 |
|---|---|---|
| Vector PSF renderer | context 生命周期短，重复初始化 | renderer/context cache |
| PSF 计算 | 频繁重算或生命周期不清晰 | physical-versioned LUT |
| Field-dependent zmap | ROI 采样和 LUT 复用矛盾 | global-field LUT |
| LUT 显存 | 全场 LUT 显存压力 | fp16 storage |
| Training step simulation | step 内重复 build / assemble | LUT lookup + cached window |
| Patch projection | Python / tensor placement 开销 | Triton fused projection |
| Physical update | posterior / projection / objective 重 | batch size + monitor / diagnostic 降频 |
| Scheduler | 曾按 optimizer step 衰减过快 | 改回 3052 epoch scheduler |

## 18. PPT 可用关键数字

```text
当前 3362 fast route:
  ROI128 / PSF51 / batch24 / steps417 / epoch scheduler

速度:
  约 2.1 min / epoch
  约 1 h / 30 epoch before first physical update
  online simulation ≈ 36 ms / step
  GMM posterior loss ≈ 21.5 ms / step
  Triton projection ≈ 0.18 ms / step

质量:
  step 2085:
    3362 Jaccard ≈ 0.225
    3052 Jaccard ≈ 0.202

  step 2502:
    3362 Jaccard ≈ 0.314
    Precision = 1.0
    Recall ≈ 0.314
    RMSE_lat ≈ 14.3 nm
    RMSE_ax ≈ 28.8 nm
```

## 19. 一句话总结

Fast route 的核心贡献是：

```text
在不删除 field-dependent vector PSF 和 physical update 的前提下，
把重复物理计算变成 physical-versioned LUT 复用，
把 projection 变成 Triton fused GPU kernel，
并修正 scheduler 以恢复 3052 级别的 early training dynamics。
```

更适合 PPT 结尾的版本：

```text
Neptune v0.3 fast route 将训练路径从“每个 batch 重复重物理仿真”，改造成“physical update 后一次性构建全场 field-dependent LUT，并在多个 epoch 内复用”的版本化仿真路线；同时通过 cached window、condition cache、Triton fused projection 和 epoch scheduler parity，在保持 3052 质量基线的前提下显著降低训练 step 开销。
```
