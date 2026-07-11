# Neptune v0.3 网络输出后处理与 LiteLoc/LUNAR 对齐实施规划

## 目标

本规划覆盖 localization network 输出之后的完整链路：NMS decode、时空切片合并、localization schema、probability selection、LUNAR-style degrid 和 reconstruction。目标是对齐 LiteLoc/LUNAR 中有明确正式代码依据的处理，并形成可验证、可追溯的 Neptune v0.3 标准路径。

## 当前状态

| 环节 | 当前状态 | 结论 |
|---|---|---|
| LiteLoc production NMS | 已固化 | formal infer 固定 candidate `p>0.3`、adjacent `p>0.6`、`sum`、final `p>0.7` |
| LiteLoc EvalMetric NMS | 已固化 | 与 formal infer 共享 NMS tensor，evaluation 固定 final `p>0.3` |
| LiteLoc weighted-offset NMS | 不采用 | LiteLoc 当前正式 infer 和训练评估调用均未使用该分支，不进入 v0.3 标准路径 |
| ROI overlap cut | 已固化 | `96 context + 80 valid core + cut8`，空间唯一覆盖和边界 golden tests 已通过 |
| 时间窗口合并 | 部分对齐 | 3-frame rolling inference，缺 frame 唯一覆盖测试 |
| Probability sweep | 已实现 | Phase 0 已统一 count weight、nm z range 和 renderer contract |
| Localization schema | 已完成 Phase 0 | H5 v0.2 已写入显式单位、`z_sig/photon_sig` 和 xy offset 字段 |
| Spatial LUNAR degrid | 已固化为默认重构坐标 | 左右通道分别使用 `6 x 12` 空间分块，每块执行 20 个 uncertainty bins、threshold 0.01；标准 filter/recon 默认读取 spatial degrid H5 |
| Raw/derived 双轨 | 已完成 | raw H5 不可变，degrid 另存派生 H5 并记录 source/contract |
| Reconstruction | 已完成 Phase 0 修正 | 默认 count weight，z 色轴和范围统一使用 nm |
| Degrid 定量验收 | 已完成 Phase 4 | NCP `prob>=0.9` 上 LiteLoc-style grid index 下降，验收通过 |
| Drift correction | 已默认启用 RCC | NCP 8000-frame 诊断确认约 184 nm y 向漂移；保留原始/degrid/RCC 三层 H5 |
| Uncertainty calibration | 不扩展为独立 Phase | 仅保留“固定半径或先做最小校准”的渲染决策 |

## 核心数据原则

标准路径必须保存不可变的原始定位：

```text
predictions_raw.h5
```

所有后处理结果均为派生数据：

```text
predictions_prob080.h5
predictions_degrid.h5
```

原始定位用于定量评估、物理模型诊断和可追溯分析。Degrid 只修正展示坐标，不得覆盖 raw localization，也不得用于宣称原始 localization accuracy 提升。

### Spatial degrid 默认修正

3371 全场统一 degrid 的全局 offset 统计接近均匀，但不同视场区域的相反 x 偏差会互相抵消，重构中仍残留一个相机像素周期的竖向光栅。实测光栅周期为 `5.057` 个 20 nm 重构像素，与 `101.11 / 20 = 5.056` 完全一致。

正式路径因此改为左右通道分别执行 `6 x 12` spatial degrid，再执行 RCC。现有 3371 H5 parity 验证中，`5.056 px` 频率功率相对频带中位数从 `513.9` 降至 `9.0`，下降约 `98.2%`；left/right localization 数量不变。原始 `predictions_merged.h5` 继续保持不可变，修正坐标写入派生 H5。

## Phase 0：输出修正与渲染契约

实施状态：已于 2026-07-10 完成代码实现与 NCP 现有 predictions 重建验证。

### 0.1 Localization 字段与单位

新输出应显式包含：

```text
x_px, y_px
x_nm, y_nm
z_nm
photon
prob
x_sig_px, y_sig_px
x_sig_nm, y_sig_nm
z_sig_nm
photon_sig
x_offset_px, y_offset_px
x_offset_nm, y_offset_nm
```

兼容字段 `z/x_sig/y_sig` 暂时保留，但 manifest 必须声明其单位，后续代码优先读取显式字段。

### 0.2 z 颜色契约

当前 predictions 中 z 的实际单位是 nm，范围约为 `-600~600 nm`。正式 renderer 默认必须使用：

```yaml
z_min_nm: -600
z_max_nm: 600
```

色条必须显示 `nm`。禁止再使用 `-0.6~0.6` 直接渲染 nm 数据。

### 0.3 Reconstruction 权重契约

新增：

```yaml
render_weight: count | photon | probability
```

语义：

- `count`：每个 localization 权重为 1，作为正式结构图默认值。
- `photon`：用于光子强度诊断。
- `probability`：用于置信度诊断，不作为正式结构图默认值。

LiteLoc NMS 的 integrated probability 可能大于 1，因此不能默认作为显示亮度。

### 0.4 Phase 0 验收

```text
1. z colorbar 显示 nm，默认范围 -600~600 nm。
2. count-weighted render 不受 integrated probability 数值缩放影响。
3. photon/probability 权重均有明确 manifest。
4. 新 H5 输出包含显式单位字段，旧 H5 仍可读取。
5. 使用现有 NCP predictions 重建 count-weighted prob0.7/0.8/0.9 公平对比图。
```

### 0.5 已落地实现

代码变化：

```text
localization/legacy_decode.py
  保存 z_sigma_nm 与 photon_sigma

infer_recon/predictions_io.py
  支持 H5 v0.2 schema、units metadata 和显式字段优先读取

infer_recon/recon/render_subpixel.py
  默认 z=-600~600 nm
  colorbar 单位 nm
  render_weight=count|photon|probability
  默认 count
  同时保存未归一化 float32 density 与 linear RGB TIFF
  保留 SMAP GUI imax_min=-3.5 parity 选项
  Neptune 正式默认使用 scalar-density q=0.997，避免异常高密度点压黑主体结构
  gamma 默认 1；density 决定亮度，RGB 仅编码 z 色相

infer_recon/recon/render_standard.py
infer_recon/filter/apply_filter_recon.py
  将新 contract 接入标准 filter/recon 路径

scripts/infer/run_3371_full8000_infer_filter_recon.py
  新 inference 输出显式 px/nm、uncertainty 和 offset 字段
```

本次复用旧 NCP H5 进行重建，因此通过兼容字段读取 `z/x_sig/y_sig/photon`；旧 H5 不会被原地改写。下一次正式 infer 将直接写出 `infer_recon_predictions_h5_v0.2`。

验证输出：

```text
output/3421_ncp_center400_right_phase0_count_weight_znm_contract_recon/
```

## Phase 1：严格固化 LiteLoc Evaluation 与 Formal Infer 的实际 NMS 契约

### 1.1 决策

Neptune v0.3 不再设计自己的统一 threshold，而是严格复刻 LiteLoc 当前实际调用链。

两条路径共享同一个 NMS 算法：

```text
candidate threshold: p > 0.3
local maximum: 3x3 max pool
probability integration: center + 4-neighbor
adjacent split: remaining p > 0.6
aggregation: p_ps1 + p_ps2，不 clamp
x: column + 0.5 + x_mu
y: row + 0.5 + y_mu
z/photon/uncertainty: 读取最终候选像素对应的网络输出
```

最终 acceptance threshold 按 LiteLoc 实际用途固定：

```text
training evaluation metrics:
  integrated p > 0.3

formal TIFF inference:
  integrated p > 0.7
```

这是两条固定 contract，不是可供用户选择的 decode mode。Evaluation 不能切成 `0.7`，formal infer 也不能切成 `0.3`。

不实现、也不保留：

```text
neptune_legacy decode mode
custom formal decode mode
liteloc_weighted_offset mode
用户可配置 formal/eval raw_th/split_th/accept_th/aggregation
```

LiteLoc `eval_utils.nms_func()` 虽然包含传入 `xo/yo/zo` 后进行 weighted-offset aggregation 的代码分支，但当前 LiteLoc `predlist()` 只向它传 probability map，并没有使用该分支。正式 `network.post_process()` 同样直接读取候选像素的 xyz/photon。因此 weighted-offset 不属于需要对齐的实际标准路径。

LiteLoc 正式 `network.post_process()` 使用 `integrated p > 0.7`，训练 `EvalMetric.predlist()` 使用 `integrated p > 0.3`。Neptune v0.3 将原样保留这一差异，不尝试统一或重新解释。

### 1.2 作用范围

共享 LiteLoc NMS implementation 用于：

```text
training validation/evaluation:
  probability map -> localization -> Jaccard/RMSE/precision/recall

formal TIFF inference:
  probability map -> localization list -> overlap cut -> reconstruction
```

明确不用于：

```text
training loss
backpropagation
optimizer.step
learning-rate scheduler
physical model update posterior selection
```

训练内部 physical update 继续使用独立的：

```text
posterior_candidate_threshold
posterior_adjacent_threshold
posterior_accept_threshold
posterior_max_emitters
```

这部分应命名为 posterior emitter selection，不再复用或暴露 formal decode mode。Phase 1 完成前后，给定相同 network output 和 posterior 配置，physical update 选择出的 emitter 必须保持一致。

### 1.3 目标代码结构

底层共享 API 收敛为：

```python
liteloc_spatial_integration_probability(probability_map)
```

该函数固定 `candidate=0.3`、`adjacent=0.6`、`aggregation=sum`，不接收 mode 或 threshold。

上层提供两个按用途命名、不可配置的入口：

```python
decode_liteloc_eval_emitters(...)         # integrated p > 0.3
decode_liteloc_formal_infer_emitters(...) # integrated p > 0.7
```

这两个入口共享同一 NMS tensor implementation，只在 LiteLoc 本身不同的 final acceptance threshold 上区分。不能用通用 `decode_mode` 或可配置 `accept_threshold` 替代。

训练内部单独保留：

```python
select_posterior_emitters(
    y_out,
    candidate_threshold=...,
    adjacent_threshold=...,
    accept_threshold=...,
    max_emitters=...,
)
```

两个函数可以复用底层 tensor helper，但 public semantics 必须分离：前者是正式 localization decode，后者是训练内部 posterior selection。

### 1.4 实施任务

#### Task 1：建立共享 LiteLoc NMS Tensor Contract

新增固定参数的 `liteloc_spatial_integration_probability()`，集中实现 LiteLoc 的 candidate、max-pool、4-neighbor integration、adjacent split 和 sum。第一步先与旧 configurable decoder 并存，避免 posterior 尚未解耦时破坏 physical update。

涉及文件：

```text
src/neptune_v03/localization/legacy_decode.py
tests/test_liteloc_nms_parity.py
```

验收：

```text
同一 probability tensor 上与 LiteLoc reference 输出逐元素一致
p=0.3、0.6 边界行为一致
sum 后 probability 允许大于 1
eval final >0.3，formal infer final >0.7
两条路径的 xyz/photon 均读取候选像素对应输出
```

#### Task 2：Evaluation 统一使用唯一 Decoder

`training/localizer_eval.py` 删除可配置 NMS 参数，固定调用 `decode_liteloc_eval_emitters()`。评估 manifest 固定记录：

```text
decode_contract=liteloc_evalmetric_nms_v1
candidate_threshold=0.3
adjacent_threshold=0.6
accept_threshold=0.3
aggregation=sum
accept_rule=>
```

训练影响边界：

```text
eval_loss 不变
training loss/backprop 不变
scheduler 不变
best checkpoint 仍按 eval_loss 选择
Jaccard/RMSE 等 localization metrics 允许因对齐 LiteLoc EvalMetric >0.3 而变化
evaluation localization count 高于 formal infer 是预期行为，与 LiteLoc 一致
```

#### Task 3：Formal Infer 统一使用唯一 Decoder

标准 infer 删除：

```text
--decode-mode
--raw-th
--split-th
--prob-threshold 作为 NMS accept threshold 的用途
DECODE_MODE
RAW_TH
SPLIT_TH
PROB_THRESHOLD
```

formal infer 固定调用 `decode_liteloc_formal_infer_emitters()`，始终生成 NMS 后 `p>0.7` 的 raw localization list。后续 `prob0.7/0.8/0.9` 仍属于 localization-list filter/view，不是 decode mode。

formal infer manifest 固定记录：

```text
decode_contract=liteloc_formal_infer_nms_v1
candidate_threshold=0.3
adjacent_threshold=0.6
accept_threshold=0.7
aggregation=sum
accept_rule=>
```

涉及文件：

```text
scripts/infer/run_3371_full8000_infer_filter_recon.py
scripts/infer/standard_channel_infer_recon.sbatch
scripts/gui/submit_training_web.py
```

#### Task 4：Physical Update Posterior 解耦

将 `localization/posterior.py` 从正式 decoder 中解耦，迁移到独立 `select_posterior_emitters()`。保持当前 posterior 参数、排序、top-k 和输出完全一致。

验收：

```text
固定 network output 下，迁移前后 posterior xyzph/mask/count 完全一致
ROI library emitter count 不变
gamma update 输入不变
formal LiteLoc NMS 常量不会覆盖 posterior thresholds
```

#### Task 5：删除旧配置与命名

删除或替换：

```text
decode_legacy_smlm_emitters
decode_mode=neptune_legacy/custom/liteloc
legacy_eval_raw_th
legacy_eval_split_th
legacy_eval_probability_threshold
标准配置中的 formal raw_th/split_th/probability_threshold
```

历史运行的 manifest/checkpoint 不修改。新 evaluation 记录 `liteloc_evalmetric_nms_v1`，新 formal infer 记录 `liteloc_formal_infer_nms_v1`。

#### Task 6：Parity Smoke 与回归验证

使用同一 checkpoint、同一 evaluation batch、同一 NCP TIFF crop 验证：

```text
evaluation 和 formal infer 调用同一 NMS tensor implementation
evaluation final acceptance 固定 >0.3
formal infer final acceptance 固定 >0.7
formal infer localization count 与当前 decode_mode=liteloc 结果一致
NCP right 预期 raw rows 对齐现有 632,810 基线
prob0.7/0.8/0.9 filter count 对齐现有结果
physical update posterior parity test 通过
```

### 1.5 实施顺序

```text
Task 1 新增共享 LiteLoc NMS tensor contract，与旧内部 helper 暂时并存
-> Task 4 解耦 posterior，保护 physical update
-> Task 2 迁移 evaluation
-> Task 3 迁移 formal infer
-> Task 5 清理旧参数和命名
-> Task 6 parity smoke
```

先解耦 posterior，再删除旧 decoder 参数，避免 formal decode 收敛时误改 physical update。

### 1.6 Phase 1 完成判据

```text
1. Evaluation 和 formal infer 共享唯一 LiteLoc NMS tensor implementation。
2. Evaluation 固定 final >0.3；formal infer 固定 final >0.7，与 LiteLoc 实际代码一致。
3. Formal CLI/config 不再暴露 decode mode 和 NMS thresholds。
4. Training loss、eval loss、scheduler、checkpoint selection 不受影响。
5. Physical update posterior 输出迁移前后完全一致。
6. 新 manifest 分别记录 LiteLoc eval/formal infer contract。
7. NCP formal infer 与当前 LiteLoc parity baseline 点数一致。
8. 全测试集、Python compile、Bash syntax 和短 infer smoke 通过。
```

### 1.7 Phase 1 实施状态

代码实施状态：已完成。

已落地内容：

```text
1. 新增唯一共享函数 liteloc_spatial_integration_probability()。
2. Evaluation 固定调用 decode_liteloc_eval_emitters()，final integrated p > 0.3。
3. Formal TIFF infer 固定调用 decode_liteloc_formal_infer_emitters()，final integrated p > 0.7。
4. 删除 formal infer 的 decode_mode/raw_th/split_th/prob_threshold NMS 控制。
5. Web panel 删除 infer prob，只保留 NMS 后的 recon prob filter。
6. Physical update posterior 独立为 select_posterior_emitters()，保留原 threshold、top-k 和输出语义。
7. 新 formal infer runtime state 和 summary 记录 decode_contract=liteloc_formal_infer_nms_v1。
```

涉及的主要文件：

```text
src/neptune_v03/localization/legacy_decode.py
src/neptune_v03/localization/posterior.py
src/neptune_v03/training/localizer_eval.py
scripts/infer/run_3371_full8000_infer_filter_recon.py
scripts/infer/standard_channel_infer_recon.sbatch
scripts/gui/submit_training_web.py
run_standard_pipeline.sh
tests/test_liteloc_nms_parity.py
tests/test_posterior_decode_parity.py
```

验证结果：

```text
pytest: 20 passed（包含 Phase 2 空间、Phase 3 degrid 和默认 reconstruction source contract tests）
Python compileall: passed
Bash syntax: passed
formal infer CLI: 不再包含旧 NMS mode/threshold 参数
GPU formal reconstruction: SLURM 3426，RTX 3090，NCP right 8000 frames
output: output/3421_ncp_center400_right_phase1_liteloc_formal_nms_v1_prob_sweep/
raw_rows: 632,810
旧 LiteLoc parity baseline raw_rows: 632,810
端到端数量 parity: exact match
elapsed_sec: 656.02
```

任务 `3426` 已正常完成。新固定 formal NMS 与旧 `decode_mode=liteloc` 基线在相同 checkpoint、相同 NCP right crop、相同 ROI96/valid80/recenter 输入下均输出 `632,810` 个 raw localizations，Phase 1 端到端数量 parity 已通过。

## Phase 2：LiteLoc Sub-FOV / Over-Cut 空间对齐

### 2.1 来源说明

Phase 2 直接对应 LiteLoc formal TIFF analyzer 的空间分块逻辑，不涉及 LUNAR degrid，也不在本阶段修改时间窗口。

```text
.local/external/LiteLoc-main/network/multi_process_deeploc.py

split_fov():
  full FOV -> sub-FOV + over_cut context

filter_over_cut():
  推理后只保留原始 sub-FOV，边界规则为 >= lower 和 < upper
```

Neptune 标准几何：

```text
network context: 96x96
valid core:      80x80
over-cut:         8 px per side
```

这与 LiteLoc `sub_fov_size=80, over_cut=8` 是同一空间设计。此前 v0.3 已有内联实现，但没有独立 contract，也没有 synthetic golden test 证明任意 FOV 和边界位置都不重不漏。

### 2.2 Phase 2 到底解决什么问题

Neptune 的定位网络一次输入局部 ROI，但正式 TIFF 是大 FOV。因此 formal infer 必须：

```text
full FOV
-> 多个 96x96 context ROI
-> 每块只保留自己负责的 valid core
-> 将 local coordinates 转回 crop/full-TIFF coordinates
-> 合并成唯一 localization list
```

Phase 1 保证每个 ROI 内的 NMS 与 LiteLoc 一致。Phase 2 保证多个 ROI 的空间合并不会制造重复、漏点、接缝或坐标偏移。

Phase 2 不是：

```text
不是新的 NMS
不是 probability filter
不是 degrid
不是时间窗口改造
不是改变网络输出
不是重新训练
```

它是纯空间 stitching contract 和对应的自动化测试。

### 2.3 空间 Tile Contract

当前标准几何为：

```text
network context ROI: 96x96
valid core:          80x80
cut edge:            8 px on each side
```

网络仍然看完整的 `96x96` 上下文，但一个 tile 只允许输出属于其 valid core 的 localization。相邻 valid core 使用半开区间：

```text
[x0, x1) and [y0, y1)
```

因此落在边界上的点只能属于右侧或下侧 tile，不能同时被两个 tile 保留。

对于不能被 80 整除的 FOV，最后一个 valid core 可以小于 `80x80`。最后一个 `96x96` context patch 向 FOV 内侧移动并 clamp 到边界，但 valid core 仍与前面的 core 无缝衔接。例如宽度 600：

```text
[0,80), [80,160), ..., [480,560), [560,600)
```

最后一段只有 40 px，但不能漏掉，也不能与 `[480,560)` 重叠。

空间 parity 的目标是：完整 FOV 中每一个像素位置恰好属于一个 valid core。这样可以避免：

```text
tile overlap 导致同一个 emitter 重复写入
tile gap 导致边界 emitter 丢失
边界归属不一致导致网格或接缝
tile-local 坐标转 full-FOV 坐标时产生固定偏移
```

### 2.4 已落地实现

新增共享模块：

```text
src/neptune_v03/infer_recon/tiling.py

build_liteloc_subfov_tiles()
emitter_in_valid_core()
tile_local_to_field_coordinates()
```

正式 infer 不再保留一套私有内联 tile 算法，而是直接调用该共享 contract。Runtime state 和 summary 新增：

```text
spatial_stitching_contract=liteloc_subfov_overcut_v1
spatial_boundary_rule=lower_inclusive_upper_exclusive
tiling_mode=edgecover
context ROI=96
valid core=80
cut edge=8
```

新增测试：

```text
1. Pixel ownership test
   任意 FOV 的每个像素被 valid-core 覆盖次数必须等于 1。

2. Boundary ownership test
   emitter 位于 x1/y1 边界时，只进入相邻 tile，不允许重复。

3. Non-divisible FOV test
   600x1200、400x400 和 413x617 非整除尺寸均不重不漏。

4. Coordinate round-trip test
   tile-local -> crop-local -> full-TIFF 坐标转换可逆，误差为 0。

5. Fixed-context validation
   crop 小于 context 或 context-valid 差值不能对称切边时直接拒绝。
```

测试覆盖尺寸：

```text
400x400
600x1200
413x617（非整除 FOV）
```

### 2.5 是否会改变现有重构结果

Phase 2 不改变 NMS、时间窗口、network output、filter 或 renderer。当前空间算法的数学行为被迁入共享模块，因此相同输入下 NCP localization count 和坐标应保持不变。

新的 NCP right 8000-frame任务用于验证这一端到端不变性。验收目标为 Phase 1 基线：

```text
raw_rows=632,810
```

已提交正式验证任务：

```text
SLURM job: 3427
GPU: NVIDIA GeForce RTX 3090
output: output/3421_ncp_center400_right_phase2_liteloc_subfov_overcut_v1_prob090/
reconstruction filter: prob >= 0.9
```

任务已正常完成：

```text
raw_rows: 632,810
Phase 1 baseline: 632,810
end-to-end parity: exact match
elapsed_sec: 634.59
```

Phase 2 仅完成代码结构收敛、显式 contract 和测试固化，没有改变正式 localization 输出。

### 2.6 Phase 2 完成判据

```text
1. 所有空间位置 valid-core coverage count == 1。
2. tile 边界使用 [lower, upper)，边界 emitter ownership count == 1。
3. full-FOV 坐标转换无固定偏移。
4. 非整除 FOV 不重不漏。
5. runtime state 明确记录 LiteLoc-style spatial stitching contract。
6. NCP right Phase 2 formal infer raw_rows 与 Phase 1 的 632,810 基线一致。
```

时间窗口策略保持当前 Neptune 实现，不属于 Phase 2。只有未来出现 frame 缺失、重复或 block-boundary 异常证据时，再启动独立的 temporal boundary audit。

## Phase 3：LUNAR-style Degrid

### 3.1 上游依据

严格参考：

```text
.local/external/LUNAR/ailoc/common/post_process.py
  histogram_equalization()
  rescale_offset()

.local/external/LUNAR/ailoc/common/analyzer.py
  rescale_bins=20
  threshold=0.01
```

LUNAR 的目的不是改变 emitter 数量或提高原始定位精度，而是修正困难条件下网络 xy offset 向像素中心聚集产生的展示网格。

### 3.2 固定算法 Contract

对每个 channel 独立执行：

```text
1. 读取 x_offset/y_offset 和 x_sig/y_sig。
2. 计算 uncertainty score：
   sqrt(x_sig^2 + sqrt(var(x_sig)/var(y_sig))^2 * y_sig^2)
3. 按 uncertainty score 分成 20 个 quantile bins。
4. bin 下界小于 threshold * sqrt(pixel_x^2 + pixel_y^2) 时不处理。
5. 对满足条件的 bin 分别执行 LUNAR 200-bin CDF histogram equalization。
6. 使用 offset 变化量更新派生 x/y 坐标。
```

正式参数：

```text
rescale_bins=20
threshold=0.01
min_bin_count=32
```

`min_bin_count=32` 是 Neptune 的稳定性保护：LUNAR 正式数据通常每个 bin 有大量定位点，但极小数据集或局部 crop 可能出现空/稀疏 bin。此时跳过该 bin，避免空 histogram 或无意义的 CDF；大样本正式路径与 LUNAR 数学一致。

### 3.3 Raw/Derived 数据边界

原始定位保持不可变：

```text
predictions_merged.h5
```

另行生成：

```text
predictions_degrid.h5
degrid_summary.json
degrid_offset_histograms.png
```

派生 H5 保持 localization row count 不变，并保持以下字段逐元素不变：

```text
frame
z/z_nm
photon
prob
x_sig/y_sig
x_sig_px/y_sig_px
x_sig_nm/y_sig_nm
z_sig/z_sig_nm
photon_sig
tile_index
```

只更新派生文件中的：

```text
x_px/y_px
x_px_full/y_px_full
x_nm/y_nm
x_nm_full/y_nm_full
x_offset_px/y_offset_px
x_offset_nm/y_offset_nm
```

原始 H5 不覆盖。派生 H5 attributes 固定记录：

```text
derived_kind=lunar_offset_degrid
source_predictions=<raw H5>
degrid_contract=lunar_rescale_offset_v1
degrid_rescale_bins=20
degrid_threshold=0.01
```

### 3.4 已落地实现

新增：

```text
src/neptune_v03/infer_recon/degrid.py

lunar_histogram_equalization()
lunar_rescale_offsets()
degrid_predictions_h5()
```

标准 `standard_channel_infer_recon.sbatch` 默认启用：

```text
DEGRID_ENABLED=true
DEGRID_RESCALE_BINS=20
DEGRID_THRESHOLD=0.01
DEGRID_MIN_BIN_COUNT=32
```

正式 infer 完成 raw H5 后自动生成 degrid 派生文件。标准 probability filter 和 reconstruction 默认读取 `predictions_degrid.h5`；原始 `predictions_merged.h5` 继续作为不可变的定量定位记录保存。

默认选择规则：

```text
DEGRID_ENABLED=true:
  reconstruction_coordinate_source=degrid
  filter/recon input=predictions_degrid.h5

DEGRID_ENABLED=false:
  reconstruction_coordinate_source=raw
  filter/recon input=predictions_merged.h5
```

启用 degrid 但派生文件不存在时直接 hard fail，不允许静默回退 raw。这样 manifest、配置和实际 reconstruction 输入不会发生漂移。

标准入口 smoke：

```text
SLURM job: 3428
frames: 256
GPU: NVIDIA GeForce RTX 3090
raw rows: 55,128
degrid rows: 55,128
processed bins: 20/20
degrid H5/summary/histogram PNG: generated
raw prob0.9 reconstruction: generated
elapsed_sec: 52.4
```

默认 degrid reconstruction source smoke：

```text
SLURM job: 3429
frames: 128
GPU: NVIDIA GeForce RTX 3090
raw rows: 28,167
degrid rows: 28,167
reconstruction_coordinate_source: degrid
filter input: right/infer/predictions_degrid.h5
prob0.9 reconstruction: generated
elapsed_sec: 43.97
```

该任务证明默认行为已经从“仅生成 degrid 文件”切换为“实际使用 degrid 坐标完成 filter/recon”。

### 3.5 NCP 实测结果

输入：

```text
output/3421_ncp_center400_right_phase1_liteloc_formal_nms_v1_prob_sweep/
  right/infer/predictions_merged.h5
```

输出：

```text
output/3421_ncp_center400_right_phase3_lunar_degrid_v1/
```

数据不变性：

```text
raw rows:    632,810
degrid rows: 632,810
frame/z/photon/prob/uncertainty/tile_index: 全部逐元素一致
```

位移规模：

```text
radial shift mean:  6.89 nm
radial shift p50:   4.73 nm
radial shift p95:  17.17 nm
radial shift max:  23.58 nm
```

offset histogram coefficient of variation，越低表示越接近均匀：

```text
x: 0.2426 -> 0.0573
y: 0.3483 -> 0.0787
```

`prob>=0.9` raw/degrid 均保留 `384,014` 个 localizations。两组重构使用完全相同的 count weight、z range、Gaussian renderer、sigma、gamma 和 scale percentile。

对比输出：

```text
raw_prob090/recon/
degrid_prob090/recon/
infer/degrid_offset_histograms.png
```

### 3.6 验收结论

```text
1. LUNAR histogram equalization 数学 parity test 通过。
2. localization count 不变。
3. 非 xy 字段逐元素不变。
4. source raw H5 不被覆盖。
5. 稀疏和低 uncertainty bin 自动跳过。
6. shift mean/p50/p95/max 和 histogram before/after 已输出。
7. offset uniformity CV 显著下降。
8. raw/degrid prob0.9 公平重构已生成。
```

左右通道必须分别 degrid，并在 channel registration 和 dual-channel matching 之前完成。

## Phase 4：Degrid 定量验收

### 4.1 精简范围

不再输出 `prob0.7/0.8/0.9 x raw/degrid` 六组图片。该设计会混合 probability selection 和坐标修正两个变量，且不能直接证明 degrid 是否有效。

Phase 4 固定使用已经存在的同一批 NCP `prob>=0.9` localizations，只比较：

```text
raw coordinates
degrid coordinates
```

两者必须满足：

```text
localization count 完全一致
frame/prob/z/photon/uncertainty 等非 XY 字段逐元素一致
只允许 XY coordinate/offset 字段变化
```

### 4.2 LiteLoc-style Grid Artifact Index

实现参考：

```text
.local/external/LiteLoc-main/utils/help_utils.py
  compute_pixel_grid_idx_fs()
```

指标定义：

```text
1. localization 按 frame 排序。
2. 按 10 个连续数据块交替拆成两组。
3. 在 0.1 camera pixel 的超分辨率网格上生成两张 count histogram。
4. 计算两张图的 split-half FRC。
5. 只读取 1 cycle/camera-pixel 附近三个频率点的最大 FRC。
6. 该值作为 camera-pixel grid artifact index，越低越好。
```

Neptune 使用 camera-pixel normalized 坐标计算，因此不需要把 `101.11 nm x 98.83 nm` 的各向异性像素压成一个虚假的标量 pixel size。时间分块和频率选择与 LiteLoc 保持一致，原始双重 Python 循环仅替换为等价的 NumPy 向量化 radial sum。

实现：

```text
src/neptune_v03/infer_recon/grid_artifact.py
tests/test_liteloc_grid_artifact.py
```

### 4.3 NCP 正式结果

输入：

```text
output/3421_ncp_center400_right_phase3_lunar_degrid_v1/
  raw_prob090/filtered_predictions.h5
  degrid_prob090/filtered_predictions.h5
```

输出：

```text
output/3421_ncp_center400_right_phase4_degrid_acceptance_audit/
  phase4_degrid_acceptance.json
  phase4_degrid_acceptance.png
```

公平性检查：

```text
raw rows:                    384,014
degrid rows:                 384,014
count parity:                passed
non-XY field parity:         passed
```

定量结果：

```text
LiteLoc-style grid index:    0.8596 -> 0.8152
x offset histogram CV:       0.2404 -> 0.0975
y offset histogram CV:       0.3072 -> 0.1089
```

验收规则及结果：

```text
degrid_grid_artifact_index < raw_grid_artifact_index
0.8152 < 0.8596: passed
```

结论：LUNAR degrid 在不改变 emitter 数量和任何非 XY 数据的前提下，显著改善 offset 分布，并降低 camera-pixel grid artifact index。该指标也可能包含真实样本在 `1 cycle/camera-pixel` 附近的周期结构，尤其不能把 NCP 的剩余绝对值全部解释为网格伪影。可靠证据是同一批 localization、同一生物结构下 raw 到 degrid 的相对下降，以及 x/y offset CV 同时下降。Degrid 是有效修正，不是对上游定位误差、样本结构或渲染问题的完全替代。

## Prediction 到 Reconstruction 的剩余差距

### 1. Probability selection 已对齐

LiteLoc formal infer 和 LUNAR 常用正式 decode 均在 integrated probability `>0.7` 后输出 localization。Neptune 标准 filter/recon、SLURM、顶层 pipeline 和 Web panel 已统一为：

```text
prob >= 0.7
```

`prob>=0.9` 仅保留为显式派生展示选项，不再是 v0.3 默认值。Phase 4 使用 `prob>=0.9` 是为了复用已有相同点集完成 raw/degrid 单变量审计，不代表当前正式默认。

当前标准保留完整的 `>0.7` formal localization，并从 degrid H5 进行重构。

### 2. 正式 renderer 已取消 uncertainty radius

当前正式标准入口统一为：

```text
render_pixel_nm=20
spot_radius_nm=28
radius_mode=fixed
Gaussian sigma=14 nm
render_weight=count
```

该配置依据 SMAP 默认 `20 nm` reconstruction pixel、`0.7 pixel` minimum Gaussian sigma。所有正式 localization 使用统一 `14 nm` sigma。`x_sig/y_sig` 继续保存在 H5 并用于 LUNAR degrid；`xy_uncertainty_mean` 仅保留为显式诊断模式。

单通道、双色、标准 SLURM、兼容 wrapper、顶层 pipeline 和 Web panel 不再隐式恢复旧的 uncertainty radius。

### 3. Reconstruction 是 Neptune 自定义展示层，不是上游严格 parity

Neptune 当前使用：

```text
20 nm render pixel
integrated Gaussian
fixed Gaussian sigma 14 nm
count weight
linear float32 density and RGB quantitative TIFF
Neptune robust density Imax: q=0.997
optional SMAP GUI parity: imax_min=-3.5, q=0.9996837722
optional fixed absolute Imax and normalization FOV
gamma=1
z=-600~600 nm color mapping; density controls value, RGB controls hue
```

定位密度与展示亮度现已分离。线性 TIFF 不做 quantile、clip 或 gamma，可用于定量分析；PNG/uint8 TIFF 只是派生展示。跨样本比较时应传入相同的 fixed Imax，单样本默认使用 SMAP quantile Imax。Gaussian 宽度和颜色映射仍属于展示 contract，因此比较算法质量时必须固定这些参数。

### 4. Degrid 依赖 uncertainty 分箱，但不会修复错误定位

Neptune 的 degrid 数学路径已与 LUNAR 对齐，并增加稀疏 bin 保护。它只重分布亚像素 offset，NCP p95 位移约 `17.17 nm`，不会增加点、修复错误 z，也不会解决模型本身的 false positive/false negative。

### 5. 时间边界存在小差异，但当前不是主问题

LiteLoc formal analyzer 对时间 block 使用自己的边界上下文处理；Neptune 使用真实重叠上下文并排除全局首尾帧。现有证据下最多影响两个全局边界帧，不足以解释主体重构质量问题。除非后续发现 frame 缺失、重复或 block-boundary 条纹，否则不修改。

### 6. RCC drift correction 已加入默认路径

NCP 8000-frame 正确 degrid H5 的 RCC 诊断测得 x 漂移约 `26 nm`、y 总跨度约 `184 nm`，校正后核孔结构收紧，因此 Neptune 标准流程默认启用 RCC。默认使用 500-frame blocks、50 nm RCC pixels、冗余 block-pair phase correlation 和 robust least-squares；原始与 degrid H5 不覆盖，新增 `predictions_degrid_rcc_corrected.h5`。

正式顺序：

```text
raw localization
-> LUNAR degrid
-> RCC drift correction
-> probability/quality filter
-> reconstruction
```

### 7. 双通道仍需独立审计 registration 和 matching

单通道 NCP Phase 4 不涉及 channel registration。对于双色结果，正确顺序仍应是：

```text
left/right 各自 decode + valid-core + degrid + RCC drift correction
-> channel registration
-> dual-channel matching
-> right-priority 或约定坐标输出
-> reconstruction
```

registration transform、匹配半径、unmatched emitter 保留规则或坐标优先级不一致，都会产生错位和网格；这些问题不能由单通道 degrid 修复，应作为单独的双通道 parity audit。

## 最终标准路径

```text
network output
-> LiteLoc production NMS
-> valid-core overlap cut
-> immutable raw localization
-> LUNAR degrid derived localization
-> RCC drift-corrected derived localization
-> integrated probability >= 0.7
-> SMAP-style fixed sigma 14 nm count-weighted reconstruction
-> immutable linear float32 density/RGB TIFF
-> SMAP quantile or explicit fixed-Imax display image
-> fixed physical z color range
-> raw/degrid 两个可追溯版本并存
```

当前剩余的后处理重点是：

```text
1. 仅在双通道任务中单独验证 registration/matching contract。
2. 跨样本或跨条件比较时，显式复用同一个 fixed Imax。
3. 对新样本记录 RCC pair correlation、轨迹范围和校正前后 QC。
```
