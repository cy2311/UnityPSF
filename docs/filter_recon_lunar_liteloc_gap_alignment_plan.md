# Neptune v0.3 Filter/Reconstruction 与 LiteLoc/LUNAR 后处理对齐规划

## 结论先行

当前 v0.3 的后处理问题不应该继续沿着 `locprec / LLrel / PSF-size` 强筛方向推进。对 NCP 这类弱小环形结构，强筛会把真实但低置信度或高不确定性的点一起删掉，导致 ring 更不明显。

LiteLoc 和 LUNAR 的主线不是“把坏点用很多阈值删干净”，而是：

```text
LiteLoc:
probability map NMS
-> p threshold
-> sub-FOV overlap cut
-> localization list

LUNAR:
LiteLoc/DeepLoc 类似推理合并
-> overlap cut
-> optional degrid / offset rescale
-> localization list / visualization
```

因此 v0.3 后续应按下面方向补齐：

```text
1. 回到 prob-only 轻筛，不默认 locprec/LLrel/PSF-size hard filter
2. 先验证/修正 LiteLoc 的 NMS / spatial integration 语义
3. 在 NMS parity 成立之后，再比较 prob 0.7 / 0.8 / 0.9
4. 实现 LUNAR-style offset degrid / rescale
5. 保持 raw localization 与 visualization localization 分离
6. 再补 drift / frame-quality diagnostics
```

## 当前背景

NCP center400 right 的 3421 训练指标并不差：

```text
JACCARD   = 0.8700787402
RMSE_LAT  = 26.59174873 nm
RMSE_AX   = 40.32974783 nm
```

但 full 8000-frame reconstruction 中核孔 ring 不够清楚。之前尝试过：

```text
locprec <= 20 / 30 / 40 nm
x_sig / y_sig gate
LLrel gate
PSF-size gate
fixed-radius rendering
frame split diagnostics
```

结果并不稳定，很多图甚至不如最初的 `prob090_no_locprec` baseline。这说明：

```text
问题不是 filter 不够严，而是后处理路线没有和 LiteLoc/LUNAR 对齐。
```

当前应将最初可接受的 baseline 固定为保底对照：

```text
predictions_merged.h5
-> prob >= 0.9
-> no locprec filter
-> no LLrel filter
-> no PSF-size filter
-> xy_uncertainty_mean render
-> median10 uncertainty cap
```

任何新后处理结果都必须和这个 baseline 并排比较，不能覆盖。

## LiteLoc 实际做了什么

参考本地代码：

```text
.local/external/LiteLoc-main/network/liteloc.py
.local/external/LiteLoc-main/network/multi_process.py
.local/external/LiteLoc-main/utils/help_utils.py
```

### 1. Probability Map NMS

LiteLoc 的核心后处理在网络输出后执行，不是在 localization list 上做复杂质量筛。

核心逻辑：

```python
p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))
pool = max_pool2d(p_clip, 3, 1, padding=1)
max_mask1 = torch.eq(p[:, None], pool)

filt = [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]
conv = conv2d(p[:, None], filt, padding=1)
p_ps1 = max_mask1 * conv

p_copy = p * (1 - max_mask1[:, 0])
max_mask2 = p_copy > 0.6
p_ps2 = max_mask2 * conv

p = p_ps1 + p_ps2
p_index = torch.where(p > 0.7)
```

语义：

| 参数 | LiteLoc 作用 |
|---|---|
| `0.3` | candidate threshold，只保留候选概率区域 |
| `3x3 max_pool` | 找局部极大值 |
| `center + 4-neighbor conv` | 把邻域概率积分到局部极大点 |
| `0.6` | adjacent split threshold，允许相邻 emitter |
| `0.7` | final accepted probability threshold |

重点：

```text
LiteLoc 主输出不是 prob >= 0.9，而是 NMS 后 p > 0.7。
```

### 2. Sub-FOV Overlap Cut

LiteLoc 大图推理使用：

```yaml
sub_fov_size: 256
over_cut: 8
```

流程：

```text
大图切成 sub-FOV
每个 sub-FOV 外扩 over_cut
网络在外扩区域也能看到完整 PSF
合并结果时只保留 original sub-FOV 内的点
overlap 区域点被去掉，避免重复和边界伪影
```

对应 v0.3：

```text
ROI input = 96
valid core = 80
cut edge = 8
```

这部分 v0.3 思路基本对齐，但需要保证 metadata 和 diagnostics 清楚记录：

```text
patch origin
valid core
cut edge
tile index
full-FOV coordinate
```

### 3. LiteLoc 输出 Localization List

LiteLoc 输出 CSV 字段类似：

```text
frame, xnm, ynm, znm, photon, prob,
x_sig, y_sig, z_sig, photon_sig,
xoffset, yoffset
```

它不会在默认路径里继续做：

```text
locprec <= 30
LLrel >= -0.8
PSFxy <= 180
fit_status required
```

这些不是 LiteLoc 的主线。

## LUNAR 实际多做了什么

参考本地代码：

```text
.local/external/LUNAR/ailoc/common/analyzer.py
.local/external/LUNAR/ailoc/common/post_process.py
.local/external/LUNAR/usages/pyscripts/lunar_usage.py
```

### 1. Divide-and-Conquer + Filter Over Cut

LUNAR 同样做：

```text
split_fov
sub_fov_size = 256
over_cut = 8
data_analyze
filter_over_cut
write localization list
```

这和 LiteLoc 的大图推理思想一致。

### 2. Degrid / Offset Rescale

LUNAR 明确提供：

```python
preds_array_re = ailoc.common.rescale_offset(
    preds_array,
    pixel_size=lunar_analyzer.pixel_size_xy,
    rescale_bins=20,
    threshold=0.01,
)
```

LUNAR 注释说明这一步用于：

```text
histogram equalization for grid artifacts removal
adjust xy offsets to reduce grid artifacts
replace xnm and ynm with x_rescale and y_rescale
```

### 3. Offset Degrid 的含义

每个 localization 点可以拆成：

```text
整数像素位置 + 像素内部 offset
```

例如：

```text
x_px = 100.72
pixel center = floor(100.72) + 0.5 = 100.5
x_offset = 100.72 - 100.5 = +0.22 px
```

理想情况下，很多 emitter 的 `x_offset / y_offset` 应该在：

```text
[-0.5, 0.5]
```

里比较均匀。

如果网络有 pixel-center bias，则会出现：

```text
offset 大量集中在 0 附近
或集中在少数几个固定 offset
```

重构图就可能有：

```text
pixel-grid artifact
网格感
结构被像素网格牵引
ring / filament 不自然
```

LUNAR 的 rescale 做：

```text
1. 计算每个点的 x_offset / y_offset
2. 按 x_sig / y_sig 不确定性分 bin
3. 对每个 uncertainty bin 内的 offset 做 histogram equalization
4. 得到新的 offset
5. 用新 offset 轻微修正 x/y 坐标
```

重点：

```text
degrid 不删点，只修正点的位置。
```

这和 strict filter 完全不同。

## v0.3 当前缺什么

### Gap 0: LiteLoc NMS / Spatial Integration Parity 尚未严格验证

v0.3 当前有 NMS-like decode，但还不能直接宣称已经和 LiteLoc 完全一致。当前实现位于：

```text
src/neptune_v03/localization/legacy_decode.py
```

核心函数：

```text
spatial_integration_probability(...)
```

它和 LiteLoc 思路接近：

```text
candidate threshold
3x3 max pool local maximum
center + 4-neighbor probability integration
adjacent split
final accept threshold
```

但目前至少有几处需要先核对：

```text
decode_legacy_smlm_emitters:
  raw_th = 0.5
  split_th = 0.6
  accept_th = 0.7
  aggregation = norm_sum
```

LiteLoc 代码中更接近：

```text
raw_th = 0.3
split_th = 0.6
accept_th = 0.7
p = p_ps1 + p_ps2
```

因此 Phase 1 不能直接叫 “LiteLoc-style prob-only sweep”。正确顺序应该是：

```text
1. 先做 LiteLoc NMS / spatial integration parity audit
2. 如果差异影响输出，先补 decode_mode=liteloc 或对齐参数
3. 再做 prob-only threshold sweep
```

否则会出现一个问题：

```text
我们以为只是在调 prob threshold，
但实际可能 decode 阶段已经和 LiteLoc 不一致。
```

### Gap 1: NMS parity 之后仍需验证 prob >= 0.9 是否偏强

v0.3 当前标准重构还额外做：

```text
filter_recon:
  prob >= 0.9
```

当前 decode 默认参数为：

```text
decode_legacy_smlm_emitters:
  raw_th = 0.5
  split_th = 0.6
  accept_th = 0.7
```

LiteLoc 更接近：

```text
NMS 后 p > 0.7 直接输出
```

因此 v0.3 的最终 recon 当前比 LiteLoc 多一层：

```text
prob >= 0.9
```

在 NCP right 当前数据中：

```text
prob >= 0.7    约 515k 点
prob >= 0.8    约 441k 点
prob >= 0.9    约 355k 点
```

这会直接删除大量弱点。对 NCP ring，这可能过强。

### Gap 2: 缺少 LUNAR-style degrid

v0.3 当前没有标准：

```text
x_offset / y_offset histogram diagnostic
offset bias summary
rescale_offset
predictions_degrid.h5
degrid vs raw recon montage
```

这是当前最值得优先补的后处理模块。

### Gap 3: Raw Localization 与 Visualization Localization 没有明确分离

LUNAR 里 degrid 主要用于 visualization / artifact reduction。它不应该覆盖原始定位结果。

v0.3 需要明确两套输出：

```text
predictions_merged.h5          # raw localization
predictions_degrid.h5          # visualization-corrected localization
```

评估、物理诊断、可追溯保存应优先使用 raw。
重构展示可以比较 raw 和 degrid。

### Gap 4: 后处理实验混入了 hard quality filter

之前尝试的：

```text
locprec <= 20/30/40
x_sig <= 0.3
y_sig <= 0.3
LLrel >= -1.0/-0.8/-0.6
psf_xy <= 170/180/190
fit_status required
```

对 NCP 不适合作为默认。

这些指标以后只保留为：

```text
diagnostics
optional soft weight
optional expert-only hard filter
```

不能作为 v0.3 NCP 默认 route。

### Gap 5: 渲染对比不公平

之前一些实验改了：

```text
fixed radius 15/20 nm
```

而 baseline 是：

```text
xy_uncertainty_mean + median10 cap
```

因此不能直接比较视觉结果。后续所有 parity sweep 必须固定 render，仅改变一个变量。

### Gap 6: 缺少 frame/drift diagnostics

这不是 LiteLoc/LUNAR 最核心的差距，但对 SMLM reconstruction 很重要。
NCP full 8000 frames 如果存在漂移或后期 bleaching，ring 会被平均糊掉。

需要作为后续 Phase 5 补充：

```text
frame block split recon
block-wise detection count
block-wise median prob/photon/sigma
optional drift correction
```

## 后续实施路线

### Phase 0: 固定 Baseline，不再覆盖

保留当前 baseline：

```text
output/3421_ncp_center400_right_infer_recon_full8000_roi96_valid80_cut8_prob090_no_locprec
```

Baseline 定义：

```text
predictions_merged.h5
-> prob >= 0.9
-> no locprec / LLrel / PSF-size hard filter
-> xy_uncertainty_mean render
-> median10 cap
```

所有新结果输出到新目录，例如：

```text
output/3421_ncp_center400_right_liteloc_lunar_postprocess_parity/
```

Phase 0 已落实为运行时保护：

```text
scripts/infer/run_3371_full8000_infer_filter_recon.py
```

默认拒绝写入任何已经存在且非空的 `--output-dir`。标准 SLURM 入口：

```text
scripts/infer/standard_channel_infer_recon.sbatch
```

同样不再提前创建 `OUTPUT_DIR`，并且默认不允许覆盖。只有显式设置：

```bash
ALLOW_OVERWRITE_OUTPUT=true
```

时才会传入 `--overwrite-output`。因此当前 baseline 目录不会被后续 parity/degrid 实验误覆盖；所有新实验必须使用新 `RUN_NAME` 或新 `OUTPUT_DIR`。

### Phase 1A: LiteLoc NMS / Spatial Integration Parity Audit

目标：先证明 v0.3 decode 和 LiteLoc 的 NMS/spatial integration 等价，或者明确差异并修正。
在这一步完成之前，不应该把后续 sweep 称为 “LiteLoc-style”。

输入：

```text
网络输出 probability map
或已保存的中间 probability tensor / 小样本推理输出
```

需要对齐的点：

```text
raw_th:
  v0.3 当前常用 0.5
  LiteLoc 使用 0.3

split_th:
  v0.3 当前 0.6
  LiteLoc 使用 0.6

accept_th:
  v0.3 当前 0.7
  LiteLoc 使用 0.7

aggregation:
  v0.3 当前有 norm_sum / clamp 语义
  LiteLoc 更接近 p_ps1 + p_ps2

boundary behavior:
  padding
  equality tie handling
  adjacent emitter split
```

建议新增或补齐：

```text
scripts/infer/compare_liteloc_nms_parity.py
```

输出：

```text
nms_parity_summary.json
nms_diff_examples.png
```

Acceptance criteria：

```text
1. 在同一 probability map 上，v0.3 decode 和 LiteLoc reference 的候选点数量、位置、prob 差异被记录
2. 如果 raw_th=0.5 是主要差异，需要提供 raw_th=0.3 / 0.5 的并排结果
3. 如果 aggregation=norm_sum 改变了输出，需要新增 decode_mode=liteloc 或参数开关
4. 只有这一步完成后，才进入 Phase 1B 的 prob-only sweep
```

### Phase 1B: NMS Parity 后的 Prob-Only Threshold Sweep

目标：判断 v0.3 的 `prob >= 0.9` 是否过强。

前提：

```text
Phase 1A 已经确认 decode parity，
或者已经将 v0.3 decode 显式切到 LiteLoc-aligned mode。
```

输入：

```text
predictions_merged.h5
```

输出：

```text
prob070_raw_uncertainty/
prob080_raw_uncertainty/
prob090_raw_uncertainty/
prob_sweep_montage.png
prob_sweep_summary.json
```

每组保持：

```text
no locprec
no LLrel
no PSF-size
no fit_status
same uncertainty render
same gamma
same scale percentile
```

只变：

```text
prob_min = 0.7 / 0.8 / 0.9
```

Acceptance criteria：

```text
1. 每组点数记录清楚
2. render 参数完全一致
3. 如果 prob0.7/0.8 ring 更好，则默认 NCP filter 应从 0.9 降到 0.8 或 0.7
4. 如果 prob0.7/0.8 噪声过多，则保留 prob0.9，但继续 degrid
```

### Phase 2: LUNAR-Style Offset Degrid / Rescale

新增脚本：

```text
scripts/infer/apply_lunar_degrid_rescale.py
```

功能：

```text
输入 predictions_merged.h5
输出 predictions_degrid.h5
输出 offset_hist_before_after.png
输出 degrid_summary.json
```

核心算法：

```python
x_offset = x_px - floor(x_px) - 0.5
y_offset = y_px - floor(y_px) - 0.5

total_sig = sqrt(x_sig ** 2 + y_sig ** 2)
bins = quantile_bins(total_sig, rescale_bins=20)

for each bin:
    if uncertainty below threshold:
        keep original offset
    else:
        x_offset_rescale = histogram_equalization(x_offset)
        y_offset_rescale = histogram_equalization(y_offset)

x_px_rescale = floor(x_px) + 0.5 + x_offset_rescale
y_px_rescale = floor(y_px) + 0.5 + y_offset_rescale
```

推荐参数对齐 LUNAR：

```text
rescale_bins = 20
threshold = 0.01
```

但 v0.3 应记录：

```text
threshold unit
pixel_size
offset histogram before/after
changed localization fraction
mean / p95 coordinate shift nm
```

重要约束：

```text
1. 不删除点
2. 不覆盖 predictions_merged.h5
3. 只输出 predictions_degrid.h5
4. manifest 明确标记 visualization-corrected
```

### Phase 3: Raw vs Degrid Fair Render Sweep

新增脚本：

```text
scripts/infer/render_liteloc_lunar_parity_sweep.py
```

输出 6 组公平对比：

```text
prob070_raw_uncertainty.png
prob080_raw_uncertainty.png
prob090_raw_uncertainty.png

prob070_degrid_uncertainty.png
prob080_degrid_uncertainty.png
prob090_degrid_uncertainty.png

montage.png
summary.json
```

固定参数：

```text
renderer = integrated_gaussian
radius_mode = xy_uncertainty_mean
uncertainty_cap_mode = median10
gamma = 1.0
scale_percentile = 99.7
render_pixel_nm = 10
```

只改变：

```text
prob_min
raw vs degrid
```

Acceptance criteria：

```text
1. degrid 图不能比 raw 明显更差
2. offset histogram 应明显更均匀
3. 如果 ring 更连续、更少 grid artifact，则 degrid 纳入标准 visualization route
4. 如果无改善，则 degrid 保留为可选诊断，不作为默认
```

### Phase 4: 标准后处理配置落盘

若 Phase 1A、Phase 1B、Phase 2、Phase 3 验证有效，标准配置建议：

```yaml
postprocess:
  decode:
    raw_th: 0.3 or 0.5
    split_th: 0.6
    accept_th: 0.7

  filter:
    mode: prob_only
    prob_min_default: 0.8
    prob_min_presets: [0.7, 0.8, 0.9]
    locprec_filter_enabled: false
    llrel_filter_enabled: false
    psf_size_filter_enabled: false
    fit_status_filter_enabled: false

  degrid:
    enabled_for_visualization: true
    method: lunar_rescale_offset
    rescale_bins: 20
    threshold: 0.01
    preserve_raw_predictions: true

  render:
    renderer: integrated_gaussian
    radius_mode: xy_uncertainty_mean
    uncertainty_cap_mode: median10
    render_pixel_nm: 10
    gamma: 1.0
    scale_percentile: 99.7

  diagnostics:
    output_prob_sweep: true
    output_raw_vs_degrid: true
    output_offset_histogram: true
```

对于 NCP，建议默认优先试：

```text
prob_min = 0.8
degrid = true
uncertainty render
```

但保留：

```text
prob_min = 0.9 raw baseline
```

作为可追溯对照。

### Phase 5: Drift / Frame Quality Diagnostics

在 LiteLoc/LUNAR parity 路线之后，再补：

```text
frame split recon:
  0-2000
  2000-5000
  5000-8000

block summary:
  detection count
  median prob
  median photon
  median x_sig/y_sig
  localization density
```

后续可以实现：

```text
block-wise drift estimation
drift-corrected predictions
frame quality weighting
```

但这不应和 degrid/prob sweep 混在一个实验里。

## 不再作为默认路线的内容

以下内容不应再作为 NCP 默认后处理：

```text
locprec <= 20/30/40 nm
x_sig <= 0.25/0.30 px
y_sig <= 0.25/0.30 px
LLrel >= -1.0/-0.8/-0.6
psf_xy <= 170/180/190 nm
require_fit_status
photon hard threshold
fixed-radius-only visualization as standard
```

这些可以保留为：

```text
expert diagnostics
debug-only filter
alternative visualization
```

但默认路径必须先对齐：

```text
LiteLoc NMS parity
prob-only threshold sweep
LUNAR degrid/rescale
same uncertainty render
```

## 实现优先级

### P0: Baseline 保护与 no-overwrite

文件：

```text
scripts/infer/run_3371_full8000_infer_filter_recon.py
scripts/infer/standard_channel_infer_recon.sbatch
```

原因：

```text
先保护现有 baseline，避免后续实验覆盖已经可用的 recon。
```

状态：

```text
已实现：默认不覆盖非空 output-dir，除非显式 ALLOW_OVERWRITE_OUTPUT=true。
```

### P1: LiteLoc NMS / spatial integration parity audit

文件：

```text
scripts/infer/compare_liteloc_nms_parity.py
```

原因：

```text
v0.3 目前是 NMS-like，不是已经证明的 LiteLoc-equivalent。
必须先确认 raw_th、split_th、accept_th、aggregation、边界行为是否一致。
```

### P2: 实现公平 prob-only parity sweep

文件：

```text
scripts/infer/render_liteloc_lunar_parity_sweep.py
```

原因：

```text
之前对比混入 fixed-radius 和 strict filter，不公平。
需要固定 render，只比较 prob 和 degrid。
```

### P3: 实现 LUNAR-style degrid 脚本

文件：

```text
scripts/infer/apply_lunar_degrid_rescale.py
```

原因：

```text
这是当前 v0.3 明确缺失、且不会删点的后处理模块。
```

### P4: 将 degrid 接入标准 infer/recon，但默认保留 raw 输出

标准输出结构：

```text
right/
  infer/
    predictions_merged.h5
    predictions_degrid.h5
    degrid_summary.json
    offset_hist_before_after.png

  filter_recon_prob080_no_locprec/
  filter_recon_prob080_no_locprec_degrid/
  postprocess_parity_montage.png
```

### P5: 再做 drift/frame quality

原因：

```text
drift 也重要，但如果先不解决 prob/degrid 公平对比，会继续误判。
```

## 成功判据

这个规划成功，不是看某一个阈值是否“更严格”，而是看：

```text
1. NCP ring 是否比 baseline 更清楚
2. localization 点数没有被大幅硬删
3. offset histogram 是否从偏置变得更均匀
4. montage 中 raw vs degrid 差异可解释
5. 所有结果都有 manifest 和 summary，可追溯
```

如果 degrid/prob sweep 后仍无改善，则结论应转向：

```text
1. right channel infer 输出本身不足
2. physical model / initial zmap / update quality 问题
3. sample drift / bleaching / frame quality 问题
4. NCP crop 或 input preprocessing 问题
```

而不是继续加强 hard filter。

## 最终路线一句话

v0.3 后处理应从“强筛坏点”改成：

```text
LiteLoc-parity NMS/spatial integration
+ prob-only localization
+ LUNAR-style offset degrid/rescale
+ raw/degrid 双输出
+ 统一 uncertainty render 公平对比
```

这样才是对齐 LiteLoc/LUNAR 的正确后处理路线。
