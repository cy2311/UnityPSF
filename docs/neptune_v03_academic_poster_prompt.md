# Neptune v0.3 学术海报生成 Prompt

本文档用于把 Neptune v0.3 工作整理成一张学术海报。目标不是做软件宣传图，而是做一张证据驱动的显微成像 / SMLM 物理建模与重建方法海报。

## 1. 海报规格

```text
尺寸: 1.2 m x 1.6 m
方向: 竖版 portrait
建议画布: 1200 mm x 1600 mm
建议导出: PDF / SVG / TIFF
建议分辨率: 300 DPI
视觉风格: Nature Methods / Nature Biomedical Engineering 风格
背景: 白色或极浅灰白
字体: Arial / Helvetica / Source Sans / 思源黑体
主色: charcoal 黑灰 + warm gray / cool gray
强调色: restrained teal-blue + muted orange, 只用于关键路径和双色结果
```

## 2. 海报主标题建议

```text
Neptune v0.3: Fast Physics-Consistent Field-Dependent SMLM Localization and Ratiometric Reconstruction from Raw TIFF Movies
```

中文副标题可以写：

```text
面向真实 raw TIFF 数据的快速场依赖 PSF 仿真、物理模型更新与双色比率重建流程
```

## 3. 一句话核心主张

```text
Neptune v0.3 在尽量保持 3052 / LUNAR-style 物理一致性的前提下，通过 physical-versioned LUT、cached-window simulation、ROI-bank gamma update 和 recentered infer/recon，把 field-dependent SMLM 训练与 8000-frame raw TIFF 重建整合成一个更快、更稳定、可诊断的标准流程。
```

海报上可以压缩为：

```text
Fast, field-dependent SMLM localization with physical-model feedback and union raw-ratio multicolor reconstruction.
```

## 4. 证据对象 Inventory

海报必须优先画真实科学对象，而不是空泛软件框图。

必须出现的证据对象：

```text
1. raw TIFF movie:
   - left / right 双通道原始显微图像
   - 8000 frames
   - raw frame thumbnails 或 raw ROI tiles

2. field-dependent physical model:
   - zmap / aberration map
   - spatially varying PSF spots
   - astigmatic PSF shape across z

3. online simulation:
   - ROI 96 x 96 input
   - valid core 80 x 80
   - stride / cut-edge inference geometry
   - global-field LUT / FP16 LUT / prewarm

4. physical update:
   - fixed ROI bank from raw TIFF
   - posterior sampling
   - gamma update
   - left / right domain independent update
   - raw vs corrected camera vs recon diagnostic strips

5. training and metrics:
   - Jaccard
   - RMSE_lat
   - RMSE_ax
   - only these three main localization metrics

6. infer / recon:
   - recentered full-frame inference
   - prob >= 0.9
   - no locprec gate by default
   - ROI 96, valid core 80, cut edge 8

7. multicolor reconstruction:
   - union left/right emitter sets
   - raw TIFF intensity measurement
   - ratio_right = I_right / (I_left + I_right)
   - threshold = 0.4
   - right-priority duplicate handling
   - final two-color emitter cloud / reconstruction image
```

建议使用的真实素材路径：

```text
双色 union raw-ratio 标准输出:
/home/guest/Others/main/race/neptune_v0.3/output/3371_union_raw_ratio_bicolor_unfiltered_thr040_right_priority/union_raw_ratio_unfiltered_thr040_right_priority.png

ratio map:
/home/guest/Others/main/race/neptune_v0.3/output/3371_union_raw_ratio_bicolor_unfiltered_thr040_right_priority/union_raw_ratio_unfiltered_thr040_right_priority_ratio_map.png

left infer/recon 输出目录:
/home/guest/Others/main/race/neptune_v0.3/output/3371_left_root_infer_recon_latestckpt_recenter_full8000_roi96_valid80_cut8_p070

right infer/recon 输出目录:
/home/guest/Others/main/race/neptune_v0.3/output/3371_right_infer_recon_latestckpt_recenter_full8000_roi96_valid80_cut8_prob090_no_locprec

fast route 速度优化说明:
/home/guest/Others/main/race/neptune_v0.3/docs/fast_route_speed_optimization_summary.md
```

## 5. 海报整体版式

建议使用 3 列竖版布局，但中央区域必须最大，避免等权重卡片网格。

```text
顶部 12-15%:
  标题、作者、单位、一句话主张、3-4 个关键数字。

左列 25%:
  问题背景 + 3052 / LUNAR / LiteLoc 对标 + 原始瓶颈。

中列 45-50%:
  Neptune v0.3 主流程大图。
  这是海报视觉中心，应占最大面积。

右列 25-30%:
  定量结果、速度收益、physical update 诊断、infer/recon 结果。

底部 18-22%:
  multicolor union raw-ratio reconstruction 大图 + takeaway。
```

顶部关键数字可写成小型 statistic strip：

```text
ROI input: 96 x 96
Valid core: 80 x 80
Raw TIFF: 8000 frames
Infer: prob >= 0.9, no locprec gate
Union ratio threshold: 0.4
Union emitters: 8.9M
Metrics: Jaccard / RMSE_lat / RMSE_ax
```

如果最后正式排版时需要更稳妥，可以把 `8.9M` 写为：

```text
~8.9M union detections
```

## 6. 分面板设计

### Panel A: Raw Data and Failure Mode

目的：说明 Neptune v0.3 面对的是双通道真实 raw TIFF，不是单纯 synthetic benchmark。

画面内容：

```text
左侧放 2-3 个 raw TIFF frame thumbnail。
每个 thumbnail 内标出 left / right channel。
旁边放几个 raw ROI tile，显示真实背景、暗 emitter、亮 emitter 和 channel intensity imbalance。
用很少文字标注:
  raw TIFF movie
  left/right channel imbalance
  field-dependent PSF
```

不要画成抽象数据入口图。必须看起来像真实显微图像证据。

### Panel B: Fast Field-Dependent Simulation Route

目的：说明速度优化不是降低物理复杂度，而是减少重复计算。

画面内容：

```text
画一个 physical version lifecycle:

physical model update
    -> new physical version
    -> prewarm global-field LUT once
    -> many epochs reuse the same LUT
    -> training step only does LUT lookup + subpixel shift + projection

同时画:
  global-field LUT slab
  zmap field
  ROI 96 x 96 crop
  valid 80 x 80 core
  cut edge 8 px
```

关键标签：

```text
Physical-versioned LUT lifecycle
Global-field FP16 LUT
Cached window / condition cache
Fused projection path
No per-step LUT rebuild
```

### Panel C: ROI-Bank Physical Model Update

目的：说明 Neptune v0.3 如何把真实 raw TIFF 反馈到物理模型。

画面内容：

```text
raw TIFF ROI bank
    -> posterior localization sampling
    -> projected emitter set
    -> gamma objective
    -> updated zmap / PSF model
    -> next physical version
```

必须明确：

```text
left and right domains are updated independently
gamma controls field-dependent physical model
heldout bank is monitor, not hard failure
diagnostic output uses raw vs corrected camera vs recon
```

建议放一条 raw / corrected / recon 的三联图：

```text
raw TIFF
corrected camera
recon + matched background visualization
```

### Panel D: Training and Inference Standard Route

目的：把训练和推理的空间语义讲清楚，避免 epoch/batch 和 ROI 语义混乱。

画面内容：

```text
Training:
  ROI 96 x 96 input
  steps_per_epoch = 417
  batch_size = 24
  LR scheduler aligned to 3052 epoch semantics
  physical update start / interval follow standard route

Inference:
  full-frame slicing with ROI 96 x 96
  valid core 80 x 80
  cut edge 8
  recenter enabled
  stitch valid cores into full reconstruction
```

建议画一个全场 FOV，被 overlapping ROI tiles 覆盖，每个 tile 只保留中心 valid core。

### Panel E: Metrics and Speed

目的：只汇报关键结果，不塞无用 metrics。

画面内容：

```text
三条曲线或三个小图:
  Jaccard vs epoch
  RMSE_lat vs epoch
  RMSE_ax vs epoch

一个小型 speed table:
  3052 baseline
  v0.3 fast route
  major optimization source
```

注意：

```text
不要把 loss、heldout、posterior diagnostic、debug profile 全都放上海报。
只保留 Jaccard / RMSE_lat / RMSE_ax。
```

如果暂时没有最终稳定数值，可以写：

```text
Insert final 3371/standard-route metrics here
```

### Panel F: Union Raw-Ratio Multicolor Reconstruction

目的：展示最终 8000-frame 双通道重建结果。

画面内容：

```text
left predictions + right predictions
    -> union duplicate suppression
    -> measure raw TIFF local intensity in left/right
    -> ratio_right = I_right / (I_left + I_right)
    -> threshold 0.4
    -> two-color reconstruction
```

必须写清楚：

```text
This is union-based, not paired-only filtering.
No locprec gate is used by default for the final multicolor reconstruction.
Duplicate detections use right-priority localization attributes.
```

建议底部放最大的一张双色重建图，右边配 ratio map 小图。

## 7. 完整生成 Prompt

下面这段可以直接给海报设计模型、视觉生成模型、PPT 自动排版工具，或作为人工排版说明。

```text
Create a portrait academic scientific poster, 1.2 m wide and 1.6 m tall, for a microscopy / single-molecule localization microscopy method named "Neptune v0.3". The poster must look like a restrained Nature Methods / Nature Biomedical Engineering scientific poster, not a startup pitch deck and not a software architecture diagram.

Main title:
"Neptune v0.3: Fast Physics-Consistent Field-Dependent SMLM Localization and Ratiometric Reconstruction from Raw TIFF Movies"

Subtitle:
"A fast field-dependent PSF simulation, physical-model gamma update, and union raw-ratio two-color reconstruction workflow for real dual-channel raw TIFF data."

Use a white or very light warm-gray background, charcoal text, thin gray divider rules, and sparse muted teal-blue and muted orange accents. Keep the visual system mostly grayscale so that microscopy evidence and metric curves carry the message. Use large whitespace and one dominant central workflow region. Avoid equal-weight card grids.

Poster size and orientation:
portrait, 1200 mm x 1600 mm, designed for PDF/SVG/TIFF export at 300 DPI.

Overall layout:
Top 12-15%: title, authors, one-sentence claim, and a thin key-number strip.
Left column: raw data problem, 3052 / LUNAR-style baseline comparison, original engineering bottlenecks.
Large central column: Neptune v0.3 physical workflow and fast route mechanism.
Right column: metrics, speed optimization summary, physical update diagnostics.
Bottom band: final union raw-ratio bicolor reconstruction and takeaway.

Key numbers to show in a compact strip:
ROI input 96 x 96; valid core 80 x 80; raw TIFF 8000 frames; inference prob >= 0.9 with no locprec gate; ratio_right threshold = 0.4; union detections about 8.9M; metrics limited to Jaccard, RMSE_lat, and RMSE_ax.

Panel A, Raw data:
Show real-looking dual-channel raw TIFF microscopy thumbnails with left and right channel labels. Include several small ROI tiles that show channel intensity imbalance, background variation, bright emitters, and dim emitters. Label the evidence as "dual-channel raw TIFF movie", "field-dependent PSF", and "left/right intensity imbalance". Do not use generic database icons.

Panel B, Fast field-dependent simulation route:
Draw a physical-version lifecycle. A physical model update creates a new physical version, then a global-field FP16 LUT is prewarmed once, then multiple training epochs reuse the same LUT. Inside each training step, show LUT lookup, subpixel shift, fused projection/placement, Poisson camera model, and GMM/posterior localization loss. Emphasize that LUT is not rebuilt per step. Show a zmap field and several PSF spot tiles whose shape changes with field position and z.

Panel C, ROI-bank physical model update:
Draw a loop from raw TIFF ROI bank to posterior localization sampling, projected emitters, gamma objective, updated zmap/PSF model, and next physical version. Show left and right domains as independent lanes, because they have independent gamma and channel-specific baseline correction. Include a diagnostic strip with three columns: raw TIFF, corrected camera, and recon plus matched background visualization.

Panel D, Training and inference geometry:
Show ROI 96 x 96 input tiles over a full FOV, with only the center 80 x 80 valid core retained and edge 8 px cut away. Show recentered full-frame inference and stitching of valid cores into a full reconstruction. Label "recenter enabled", "prob >= 0.9", "no locprec gate", "valid-core stitching".

Panel E, Quantitative results:
Use only three main localization metrics: Jaccard, RMSE_lat, and RMSE_ax. Draw clean line plots versus epoch, with a quiet comparison to the 3052 reference route where available. Add a small speed table that attributes time savings to physical-versioned LUT prewarm, cached window / condition cache, fused projection path, reduced redundant diagnostics, and cleaned physical update scheduling. Do not include noisy internal metrics.

Panel F, Union raw-ratio bicolor reconstruction:
Make this the bottom visual payoff. Show left prediction set and right prediction set entering a union duplicate-suppression step, then raw TIFF intensity measurement in both channels, then ratio_right = I_right / (I_left + I_right), threshold 0.4, and final two-color reconstruction. Make clear that this is union-based, not paired-only filtering. For duplicate detections, keep right-channel position, z, probability, and precision. Place the final bicolor emitter cloud as a large microscopy reconstruction image, with a smaller ratio map beside it.

Scientific evidence objects that must be visible:
raw TIFF frame tiles; left/right channel lanes; zmap or field map; astigmatic PSF spot tiles at different z positions; ROI 96 x 96 and valid 80 x 80 geometry; physical-versioned LUT block; posterior sampling / gamma update loop; Jaccard / RMSE_lat / RMSE_ax plots; final two-color union raw-ratio reconstruction.

Use concise labels only. The poster should still be readable if most text is removed because the image tiles, PSF tiles, ROI geometry, loop arrows, and metric plots carry the story. Use thin arrows and restrained lane boundaries. Avoid decorative icons, gradient backgrounds, abstract neural-network boxes, molecule clip-art, dashboard-like UI widgets, or colorful AI-art collage.

Bottom takeaway line:
"Neptune v0.3 keeps the field-dependent physical model in the loop while moving repeated PSF computation out of the training step, enabling faster training and full-frame raw-TIFF ratiometric reconstruction."
```

## 8. 中文版本 Prompt

如果需要给中文设计师或中文排版模型，可以使用下面版本。

```text
请制作一张竖版学术海报，尺寸为 1.2 m x 1.6 m，主题是 Neptune v0.3：面向真实双通道 raw TIFF 显微电影的快速场依赖 SMLM 定位训练、物理模型更新和双色比率重建流程。

整体风格参考 Nature Methods / Nature Biomedical Engineering 的克制科学海报：白色或极浅灰白背景，charcoal 黑灰文字，细灰色分割线，少量 muted teal-blue 和 muted orange 强调色。不要做成软件宣传图、创业公司 infographic、深色科技风或花哨 dashboard。海报应由真实科学证据对象驱动，包括 raw TIFF 图像、PSF 光斑、zmap、ROI 切片、metric 曲线和最终重建图。

标题:
Neptune v0.3: Fast Physics-Consistent Field-Dependent SMLM Localization and Ratiometric Reconstruction from Raw TIFF Movies

副标题:
面向真实 raw TIFF 数据的快速场依赖 PSF 仿真、物理模型更新与双色比率重建流程

核心主张:
Neptune v0.3 在保持 3052 / LUNAR-style 物理一致性的基础上，通过 physical-versioned LUT、cached-window simulation、ROI-bank gamma update 和 recentered infer/recon，把 field-dependent SMLM 训练和 8000-frame raw TIFF 重建整合成更快、更稳定、可诊断的标准流程。

海报布局:
顶部 12-15% 放标题、作者、单位、一句话主张和关键数字条。
左列放 raw TIFF 问题背景、双通道强度不均衡、field-dependent PSF、3052 / LUNAR-style 对标和原始工程瓶颈。
中间最大区域放 Neptune v0.3 主流程图，必须是海报的视觉中心。
右列放三项核心 metrics、速度优化摘要和 physical update 诊断图。
底部放最终 union raw-ratio 双色重建大图和 takeaway。

关键数字条:
ROI input 96 x 96; valid core 80 x 80; raw TIFF 8000 frames; infer prob >= 0.9; no locprec gate; ratio_right threshold = 0.4; union detections about 8.9M; metrics only Jaccard / RMSE_lat / RMSE_ax.

Panel A: Raw data and problem.
展示真实感的 left/right 双通道 raw TIFF frame thumbnails 和 raw ROI tiles，能看出背景、暗 emitter、亮 emitter、左右通道强度不均衡。标签只保留 raw TIFF movie、left/right channel imbalance、field-dependent PSF。

Panel B: Fast physical simulation route.
画出 physical-version lifecycle：physical model update 产生新 physical version，然后 global-field FP16 LUT 只 prewarm 一次，之后多个 epoch 复用同一个 LUT。training step 内只做 LUT lookup、subpixel shift、fused projection/placement、Poisson camera model 和 GMM/posterior localization loss。强调 no per-step LUT rebuild。旁边画 zmap field 和不同 z / field position 下的 astigmatic PSF spot tiles。

Panel C: ROI-bank physical model update.
画 raw TIFF ROI bank -> posterior localization sampling -> projected emitters -> gamma objective -> updated zmap/PSF model -> next physical version 的闭环。left 和 right 画成两条独立 lane，因为左右通道有独立 gamma 和 channel-specific baseline correction。放一个三联诊断图：raw TIFF、corrected camera、recon + matched background visualization。

Panel D: Training and inference geometry.
展示 full FOV 上的 ROI 96 x 96 overlapping tiles，每个 tile 只保留中心 valid core 80 x 80，cut edge 8 px。展示 recentered full-frame inference 和 valid-core stitching。标注 recenter enabled、prob >= 0.9、no locprec gate、valid-core stitching。

Panel E: Quantitative results.
只展示 Jaccard、RMSE_lat、RMSE_ax 三个核心指标，做简洁的 epoch 曲线，并可与 3052 reference route 做轻量对比。旁边放一个小 speed table，说明提速来自 physical-versioned LUT prewarm、cached window / condition cache、fused projection path、减少重复 diagnostics、clean physical update scheduling。不要放杂乱的 loss、heldout、posterior debug、profile internal metrics。

Panel F: Union raw-ratio bicolor reconstruction.
底部做成视觉 payoff：left prediction set + right prediction set -> union duplicate suppression -> 从 raw TIFF 在左右通道重新测局部强度 -> ratio_right = I_right / (I_left + I_right) -> threshold 0.4 -> final two-color reconstruction。必须写清楚这是 union-based，不是 paired-only filtering。duplicate detections 使用 right-priority 的 position、z、probability 和 precision。放大最终双色 emitter cloud，并在旁边放 ratio map。

禁止元素:
不要用通用神经网络大方框主导画面。
不要用 startup pitch deck 风格。
不要用装饰性分子、随机粒子、渐变背景、霓虹科技风。
不要把 metric 做成复杂 dashboard。
不要写 paired-only ratio reconstruction。
不要把 locprec <= 40 写成默认标准。
不要把 epoch/batch 训练语义混在一起。

底部 takeaway:
Neptune v0.3 keeps the field-dependent physical model in the loop while moving repeated PSF computation out of the training step, enabling faster training and full-frame raw-TIFF ratiometric reconstruction.
```

## 9. 海报文字建议

### Problem

```text
Real dual-channel SMLM movies contain field-dependent aberrations, channel-specific background / intensity imbalance, and spatially varying PSF shapes. A useful training route must preserve these physical effects while avoiding repeated vector-PSF computation inside every training step.
```

### Method

```text
Neptune v0.3 introduces a physical-versioned fast route: after each physical model update, the current field-dependent PSF model is materialized into a reusable global-field LUT. Training then reuses this LUT across epochs and performs only lightweight lookup, shift, projection, camera simulation, and localization loss.
```

### Physical Update

```text
Raw-TIFF ROI banks provide the feedback signal for gamma-based physical model updates. Left and right domains are optimized independently with channel-specific baseline correction, while diagnostic strips compare raw TIFF, corrected camera observations, and reconstructed images.
```

### Reconstruction

```text
Full-frame inference uses recentered ROI slicing with 96 x 96 inputs and 80 x 80 valid cores. Multicolor reconstruction is performed by unioning left/right emitter sets, re-measuring local raw-TIFF intensities, and assigning color using ratio_right with a threshold of 0.4.
```

### Takeaway

```text
Neptune v0.3 converts a slow field-dependent physical training loop into a reusable-versioned simulation and reconstruction workflow while preserving the physical feedback path needed for real raw TIFF SMLM data.
```

## 10. Negative Prompt

```text
Do not create a generic AI workflow diagram.
Do not use a dark cyberpunk background.
Do not fill the poster with decorative molecules, random particles, or abstract glowing neural networks.
Do not use equal-weight colorful cards.
Do not make the method look like a web dashboard.
Do not describe the multicolor method as paired-only filtering.
Do not show locprec <= 40 as the default final reconstruction gate.
Do not include many internal debug metrics.
Do not hide the real microscopy evidence.
Do not omit field-dependent PSF / zmap.
Do not omit ROI 96 x 96 and valid 80 x 80 geometry.
```

## 11. 最终检查清单

正式出图前检查：

```text
[ ] 海报尺寸是 1.2 m x 1.6 m 竖版。
[ ] 中央区域是 Neptune v0.3 主流程，不是通用软件框图。
[ ] raw TIFF / PSF / zmap / ROI geometry / recon 都有真实视觉载体。
[ ] 指标只保留 Jaccard、RMSE_lat、RMSE_ax。
[ ] 写清楚 ROI 96 x 96、valid core 80 x 80、cut edge 8。
[ ] 写清楚 infer 使用 recenter。
[ ] 写清楚 final infer/recon 默认 prob >= 0.9, no locprec gate。
[ ] 写清楚 multicolor 是 union raw-ratio，不是 paired-only。
[ ] 写清楚 ratio_right threshold = 0.4。
[ ] 写清楚 left/right physical update 是独立 domain。
[ ] 没有把 batch-budget 旧语义和 3371 标准 epoch route 混在一起。
[ ] 没有放太多内部 debug / profile 指标。
[ ] 底部有明确 takeaway。
```
