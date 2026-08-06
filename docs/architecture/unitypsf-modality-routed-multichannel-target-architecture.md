# UnityPSF 改造完成后的完整项目蓝图

- 文档状态：目标架构，后续实现和验收的完成态定义
- 日期：2026-08-05
- 调研用途：向其他 AI 或合作者完整说明模型，并据此调研最能验证方法价值的生物实验
- 独立调研简报：[UnityPSF 生物实验调研简报](../research/unitypsf-biological-experiment-research-brief.md)
- 实施依据：[按模态路由的多通道修正规划](../plans/unitypsf-modality-routed-multichannel-correction-plan.md)
- 保留决策：[ADR 0003：单文件 joint checkpoint](../adr/0003-single-joint-checkpoint.md)

> 本文描述全部改造完成后的 UnityPSF，不代表当前代码已经达到该状态。当前代码仍有
> `(modality, channel_id)` 完整网络实例、四实例 checkpoint 和缺少正式 held-out eval
> 等旧行为。只有本文列出的验收条件全部通过后，才能更新 README 并冻结新 baseline。

## 1. 项目定位

UnityPSF 是面向单分子定位显微成像的多模态 PSF 基础模型。它使用一个模型对象和一个
checkpoint，统一处理：

- 2D emitter PSF；
- astigmatism PSF；
- double-helix PSF；
- 每种 PSF 下的单通道或 left/right 多测量通道。

UnityPSF 要回答的核心科学问题是：

> 能否用一套统一的稀疏 MoE 模型，在保留不同 PSF 物理模型和解码方式的前提下，处理
> 多种 PSF 形态，并让同一 PSF 模态自然支持多个测量通道？

项目最终对外只暴露：

```text
一个 UnityPSF 模型
一个 unitypsf_joint.ckpt
一个 localize(...) API
一套训练、推理、评估和可视化流程
```

内部不同模态拥有完整 expert；同一模态的不同测量通道共享网络参数，但保留各自独立的
物理状态。

## 2. 最终模型结构

```text
Input
  +-- temporal image frames: (N, T, H, W)
  +-- modality metadata: emitter_2d | astigmatism | double_helix
  +-- channel_id: main | left | right | custom
  +-- channel physical condition
          |
          v
ModalityRouter                         sparse top-1 route
  +-- Emitter2DExpert
  |     +-- preprocessing
  |     +-- full localization backbone
  |     +-- FiLM(condition + optional channel one-hot)
  |     +-- detection/xy/photon/background heads
  |     +-- 2D loss and decoder
  |
  +-- AstigmatismExpert
  |     +-- preprocessing
  |     +-- full localization backbone
  |     +-- FiLM(zernike/field/channel condition)
  |     +-- detection/xy/z/photon/background heads
  |     +-- astigmatism loss and decoder
  |
  +-- DoubleHelixExpert
        +-- preprocessing
        +-- full localization backbone
        +-- FiLM(calibration/field/channel condition)
        +-- common localization heads
        +-- lobe-angle/lobe-separation heads
        +-- DH physics, loss and decoder
          |
          v
Unified SMLM Output Contract
  +-- p
  +-- photon mean/sigma
  +-- x/y/z mean/sigma
  +-- background
  +-- modality-specific auxiliary output
```

这是确定性、稀疏、top-1 的真正 MoE：一次请求只执行选中的 PSF 模态 expert，未选中的
expert 不参与 forward，也不产生梯度。模态 metadata 是第一版正式路由来源；图像自动模态
检测器是后续扩展，不阻塞基础模型训练。

## 3. 路由与通道边界

### 3.1 顶层路由

顶层 router 的唯一专家选择键是：

```text
PSFModality
```

正式专家注册表为：

```text
experts["emitter_2d"]
experts["astigmatism"]
experts["double_helix"]
```

以下形式不得再作为正式模型注册表：

```text
experts["emitter_2d:left"]
experts["emitter_2d:right"]
experts["astigmatism:left"]
experts["astigmatism:right"]
```

### 3.2 通道选择

`channel_id` 不是 expert 路由键，而是已选 expert 的运行条件：

```text
router.resolve(modality) -> modality expert
modality expert.resolve_channel(channel_id) -> channel context
```

通道 context 包含：

- raw TIFF crop；
- measurement channel metadata；
- anchor profile；
- peak zmap；
- coefficient/gamma map；
- condition provider；
- calibration；
- physical version 和完整性 hash。

同一模态内，left/right 不复制 backbone、FiLM 或 heads。它们通过物理 condition 和可选
channel one-hot 让共享网络识别通道差异。

### 3.3 外部 API

外部 API 保持显式、稳定：

```python
model = UnityPSF.from_checkpoint("unitypsf_joint.ckpt", device="cuda:0")

result = model.localize(
    images,
    modality="astigmatism",
    channel_id="left",
    conditions=conditions,
)
```

执行顺序是：

```text
validate input
  -> resolve modality expert
  -> validate channel is supported by that expert
  -> load channel physical state
  -> build/validate FiLM condition
  -> run exactly one modality expert
  -> decode common output
  -> attach modality/channel semantics
```

错误 modality、未知 channel、缺失 calibration、condition 维度错误或 checkpoint 能力不匹配
都必须在边界处明确失败，禁止静默 fallback 到其他 expert 或 CPU。

## 4. 三个规范 Expert

### 4.1 Emitter2DExpert

职责：定位没有有效轴向编码的 2D emitter PSF。

- 使用完整 localization backbone、FiLM 和独立 heads；
- `z_mu=0`，并输出 `z_valid=false`；
- z loss 不参与反向传播；
- 默认物理起点是 zero-aberration focal PSF；
- left/right 可拥有不同 bead calibration 和成像 crop；
- 多通道数据共同训练同一个 Emitter2DExpert。

### 4.2 AstigmatismExpert

职责：处理利用像散编码 z 的 PSF。

- 默认 anchor 为 `Z(2,2)=99 nm` 的已验证物理配置；
- left/right 分别从自己的 raw crop 构建 peak zmap；
- left/right 分别维护 gamma、coefficient map 和 physical version；
- 两个通道共享一套 localization backbone、FiLM 和输出 heads；
- FiLM 输入携带对应通道的 Zernike/field condition；
- 两个通道的数据共同更新同一个 AstigmatismExpert optimizer。

### 4.3 DoubleHelixExpert

职责：处理 double-helix PSF 的双叶几何和轴向解码。

- 复用 `optics/psf/double_helix/` 中唯一的物理和 calibration 实现；
- 预测公共定位量以及 lobe angle、lobe separation 等辅助量；
- 通过已验证的 angle-to-z calibration 解码 z；
- main/left/right 分别保存自己的 calibration 和有效 z 范围；
- calibration 缺失、越界或低置信度时明确拒识；
- 在真实 DH 数据通过科学验收前，不进入正式 release checkpoint。

## 5. 多通道联合训练

多通道语义对齐 Neptune v0.3：一个模态只有一个训练 runtime。

```text
ChannelBatchProvider(left)  --+
                              +--> balanced channel scheduler
ChannelBatchProvider(right) --+             |
                                            v
                                  one modality expert
                                  one optimizer
                                  one scheduler
                                  one model checkpoint
```

训练约束：

1. left/right 数据按 step 或 sequence 交替，也允许 cached-window batch 在边界处混合通道。
2. 不要求每个 optimizer step 都严格包含成对的 left/right 样本。
3. 每个 epoch 必须覆盖全部启用通道，并记录实际 sample/step 数。
4. 每个样本必须携带 channel identity 和对应物理 condition。
5. 所有通道 loss 都对同一套模态网络参数反向传播。
6. 各通道的物理更新只允许写入自己的 channel context。
7. 第一版不引入跨通道 consistency loss，避免增加未经验证的科学假设。

训练日志既要保留模态聚合指标，也要保留 per-channel 指标。不得只画一条平均 loss 掩盖
某个通道退化。

## 6. Expert Parallel

Expert Parallel 是训练执行策略，不改变 UnityPSF 的模型身份：

```text
GPU 0 / rank 0 -> Emitter2DExpert
                   +-- main/left/right channel batches

GPU 1 / rank 1 -> AstigmatismExpert
                   +-- main/left/right channel batches

GPU 2 / rank 2 -> DoubleHelixExpert
                   +-- enabled only after DH data gate
```

每个 rank：

- 拥有一个模态 expert；
- 拥有该 expert 唯一的 optimizer、scheduler、AMP scaler 和 RNG；
- 独立推进训练，不执行跨模态梯度同步；
- 保存模态级 resume checkpoint 和完成状态；
- 可以从自身 checkpoint 恢复而不重训其他模态。

训练阶段不得依赖默认 10 分钟 NCCL collective。模态训练完成后，由控制面确认所有必需
模态状态，再由 coordinator 组装、校验、原子发布 joint checkpoint。不同模态耗时差异不能
导致已完成 rank 被 watchdog 杀死。

## 7. Joint Checkpoint

正式发布单位仍是一个文件：

```text
unitypsf_joint.ckpt
  +-- metadata
  |     +-- model_family: UnityPSF
  |     +-- supported_modalities
  |     +-- supported_channels_per_modality
  |     +-- code/schema version
  |
  +-- router
  |     +-- type: deterministic
  |     +-- key: modality
  |     +-- mode: hard_top1
  |
  +-- experts
  |     +-- emitter_2d
  |     |     +-- model_config
  |     |     +-- model_state_dict
  |     |     +-- input/output contracts
  |     |
  |     +-- astigmatism
  |     +-- double_helix                 # only after validation
  |
  +-- channel_states
  |     +-- emitter_2d/{main,left,right}
  |     +-- astigmatism/{main,left,right}
  |     +-- double_helix/{main,left,right}
  |
  +-- calibration
  +-- provenance
  +-- integrity
  +-- training_state                     # resume role only
```

`release` checkpoint 不包含 optimizer；`resume` checkpoint 对每个模态保存：

- optimizer；
- scheduler；
- AMP scaler；
- epoch/global step；
- channel scheduler state；
- RNG state；
- physical state version。

checkpoint 必须能够在移动到新路径后独立加载。任一 expert、channel state、calibration 或
metadata 被修改后，完整性校验必须失败。

## 8. 推理与重建

单次推理流程：

```text
raw TIFF
  -> modality resolution
  -> channel crop
  -> temporal window/preprocessing
  -> channel physical condition
  -> one routed modality expert
  -> common decoder
  -> filter
  -> full-frame coordinate restoration
  -> per-channel localization/reconstruction
```

双通道流程：

```text
left crop  -> same modality expert(channel=left)  -> left localizations
right crop -> same modality expert(channel=right) -> right localizations
                                                     |
                                                     v
                                  registration / union / multicolor reconstruction
```

left/right 必须使用相同的坐标、photon、z 方向和输出 contract；物理状态和 calibration 不得
串用。推理结果保留 per-channel 文件，然后才进入 union 或 ratiometric reconstruction。

## 9. Eval 与可视化

正式训练必须使用按 acquisition group 隔离的 train/validation/test split。固定验证样本在
训练开始前写入 manifest，训练过程中不得根据结果替换样本。

每个 modality/channel 至少输出：

- train loss 和 `eval_loss`；
- precision、recall、Jaccard；
- `RMSE_XY_nm`；
- `RMSE_Z_nm`，Emitter2D 标记为不适用；
- photon relative error；
- route count、optimizer step、sample count；
- throughput、峰值显存和 checkpoint hash。

正式报告必须包含：

```text
00_input_audit
01_psf_patch_montage
02_route_and_step_balance
03_train_validation_curves
04_prediction_gt_overlay_<modality>_<channel>
05_error_by_z_<modality>_<channel>
06_reconstruction_<modality>_<channel>
07_physical_state_<modality>_<channel>
08_cross_modality_scorecard
report.html
summary.json
```

真实数据没有 GT 时，只报告可被真实数据支持的量，并重点检查结构连续性、背景假阳性、
网格伪影、重复定位和 reconstruction；不得伪造 precision 或 RMSE。

## 10. 改造完成后的代码边界

```text
unity/
  +-- configs/
  |     +-- modality expert configs
  |     +-- multichannel layouts
  |     +-- joint training/eval configs
  |
  +-- src/unity_psf/
  |     +-- contracts/
  |     |     +-- modality.py             modality/channel/input contracts
  |     |     +-- checkpoint.py           modality checkpoint contract
  |     |     +-- joint_checkpoint.py     one-file UnityPSF contract
  |     |
  |     +-- models/
  |     |     +-- unity_psf.py            public model API
  |     |     +-- psf_moe/
  |     |           +-- router.py         modality-only sparse router
  |     |           +-- experts/
  |     |                 +-- emitter_2d.py
  |     |                 +-- astigmatism.py
  |     |                 +-- double_helix.py
  |     |
  |     +-- training/
  |     |     +-- channel_context.py      per-channel physical state
  |     |     +-- multimodal_joint.py     modality-level schedules
  |     |     +-- loop.py                 shared train/eval loop
  |     |     +-- runtime.py              one runtime per modality
  |     |
  |     +-- localization/                 data, losses, decoding and eval
  |     +-- optics/psf/                   PSF renderers and calibration
  |     +-- peak/                         per-channel peak-zmap bootstrap
  |     +-- gamma_update/                 per-channel physical updates
  |     +-- infer_recon/                   inference/filter/reconstruction
  |     +-- reporting/                     fixed scientific report pack
  |     +-- cli/                           stable public entry points
  |
  +-- tests/
  |     +-- contracts/
  |     +-- models/
  |     +-- training/
  |     +-- integration/
  |     +-- baseline/
  |
  +-- docs/
  +-- scripts/
  +-- output/                              ignored runtime artifacts
  +-- logs/                                ignored SLURM logs
  +-- .local/                              ignored caches and temporary state
  +-- pyproject.toml
  +-- README.md
```

`training.multichannel` 中以“每通道独立完整网络”为前提的接口只能保留为旧 checkpoint
兼容层，不能继续作为正式训练入口。正式入口必须创建 modality-level runtime。

## 11. 正式工作流

### 11.1 双模态阶段

在 DH 数据就绪前，正式模型支持：

```text
Emitter2DExpert(left + right)
AstigmatismExpert(left + right)
```

两张 GPU 分别训练两个模态，完成后生成一个双模态 `unitypsf_joint.ckpt`。该阶段必须先
通过 held-out eval 和人工可视化，再冻结 baseline。

### 11.2 三模态阶段

DH raw TIFF、bead calibration、z/angle metadata 和 train/validation/test split 到位后：

```text
双模态 checkpoint schema
  -> 加入 DoubleHelixExpert
  -> 三 GPU Expert Parallel
  -> 三模态 joint checkpoint
  -> 三模态 eval/report
```

加入 DH 不改变顶层 API、router key 或已有 expert checkpoint 结构。

## 12. 迁移与历史产物

SLURM job `4525` 的四个通道独立 checkpoint：

- 保留原始日志、metrics 和 checkpoint；
- 标记为 `independent-channel ablation`；
- 不恢复为新的正式 baseline；
- 不直接平均或合并权重生成共享模态 expert；
- 可用于比较“通道独立网络”和“模态内共享网络”的参数量、精度和泛化差异。

旧 `unity_psf.joint_checkpoint.v1` 可提供只读兼容导入，但导入结果必须明确标记为 legacy，
不能伪装成新架构训练得到的 checkpoint。

## 13. 不可破坏的架构约束

1. 顶层 router 只按 modality 选择完整 expert。
2. channel 只选择 expert 内部的物理上下文，不选择另一套完整网络。
3. 一个模态只有一个 model、optimizer 和 scheduler。
4. 同一模态的所有启用通道共同更新该模型。
5. 通道的 peak-zmap、gamma、calibration 和 physical version 必须独立。
6. 不同 PSF 模态不共享完整 localization backbone。
7. 一次 forward 只激活一个模态 expert。
8. 一个 release checkpoint 必须自包含所有已声明支持的模态和通道状态。
9. 未通过数据和科学验收的模态不得写入 supported modalities。
10. CPU smoke、训练 loss 或路由 smoke 不得冒充正式科学 eval。

## 14. 项目完成标准

只有以下条件全部满足，才能宣称上述改造完成：

- 代码中不再以 `(modality, channel_id)` 注册正式完整网络；
- left/right 调用解析到同一个模态模型对象；
- left/right batch 都能对同一模型产生独立可验证的梯度更新；
- left/right physical state 仍然隔离且完整性 hash 不同；
- 双模态 checkpoint 只有两套完整网络，三模态 checkpoint 只有三套完整网络；
- 模态级 resume 能跳过已完成模态，只恢复未完成模态；
- Expert Parallel 模态耗时差超过 10 分钟时不会触发 NCCL timeout；
- 一个 checkpoint 能在新路径加载并连续推理所有已支持 modality/channel；
- 固定 held-out eval、全部强制图、`summary.json` 和 `report.html` 均生成；
- 双模态真实数据结果通过人工科学验收；
- DH 数据就绪并单独通过验收后，才升级三模态声明；
- README、ADR、配置示例和 CLI 帮助全部与本架构一致；
- Neptune v0.3 未被修改，仍可独立安装和运行。

完成这些条件后，UnityPSF 才是：

> 一个 checkpoint、一个模型、按 PSF 模态稀疏路由、每个模态内部联合处理多测量通道的
> 多模态 PSF 基础模型。

## 15. 给外部 AI 的生物实验调研说明

本节可以单独交给负责文献和实验调研的 AI。它描述的是拟完成的方法及需要验证的科学
价值，不代表这些价值已经被实验结果证明。

### 15.1 两句话说明模型

现有 SMLM 定位方法通常围绕一种 PSF 和一套测量通道单独训练，切换 2D emitter、
astigmatism 或 double-helix PSF 时，需要管理不同模型、checkpoint、物理校准和推理流程。
UnityPSF 用一个按 PSF 模态稀疏路由的 MoE checkpoint 统一三类完整定位 expert，并让每个
模态 expert 在共享网络参数的同时联合处理 left/right 多测量通道及其独立物理状态。

### 15.2 方法到底新在哪里

需要调研和验证的不是“能否用神经网络做一次定位”，而是以下组合能力是否具有实际价值：

1. **一个模型覆盖多种 PSF。** 同一个 `unitypsf_joint.ckpt` 包含 2D、像散和 DH 三个完整
   expert，实验切换 PSF 时不再切换外部软件体系。
2. **稀疏模态路由。** 每次只激活一个 PSF expert，保留专业模型能力，避免让一个共享
   backbone 强行拟合差异很大的 PSF 几何。
3. **模态内多通道联合学习。** 同一种 PSF 的 left/right 数据共享网络和 optimizer，物理
   calibration、peak-zmap 和 gamma 仍按通道独立。
4. **物理条件进入 FiLM。** 网络不仅看到图像，还看到与通道和空间位置对应的物理 condition。
5. **一个可移动 checkpoint。** 模型参数、模态能力、通道物理状态、calibration、版本和
   hash 在一个发布文件中统一管理。
6. **可扩展基础模型。** 新增 DH 数据时增加一个模态 expert，不改变顶层 API，也不破坏
   已有 2D/astigmatism 能力。

### 15.3 调研时必须区分的三种对象

| 对象 | 网络参数关系 | 物理状态关系 | 用途 |
| --- | --- | --- | --- |
| 不同 PSF 模态 | 完整 expert 相互独立 | 各自独立 | 2D、像散、DH 的专业定位 |
| 同一模态 left/right | 共享同一 expert 网络 | 各通道独立 | 多通道联合训练和推理 |
| 顶层 UnityPSF | 一个模型对象和 checkpoint | 汇总全部已验证状态 | 统一加载、路由、评估和部署 |

外部 AI 不得把 UnityPSF 描述成：

- 一个共享 backbone 后面挂三个小输出头；
- 把 left/right 当成两个 PSF 模态；
- 对每张图片同时执行所有 expert 的 dense ensemble；
- 已经训练完成的三模态模型；
- 已经证明优于所有单模态方法的模型。

### 15.4 生物实验需要验证什么

候选生物实验至少应验证下列一个核心命题，最好同时覆盖两个以上：

1. **多 PSF 通用性：** 同一个生物结构分别使用 2D、像散和 DH 成像时，UnityPSF 能否
   在一个模型中保持接近专业单模态模型的定位质量？
2. **轴向范围互补：** astigmatism 适合的常规 3D 范围与 DH 的扩展轴向范围，能否在同一
   分析体系下覆盖从薄层结构到较厚三维结构？
3. **多通道价值：** left/right 共享模态网络是否比两个完全独立网络获得更好的样本效率、
   鲁棒性或跨通道一致性，同时不损害各通道的独立校准？
4. **跨样本泛化：** 联合多模态训练是否改善模型从 calibration beads/模拟数据迁移到真实
   细胞结构的能力？
5. **工作流价值：** 一个 checkpoint 是否显著降低模型选择、部署、版本管理和复现实验的
  复杂度，而不增加不可接受的显存和推理开销？
6. **边界与拒识：** 模态 metadata 错误、calibration 失配、低信号、过曝和未知 PSF 时，
   系统能否明确失败或拒识，而不是给出看似合理的错误定位？

### 15.5 候选生物实验方向

以下只是供文献调研排序的候选，不是已经选定的实验：

| 候选结构或过程 | 可能验证的价值 | 可能使用的 PSF | 主要评价依据 |
| --- | --- | --- | --- |
| 微管或微管组织网络 | 2D 连续性、3D 走向、不同深度范围和现有数据延续性 | 2D、像散、DH | 线宽、连续性、分叉、轴向偏差、重建伪影 |
| 核孔复合体 | 已知几何结构可作为精度和偏差 proxy | 2D、像散，必要时 DH | 环直径、双环间距、圆度、重复结构一致性 |
| 网格蛋白包被小窝 | 从近二维平面到三维弯曲结构的形态恢复 | 2D、像散 | 直径、曲率、深度分布、结构分类稳定性 |
| 线粒体、内质网及接触位点 | 厚三维细胞区域和多结构空间关系 | 像散、DH，可加双通道 | 接触距离、网络连续性、轴向覆盖和多色配准 |
| 突触前后蛋白纳米组织 | 高密度、多色、多通道和轴向分层 | 2D、像散、DH | 簇大小、相对位移、层间距离、通道串扰 |
| 细胞膜受体聚簇或内吞 | 2D 高通量与 3D 状态切换 | 2D、像散 | cluster size、density、深度、动态或状态差异 |
| 活细胞细胞器或膜动力学 | 通用模型在速度、光毒性和时间分辨率约束下的价值 | 2D 或像散，DH 需评估光子预算 | 定位率、轨迹/结构连续性、时间分辨率、光漂白 |

调研 AI 必须判断每个候选是否真的需要多 PSF 或多通道。仅仅“可以用 SMLM 看见”不足以
证明它适合作为 UnityPSF 的核心验证实验。

### 15.6 优先选择实验的标准

每个候选实验按以下维度评分：

| 维度 | 核心问题 |
| --- | --- |
| 科学问题强度 | 生物结论是否重要，而不是纯算法展示？ |
| 多 PSF 必要性 | 使用两种或三种 PSF 是否回答了单一 PSF 难以回答的问题？ |
| 多通道必要性 | left/right 或多色测量是否是实验本身需要，而不是人为增加复杂度？ |
| 定量 ground-truth proxy | 是否有已知几何、DNA origami、校准结构或独立测量可验证偏差？ |
| 样本可得性 | 实验室是否容易获得细胞系、标记、原始 TIFF 和 calibration 数据？ |
| 光子预算 | DH 或多通道分光后是否仍有足够光子支持可靠定位？ |
| 配准难度 | 通道和模态之间是否能可靠配准并记录误差？ |
| 实施周期 | 能否先完成两周 pilot，再扩展为正式实验？ |
| 论文说服力 | 结果是否能同时说明方法正确、通用且有生物价值？ |

优先级最高的方案应包含：

- 一个带几何或独立测量 proxy 的方法学验证样本；
- 一个真实细胞结构实验；
- 一个能体现多 PSF 或多通道独特价值的实验，而不是三个互不相关的演示。

### 15.7 调研必须比较的 Baseline

外部 AI 应查找并为每种实验推荐公平 baseline：

1. 每种 PSF 对应的专业单模态方法或模型；
2. 同一架构但每个通道完全独立训练的模型，`4525` 可作为工程消融起点；
3. 模态内 left/right 联合训练的 UnityPSF expert；
4. 不使用 FiLM 物理 condition 的消融；
5. 不进行多模态联合管理、分别加载 checkpoint 的传统工作流；
6. 领域内常用的非深度学习或分析软件基线，在其适用范围内比较。

比较必须使用相同数据 split、相同检测阈值选择规则和相同 matching/evaluation protocol。
不能通过给 UnityPSF 更多训练数据或更宽松阈值制造不公平优势。

### 15.8 建议的实验层级

```text
Level 1: calibration/geometry validation
  beads or DNA-origami-like reference
  -> 验证 x/y/z 偏差、PSF 模态、通道和 calibration

Level 2: fixed-cell structural validation
  known cellular nanostructure
  -> 验证真实背景、密度、结构连续性和可重复性

Level 3: multimodal or multichannel biological question
  one biological system where PSF depth/channel complementarity matters
  -> 验证 UnityPSF 相对单模型工具链的真正实验价值

Level 4: robustness and generalization
  microscope/date/sample/label shift
  -> 验证基础模型与物理 condition 的跨域能力
```

第一篇方法工作不必同时完成所有 Level 4 内容，但至少应完成 Level 1、Level 2，并提供一个
可信的 Level 3 实验。

### 15.9 外部 AI 的预期调研输出

调研结果不应只列论文。最终应提交一张排序表，每个候选包含：

- 生物问题和为什么重要；
- 推荐样本、标记策略和固定/活细胞条件；
- 推荐 PSF 模态及为什么需要该模态；
- 是否需要 left/right、多色或分光通道；
- calibration 和 ground-truth proxy；
- 需要的 raw 数据、metadata 和最小样本量；
- 专业 baseline 和评价协议；
- UnityPSF 预期贡献及可能失败原因；
- 两周 pilot 设计；
- 完整实验的设备、时间、试剂和数据规模；
- 关键参考文献及其 DOI/URL；
- 总体优先级：高、中、低。

调研还必须回答：

1. 哪一个实验最能证明“一个模型通吃多种 PSF”不是工程包装？
2. 哪一个实验最能证明“模态内多通道共享网络”优于独立通道网络？
3. 哪一个实验真正需要 DH 的扩展轴向范围？
4. 哪些候选只能证明定位精度，不能证明生物学价值？
5. 哪些实验在现有数据和设备条件下最容易先做出可信 pilot？

### 15.10 可直接交给其他 AI 的调研任务

```text
请基于本文件描述的 UnityPSF 目标模型，系统调研最适合验证其方法学与生物学价值的
SMLM 生物实验。UnityPSF 是一个单 checkpoint、按 PSF modality 稀疏 top-1 路由的
MoE：Emitter2DExpert、AstigmatismExpert 和 DoubleHelixExpert 是三套完整定位网络；
同一模态的 left/right 测量通道共享网络和 optimizer，但各自保留独立 crop、peak-zmap、
gamma、calibration 和 physical state。第一阶段已有 2D 与 astigmatism 方向，DH 数据仍在
准备中。

请不要泛泛列出可以做 SMLM 的生物样本。请寻找真正能体现以下至少一项价值的实验：
多 PSF 统一模型、astigmatism 与 DH 轴向范围互补、同模态多通道联合学习、物理 FiLM
conditioning、单 checkpoint 可复现工作流、跨样本或跨显微镜泛化。

请调研并排序至少 8 个候选实验，重点覆盖微管、核孔复合体、网格蛋白包被小窝、
线粒体-内质网接触、突触纳米组织、膜受体聚簇等方向，但允许依据文献删除不合适候选或
加入更强候选。每个候选必须给出：生物问题、样本与标记、所需 PSF 和通道、为何单一 PSF
不够、ground-truth proxy、推荐 baseline、评价指标、数据规模、主要风险、两周 pilot、完整
实验成本和关键文献 DOI/URL。

最后给出三个明确推荐：最稳妥的方法学验证、最有生物学价值的主实验、最能体现 DH
不可替代性的扩展实验。明确区分文献证据、合理推断和仍需实验确认的假设。
```
