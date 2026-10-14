# 方案 A：共享 Backbone 的双分支条件化定位模型

- 日期：2026-10-13
- 状态：提案，尚未实现或进行速度／定位质量验收
- 范围：以 job 5705 的双 domain soft-MoE 路线为参照
- 本文不替代正式配置、已接受的 ADR 或历史 checkpoint

## 1. 目标与边界

本方案希望减少两个完整 DoubleUNet expert 的重复计算，同时保留两套分支特异的
FiLM 和输出 heads，以及 soft gate 加权输出。核心变化是让两个分支共用一次
backbone 特征计算，而不是每步分别计算两套 backbone。

这是模型结构变化，不是与当前模型严格等价的性能重构。必须独立命名、重新训练、
进行分通道质量验证，不能直接替换 job 5705 的正式路线并声称科学性能不变。

此前讨论中的“方案 A”包含两种不同方向，本文明确区分：

| 方向 | 含义 | 是否改变参数共享关系 |
| --- | --- | --- |
| A1：执行优化 | 保留独立专家，尝试批量执行、编译或 grouped 运算 | 不主动改变；需验证数值及状态语义 |
| A2：共享 backbone | 一套 backbone 加两套 FiLM／heads | 是，本文的主要提案 |

A1 的收益不能套用 A2 的计算量模型；它需要独立 benchmark。

## 2. 当前模型与计算重复

当前 `SoftMoEFiLMExperts` 深拷贝 base model，创建两个完整专家。每步两个专家
接收同一个图像和条件，分别执行 forward，然后按 gate 权重加权。

```text
单通道三帧时间窗口 image + condition
    ├── DoubleUNet core 0 → FiLM 0 → heads 0 → 输出激活 → y0
    ├── DoubleUNet core 1 → FiLM 1 → heads 1 → 输出激活 → y1
    └── gate(condition) → w0, w1

y = w0 × y0 + w1 × y1
```

这里的三帧输入是时间上下文，不是 left/right 两个测量通道。训练按 domain
交替获取单通道 batch，两个专家都处理当前 batch；两条梯度路径均参与反向。

DoubleUNet core 内部包含对时间帧应用的 `unet_shared` 和时间特征融合的
`unet_union`。名称中的 `shared` 原本指时间帧共享，不代表两个 soft-MoE
专家已经共用参数。当前 FiLM 位于完整 core 之后，这为本提案提供了清晰拆分点。

源码依据：

- `src/unity_psf/localization/soft_moe.py:33`：完整专家的复制。
- `src/unity_psf/localization/soft_moe.py:68`：双专家执行及 soft 混合。
- `src/unity_psf/localization/film.py:96`：core、FiLM、heads 的执行顺序。
- `src/unity_psf/localization/smlm_unet.py:472`：DoubleUNet core 的时间上下文处理。
- `.local/tmp/unitypsf_roi_gamma_resolved_5705.yaml:98`：5705 的训练输入与路由配置。

## 3. A2 的目标结构

```text
单通道三帧时间窗口 image
    ↓
Shared DoubleUNet core（只计算一次）
    ↓
features
    ├── branch FiLM 0(condition) → heads 0 → 原输出激活 → y0
    └── branch FiLM 1(condition) → heads 1 → 原输出激活 → y1

gate(condition) → w0, w1
y = w0 × y0 + w1 × y1
```

第一版应保留以下契约，避免同时改变多个实验变量：

- 原三帧时间窗口、单通道 batch、domain 条件和采样预算。
- 两套独立 FiLM、heads，以及原 soft gate；不混入 hard/top-1 routing。
- 每个分支先进行原输出非线性，再混合输出；不改成先混合 logits。
- 原 10-channel 输出顺序、target 单位、GMM 和背景损失及解码流程。
- 各通道独立的 crop、物理状态、gamma、标定和指标。

共享 backbone 收到两个分支回传梯度的合计，而不是只训练对应 domain 的分支。
共享参数意味着原先两套 backbone 的表达能力受到约束；即使初始化一致，原模型
训练后两套权重也可能不同。因此不能用缓存一套专家特征的方法声称与原模型等价。

共享的 BatchNorm 状态、dropout 随机性及两个 domain 的数据分布都需要单独检查。
一套 backbone 的统计状态与原两套状态不同，不应在本实验中再静默更换归一化。

## 4. 科学问题：共享训练不等于双通道观测融合

当前模型与 A2 都能利用两个通道的训练数据学习参数，但单次神经网络推理仍只
观察一个通道。可称为“跨通道共同训练／共享表示／条件化定位”，不能仅凭存在
两个分支声称“同一分子的双通道观测联合提高了定位精度”。共享是否有益也是
待验证假设，存在负迁移或通道特异信息被压缩的风险。

真正检验双通道互补信息，需要在同一 emitter 的配对观测上，将两路信息用于
一个联合定位估计。实现方式不限于神经网络 feature fusion：

1. 独立定位后配准、关联，再进行不确定性感知的联合位置估计。
2. 用两通道 raw image 和各自 PSF 做共享坐标的联合物理／概率 refinement。
3. 显式双通道输入的神经网络融合模型。

前两种同样可以联合利用双通道信息；不能将“没有网络融合”误写成“整个方法
无法利用双通道信息”。现有 ADR 0004 已选择 channel-linked 路线，A2 可作为
其单通道神经网络前端的候选，但不能替代关联和联合 refinement 阶段。

关联前提包括可靠配对、坐标配准、时间对应关系以及确实同时可见的分子。
两个通道若标记不相交结构，不能按同一 emitter 联合估计；共享权重也不会自动
解决这个问题。5705 本身的产物不能证明 channel-linked 阶段已经执行。

参考：`docs/adr/0004-channel-linked-refinement-without-network-fusion.md`、
`docs/architecture/channel-linked-self-calibrating-localization.md`。

## 5. 速度预期：条件估算，不是实测承诺

已有报告记录 30 epoch 速度基准为 15,000 steps、1,279.26 s，即 85.28 ms/step。
此前不同 profiling run 的量级包括 forward + loss 26–28 ms、backward 54–57 ms、
GMM loss 区块 4–10 ms、取 batch 6–10 ms、optimizer 数 ms。

这些不是同一个 run 的可加总分解；GMM forward 已包含在 forward + loss 中。
backward 包括网络与 GMM 的反向，现有数字不能分离两者，也不能据此断言所有
54–57 ms 都来自双 backbone。先前用不同 run 的分段值除以 85.28 ms 得出的
占比不能作为严格 profiling 结果使用。

令同一条件下原训练步时间为 `T0 = 2C + R`，其中 `C` 是一套独立 core 的
forward/backward 成本，`R` 是两个分支、GMM、搬运及其他不变成本。理想化的
A2 为 `T1 ≈ C + R`。若双 core 占原 step 的比例为 `f = 2C/T0`，则：

```text
时间缩短比例 ≈ f / 2
吞吐加速倍数 ≈ 1 / (1 - f / 2)
```

| 假设双 core 占原 step | 理想时间缩短 | 理想吞吐倍数 | 用 85.28 ms 作示例的 T1 |
| --- | --- | --- | --- |
| 50% | 25% | 1.33× | 63.96 ms |
| 70% | 35% | 1.54× | 55.43 ms |
| 90% | 45% | 1.82× | 46.90 ms |

表格是敏感性分析，不是对实际占比的测量，也不代表保证得到该速度。实际共享
core 的 backward、两分支梯度累加、GPU kernel 效率和 optimizer 成本会变化。
在只去掉一套同宽 core 的理想模型中，收益趋近 2×；这不是硬件层面的严格上限。

job 5705 的约 277.5 ms/step 是整次运行摊销，包含 gamma、评估、诊断和写盘。
不能乘上述倍数来承诺整次 300 epoch 的 wall time，也不能把 batch 10 的
85.28 ms 基准直接视为 batch 64 的 5705 单步时间。

来源：`docs/performance/unitypsf-v03-liteloc-training.md`、
`docs/performance/slurm-5705-full-training-report.md`。

## 6. 实验与验收

工程比较先固定 GPU、软件环境、batch size、输入大小、target 数量分布、dtype、
AMP、cache、损失、optimizer 和训练步数；只改变 backbone 共享关系。

1. 保留原双完整专家 soft-MoE 作为 baseline。
2. 实现 A2 独立模型入口及独立 checkpoint schema，不覆盖正式路线。
3. 使用已物化 batch 和充分 warm-up，在 GPU 上比较 forward、loss、backward、
   optimizer、端到端 step 中位数／高分位数，以及峰值显存。
4. 单独记录 core 和 heads 成本，确认节省来源；不能仅比较 FLOPs／参数量。
5. 按原训练预算比较完整 wall time、收敛曲线及达到相同质量门槛所需时间。

科学比较使用多随机种子、固定 held-out 条件，分别报告 left/right 的 precision、
recall、Jaccard、横向／轴向误差和不确定性校准；不得只报告两通道平均。
质量容忍阈值应在正式实验前明确，速度更快但某个通道明显退化不算通过。

双通道信息收益另做配对数据对照：单通道定位、配准后的独立结果、关联联合
坐标估计、可选 raw-pixel 联合 refinement。保持 emitter、光子预算和测试数据
可比，不把更多总光子或筛选更容易的 matched emitters 混同为架构收益。

## 7. 当前决定

仅保存提案，不改训练代码、不启动实验、不宣称速度或精度增益已得到验证。
优先测试 A2 是否减少单步计算且保持分通道质量；双通道科学结论仍以配对观测
的联合估计对照为依据。GMM 分块、自定义反向、kernel 融合和同步优化属于
独立实验变量，不在第一轮 A2 对照中同时引入。
