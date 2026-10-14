# UnityPSF 生物实验调研简报

- 文档用途：让负责文献检索和实验设计的 AI 准确理解 UnityPSF，并调研最能验证其方法价值的生物实验
- 文档状态：目标模型说明与调研任务书，不是已经取得的实验结论
- 日期：2026-08-05
- 完整工程蓝图：[UnityPSF 按模态路由、多通道目标架构](../architecture/unitypsf-modality-routed-multichannel-target-architecture.md)
- 实施规划：[按模态路由的多通道修正规划](../plans/unitypsf-modality-routed-multichannel-correction-plan.md)

> 重要：本文的“最终 UnityPSF”指改造和科学验收完成后的目标模型。当前工程已经具备
> 多 expert、显式路由、joint checkpoint、双模态训练和报告等基础能力，但仍在把旧的
> `(modality, channel_id)` 完整网络实例改为“一个模态一个 expert、通道只选择该 expert
> 内部物理上下文”。Double Helix 真实数据和 calibration 也尚未完成科学验收。

## 1. 一句话定义

UnityPSF 是面向单分子定位显微成像（SMLM）的稀疏多专家模型：它在一个模型对象和一个
checkpoint 中容纳 2D emitter、astigmatism 和 double-helix 三类完整定位 expert，根据
PSF 模态每次只激活其中一个，并在每个模态内部联合处理具有独立物理校准的多个测量通道。

它希望解决的问题不是“再训练一个定位网络”，而是：

> 能否用一个统一模型覆盖多种物理形态明显不同的 PSF，同时保留各模态的专业定位能力、
> 物理校准和解码规则，从而形成一个可复现、可扩展的多模态 PSF 基础模型？

## 2. 为什么需要 UnityPSF

现有 SMLM 工作流通常围绕某一种 PSF 单独建立：

- 一套数据预处理；
- 一个定位模型和 checkpoint；
- 一套 calibration、物理参数和解码器；
- 一套训练、评估和推理脚本。

实验从 2D PSF 切换到 astigmatism 或 double-helix PSF 时，往往同时切换模型、配置、
校准文件和推理入口。多测量通道又会进一步增加独立模型和版本的数量。这带来三个问题：

1. 不同 PSF 的模型和物理状态难以统一管理与复现；
2. 同一模态的多个通道被当成互不相关的任务，无法利用共同的成像规律；
3. 新增 PSF 模态时，通常需要重新建立一套外围工具链。

UnityPSF 的目标是把这些能力收敛到一个稳定接口和一个自包含的发布 checkpoint 中，同时
避免用一个完全共享的 backbone 强行拟合差异很大的 PSF 几何。

## 3. 最终模型结构

```text
输入图像 + modality + channel_id + channel physical condition
                           |
                           v
                ModalityRouter
          deterministic sparse top-1 route
              /             |             \
             /              |              \
            v               v               v
 Emitter2DExpert   AstigmatismExpert   DoubleHelixExpert
 preprocessing     preprocessing       preprocessing
 full backbone     full backbone       full backbone
 FiLM              FiLM                FiLM
 2D heads          x/y/z heads         common + lobe heads
 2D decoder        astig decoder       DH physics/decoder
             \              |              /
              \             |             /
               v            v            v
              Unified SMLM Output Contract
```

一次 forward 只执行一个 expert。三个 expert 是三套完整定位网络，不是一个共享 backbone
后面挂三个小输出头，也不是三个预测结果的 dense ensemble。

### 3.1 顶层路由

顶层只使用 `PSFModality` 选择 expert：

```text
emitter_2d   -> Emitter2DExpert
astigmatism  -> AstigmatismExpert
double_helix -> DoubleHelixExpert
```

第一版正式模型使用采集 metadata 或配置中的显式模态标签。自动从 raw TIFF 判断 PSF
模态是后续能力，不是当前方法成立的前提。路由错误时必须显式失败，不能静默切换 expert。

### 3.2 通道不是 expert

`left`、`right`、`main` 表示同一 PSF 模态下的测量通道，不表示新的 PSF 模态，也不应
选择另一套完整网络：

```text
router.resolve(modality) -> one modality expert
expert.resolve_channel(channel_id) -> one channel physical context
```

同一模态的 left/right：

- 共享完整 backbone、FiLM、heads、optimizer 和 scheduler；
- 共同参与该模态 expert 的反向传播；
- 各自保留 crop、anchor、peak-zmap、gamma/coefficient map、calibration、condition
  provider、physical version 和完整性 hash；
- 可以从相同初始参数开始，但训练后仍属于同一个共享模态网络，而不是两个独立网络。

这里的 left/right 是测量通道，不应被调研 AI 自动解释为两种荧光颜色。它可以来自分光、
双视场、双臂光路或其他实验布局；是否对应多色成像必须根据具体数据和光路确认。

### 3.3 FiLM 的作用

每个 expert 内部保留 FiLM。FiLM condition 可以携带与样本、空间位置和通道相关的物理
信息，例如 Zernike/field condition、anchor profile、calibration 或 channel identity。

因此网络既看到图像，也知道该图像对应的物理成像条件。不同通道共享网络参数，但通过
不同 condition 和物理状态保留真实光路差异。

## 4. 三个 expert 分别做什么

### 4.1 Emitter2DExpert

- 面向没有有效轴向编码的常规 2D emitter PSF；
- 使用完整 localization backbone、FiLM 和独立输出 heads；
- 预测检测概率、x/y、光子数和背景等公共量；
- `z` 不作为有效定位结果，z loss 不参与反向传播；
- 单通道或 left/right 多通道数据共同训练同一个 2D expert；
- 合理物理起点是 zero-aberration focal PSF，之后由真实 calibration 修正。

### 4.2 AstigmatismExpert

- 面向通过 x/y 方向宽度变化编码轴向位置的 astigmatism PSF；
- 默认从已经验证的 `Z(2,2)=99 nm` anchor 配置开始；
- 预测检测概率、x/y/z、光子数、背景和不确定度；
- left/right 分别从自己的 raw crop 构建 peak-zmap；
- left/right 分别维护 gamma、coefficient map、calibration 和物理版本；
- 所有通道共同更新一套 AstigmatismExpert 网络参数。

### 4.3 DoubleHelixExpert

- 面向利用双叶旋转角和几何关系编码 z 的 double-helix PSF；
- 使用独立完整 backbone 和 FiLM；
- 除公共定位量外，预测 lobe angle、lobe separation 等辅助量；
- 使用专门的 DH calibration、physics 和 angle-to-z decoder；
- 每个测量通道保留独立 calibration、有效 z 范围和置信度边界；
- calibration 缺失、超出有效范围或置信度不足时应拒识；
- 只有真实 DH 数据和 calibration 通过科学验收后，才能宣称三模态支持。

## 5. 输入、输出和 checkpoint

### 5.1 输入

模型推理的最小语义输入是：

```text
images: temporal image frames, typically (N, T, H, W)
modality: emitter_2d | astigmatism | double_helix
channel_id: main | left | right | custom
conditions: channel- and position-specific physical condition
```

### 5.2 统一输出

所有 expert 映射到统一的 SMLM 输出契约，至少包括：

- 检测概率 `p`；
- x/y/z 坐标及其不确定度；
- photon 和 background；
- modality、channel 和 calibration 语义；
- 模态特有辅助量，例如 DH lobe geometry；
- 对 2D expert 明确标记 `z_valid=false`。

### 5.3 一个 joint checkpoint

正式发布物是一个 `unitypsf_joint.ckpt`。它自包含：

- 所有已经验收的模态 expert 参数；
- 每个模态支持的 channel 能力；
- 各通道独立物理状态和 calibration；
- checkpoint schema、版本、来源和完整性 hash；
- 必要的训练和评估 provenance。

双模态 checkpoint 只应包含两套完整网络，三模态 checkpoint 只应包含三套完整网络；不会
因为 left/right 而复制完整 expert。

## 6. 训练方式

### 6.1 模态内多通道联合训练

每个模态只有一个训练 runtime、一个 optimizer 和一个 scheduler。left/right batch 可以
按 step 或 sequence 交替，也可以在明确边界处组成混合 batch，但每个 epoch 必须覆盖所有
启用通道并记录每个通道的样本数、step 数和指标。

所有通道的 loss 都更新同一套模态网络参数；物理状态更新只能写入对应通道自己的 context。
第一版不强制加入跨通道 consistency loss，以免引入未经验证的物理假设。

### 6.2 多 GPU Expert Parallel

训练时可以让不同 GPU 分别负责不同 PSF 模态，例如：

```text
GPU 0 -> Emitter2DExpert(left + right)
GPU 1 -> AstigmatismExpert(left + right)
GPU 2 -> DoubleHelixExpert(left + right)
```

每张 GPU 在负责的模态内联合训练其全部通道，最后由控制面组装一个 joint checkpoint。
Expert Parallel 是提高训练效率和避免模态相互等待的工程策略，不是 UnityPSF 的主要科学
创新，也不等于不同 expert 之间共享梯度或表征。

## 7. 它为什么属于 MoE，以及它不是什么

UnityPSF 符合稀疏 MoE 的关键定义：

- 有多个具有不同专长的完整 expert；
- router 为每个请求选择 expert；
- 每次只激活一个 expert；
- 未选中的 expert 不参与 forward，也不产生梯度；
- expert 的专业化边界对应明确的 PSF 物理模态。

但需要诚实说明：

- 第一版是 metadata 驱动的确定性 hard routing，不是学习型 router；
- 不同模态 expert 之间不共享完整 backbone，也没有跨 expert 的表征迁移；
- 一个 joint checkpoint 本身不自动证明多模态联合学习提高了定位精度；
- Expert Parallel 只并行训练各 expert，不意味着 expert 之间进行了 DDP 式参数同步；
- “基础模型”是项目目标，必须通过跨样本、跨设备或跨条件泛化证据来支撑，不能仅凭模型
  命名或 checkpoint 封装宣称。

因此，UnityPSF 最稳健的第一层科学主张是：

> 在一个稀疏路由模型和单一 checkpoint 中统一管理多类 PSF 的专业定位能力，并保持与
> 单模态专用模型相当的性能、清晰的物理边界和更好的实验工作流一致性。

更强的主张，例如“多模态训练提高未知样本泛化”或“共享通道网络提高定位精度”，都需要
专门消融和生物实验支持。

## 8. 当前真实状态

调研 AI 必须区分目标模型和当前结果：

| 能力 | 当前状态 | 可以如何表述 |
| --- | --- | --- |
| 三类 expert contract 和物理边界 | 已规划，部分工程实现 | 已定义目标架构，不能说三模态已完成 |
| joint checkpoint、显式路由和回载 | 已有工程基础 | 已通过工程 smoke，不等于科学性能验收 |
| Emitter2D + Astigmatism 双模态 | 已有训练链路 | 正在改为模态级多通道共享网络并重训 |
| left/right 多通道 | 旧代码仍按通道复制完整网络 | 目标是同模态共享网络、物理状态独立 |
| Double Helix 物理和 calibration 基础 | 已迁移部分模块 | 真实 DH 数据、完整 expert 和科学验收未完成 |
| held-out 科学评估 | 尚不完整 | 训练 loss 不能作为定位性能结论 |
| 自动 PSF 模态检测 | 后续扩展 | 第一版依赖显式 metadata 路由 |

SLURM 作业 `4525` 使用 left/right 四个独立完整网络训练，只能作为
`independent-channel ablation`，不能代表最终 UnityPSF 架构。

## 9. 生物实验需要验证的科学命题

候选实验不必同时覆盖全部命题，但主实验至少应直接支持两个：

1. **专业能力保留：** joint model 中每个 expert 是否达到对应单模态专业模型的定位质量？
2. **多 PSF 通用性：** 同一个 checkpoint 是否能可靠处理 2D、astigmatism 和 DH 数据？
3. **轴向范围互补：** astigmatism 与 DH 是否在同一分析体系中覆盖不同深度和光子条件？
4. **模态内多通道价值：** 共享模态网络是否优于 left/right 完全独立训练，至少在样本效率、
   鲁棒性或跨通道稳定性中的一项成立？
5. **物理 conditioning 价值：** FiLM physical condition 是否降低场依赖误差或 calibration
   shift，而不是只增加模型复杂度？
6. **泛化：** 模型能否跨细胞、成像日期、视野、标记密度或显微镜保持性能？
7. **工作流价值：** 单 checkpoint 是否降低部署、模型选择、版本管理和复现成本？
8. **失败边界：** 错误 modality、calibration 失配、低信号、过曝和未知 PSF 时能否拒识？

## 10. 生物实验调研的优先方向

下表只是检索起点。调研 AI 应依据文献证据重新排序、删除弱候选并增加更强候选。

| 候选对象 | 可能验证的价值 | 可能使用的 PSF | 可量化结构或生物读出 |
| --- | --- | --- | --- |
| 微管网络 | 连续结构、不同深度、现有 SMLM 基线丰富 | 2D、astigmatism、DH | 线宽、连续性、分叉、轴向偏差 |
| 核孔复合体 | 已知重复几何可作为精度 proxy | 2D、astigmatism，按需要加入 DH | 环直径、圆度、双环间距 |
| 网格蛋白包被小窝 | 从近二维到三维弯曲形态 | 2D、astigmatism | 直径、曲率、深度、形态分类 |
| 线粒体与内质网接触位点 | 厚三维区域和多结构空间关系 | astigmatism、DH、多通道 | 接触距离、网络连续性、轴向覆盖 |
| 突触纳米组织 | 高密度、轴向分层和多靶标关系 | 2D、astigmatism、DH、多通道 | 簇大小、相对位移、层间距离 |
| 膜受体聚簇或内吞 | 2D 高通量与 3D 状态变化 | 2D、astigmatism | cluster size、density、深度分布 |
| 活细胞细胞器或膜动力学 | 时间分辨率、光毒性和部署速度 | 2D、astigmatism；DH 需论证光子预算 | 定位率、轨迹/结构连续性、动态时间尺度 |

候选实验只有“可以使用 SMLM”还不够。它必须说明为什么需要多 PSF、为什么需要多通道，
或者为什么统一 checkpoint 能改变真实实验结论或可复现性。

## 11. 实验优先级判断标准

每个候选按 1-5 分评分，并写出证据来源：

| 维度 | 需要回答的问题 |
| --- | --- |
| 生物问题强度 | 是否能产生重要生物结论，而不只是漂亮重建图？ |
| 多 PSF 必要性 | 两种或三种 PSF 是否提供单一 PSF 无法提供的信息？ |
| 多通道必要性 | 多通道是否来自真实实验需求，而不是人为增加复杂度？ |
| 定量 proxy | 是否有已知几何、DNA origami、beads 或独立测量用于判断偏差？ |
| 数据可得性 | 是否容易获得原始 TIFF、标记、细胞系和 calibration 数据？ |
| 光子预算 | 分光或 DH 后是否仍能可靠定位？ |
| 配准风险 | 通道和 PSF 模态之间能否可靠配准并报告误差？ |
| 实施周期 | 两周内能否完成有判别力的 pilot？ |
| 论文说服力 | 能否同时证明方法正确、通用并有生物价值？ |
| 失败可解释性 | 阴性结果能否区分数据、光学、路由和模型问题？ |

## 12. 必须设置的 baseline 和消融

调研方案至少考虑：

1. 每种 PSF 的领域内专业单模态模型或传统软件；
2. 与 UnityPSF expert 同架构、同数据量的独立单模态 checkpoint；
3. left/right 各自训练完整网络的 independent-channel baseline；
4. 同模态 left/right 共享网络的目标 UnityPSF expert；
5. 去除 FiLM physical condition 的消融；
6. 一个 checkpoint 与多个分散 checkpoint 的工作流和复现性比较；
7. 在适用范围内加入非深度学习定位方法。

所有比较必须使用相同数据 split、检测阈值选择规则、matching protocol 和后处理。数据应按
细胞、视野或采集批次切分，避免相邻帧或同一结构泄漏到训练集和测试集。

## 13. 评价指标应该分三层

### 13.1 定位和重建指标

- detection precision、recall、Jaccard/F1；
- x/y/z bias、RMSE 和不确定度校准；
- localization density、重复检测率和漏检率；
- 不同光子数、背景、密度和视野位置下的性能曲线；
- 推理显存、吞吐量、单模态激活参数量和 checkpoint 大小。

### 13.2 结构指标

- 线宽、环直径、圆度、层间距、曲率、cluster size 等对象特异指标；
- 同一结构跨 PSF、跨通道和跨日期的重复性；
- 结构连续性、伪影率和轴向覆盖范围；
- 与已知几何或独立成像测量的一致性。

### 13.3 生物学读出

- UnityPSF 是否改变或增强了可回答的生物问题；
- 生物效应量、置信区间、样本间变异和统计功效；
- 结果是否在独立细胞、独立制备和独立采集日期重复；
- 多 PSF 或多通道是否带来单一 PSF 无法获得的结论。

训练 loss 不能替代上述任一层 held-out 指标。

## 14. 推荐的验证层级

```text
Level 1: beads / calibration / geometry reference
         验证定位偏差、通道物理状态、PSF 模态和 calibration

Level 2: fixed-cell known structure
         验证真实背景、标记密度、结构连续性和重复性

Level 3: one biological question requiring PSF or channel complementarity
         验证 UnityPSF 相对多个独立工具链的实际科学价值

Level 4: microscope/date/sample/label shift
         验证“基础模型”和 physical conditioning 的泛化主张
```

第一篇方法工作至少应完成 Level 1、Level 2，并提供一个可信的 Level 3。Level 4 可以分阶段
完成，但没有任何跨域结果时应谨慎使用“基础模型”这一强表述。

## 15. 外部 AI 必须交付什么

调研结果需要给出至少 8 个候选实验的排序表。每个候选必须包含：

- 生物问题及其重要性；
- 推荐样本、细胞系、标记靶点、探针和固定/活细胞条件；
- 推荐 PSF 模态，以及为什么需要这些模态；
- 是否需要 left/right、分光或多色通道；
- 光路、calibration、配准和 ground-truth proxy；
- 所需 raw TIFF、metadata、训练/验证/测试规模和生物重复数；
- 领域内专业 baseline、消融和公平评价协议；
- 定位、结构和生物学三层指标；
- UnityPSF 可能带来的贡献；
- 主要失败模式和替代方案；
- 两周 pilot 的最小实施方案；
- 完整实验所需设备、试剂、采集时间、计算量和预计周期；
- 关键参考文献、DOI 或稳定 URL；
- 文献事实、合理推断和待验证假设的明确区分；
- 总体优先级和评分理由。

最后必须明确推荐三个实验：

1. 最稳妥的方法学/几何验证；
2. 最有生物学价值、最适合作为主结果的实验；
3. 最能证明 Double Helix 扩展轴向范围不可替代的实验。

同时回答：

1. 哪个实验最能证明“一个模型处理多种 PSF”不是简单工程包装？
2. 哪个实验最能证明“模态内多通道共享网络”优于独立通道网络？
3. 哪个实验真正需要 DH，而不是 astigmatism 已经足够？
4. 哪些实验只能证明定位精度，不能证明生物学价值？
5. 哪些方案可以利用现有 Origami 2D 和 astigmatism 数据立即开始？

## 16. 可直接交给其他 AI 的完整调研提示词

```text
你是一名熟悉单分子定位显微成像、PSF engineering、生物样本制备和定量实验设计的科研
顾问。请根据下面的模型定义，系统调研最能验证 UnityPSF 方法学和生物学价值的实验。

UnityPSF 是面向 SMLM 的稀疏多专家模型。最终模型只发布一个 unitypsf_joint.ckpt，并
包含三个完整、相互独立的定位 expert：Emitter2DExpert、AstigmatismExpert 和
DoubleHelixExpert。顶层 ModalityRouter 根据显式 PSF modality 做确定性 sparse top-1
路由，每次只执行一个 expert。第一版不是学习型 router，也不是共享 backbone 后接三个小
head，更不是同时运行三个 expert 的 ensemble。

每个 expert 内部都有自己的 preprocessing、完整 localization backbone、FiLM、heads、
loss 和 decoder。Emitter2DExpert 只提供有效 x/y 定位；AstigmatismExpert 使用像散编码 z；
DoubleHelixExpert 还具有 lobe angle、lobe separation、DH calibration、physics 和
angle-to-z decoder。

同一 PSF 模态可以有 main、left、right 等测量通道。channel_id 不选择另一套完整网络，
而是在选中的模态 expert 内选择物理 context。同一模态 left/right 共享 backbone、FiLM、
heads 和 optimizer，共同反向传播；但各自保存 crop、anchor、peak-zmap、gamma/coefficient
map、calibration、condition provider 和 physical version。left/right 是测量通道，不一定
等于两种荧光颜色，必须依据具体光路判断。

FiLM 向网络注入与通道、空间位置和 calibration 相关的物理 condition。训练可以使用
Expert Parallel，让不同 GPU 分别训练不同模态，再组装一个 joint checkpoint；这只是训练
策略，不代表不同模态 expert 共享梯度或表征。

当前状态必须与目标区分：2D emitter 和 astigmatism 已有工程训练链路，但模态内多通道
共享网络改造仍待完成和重训；现有 job 4525 是 left/right 独立完整网络，只能作为消融；
Double Helix 物理模块已部分迁移，但真实 DH 数据、完整 calibration 和科学验收尚未完成；
当前训练 loss 不是 held-out 科学指标。现有 Origami 2D 数据和 astigmatism 数据可以先用于
双模态 pilot，之后再扩展 DH。

请不要泛泛列出“可以用 SMLM 观察”的样本。候选实验必须真正检验至少一项：
1. 一个 checkpoint 保留三种 PSF 专业模型的能力；
2. 2D、astigmatism 和 DH 在轴向范围、光子预算或生物问题上的互补；
3. 同模态多通道共享网络相对独立通道网络的样本效率、鲁棒性或稳定性；
4. FiLM physical conditioning 对场依赖像差或 calibration shift 的价值；
5. 跨细胞、视野、日期、标记或显微镜的泛化；
6. 错误模态、calibration 失配、低信号和未知 PSF 的拒识；
7. 单 checkpoint 对复现性和实验工作流的实际改善。

请至少调研并排序 8 个候选。优先检索但不限于：微管网络、核孔复合体、网格蛋白包被
小窝、线粒体-内质网接触位点、突触纳米组织、膜受体聚簇或内吞、活细胞细胞器动力学。
如果文献证据表明某个候选不适合，请删除并说明原因；如果存在更强候选，请替换。

对每个候选请给出：生物问题；样本/细胞系/靶点/探针；固定或活细胞条件；推荐 PSF；
通道需求；为什么单一 PSF 或独立模型不够；光路和 calibration；ground-truth proxy；原始
数据与 metadata；最小训练、验证、测试规模和生物重复；专业 baseline 与消融；定位、结构
和生物学三层指标；主要风险；两周 pilot；完整实验成本和周期；关键文献 DOI/URL。

比较必须采用相同数据 split、阈值选择、matching protocol 和后处理，并防止同一细胞、
视野或相邻帧跨集合泄漏。请明确区分文献事实、合理推断和待验证假设，不得把 UnityPSF 的
目标能力写成已经证明的结果。

最后请给出三个明确推荐：
A. 最稳妥的方法学/几何验证；
B. 最有生物学价值的主实验；
C. 最能体现 Double Helix 不可替代性的扩展实验。

另外请明确判断：哪个方案可以利用当前 Origami 2D + astigmatism 数据立即启动，哪些必须
等待真实 DH 数据；哪个实验最能说服审稿人 UnityPSF 是多模态 PSF 模型，而不只是把多个
网络装进同一个 checkpoint。
```

## 17. 调研结论的使用边界

外部 AI 的输出用于形成候选实验和文献依据，不能替代以下工作：

- 对实验室实际显微镜光路、激光、相机和滤光片配置的核对；
- 对可获得样本、标记效率、光毒性和光子预算的实测；
- 对原始数据许可、患者/动物/细胞伦理和生物安全要求的确认；
- 与具体生物方向合作者共同确认科学问题和效应量；
- 用 pilot 数据做功效分析后再决定正式样本量。

最终实验应由“文献证据 + 设备可行性 + 数据可得性 + pilot 结果”共同决定，而不是只依据
语言模型给出的候选排序。
