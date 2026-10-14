# UnityPSF v0.3 可选择模态与通道的训练方案

状态：CLI 选择器已实现并通过单模态、双模态与双通道 GPU smoke；长期科学质量验证仍需正式训练。实际入口为 `unity-psf-train plan|smoke|run|resume`，通过重复 `--route` 指定模态和通道。范围：训练配置、启动、物理输入预检、验证和发布合同；本方案不执行 DH 校准，也不改变已完成实验的结果。

## 1. 目标与术语

用户应能选任意非空模态集合：`emitter_2d`、`astigmatism`、`double_helix`。2D 和 Astigmatism 各支持单通道或双通道训练：可选 `left`、`right` 或 `[left, right]`；默认正式方案使用两个通道。DH 保持单通道，只接受 `main`。单独训练时也产出可由 `UnityPSF.from_checkpoint` 加载的统一 checkpoint。

“通道”在用户入口中专指相机的测量通道，如 `main`、`left`、`right`。`online_generation.channels: 3` 是连续输入帧数，必须仍由 `input_frame_spec.input_frame_channels` 管理，不暴露为测量通道数。现有一致性检查在 [runtime/config.py](../../src/unity_psf/localization/runtime/config.py:509)。

期望的物理初始化如下：

| 模态 | 默认初始化 | 外部输入 | 前置步骤 |
| --- | --- | --- | --- |
| 2D | **圆形 PSF、50 nm 光斑半径 anchor**；需要样本系数图时可显式覆盖 | 默认 anchor-only；`--map` 可替换为指定通道的 sample coefficient map | 不要求 beads 校准 |
| Astigmatism | 当前 660 nm、100 nm 像散 anchor | anchor 本身无需外部文件；可显式选择样本生成的系数图方案 | 不要求 beads 校准 |
| DH | beads 校准产生的 **PPG/PCA 图** | `main` 通道的一份 `ppg.pca_map.v1` 文件及其校准来源记录 | 必须先完成 beads 校准并通过质量门禁 |

这里的 2D anchor 是显式的 `circular_2d_50nm` profile，光斑半径以 Gaussian sigma 等效值记录为 50 nm；它不再使用未命名的 `zero` profile。需要样本场时通过 CLI 的 `--map modality:channel=PATH` 显式选择。Astigmatism 的默认 `astigmatism_660nm` profile 为 660 nm、100 nm。两者的 anchor 来源和数值都会写入生成的计划及 checkpoint 元数据。

## 2. 已核实的现状

1. 联合入口 `_instance_specs` 强制同时出现 2D 和 Astigmatism；`train-modality-joint` 也复用该解析器，因而单模态、2D+DH、Astigmatism+DH 不能由该入口启动。[入口](../../src/unity_psf/training/entrypoints/train_joint.py:60)、[模态联合入口](../../src/unity_psf/training/entrypoints/train_modality_joint.py:568)。
2. 正式运行审计固定要求 2D/Astigmatism 各有 `left/right`，DH 有 `main`。[审计](../../src/unity_psf/training/entrypoints/train_modality_joint.py:190)。旧的 2D 单通道正式 YAML 虽存在，按当前模态联合正式审计仍不构成可用的单通道正式路线；目标方案要把单通道作为明确的选择合同，而不是复制双通道配置后隐式丢弃另一侧。
3. Expert Parallel 按模态分组，`world_size` 必须等于所选模态数；每个 rank 内可训练该模态的多个通道。这部分可复用。[入口](../../src/unity_psf/training/entrypoints/train_modality_expert_parallel.py:556)。
4. 当前三模态正式实验的 2D/Astigmatism 引用每通道样本生成的系数图；正式审计还强制文件存在、记录哈希。[2D 配置](../../configs/modalities/emitter_2d/emitter_2d_dual_channel_300epoch.yaml:131)、[Astigmatism 配置](../../configs/modalities/astigmatism/astigmatism_dual_channel_300epoch.yaml:130)、[审计](../../src/unity_psf/training/entrypoints/train_modality_joint.py:336)。因此“无需 beads”与“当前正式配置完全无需外部图”不是同一句话。
5. 物理模拟器已有无系数图时的 2D 零像差和 Astigmatism anchor 生成路径，但现有无图 smoke 配置是 CPU 小尺寸验证；正式 CUDA 训练尚需单独验证。[online.py](../../src/unity_psf/localization/data/online.py:1691)、[2D smoke](../../configs/modalities/emitter_2d/emitter_2d_single_channel_smoke.yaml:65)、[Astigmatism smoke](../../configs/modalities/astigmatism/astigmatism_single_channel_smoke.yaml:67)。
6. 当前三模态 DH 实例指向 vector CUDA 系数图和 complex pupil；它**不是 PCA 路线**。[当前 DH 配置](../../configs/modalities/double_helix/double_helix_raw_tiff_300epoch.yaml:44)。仓内另有 `ppg_direct` 配置、`ppg.pca_map.v1` 渲染器和 beads→PCA 生成器。[PPG 配置](../../configs/modalities/double_helix/ppg_physical.yaml:1)、[PCA 生成器](../../src/ppg/beads_initialization.py:284)、[渲染器](../../src/unity_psf/localization/simulator.py:466)。目标路线应显式切换到这一合同，不能把 vector 图改名为 PCA 图。
7. 正式训练由 `unity-psf-train` 解析选择合同并动态分配 CUDA rank；旧的固定三模态 raw-TIFF 启动脚本已移除。`formal_gate` 仅保留为历史输入和 GPU 收据工具，不再拥有正式训练入口。
8. v2 模态 checkpoint 元数据要求模态集合非空、各模态通道集合非空；分片组装也接受任意非空模态集合。限制主要在入口和正式合同，而不在 checkpoint 结构。[checkpoint 合同](../../src/unity_psf/contracts/modality_joint_checkpoint.py:67)、[组装](../../src/unity_psf/training/modality_joint.py:555)。

## 3. 用户入口与配置合同

早期设计考虑 YAML 训练选择规格；当前按用户决定实现 CLI 参数入口。以下 YAML 仅为设计阶段的规格草图，不作为用户入口：

```yaml
schema_version: unitypsf.training_selection.v1
run:
  epochs: 300
  execution: expert_parallel
  output: output/unitypsf
modalities:
  emitter_2d:
    channels: [left, right]
    physics:
      anchor: circular_2d_50nm
      radius_nm: 50.0
      field_source: anchor_only
  astigmatism:
    channels: [left, right]
    physics:
      anchor: astigmatism_660nm_anchor
      field_source: anchor_only
```

选择 DH 时增加其 beads/PCA 输入；没有选择 DH 时不出现 DH 字段：

```yaml
  double_helix:
    channels: [main]
    physics:
      initialization: beads_ppg_pca
      ppg_map: project://.local/calibration/dh/ppg_map_initial.npz
      calibration_manifest: project://.local/calibration/dh/manifest.json
```

DH 的 `channels` 固定只能是 `[main]`；`[left]`、`[right]` 或两个通道都应在计划阶段拒绝。2D 和 Astigmatism 的单通道选择沿用同一模态模型，只 materialize 被选择的通道状态；双通道选择则保留 left/right 的独立 crop、conditioning、物理状态和指标。

2D 默认使用 `circular_2d_50nm` anchor-only 物理场；`--map` 可显式覆盖单个通道为样本系数图。Astigmatism 默认使用固定 100 nm anchor，也可用 `--map` 显式选择样本图。系数图是绝对值，不与固定 anchor 叠加。

已实现的命令为：

```bash
unity-psf-train plan --route emitter_2d:left,right --route astigmatism:left,right --run-id experiment-001
unity-psf-train smoke --route emitter_2d:left,right --run-id smoke-001
unity-psf-train run --route emitter_2d:left,right --route astigmatism:left,right --run-id experiment-001
unity-psf-train resume --route emitter_2d:left,right --route astigmatism:left,right --run-id experiment-001
```

`plan` 只解析和预检，打印选中实例、物理初始化、所需文件、GPU 数、每通道 step budget、输出目录及配置签名，不申请 GPU、不创建训练结果。`run` 和 `smoke` 使用相同解析结果；smoke 仅缩小训练预算。运行时产生不可变的 resolved 配置和输入哈希清单。不得让启动脚本重写训练语义。

### 配置所有权

- 实验规格拥有模态集合、测量通道集合、执行方式、种子和预算。
- 模态配置拥有网络、loss、光学参数、anchor 类型及允许的物理输入类型。
- 通道绑定拥有 crop、相机参数和该通道可选的样本图路径。
- DH 校准清单拥有 beads 数据来源、PCA 文件、schema、质量指标和哈希。
- 启动器只处理资源、运行目录和命令；不硬编码某个模态。

解析阶段从所选模态和通道生成现有 `instances` 结构。例如 `2D:[left,right] + Astigmatism:[left,right]` 生成四个通道状态、两个模态 rank；`DH:[main]` 始终只有一个通道状态。执行路径继续调用现有 modality-owned trainer；单模态也产出同一 v2 checkpoint schema，而不是旁路到普通实例 checkpoint。单进程与 Expert Parallel 的模型、路由、物理状态及发布合同应一致。

## 4. 物理输入与校准门禁

### 2D

默认使用 `circular_2d_50nm` anchor-only 物理场，并在 resolved 配置和
checkpoint provenance 中记录 `radius_nm=50.0`。需要样本校准时，仅要求本次选中通道的图，记录路径与 SHA256；保留 2D z head/z loss 关闭和 nuisance z 范围合同。

### Astigmatism

默认以 `astigmatism_660nm` 的 100 nm anchor 初始化。`anchor_only` 无需预校准文件；选择 `anchor_then_sample_map` 时每个通道绑定独立图。当前项目已有关于 fixed anchor 与 residual map 重复叠加的风险记录；必须在 resolved 配置、物理状态和导出物中记录图是否已含 anchor，禁止重复相加。[现有分析](../anchor_physical_update_metrics_summary.md#2-anchor-语义尚未完全统一)

### DH

默认 `beads_ppg_pca`：先用 beads 生成 `ppg.pca_map.v1`，再在 `main` 通道训练中使用 PCA mean、components 和空间 coefficient field；FiLM 条件字段为 `field_x/field_y/ppg_pca_0..3`。预检核对文件 schema、必要数组形状与有限值、beads 校准清单、质量门禁和 SHA256。DH 不建立 left/right 双通道合同；将来若出现其他 DH 采集侧，应作为新的显式模态合同，而不是扩展本方案的 DH 通道列表。当前 vector+pupil 方案作为显式 legacy 方案保留，不作为新入口默认值，也不能与 PCA 输入混用。

输入预检应从**已解析的所选实例及其物理方案**收集引用文件；没有选 DH 不检查 beads/PCA。输入签名覆盖 resolved 配置和所有实际使用的外部文件，包括 `ppg_map`，而非仅覆盖路径中带 `formal-inputs` 的文件。resume 时必须核对签名、模态集合、通道集合、物理方案与 checkpoint 元数据。

## 5. 执行、产物与验证

Expert Parallel 保持“一模态一 rank”，因此所需 GPU 数等于本次选择的模态数；一个模态的多个通道在其 rank 内共享模型和优化器。单 GPU 单模态、双 GPU 双模态、三 GPU 三模态均有各自 smoke。单进程入口可在一张 GPU 上顺序训练多个模态，但应使用同一规格与发布合同。GPU 数不足时在启动前报错。

所有运行统一写出：resolved 配置、输入清单与哈希、每模态每通道指标、rank 身份、resume 分片、v2 release checkpoint、已训练路由清单。checkpoint 的 `supported_modalities` 和 `supported_channels_per_modality` 必须恰好等于本次选择；推理请求未训练的路由应明确失败。既有 v1/v2 checkpoint 的加载语义不因新入口而改变。

正式门禁只审核本次选择的路由和物理输入。smoke 收据绑定训练签名、checkpoint 哈希及所选通道，逐通道检查有限 loss、优化器更新、held-out loss、checkpoint 与恢复分片。2D 使用样本图，Astigmatism 的 anchor-only 路线仍需短程训练和 held-out 审查。

## 6. 分阶段实施与验收

每阶段控制在最多七个代码或配置文件内，阶段完成后验证，再进入下一阶段。

| 阶段 | 修改范围 | 可验收结果 |
| --- | --- | --- |
| 1. 选择规格与计划器 | 新 schema、解析器、`plan` CLI、文档示例 | 解析全部非空模态组合；拒绝重复通道、未知模态、缺 crop、错误预算；`plan` 输出实例与 GPU 数；不启动训练 |
| 2. 解除入口固定组合 | `_instance_specs`、模态联合入口及其定向测试 | 单模态、任意双模态、三模态均可建立 runtime；单模态产出 v2 checkpoint；已存在的双/三模态配置仍能解析 |
| 3. 通道与 anchor 合同 | 正式审计、2D/Astigmatism 物理方案配置和测试 | 2D/Astigmatism 可选一个或两个测量通道；DH 仅 `main`；2D anchor 与三模态现值一致；Astigmatism anchor 不重复叠加；所选通道之外无需图 |
| 4. DH PCA 接入 | PPG 配置绑定、PCA 输入预检、DH 定向 smoke | 有 beads/PCA 清单时 DH `main` 能训练并发布；缺清单或错误 schema 仅阻止含 DH 的计划；任何非 `main` DH 通道在计划阶段拒绝；vector 旧方案需显式选择 |
| 5. 动态门禁与启动 | `formal_gate`、通用 SLURM 启动器、签名/收据测试 | 1/2/3 GPU 资源与 rank 数匹配；无 DH 的运行不执行 DH 预检；resume 拒绝输入或路由漂移 |
| 6. 端到端验证与文档迁移 | 组合 smoke、推理路由、README/配置文档 | 下表各代表组合从计划、训练、发布到加载通过；旧固定脚本标记为历史入口 |

最小组合验收矩阵：

| 组合 | 测量通道 | 预期 GPU | 关键断言 |
| --- | --- | ---: | --- |
| 2D | `left` 或 `left,right` | 1 | circular 50 nm anchor；无 DH/Beads 输入；checkpoint 只包含所选 2D 通道 |
| Astigmatism | `left`、`right` 或 `left,right` | 1 | 一个共享模态模型；所选通道各自有状态；100 nm anchor |
| DH | `main` | 1 | 必须通过 beads/PCA 输入门禁；checkpoint 记录 PPG/PCA schema 和输入哈希 |
| 2D + Astigmatism | `left,right` / `left,right` | 2 | 四个通道状态、两个模态；没有 DH 输入检查 |
| 2D + DH | `left` / `main` | 2 | 可通过配置解析；只有 DH 要 beads/PCA |
| Astigmatism + DH | `left,right` / `main` | 2 | 两模态可训练、通道指标分离 |
| 三模态 | `left,right` / `left,right` / `main` | 3 | 五实例、三模态；各模态采用上述指定物理方案 |

矩阵中的 CPU/小预算测试用于验证路由和合同；正式完成标准仍包括相应 GPU smoke、有限优化更新、held-out 评估以及从发布 checkpoint 对全部已训练路由进行推理。已覆盖的用例可复用现有测试，不为 YAML 文本本身重复造测试。

## 7. 迁移与风险

现有三模态实验配置继续作为可复现实验快照。新入口先读取并生成相同实例清单，再逐步替换固定启动脚本。现有单实例 checkpoint 不自动冒充 v2 多模态 release；如需导入，应由显式转换工具校验模型、输入帧和通道物理状态。历史结果的比较必须注明是否使用样本图、DH 是 vector 还是 PCA、相机和 z 范围是否相同。

最大科学风险是把“2D/Astigmatism 不需 beads”误解成“移除样本图后仍是同一正式实验”，以及把当前 DH vector 结果误称 PCA。两种 anchor-only 路线和 DH PCA 路线都要有独立的配置签名、物理 provenance 与验证结果。DH 校准产物尚未准备好时，2D/Astigmatism 的实现和验证可以独立推进，DH 相关组合保持明确的待验证状态。
