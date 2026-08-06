# UnityPSF 任务 11-17：单模型、双模态、多通道实施计划

- 状态：双模态 + 多通道工程链路已完成；真实数据科学 baseline 待验收
- 日期：2026-08-04
- 上游冻结点：任务 10 baseline
- 架构决策：[`ADR 0003`](../adr/0003-single-joint-checkpoint.md)
- 可视化验收：[`UnityPSF 可视化训练与验收协议`](unitypsf-visible-training-validation.md)

## 0. 2026-08-04 实施状态

| 任务 | 当前状态 | 已完成 | 尚未完成 |
| --- | --- | --- | --- |
| 11 | 工程完成 | `unity_psf.joint_checkpoint.v1`、原子保存、嵌套完整性校验、`inspect/verify/assemble` | 真实 peak-zmap/gamma 随正式实例写入后的数据级验收 |
| 12 | 工程完成 | 一个 `UnityPSF`、一个 checkpoint、精确 `(modality, channel_id)` 路由、eager/lazy 回载 | 真实大图推理的性能与显存评估 |
| 13 | 报告工程完成 | 固定图组、`summary.json`、`figure_index.json`、静态 `report.html` | 真实 Astig left/right 等价性、peak-zmap/gamma 图和人工科学验收 |
| 14 | 专家与数据 contract 完成 | 完整 `Emitter2DExpert`、z-loss mask、Origami manifest/split contract | Origami 固定 quicklook、正式训练和科学指标 |
| 15 | 等待数据 | DH 物理/calibration 基础模块已迁入 | 完整 DH expert、真实训练和科学验收 |
| 16A-C | 工程完成 | round-robin、3-rank Expert Parallel、barrier 后 joint commit、回载 smoke | 使用真实 Origami + Astig left/right 的正式联合训练 |
| 16D | 未开始 | checkpoint schema 已可扩展 | DH 加入三模态训练 |
| 17 | 未开始 | 显式确定性路由可用 | raw TIFF detector、置信度校准和拒识 |

工程验收凭据：SLURM job `4513` 在 3 张 RTX 3090 上分别运行
`Emitter2D(main)`、`Astigmatism(left)` 和 `Astigmatism(right)`。三个 rank 均完成独立
forward/backward/optimizer step，rank 0 原子发布并重新加载：

```text
output/unitypsf/dual-modality-ep-4513/checkpoints/unitypsf_joint.ckpt
SHA-256: 4e8a370dd8b15ea69836c2d0500588799304802f6d1a4054951b64e49209928b
report: output/unitypsf/dual-modality-ep-4513/report/report.html
```

该作业使用 8x8 合成 smoke 输入，只冻结工程行为，不冻结科学性能。不得把它命名为
`baseline-dual-modality-multichannel.md`；第一份正式科学 baseline 必须等待真实
Origami 和 Astigmatism left/right 数据验收。

## 1. 本阶段要交付什么

第一正式里程碑不是等待三模态数据齐全，而是先完成可训练、可观察、可复现的
**双模态 + 多通道 UnityPSF**：

```text
unitypsf_joint.ckpt
    |
    +-- UnityPSF
            +-- hard router(modality, channel_id)
            +-- Emitter2DExpert(main)        <- Origami 2D
            +-- AstigmatismExpert(left)      <- 独立网络/FiLM/zmap/gamma
            +-- AstigmatismExpert(right)     <- 独立网络/FiLM/zmap/gamma
            +-- one localization output contract
```

用户只加载一个 checkpoint、得到一个模型对象、调用一个 API。内部 expert 与 channel
实例仍然完全独立。Double Helix 的 contract 和 checkpoint schema 在这一阶段预留，
但在真实数据和科学验收到位前，不写入空 expert，也不宣称三模态支持。

## 2. 依赖顺序

```text
任务 10 baseline
    -> 任务 11 joint checkpoint
    -> 任务 12 顶层 UnityPSF 模型与加载 API
    -> 任务 13 像散可视化等价性
    -> 任务 14 Emitter2D + Origami
    -> 任务 16A-C 双模态 + 多通道联合训练
    -> 第一正式里程碑冻结

Double Helix 数据就绪
    -> 任务 15 DH 完整专家
    -> 任务 16D 三模态扩展
    -> 任务 17 三分类 detector 扩展
```

当前显式确定性路由已完成。任务 17 的图像 detector 不阻塞双模态工程发布。

## 3. 任务 11：定义并实现单文件 joint checkpoint

### 目标

将任务 1-10 已有的 prototype、instance checkpoint、peak zmap、gamma 和训练状态，
组装为一个物理文件 `unitypsf_joint.ckpt`。该文件是一个完整 UnityPSF 模型快照，
不是指向多个模型文件的 manifest。

### 实施小步

1. 定义 `unity_psf.joint_checkpoint.v1` schema 和不可变 metadata dataclass。
2. 定义顶层字段：模型身份、支持模态、公共 contracts、router、expert registry、
   calibration、provenance、training state 和 integrity table。
3. 定义实例键 `(modality, channel_id)`，禁止同一键重复或缺少 channel metadata。
4. 将 peak zmap、gamma、DH LUT 等推理必需状态序列化进 checkpoint；原始 TIFF 只记
   manifest/hash。
5. 实现从 v2 prototype/instance checkpoint 导入的 assembler，作为任务 1-10 的迁移桥。
6. 保存时先写同目录临时文件，重新加载并验证后执行原子替换。
7. 定义 `release` 与 `resume` 两种 payload role；两者都是完整 joint checkpoint，
   `resume` 额外包含 optimizer、scheduler、scaler、RNG 和 step。
8. 定义两模态 checkpoint 的 unsupported-modality 行为；不得创建空 DH 状态。

### 产出物

- joint checkpoint contract、validator、assembler 和 loader；
- `unity-psf-checkpoint inspect/assemble/verify` CLI；
- Astigmatism left/right 到单文件 checkpoint 的 round-trip fixture。

### 可视化交付

- `checkpoint_inventory.png`：一个文件内部的模态、通道、参数量、物理状态和 hash；
- `checkpoint_inventory.json`：机器可读版本；
- 清楚标记 supported、unsupported、prototype 和 trained instance。

### 验收条件

- 移动单个 `.ckpt` 文件后无需任何外部模型文件即可加载。
- left/right state、peak zmap 和 gamma 仍然不同且 hash 可验证。
- 任意嵌套 tensor 或 metadata 被修改后校验失败。
- release checkpoint 不携带 optimizer；resume checkpoint 可完整恢复。
- 旧 v2 checkpoint 只作为导入来源，不再是正式发布接口。

### 验证

```bash
pytest -q tests/contracts/test_joint_checkpoint.py tests/integration/test_joint_checkpoint_assembly.py
```

预计每个实现批次修改 4-5 个文件。任务完成后冻结 `baseline-task11.md`。

## 4. 任务 12：实现一个顶层 `UnityPSF` 模型

### 目标

提供一个真实的顶层模型对象和稳定 API：

```python
model = UnityPSF.from_checkpoint("unitypsf_joint.ckpt", device="cuda:0")
result = model.localize(images, modality="astigmatism", channel_id="left")
```

### 实施小步

1. 实现 `UnityPSF(nn.Module)`，拥有 input contract、router、expert registry 和 output contract。
2. registry 使用 `(PSFModality, channel_id)` 唯一定位完整实例。
3. 提供 `load_mode=eager|lazy`。lazy 只将选中实例 materialize 到 GPU，但逻辑模型身份不变。
4. `forward/localize` 必须先解析 modality/channel，再执行唯一 expert。
5. 将各 expert 输出标准化为统一 localization contract，同时保留 modality auxiliary。
6. forward hook 验证未选 expert 没有执行；显存统计验证 lazy load 有效。
7. 推理脚本改成顶层 API 的薄调用；旧显式 per-expert checkpoint 入口保留只读兼容。
8. 加入 `model.describe()`，输出支持模态、通道、实例状态和 checkpoint hash。

### 可视化交付

- `route_trace.png`：固定样本从输入 contract 到选中 expert 再到公共输出；
- `activation_audit.png`：每次调用的 expert forward 次数和未激活 expert 状态；
- 2D/Astig 尚未齐全时，未实现模态必须明确显示 `unsupported`，不能静默 fallback。

### 验收条件

- 一个模型对象可连续处理 Astigmatism left 和 right，且每次只激活正确实例。
- eager/lazy 输出在容差内一致。
- 错误模态、错误 channel、模态冲突和缺失 calibration 在边界处失败。
- API 不暴露内部文件路径或要求用户手动选择 expert checkpoint。

### 验证

```bash
pytest -q tests/models/test_unity_psf.py tests/infer_recon/test_unity_model_loading.py
```

依赖任务 11。完成后冻结顶层 API contract，后续 expert 不得改变调用方式。

## 5. 任务 13：像散单/双通道等价性与第一套可视化报告

### 目标

证明 joint checkpoint 和顶层模型没有破坏当前 Astigmatism main/left/right 的训练、
推理、物理状态和重建行为，并落地通用 `report.html` 生成器。

### 实施小步

1. 使用任务 1 冻结的配置、seed 和容差运行 CPU contract parity。
2. 使用固定小规模 SLURM GPU smoke 验证 forward、loss、backward、resume 和 inference。
3. 比较 crop、10-channel 输出、解码、过滤、坐标回填、重建和 union。
4. 比较导入前后的 peak zmap、gamma、physical version 和 checkpoint lineage。
5. 实现通用 metrics reader、固定样本选择和静态 `report.html`。
6. 将所有图按 `astigmatism/main|left|right` 分组，禁止只显示 union。

### 强制图

- Astig 原始帧和 left/right crop；
- PSF patch 随 z 的 montage；
- train/validation loss；
- prediction/GT overlay 或真实数据定位叠加；
- x/y/z error by z-bin；
- left/right reconstruction 与 union；
- left/right peak zmap 和 gamma 差值图。

### 验收条件

- 工程 contract 与任务 10 baseline 一致。
- left/right 图像、物理状态和训练轨迹保持独立。
- `report.html` 在无服务器条件下可直接打开，所有图有 config/epoch/hash 来源。
- 人工查看后确认没有空图、全亮/全暗、明显坐标翻转或通道串写。

### 验证

```bash
pytest -q tests/baseline tests/integration/test_astigmatism_parity.py tests/reporting
```

随后提交固定 seed 的 SLURM GPU smoke。完成后冻结 `baseline-task13.md` 并经过人工验收门。

## 6. 任务 14：完整 `Emitter2DExpert` 与 Origami 接入

### 目标

将当前轻量 2D adapter 升级为完整 expert，并使用现有 Origami 数据建立真实 2D
输入、训练、推理和可视化闭环。

### 数据边界

- 数据根：`datasets/training_sets/origami_2d`；
- 首先生成只读 data manifest，记录 OME-TIFF、acquisition group、波长/曝光信息和 hash；
- 按 acquisition group 划分集合，禁止同一 spool 的帧跨 train/validation/test；
- 第一版使用 `channel=main`；现有多个 spool/波长不能在没有配准证据时伪装成 left/right 配对；
- Emitter2D 仍使用统一 ChannelLayout，后续有真实 paired 2D 数据时无需改模型接口。

### 实施小步

1. 实现独立 preprocessing、完整 DoubleUNet、FiLM、heads、loss 和 decoder。
2. 定义 2D condition schema、photon/background 单位与 `z=0, z_valid=false`。
3. 使用合成 2D 数据做单 batch overfit，先证明优化链路正确。
4. 为 Origami 建立固定 crop、代表性帧和真实数据 inference/reconstruction quicklook。
5. 训练过程中独立记录 Emitter2D loss 和 optimizer step，不与 Astig loss 混合解释。
6. 将训练后的 `Emitter2DExpert(main)` 写入同一个 joint checkpoint schema。

### 强制图

- Origami 数据审计、crop 和强度分布；
- 代表性 PSF patch montage；
- 2D 专家训练曲线；
- detection map 与定位叠加；
- 原始结构与 localization reconstruction；
- photon/background 和局部密度分布；
- 模拟数据的 precision/recall 与 x/y error。

### 验收条件

- 合成数据可稳定 overfit，真实 Origami 输出非空且结构合理。
- 2D z loss 被屏蔽，输出明确携带 `z_valid=false`。
- 2D 参数不与 Astigmatism 实例共享。
- 一个 joint checkpoint 可同时描述 Astig left/right 与 Emitter2D main。
- 用户人工查看 Origami 报告后才进入双模态联合训练。

### 验证

```bash
pytest -q tests/models/test_emitter_2d_expert.py tests/integration/test_emitter_2d_training.py tests/data/test_origami_manifest.py
```

正式训练通过 SLURM GPU 执行。完成后冻结 `baseline-task14.md`。

## 7. 任务 15：完整 `DoubleHelixExpert`，等待真实数据门

### 状态

接口设计可以提前完成，正式训练与科学验收等待 Double Helix 样本和 calibration
数据到位。该任务不阻塞双模态 + 多通道里程碑。

### 数据就绪门

开始正式实现前必须确认：

- raw TIFF、bead calibration stack 和 z/angle metadata；
- acquisition、显微镜、波长、像素尺寸和 z-step；
- train/validation/test 分组策略；
- 已迁入 `optics/psf/double_helix` 的解析 anchor 可重建代表性数据。

### 实施小步

1. 定义 DH 输入、condition、auxiliary output 和 calibration contract。
2. 实现完整 backbone、FiLM、公共定位 heads 和 lobe geometry heads。
3. 复用唯一 DH optics/calibration 源，禁止在 expert 内复制解析代码。
4. 实现 lobe angle/separation loss、angle-to-z LUT 和有效范围拒识。
5. 先完成合成 overfit 和 calibration round trip，再运行真实数据训练。
6. 将 DH main/left/right 作为可选实例写入同一 joint checkpoint schema。

### 强制图

- raw/calibration/reconstruction z-stack；
- lobe angle 与 z、separation 与 z；
- calibration residual 与有效范围；
- 定位 overlay、z-bin error 和重建图；
- 不同通道独立 physical state。

### 验收条件

- 数据就绪门未满足时只允许 contract 测试，不允许宣称 DH 支持。
- 合成和真实数据分别通过，calibration 缺失或越界必须拒识。
- 三模态 joint checkpoint round trip 成功。

### 验证

```bash
pytest -q tests/models/test_double_helix_expert.py tests/optics/test_double_helix_calibration.py tests/integration/test_double_helix_training.py
```

## 8. 任务 16：统一硬路由与双模态/多通道联合训练

### 目标

在一个父训练 run 中训练多个模态和多个通道，并在同步边界保存一个完整 joint
checkpoint。模型层面始终是一个 `UnityPSF`；Expert Parallel 只是可选执行策略。

### 16A：确定性路由和数据调度

1. 实现 `ModalityResolver`，优先级为显式 config、TIFF metadata、calibration metadata、detector。
2. 实现 modality-homogeneous microbatch contract；首版不在一个 microbatch 混合 PSF。
3. 实现 `MultimodalTrainingPlan`，为 `(emitter_2d, main)`、
   `(astigmatism, left)`、`(astigmatism, right)` 生成独立数据流和 optimizer owner。
4. 定义每个 global epoch 的 per-instance step budget，记录实际 step 和样本数。
5. 禁止将不同尺度的 expert loss 直接求和后作为唯一训练质量判断。

### 16B：两种执行模式

先实现可审计基线，再启用多 GPU：

- `round_robin`：单进程/单 GPU 按固定 schedule 轮转 expert instance；
- `expert_parallel`：一个 rank/GPU 拥有一个或一组 expert instance，各自 forward、backward 和 optimizer；
- 两种模式使用相同数据计划、路由 contract、epoch 定义和 checkpoint schema；
- 正式双模态多通道建议映射为 rank 0 = Emitter2D main，rank 1 = Astig left，rank 2 = Astig right；
- 不做跨 expert 梯度同步，只有 metrics 汇总和 checkpoint barrier。

### 16C：joint checkpoint commit

1. 所有 rank 到达 epoch/milestone barrier 后，将 CPU state 和完成状态交给 coordinator。
2. 任一 required instance 失败或缺席时，不得写出 `complete` release checkpoint。
3. rank 0 组装、验证并原子提交一个 `unitypsf_joint.ckpt`。
4. 保存后立即用新的顶层 `UnityPSF.from_checkpoint(...)` 做每模态/通道 smoke inference。
5. checkpoint 选择基于固定 validation scorecard；任何模态低于最低门槛都不能被总平均掩盖。

### 16D：从双模态升级三模态

任务 15 完成后，在 training plan 中增加 DH main 或 left/right。不得修改顶层 API、
joint checkpoint schema 或既有两模态实例键，只扩展 `supported_modalities` 和 registry。

### 双模态 + 多通道验收条件

- 一个正式 SLURM run 同时推进 Emitter2D main、Astig left 和 Astig right。
- 三个实例 step、loss、显存和结果图可分别查看。
- 一个 joint checkpoint 可依次推理 Origami 2D、Astig left 和 Astig right。
- 未选 expert 不 forward、不产生 gradient、不更新 optimizer。
- left/right peak zmap、gamma、参数和 checkpoint state 保持独立。
- `report.html` 同时展示 2D、Astig left、Astig right，且没有只报平均分。
- 用户完成视觉验收后冻结 `baseline-dual-modality-multichannel.md`。

### 验证

```bash
pytest -q tests/models/test_modality_router.py tests/training/test_multimodal_plan.py tests/distributed/test_expert_parallel.py tests/integration/test_dual_modality_multichannel.py
```

正式 CUDA smoke 和训练必须通过 SLURM GPU，报告 job ID、rank/GPU 证据和 joint
checkpoint hash。

## 9. 任务 17：raw TIFF 模态检测器与拒识

### 定位

detector 是 metadata 缺失时的 fallback，不是 localization backbone，也不阻塞第一版
双模态 deterministic routing。

### 17A：双模态 detector

1. 使用 Origami 2D 与 Astig raw TIFF/patch 建立 acquisition-group 隔离的数据 manifest。
2. 加入低信号、过曝、异常离焦和未知数据作为 reject/OOD。
3. 输出概率、confidence、accepted/rejected 和 detector version。
4. 在 validation set 校准温度和阈值，test set 只做最终评估。
5. 未见过的 DH 在三分类模型完成前必须倾向 reject，不能强制归入 2D 或 Astig。

### 17B：三模态扩展

DH 数据就绪并完成任务 15 后，增加 `double_helix` 类，重新按 acquisition group 划分
并校准。旧双分类 detector 保留版本，不能原地覆盖后丢失 provenance。

### 强制图与指标

- confusion matrix；
- confidence reliability diagram 和 ECE；
- confidence 分布与拒识阈值；
- coverage-risk curve；
- 按显微镜、采集批次和模态分面的错误样本 montage；
- 错误路由对 localization 的实际代价。

### 验收条件

- 数据划分无 acquisition 泄漏。
- 低置信度和未知 PSF 能拒识。
- 显式 config/metadata 与 detector 冲突时停止，不静默覆盖。
- detector state 和阈值写入同一个 joint checkpoint。
- 所有正式指标来自 SLURM GPU 运行并可复现。

### 验证

```bash
pytest -q tests/modality_detection tests/integration/test_detector_routing.py
```

## 10. 阶段验收顺序

1. 已完成任务 11：单文件 checkpoint contract、CLI 和完整性测试。
2. 已完成任务 12：一个模型对象、三实例精确路由和 checkpoint 回载。
3. 已完成任务 13 的报告工程；下一步接入真实 Astig left/right 物理状态并人工验收。
4. 已完成任务 14 的专家和 manifest contract；下一步运行 Origami quicklook 与训练。
5. 已完成任务 16A-C 的合成工程 smoke；真实双模态训练后才能冻结第一科学里程碑。
6. DH 数据到位后实施任务 15 和任务 16D。
7. 任务 17 在真实两模态数据稳定后进行，detector 不阻塞前述里程碑。

任何一步的图像结果不合理，即使单元测试全部通过，也停在当前任务排查，不进入下一项。
