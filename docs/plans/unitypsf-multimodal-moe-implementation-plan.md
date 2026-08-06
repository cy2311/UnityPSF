# UnityPSF 多模态 PSF MoE 详细实施计划

- 状态：任务 11、12、16A-C 工程完成；任务 13、14 完成工程 contract，等待真实数据科学验收
- 日期：2026-08-04
- 范围：二维发射体、像散、双螺旋三种 PSF
- 架构依据：`../ideas/unitypsf-modality-routed-film-moe.md`
- 单模型决策：`../adr/0003-single-joint-checkpoint.md`
- 任务 11-17 细化：`unitypsf-tasks11-17-joint-model.md`
- 可视化验收：`unitypsf-visible-training-validation.md`
- 实施原则：每完成 2-3 个任务必须验证并经过阶段验收，再进入下一阶段

## 1. 最终目标

UnityPSF 的目标不是把多个小型输出头挂在一个共享网络后面，而是建立三个拥有完整定位能力的规范专家：

1. `Emitter2DExpert`
2. `AstigmatismExpert`
3. `DoubleHelixExpert`

每个专家内部都拥有自己的数据预处理、完整定位主干网络、FiLM、输出头、损失函数和解码器。双螺旋专家还拥有双螺旋专用的标定、物理模型和叶瓣几何接口。

顶层路由器只选择当前任务需要的 PSF 专家。硬路由确定后，未被选中的专家不参与模型构造、前向传播、反向传播和参数更新。

```text
PSF 模态解析器
    |
    +-- Emitter2DExpert
    |
    +-- AstigmatismExpert
    |      |
    |      +-- channel=main
    |      +-- channel=left
    |      +-- channel=right
    |
    +-- DoubleHelixExpert

每个通道实例独立拥有
    +-- 完整模型和 FiLM 参数
    +-- raw crop 和 peak zmap
    +-- gamma 与物理状态
    +-- optimizer 与 scheduler
    +-- 训练 checkpoint
```

UnityPSF 对外提供一个 `unitypsf_joint.ckpt`。用户只加载一个 checkpoint、得到一个
顶层 `UnityPSF` 模型，并通过一个 API 完成模态和通道路由。专家、通道实例与标定
状态是 joint checkpoint 内部的条件模块，不是用户分别管理的模型文件。

## 2. 当前实现与目标架构的差距

截至 2026-08-04，目标架构的核心工程边界已经落地：

- 正式路径不再使用共享 image stem；`Emitter2DExpert` 和两个
  `AstigmatismExpert` 实例分别拥有完整 backbone 与 FiLM。
- 输入帧与测量通道 contract 已分离，left/right 是同一规范像散专家的完整独立实例。
- 顶层 `UnityPSF` 使用 `(modality, channel_id)` 做精确硬路由，输出统一的 10-channel
  localization contract。
- `unity_psf.joint_checkpoint.v1` 将三个实例和推理所需状态放进一个物理文件，并提供
  原子保存、嵌套完整性校验和顶层回载 API。
- 单进程 round-robin 与 3-rank Expert Parallel 已使用相同训练计划和 checkpoint schema。
- UnityPSF 测试目录已覆盖 contracts、专家独立性、训练、联合 checkpoint、报告和 CLI。

剩余差距主要是科学验证，不是模型身份或训练控制面的缺口：

- 需要对真实 Origami 生成固定 frame/crop quicklook，完成 2D 推理、重建和正式训练。
- 需要将 Astigmatism left/right 的真实独立 peak-zmap、gamma 和 physical state 写入同一
  joint checkpoint，并检查各自的 z 误差与重建。
- 需要用真实两模态数据运行联合训练，经过人工图像验收后冻结第一科学 baseline。
- Double Helix expert、三模态扩展和 raw TIFF detector 仍等待相应数据与独立验收。

旧 SoftMoE 路径继续保留为兼容与对照基线，不能被表述为当前正式 UnityPSF 架构。

## 3. 必须固定的架构边界

### 3.1 输入帧与测量通道必须分离

以下两个概念不能再使用同一个 `channels` 字段表达：

- `input_frame_channels`：一次输入模型的时间帧数量，例如当前的 3-frame window。
- `measurement_channels`：显微镜测量通道，例如 `main` 或 `left/right`。

正确示例：

```yaml
input:
  input_frame_channels: 3

measurement:
  channels:
    - id: left
      crop: [0, 0, 128, 128]
    - id: right
      crop: [128, 0, 128, 128]
```

这里每个测量通道都向自己的模型实例提供 3 帧输入。双通道不等于模型 tensor 的通道维度为 2。

### 3.2 规范专家与通道实例必须分离

`AstigmatismExpert` 只有一个类和一个基础 checkpoint。left/right 不是两个预定义专家，而是同一个规范专家的两个运行时实例。

```text
                         astigmatism_base.ckpt
                                  |
                     分别构造两个全新模型对象
                                  |
                     严格加载同一个 state_dict
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
AstigmatismExpert(channel=left)   AstigmatismExpert(channel=right)
                 |                                 |
        astigmatism_left.ckpt            astigmatism_right.ckpt
```

不使用已经上 GPU 的模型对象进行浅复制，也不允许两个实例共享 `Parameter` 或 tensor storage。正确方法是重新调用模型工厂构造两个模型，然后分别执行 `load_state_dict(..., strict=True)`。

### 3.3 PSF 路由与 FiLM 分工固定

- PSF 路由器负责在 `emitter_2d`、`astigmatism`、`double_helix` 中选择一种专家。
- 通道编排器负责为选中的专家创建 `main`、`left`、`right` 或自定义实例。
- FiLM 只在专家内部表达 Zernike、视场位置、标定参数和采集条件。
- 实例已经绑定 channel 后，不再向 FiLM 追加 left/right one-hot。

首版按整份 TIFF 或一次训练/推理运行做硬路由，不在同一个 batch 内混合不同 PSF 模态。batch 内软路由只作为后续消融实验。

### 3.4 原型 checkpoint 与实例 checkpoint 分离

- `prototype`：规范专家的基础 checkpoint，例如 `astigmatism_base.ckpt`。
- `instance`：绑定具体测量通道并独立训练后的 checkpoint，例如 `astigmatism_left.ckpt`。

每个实例 checkpoint 必须记录 `parent_checkpoint_hash`。普通通道训练不得回写原型 checkpoint。将某个训练实例提升为新原型必须通过单独的显式命令完成。

### 3.5 统一定位输出与模态扩展分离

像散专家第一阶段继续使用当前 10-channel SMLM 输出、损失和解码语义，避免同时重写网络与科学 contract。

- 二维发射体专家要明确 z 字段的语义：解码结果中可固定 `z=0`，同时设置 `z_valid=false`，训练时屏蔽 z 损失。
- 双螺旋专家保留公共定位结果，并通过专用辅助输出承载叶瓣角度和叶瓣间距。
- 三种专家最终必须输出一致的 x/y 坐标、photon 单位、置信度和重建接口。

## 4. 总体依赖顺序

```text
冻结当前基线
    -> 修正模态与通道 contract
    -> 定义 checkpoint 和模型包 contract
    -> 实现完整 AstigmatismExpert
    -> 实现原型到实例的独立复制
    -> 打通像散单通道训练闭环
    -> 打通像散双通道独立训练
    -> 接入模型包推理
    -> 通过像散等价性验收
    -> 实现二维发射体和双螺旋专家
    -> 接入统一硬路由
    -> 训练 TIFF 模态检测器
    -> 完成科学实验与消融
```

## 5. 阶段 0：冻结现有基线

### 任务 1：建立像散基线验收凭据

**目标：** 在修改模型和 contract 前，用最小测试集固定当前可工作的行为，保证后续能判断是正常迁移还是科学行为回归。

**前置输入：**

- 当前 `active_smlm_soft_moe_double_unet` 配置。
- 当前 left/right crop 配置。
- 当前 10-channel 输出、过滤、重建和 union 逻辑。

**实施细节：**

1. 新建 `tests/baseline/`，只编写 UnityPSF 迁移需要的测试，不复制 Neptune v0.3 的全部 tests。
2. 使用固定 seed 的小型合成输入，记录模型输入 shape、输出 shape、输出字段顺序和解码后字段。
3. 固定 left/right crop 的原图坐标偏移规则，特别验证 right crop 输出回到全图坐标时的 x 偏移。
4. 固定 z、photon、background 的归一化和反归一化规则。
5. 固定 left/right localization 合并和重建入口的输入输出 schema。
6. 对需要数值比较的项目建立容差文件；确定性 contract 使用精确比较，浮点输出使用明确的 `rtol/atol`。

**产出物：**

- `tests/baseline/test_astigmatism_runtime_contract.py`
- `tests/baseline/test_dual_channel_coordinate_contract.py`
- `tests/fixtures/astigmatism_baseline.yaml`
- 小型、可提交且不包含真实敏感数据的测试 fixture

**验收条件：**

- 当前 SoftMoE 路径仍能完成一次 forward、loss 和 backward。
- left/right 坐标、z/photon scaling、过滤和 union contract 均有断言。
- 测试不依赖 Neptune v0.3 的 import。
- Neptune v0.3 没有任何文件变更。

**验证命令：**

```bash
pytest -q tests/baseline
```

**依赖：** 无。

**预计涉及文件：** 3-5 个。

**范围边界：** 本任务只建立观察和测试，不修复现有行为。

### 任务 2：拆分输入帧与测量通道 contract

**目标：** 删除 `ASTIGMATISM.required_channels == 2` 的错误语义，建立可被训练、推理、checkpoint 和模型包共同使用的通道描述。

**实施细节：**

1. 在 `contracts/modality.py` 中保留 `PSFModality`，删除或废弃 `required_channels`。
2. 定义 `InputFrameSpec`，至少记录帧数和帧顺序。
3. 定义 `MeasurementChannelSpec`，至少记录 `channel_id`、crop、anchor profile 和可选 calibration 引用。
4. 定义 `ChannelLayout`，负责校验 channel ID 唯一性、crop 合法性和确定性顺序。
5. 定义 `ExpertInstanceSpec`，记录 `expert_type`、`instance_id`、`channel_id` 和原型引用。
6. 配置解析层将旧的 `channels` 解释为 `input_frame_channels`，并对旧配置发出一次清晰的兼容提示；新配置只写新字段。

**产出物：**

- 稳定的通道与实例 dataclass/API。
- 新旧配置到统一内部 contract 的解析规则。
- 单通道、双通道和自定义通道测试。

**验收条件：**

- `input_frame_channels=3` 可与一个或多个测量通道同时存在。
- 不新增 `LeftAstigmatismExpert` 或 `RightAstigmatismExpert` 类。
- 重复 channel ID、负 crop、越界 crop 和空 layout 在边界处失败。
- 模态枚举不再决定测量通道数量。

**验证命令：**

```bash
pytest -q tests/contracts/test_modality.py
```

**依赖：** 任务 1。

**预计涉及文件：**

- `src/unity_psf/contracts/modality.py`
- `src/unity_psf/contracts/__init__.py`
- `tests/contracts/test_modality.py`

**预计规模：** 小，3 个文件。

### 验收门 A：基线与术语

- 基线测试全部通过。
- 新 contract 能明确回答“3 帧输入、2 个测量通道会创建几个模型实例”。
- 新增代码没有引用 Neptune v0.3。
- 人工确认通道术语后再进入 checkpoint 和模型改造。

## 6. 阶段 1：建立原型和实例语义

### 任务 3：定义 checkpoint v2 与模型包 contract

**目标：** 收敛新架构的 checkpoint 写入格式，同时保留现有格式的只读兼容能力。

**实施细节：**

1. 定义 `CHECKPOINT_SCHEMA_VERSION = unity_psf.checkpoint.v2`。
2. v2 统一使用 `model_state_dict`、`optimizer_state_dict`、`scheduler_state_dict`，与当前训练 loop 的字段保持一致。
3. metadata 至少记录：
   - `checkpoint_role: prototype | instance`
   - `expert_type`
   - `model_config`
   - `input_frame_spec`
   - `instance_id` 和 `channel_spec`
   - `condition_schema`
   - `parent_checkpoint_hash`
   - `code_version` 和 schema version
4. loader 按顺序识别 v2、当前训练 legacy payload 和早期 `psf_moe` v1 payload。
5. 新路径只写 v2；旧格式只能读取，不再新增旧格式文件。
6. 文件 hash 使用完整 checkpoint 文件的 SHA-256；实例 metadata 保存其原型文件 hash。
7. 定义 `unity_psf.bundle.v1` 的 `manifest.yaml` schema，所有内部路径必须相对模型包根目录。
8. 加载 manifest 时拒绝绝对路径、`..` 路径穿越、缺失文件和 hash 不匹配。

**产出物：**

- v2 checkpoint metadata 和 loader。
- 模型包 manifest dataclass、保存器和校验器。
- legacy 格式兼容测试。

**验收条件：**

- prototype 和 instance checkpoint 可分别保存和加载。
- instance 缺少 `parent_checkpoint_hash` 时校验失败。
- 移动整个模型包目录后仍能加载。
- 旧 checkpoint 可读取，但不会被静默解释成 v2。

**验证命令：**

```bash
pytest -q tests/contracts/test_checkpoint.py tests/contracts/test_bundle.py
```

**依赖：** 任务 2。

**预计涉及文件：**

- `src/unity_psf/contracts/checkpoint.py`
- `src/unity_psf/contracts/bundle.py`
- `src/unity_psf/contracts/__init__.py`
- 两个对应测试文件

**预计规模：** 中，5 个文件。

### 任务 4：实现完整的 `AstigmatismExpert`

**目标：** 让像散专家从输入图像开始拥有完整定位网络，不再依赖共享 `SharedPSFStem`。

**实施细节：**

1. 复用 `localization/smlm_unet.py` 中的 `DoubleUNet` 和 `film.py` 中的现有 FiLM 组件。
2. `AstigmatismExpert` 内部封装完整 `FiLMConditionedDoubleUNet`。
3. 第一版 forward contract 与当前训练保持一致：

```text
(images[N,C,H,W], conditions[N,D])
    -> localization[N,10,H,W]
```

4. 不在本任务重写 `ActiveSMLMLoss`、`decode_smlm_output` 或 10-channel 含义。
5. 明确 condition schema 的字段顺序和维度，模型构造时将其写入属性和 checkpoint metadata。
6. 禁止像散专家从 `SharedPSFStem` 接收 feature tensor；它必须接收原始预处理后的 image tensor。
7. 原有轻量 `AdaptedPSFExpert` 暂时保留给兼容测试，不作为新像散训练入口。

**产出物：**

- 可独立构造和训练的完整 `AstigmatismExpert`。
- `astigmatism_base.ckpt` 的模型结构定义。
- shape、梯度和 state round-trip 测试。

**验收条件：**

- 专家拥有完整 backbone 和自己的全部 FiLM 参数。
- 输出能直接进入现有 localization loss 和 decoder。
- 固定输入下保存、加载前后的输出一致。
- 参数列表中不存在对顶层共享 stem 的引用。

**验证命令：**

```bash
pytest -q tests/models/test_astigmatism_expert.py
```

**依赖：** 任务 2。

**预计涉及文件：**

- `src/unity_psf/models/psf_moe/experts/astigmatism.py`
- `src/unity_psf/models/psf_moe/base.py`
- 专家 exports
- `tests/models/test_astigmatism_expert.py`

**预计规模：** 中，6 个文件。

### 任务 5：实现原型到实例的独立复制

**目标：** 从一个 `astigmatism_base.ckpt` 安全创建一个或多个完全独立的通道实例。

**实施细节：**

1. 新建实例工厂，输入为原型 checkpoint、`ExpertInstanceSpec` 和目标 device。
2. 工厂先读取并校验 metadata，再根据 `model_config` 构造全新模型。
3. 使用 `load_state_dict(..., strict=True)` 加载原型参数。
4. 构造实例后写入只读的 `expert_type`、`instance_id`、`channel_id` 和 `parent_checkpoint_hash`。
5. optimizer 必须在实例创建完成后，从该实例的 `parameters()` 单独构造。
6. 禁止通过模块浅复制产生实例；`copy.deepcopy` 只可作为测试对照，不作为正式 API。
7. 提供参数 hash 辅助函数，用于验证两个刚创建的实例初值相同。

**产出物：**

- `create_expert_instance_from_prototype(...)` 一类明确的工厂 API。
- 实例独立性和 lineage 测试。

**验收条件：**

- left/right 初始 state_dict 逐项相等。
- 对应 Parameter 对象不同，底层 `data_ptr()` 不同。
- left 完成一次 optimizer step 后，right 和 prototype 文件不变。
- 错误专家类型或不匹配模型配置必须严格加载失败。

**验证命令：**

```bash
pytest -q tests/models/test_expert_instances.py
```

**依赖：** 任务 3、任务 4。

**预计涉及文件：**

- `src/unity_psf/models/psf_moe/instances.py`
- `src/unity_psf/models/psf_moe/__init__.py`
- `tests/models/test_expert_instances.py`

**预计规模：** 小，3 个文件。

### 验收门 B：完整专家与复制独立性

- 完整 `AstigmatismExpert` 可执行 forward、loss 和 backward。
- `astigmatism_base.ckpt` 可严格加载。
- left/right 参数值初始相同，但对象、storage 和 optimizer 完全独立。
- 未通过本验收门前，不接入高保真训练。

## 7. 阶段 2：打通像散单通道闭环

### 任务 6：注册单通道像散运行时

**目标：** 先完成最小的单通道垂直闭环，验证新专家能够复用当前训练设施。

**实施细节：**

1. 在模型 registry 中增加 `astigmatism_expert`，旧的 `active_smlm_soft_moe_double_unet` 保持不变。
2. runtime config 输出 `input_frame_spec`、`channel_layout` 和 `expert_instance`。
3. 单通道默认使用 `instance_id=main`、`channel_id=main`。
4. `TrainerRuntime` 继续只拥有一个 model、optimizer、scheduler 和 batch provider；不在此层加入多 optimizer 支持。
5. online batch provider 只接收当前实例需要的一份物理系数图。
6. 实例绑定后移除 `domain_onehot`；FiLM condition 只保留物理和采集条件。
7. 提供一份最小 `configs/modalities/astigmatism/astigmatism_single_channel_smoke.yaml`。

**产出物：**

- 可由现有训练入口构建的单通道像散 runtime。
- 新配置到 runtime contract 的测试。

**验收条件：**

- 构造 runtime 时只有一个完整像散专家和一个 optimizer。
- 一个 batch 可完成 forward、loss、backward 和 optimizer step。
- 模型输入帧数只取决于 `input_frame_channels`。
- 旧 SoftMoE 配置仍解析为旧模型。

**验证命令：**

```bash
pytest -q tests/training/test_astigmatism_runtime.py
```

随后使用 `configs/modalities/astigmatism/astigmatism_single_channel_smoke.yaml` 运行一次隔离的 smoke。

**依赖：** 任务 2、任务 4、任务 5。

**预计涉及文件：**

- `src/unity_psf/localization/model.py`
- `src/unity_psf/localization/runtime_config.py`
- 一个 smoke config
- 一个测试文件

**预计规模：** 中，4 个文件。

### 任务 7：建立单实例物理状态上下文

**目标：** 将 raw crop、peak zmap、gamma 和 condition store 从“多 domain 共享对象”改为“一个实例拥有一套上下文”。

**实施细节：**

1. 新建 `ChannelTrainingContext`，包含实例描述、crop、anchor profile、peak zmap 路径、当前 coeff map、condition store 和物理状态路径。
2. `run_high_fidelity.py` 只增加薄接线，不在这个大文件中继续堆积通道分支。
3. peak bootstrap 读取当前 channel crop，只输出到当前 run layout。
4. 将 99 nm anchor 移入命名明确的 660 nm astigmatism profile；其他波长和 PSF 不继承此常量。
5. 单实例 `ConditioningProviderStore` 只包含一个物理状态，不再等待多个 domain 的延迟提交。
6. gamma 更新采用临时文件加原子替换，避免中断时留下半写状态。
7. manifest 同时记录初始和最新 physical-state hash。

**产出物：**

- `ChannelTrainingContext`。
- 单实例 peak/gamma 状态目录和 manifest 字段。
- 单实例物理状态恢复测试。

**验收条件：**

- `main` 只从自己的 raw crop 构建 peak zmap。
- gamma 更新只改变当前实例的 coeff map 和 condition store version。
- checkpoint 中引用的物理 artifact 均存在且 hash 一致。
- 进程在 gamma 写入中断时，旧状态仍可恢复。

**验证命令：**

```bash
pytest -q tests/training/test_channel_physical_context.py
```

**依赖：** 任务 6。

**预计涉及文件：**

- `src/unity_psf/training/channel_context.py`
- `src/unity_psf/training/run_high_fidelity.py`
- 一份 astigmatism profile
- 一个测试文件

**预计规模：** 中，6 个文件。

### 任务 8：保存和恢复实例 checkpoint

**实施状态：已完成。** 任务 8 的冻结记录见 [`docs/architecture/baseline-task8.md`](../architecture/baseline-task8.md)。

**目标：** 让任意通道实例能够在不依赖其他通道的情况下完整恢复训练。

**实施细节：**

1. 训练 loop 增加 v2 metadata 写入入口，不让 loop 自己推断专家和通道身份。
2. checkpoint 保存 model、optimizer、scheduler、epoch、global step、随机数状态和当前 physical-state 引用。
3. resume 时先校验专家类型、实例 ID、channel ID 和父原型 hash，再加载训练状态。
4. 如果用户显式允许从同一原型重新开始新通道，只加载 model state，不加载旧实例 optimizer。
5. 将“完整续训”和“从权重初始化新实例”定义为两个不同 API，禁止自动猜测。
6. 保留 legacy checkpoint 恢复分支，并在 manifest 标记其来源格式。

**已实现的接口：**

- `TrainingConfig.checkpoint_metadata` 和 `EpochTrainingConfig.checkpoint_metadata`：由调用方显式提供 v2 `CheckpointMetadata`，训练 loop 不猜测专家、实例或通道身份。
- `resume_training_checkpoint(...)`：完整恢复 model、optimizer、scheduler、AMP scaler、epoch、step、global step、RNG 和 physical-state 引用。v2 instance 必须同时校验 `expert_type`、`instance_id`、`channel_id`、`parent_checkpoint_hash`。
- `initialize_model_from_checkpoint(...)`：只从 prototype checkpoint 加载 model state，不加载 optimizer、scheduler、scaler、计数器、RNG 或旧实例 physical state。
- `load_training_checkpoint(...)`：保留旧入口，legacy payload 走兼容分支；高保真运行的 manifest 记录 `checkpoint_format`。
- v2 checkpoint 继续保留 `epoch`、`step_count`、`global_step`、`model_state_dict` 等顶层训练字段，并使用同目录临时文件加原子替换。

**产出物：**

- v2 instance checkpoint 写入和恢复流程。
- 完整续训与仅权重初始化两种明确路径。
- Python、NumPy、Torch CPU/CUDA RNG 状态的保存与恢复。
- legacy 来源格式的 manifest 标记。

**验收条件：**

- 保存后恢复继续训练的 global step、scheduler 和 optimizer 一致。
- 固定 seed 下，中断续训与不中断训练结果一致。
- left checkpoint 不能以“完整续训”方式加载到 right 实例。
- 普通训练不会修改 `astigmatism_base.ckpt`。

**验证命令：**

```bash
pytest -q tests/training/test_instance_checkpoint_resume.py
```

任务 8 定向测试结果：`7 passed, 1 warning`。任务 7 physical-state、baseline runtime 和 checkpoint contract 回归结果：`14 passed, 1 warning`。全量测试结果：`49 passed, 5 warnings`；high-fidelity 单通道 smoke 和 legacy resume smoke 均退出码 0。warning 仅来自当前环境 CUDA 探测及既有 `vector_psf.py` tensor 构造，不是任务 8 新增失败。

当前完整续训粒度为 epoch 边界；若要支持 epoch 内中断，需要为 batch provider 增加可恢复 cursor/state，并另立任务验收。

**依赖：** 任务 3、任务 7。

**预计涉及文件：**

- `src/unity_psf/training/loop.py`
- `src/unity_psf/training/run_high_fidelity.py`
- `src/unity_psf/training/__init__.py`
- `src/unity_psf/contracts/checkpoint.py`（复用既有 v2 contract）
- `tests/training/test_instance_checkpoint_resume.py`

**预计规模：** 中，5 个代码/测试文件，另有 2 份验收文档。

### 验收门 C：像散单通道闭环

以下流程必须端到端通过：

```text
astigmatism_base.ckpt
    -> AstigmatismExpert(channel=main)
    -> main raw crop
    -> main peak zmap
    -> main gamma 和 localization training
    -> astigmatism_main.ckpt
    -> 独立 resume
    -> 独立 inference
```

验收时同时确认旧 SoftMoE smoke 仍然通过。

## 8. 阶段 3：实现像散双通道独立训练

### 任务 9：实现多通道编排器

**目标：** 让双通道成为两个独立单通道运行时的组合，而不是在一个模型内增加两个 domain。

**实施细节：**

1. 新建 `MultichannelTrainingPlan`，根据 `ChannelLayout` 生成有序的 `ChannelRunSpec`。
2. 每个 `ChannelRunSpec` 只包含一个 channel 的 crop、原型 checkpoint、run name、seed 和输出目录。
3. 本地模式按顺序调用两个独立运行；SLURM 模式生成两个独立命令或 job-array 条目。
4. 两个运行分别调用现有单模型 `build_trainer_runtime()`，不修改 `TrainerRuntime` 为多 optimizer 容器。
5. 默认使用相同的原型 hash 和初始化 seed；允许明确配置不同 seed，但必须写入 manifest。
6. 统一父 run 只收集状态，不持有两个模型对象，也不执行联合 backward。
7. 父 run 状态明确区分 `pending`、`running`、`completed` 和 `failed`。

**产出物：**

- 多通道训练计划生成器。
- 本地顺序执行入口。
- SLURM channel job 规格。

**验收条件：**

- left/right 拥有不同 model 对象、optimizer、scheduler、run layout 和 checkpoint 路径。
- left 训练一步不会改变 right 的任何训练状态。
- 一个 channel 失败时另一个已完成结果仍保留。
- 编排器不合并 batch、loss 或梯度。

**验证命令：**

```bash
pytest -q tests/training/test_multichannel_orchestrator.py
```

**依赖：** 验收门 C。

**预计涉及文件：**

- `src/unity_psf/training/multichannel.py`
- 一个正式 CLI 模块
- `pyproject.toml`
- 一个测试文件

**预计规模：** 中，4 个文件。

**实现记录（2026-08-02）：**

- 已新增 `training.multichannel` 公共编排器。它为一个 PSF 模态生成独立的
  `left/right`（或任意显式 channel）`ChannelRunSpec`；每个 spec 拥有独立的
  `instance_id`、seed、crop、run 目录、配置路径和入口参数。
- `build_multimodal_training_plans(...)` 可以同时生成
  `emitter_2d`、`astigmatism`、`double_helix` 三个互不共享父目录的计划。
  统一 run name 会自动追加模态后缀，避免覆盖。
- `unity-psf-train-multichannel` 已提供 `plan`、`local`、`slurm` 三种模式。
  CLI 会把基础 YAML 物化为每个 channel 的单通道 `config.yaml`，并写入该
  channel 的独立 run 目录；local 使用独立子进程，SLURM 使用独立脚本。
- 父计划只写 `multichannel_manifest.json`、汇总状态和生成调度描述，不持有
  多个模型、optimizer、scheduler，也不合并 batch、loss 或 gradient。
- unknown channel 的 seed、prototype 和 extra args 会在计划构造阶段显式拒绝。
- 当前任务 9 没有修改 peak zmap、gamma 或 physical-state 的实际隔离逻辑；这些
  保留给任务 10，避免把编排边界和物理状态迁移混在同一个变更中。

**实现文件：**

- `src/unity_psf/training/multichannel.py`
- `src/unity_psf/training/__init__.py`
- `src/unity_psf/cli/multichannel.py`
- `tests/training/test_multichannel_orchestrator.py`
- `tests/cli/test_multichannel.py`
- `pyproject.toml`

**任务 9 验证结果：** 定向测试 `10 passed`；完整测试见任务 9 baseline。

### 任务 10：隔离 left/right 的 peak zmap 与 gamma

**目标：** 保持当前 left/right 各自拟合物理状态的科学语义，同时改用新的实例边界。

**实施细节：**

1. left raw crop 只写入 `channels/left/`，right raw crop 只写入 `channels/right/`。
2. 两个 channel 可引用同一个 anchor profile，但 peak zmap 必须分别计算和保存。
3. 每个 channel 使用自己的 `ChannelTrainingContext`、gamma optimizer 和更新计数。
4. 每个 checkpoint 只嵌入或引用当前 channel 的 physical state。
5. 父编排器在两个 channel 都结束后才生成最终完整模型包；部分成功时生成明确的 incomplete 状态。
6. SLURM 任务使用独立输出目录，避免并发写入同一个 `current_physical_state.json`。

**预期目录：**

```text
run/
    channels/
        left/
            checkpoints/
            metadata/current_physical_state.json
            stages/peak/
        right/
            checkpoints/
            metadata/current_physical_state.json
            stages/peak/
```

**验收条件：**

- 两个 crop 不同时，peak zmap artifact 和 hash 独立。
- left resume 不读取 right 的 coeff map 或 gamma step。
- 两个 SLURM 任务并发执行时不写同一文件。
- 任一 channel 的 artifact 被替换后，模型包 hash 校验失败。

**验证命令：**

```bash
pytest -q tests/training/test_multichannel_physical_isolation.py
```

GPU smoke 必须通过 SLURM 启动，不在登录节点直接运行 CUDA 训练。

**依赖：** 任务 9。

**预计涉及文件：**

- `src/unity_psf/training/channel_context.py`
- `src/unity_psf/training/run_high_fidelity.py`
- `src/unity_psf/localization/runtime_config.py`
- `src/unity_psf/cli/multichannel.py`
- `tests/training/test_multichannel_physical_isolation.py`
- `tests/cli/test_multichannel.py`

**预计规模：** 中，6 个文件。

**实现记录（2026-08-04）：**

- 每个 channel 使用自己的 `channels/<channel>/` run 目录和
  `metadata/current_physical_state.json`，不再共享或覆盖另一个 channel 的
  physical-state 文件。
- `ChannelTrainingContext` 在 checkpoint extra 和 resume 时校验
  `expert_type`、`instance_id`、`channel_id`，并验证 peak zmap 文件的 SHA-256。
  替换或串用 left/right physical state 会直接失败。
- 有 expert instance 的 emitter、astigmatism 和 double-helix runtime 都会生成
  统一的 channel contract，并创建独立的 `ChannelTrainingContext`；单 channel
  provider/model 的 domain_count 和 condition_dim 会同步收敛。
- 新 physical-state 的 coefficient map 必须绑定当前 channel；checkpoint extra
  拒绝错误 channel、多个 map 和缺失 artifact hash。legacy 外部 map 仍可读取，但
  不能跨 channel 载入。
- 单通道 peak bootstrap、coefficient map、gamma ROI domain 和
  `ConditioningProviderStore` 都绑定到当前 channel；无法唯一绑定时显式报错。
- `unity-psf-train-multichannel` 物化配置时会过滤当前 channel 之外的
  `real_tiff_wake.domains`、coefficient maps、LUT zmap、ROI-bank base maps、
  ROI-bank source domains 和 `auto_build_domains`。
- 多候选物理 domain 没有当前 channel 的唯一匹配时拒绝生成配置；只有一个候选
  时显式绑定并规范化名称为当前 channel。
- 父编排器继续保持独立模型、optimizer、scheduler、batch、loss、gradient 和
  checkpoint；任务 10 没有组装最终模型包。

**任务 10 验证结果：**

```text
CLI 与任务 9/10 定向测试：37 passed, 5 warnings
全量 UnityPSF 测试：73 passed, 5 warnings
```

warning 来自当前环境的 CUDA 探测和已有 `vector_psf.py` tensor 构造，不是
任务 10 新增失败。

任务 10 的冻结凭据见 `docs/architecture/baseline-task10.md`。从任务 11 起采用
[`ADR 0003`](../adr/0003-single-joint-checkpoint.md)：正式交付物是一个物理文件
`unitypsf_joint.ckpt`，加载后得到一个顶层 `UnityPSF` 模型。

任务 11-17 的唯一有效实施版本见
[`unitypsf-tasks11-17-joint-model.md`](unitypsf-tasks11-17-joint-model.md)。第一个正式
里程碑是双模态 + 多通道：Origami `Emitter2D(main)` 与 Astigmatism `left/right`
在一个父训练 run 中推进，最后提交同一个 joint checkpoint。Double Helix 不阻塞该
里程碑，待数据就绪后扩展为三模态。

训练和推理的强制可视化产物见
[`unitypsf-visible-training-validation.md`](unitypsf-visible-training-validation.md)。每个任务
必须同时提供工程、数值和视觉证据。

**任务 11-17 当前执行快照：**

- 任务 11：工程完成。joint checkpoint contract、assembler、CLI、原子保存和完整性校验已通过。
- 任务 12：工程完成。一个顶层模型可从同一 checkpoint 连续路由三条实例路径。
- 任务 13：可视化报告工程完成；真实 Astig 等价性和物理状态图仍待验收。
- 任务 14：完整 Emitter2D 与 Origami manifest/split contract 完成；真实 quicklook 和训练待执行。
- 任务 15：等待 Double Helix 真实样本和 calibration 数据。
- 任务 16A-C：工程完成。SLURM job `4513` 在 3 张 RTX 3090 上完成 Expert Parallel
  synthetic smoke，发布的 joint checkpoint SHA-256 为
  `4e8a370dd8b15ea69836c2d0500588799304802f6d1a4054951b64e49209928b`。
- 任务 16D 和任务 17：未开始，不阻塞当前双模态工程里程碑。

job `4513` 只证明三路训练、barrier、单文件提交、回载路由和报告生成正确，不证明
真实样本上的定位精度。第一科学 baseline 仍需 Origami main 与 Astigmatism left/right
真实数据共同通过。

<details>
<summary>已归档的旧版任务 11-17（目录 bundle/manifest 方案，不得据此实施）</summary>

以下内容仅用于追踪规划历史，已由上面的单 joint checkpoint 计划取代。

### 任务 11：组装 UnityPSF 模型包

**目标：** 将规范原型、通道实例和标定产物组织为一个可移动、可验证的交付物。

**实施细节：**

1. bundle builder 接收已完成的 channel run，不从任意目录猜测文件。
2. 正式导出默认复制需要交付的文件，避免模型包依赖原 run 目录。
3. 所有文件先复制到临时 bundle 目录，完成 hash 校验后再原子重命名为最终目录。
4. manifest 记录模态、通道顺序、原型路径、实例路径、calibration 路径、schema、代码版本和 SHA-256。
5. loader 对 manifest 中每个文件执行存在性、路径边界和 hash 校验。
6. bundle 不合并 left/right 权重；统一性由 manifest 提供，而不是依靠一个巨大的 monolithic checkpoint。

**产出物：**

- `build_unity_bundle(...)`。
- `load_unity_bundle(...)`。
- 可移动的像散单通道和双通道模型包。

**验收条件：**

- 移动整个模型包后仍可加载。
- 删除、修改或交换 left/right checkpoint 时校验失败。
- manifest 不包含本机绝对路径。
- incomplete channel run 不能生成标记为 complete 的模型包。

**验证命令：**

```bash
pytest -q tests/contracts/test_bundle.py tests/integration/test_astigmatism_bundle.py
```

**依赖：** 任务 9、任务 10。

**预计涉及文件：**

- `src/unity_psf/contracts/bundle.py`
- `src/unity_psf/runtime/bundle_builder.py`
- 一个 CLI 接线文件
- 一个集成测试文件

**预计规模：** 中，4 个文件。

### 验收门 D：像散双通道交付物

双通道运行必须产生：

```text
astigmatism_left.ckpt
astigmatism_right.ckpt
calibration/astigmatism_left/...
calibration/astigmatism_right/...
manifest.yaml
```

两个 checkpoint 必须具有相同 `parent_checkpoint_hash`，但拥有不同的 `instance_id`、channel metadata、训练状态和物理状态。

## 9. 阶段 4：接入推理并完成像散等价性验收

### 任务 12：实现模型包感知的推理加载器

**目标：** 将当前推理脚本中的直接 `torch.load` 下沉为可复用 package API，并根据模态和 channel 选择实例。

**实施细节：**

1. 提供类似 `load_expert_for_inference(bundle, modality, channel_id, device)` 的稳定 API。
2. loader 先验证 bundle，再读取指定实例的 checkpoint metadata。
3. 根据 checkpoint 中的 `model_config` 构造模型并执行 strict state load。
4. left TIFF crop 只能绑定 left checkpoint，right 同理。
5. 兼容当前显式 `--checkpoint` 参数；显式 checkpoint 模式必须要求用户同时给出或确认 channel。
6. 将 `scripts/infer/run_3371_full8000_infer_filter_recon.py` 中的模型加载逻辑替换为薄调用。
7. 推理输出保留当前 per-channel localization 文件，再进入现有 filter/reconstruction/union 流程。

**产出物：**

- 模型包推理加载 API。
- 旧脚本的薄兼容接线。
- 单通道、双通道和 legacy checkpoint 加载测试。

**验收条件：**

- left 只加载 left 实例，right 只加载 right 实例。
- 不存在的 channel、错误模态和 hash 不匹配明确失败。
- 模型构造参数不由推理脚本硬编码。
- 现有显式 checkpoint 命令仍能运行。

**验证命令：**

```bash
pytest -q tests/infer_recon/test_model_loading.py
```

**依赖：** 任务 11。

**预计涉及文件：**

- `src/unity_psf/infer_recon/model_loading.py`
- `src/unity_psf/infer_recon/__init__.py`
- 当前正式 inference 脚本
- 一个测试文件

**预计规模：** 中，4 个文件。

### 任务 13：通过像散单通道与双通道等价性验收

**目标：** 证明新实例架构没有破坏当前已经成立的定位和重建行为。

**比较维度：**

1. 配置解析结果。
2. 模型输入和 10-channel 输出 shape。
3. fixed-seed 单步 loss 和梯度是否在基线容差内。
4. left/right crop 及回到全图坐标后的偏移。
5. detection、x/y、z、photon、sigma、background 解码。
6. 每个 channel 的过滤结果。
7. 每个 channel 的重建图。
8. left/right union localization 数量和坐标。
9. checkpoint 保存、加载和 resume。
10. peak zmap 与 gamma artifact 的实例隔离。

**实施细节：**

1. CPU contract 测试先运行，GPU 训练只验证小规模端到端流程。
2. 使用任务 1 冻结的容差；不得在看到新结果后随意放宽。
3. 网络结构有意变化导致数值不可能逐点相同时，比较科学指标和 contract，而不是伪造逐元素一致。
4. 保存每次验收的 config、seed、代码版本、指标 JSON 和 artifact hash。
5. 将失败分为 contract 回归、数值偏差、训练不稳定和物理状态错误四类。

**验收条件：**

- schema、单位、坐标方向和 crop offset 完全一致。
- 单通道 main 流程独立通过。
- 双通道 left、right 和 union 分别通过。
- 两个实例可独立保存、加载和恢复。
- 旧 SoftMoE 路径仍可运行，直到明确决定弃用。

**验证命令：**

```bash
pytest -q tests/baseline tests/integration/test_astigmatism_parity.py
```

随后提交固定 seed 的 SLURM smoke 作业，并检查训练、推理、过滤和重建全部退出码。

**依赖：** 任务 12。

**预计涉及文件：** 3-5 个测试、fixture 和 smoke config 文件。

**范围边界：** 本任务只确认像散迁移，不开始二维发射体和双螺旋实现。

### 验收门 E：像散正式通过

只有以下条件全部满足后，才能继续另外两种 PSF：

- 单通道像散闭环通过。
- 双通道 left/right 独立性通过。
- union reconstruction contract 通过。
- bundle 移动和 hash 校验通过。
- legacy 路径没有被提前删除。

## 10. 阶段 5：补齐另外两个完整专家

### 任务 14：实现完整的 `Emitter2DExpert`

**目标：** 使用相同完整专家接口支持普通二维发射体 PSF。

**实施细节：**

1. 专家拥有自己的 preprocessing、完整 backbone、FiLM、heads、loss 和 decoder。
2. 不复用其他专家的 Parameter；可以复用经过验证的网络类和公共工具函数。
3. 定义二维模态的 condition schema，去除不需要的像散或 DH 条件。
4. 定义 z 语义：训练时屏蔽 z loss，解码时输出 `z=0` 和 `z_valid=false`。
5. 保持 x/y、photon、background 和 uncertainty 与公共输出 contract 一致。
6. 生成独立的 `emitter_2d_base.ckpt`，其 metadata 标记为 prototype。
7. 用小型合成数据先做单 batch overfit，再做独立验证集评估。

**产出物：**

- 完整 `Emitter2DExpert`。
- 二维发射体 loss/decoder 配置。
- `emitter_2d_base.ckpt` 生成入口。

**验收条件：**

- 专家可独立训练、保存、恢复和推理。
- z 字段不会参与二维定位的错误监督。
- checkpoint 不包含像散或双螺旋专家参数。
- 小型合成数据可被稳定 overfit。

**验证命令：**

```bash
pytest -q tests/models/test_emitter_2d_expert.py tests/integration/test_emitter_2d_training.py
```

**依赖：** 验收门 E。

**预计涉及文件：**

- `src/unity_psf/models/psf_moe/experts/emitter_2d.py`
- 一个 loss/decoder 接线文件
- registry/config
- 对应测试

**预计规模：** 中，4-5 个文件。

### 任务 15：实现完整的 `DoubleHelixExpert`

**目标：** 将已经迁入 UnityPSF 的双螺旋物理和标定能力接到完整定位专家中。

**实施细节：**

1. 专家拥有自己的 preprocessing、完整 backbone、FiLM 和公共定位输出。
2. 增加双螺旋辅助输出：至少包括 `lobe_angle` 和 `lobe_separation`。
3. 明确从叶瓣几何到 z 的 calibration contract，包括单位、角度周期、有效范围和外推策略。
4. 将 `src/unity_psf/optics/psf/double_helix/` 作为唯一 DH 物理实现来源，不在 expert 内复制标定算法。
5. calibration artifact 独立版本化并写入模型包 hash。
6. loss 分为公共定位项与 DH 几何项；每项权重进入 config 和 checkpoint metadata。
7. decoder 输出公共 localization 结果，同时保留 DH 质量诊断字段。
8. 生成独立的 `double_helix_base.ckpt`。

**产出物：**

- 完整 `DoubleHelixExpert`。
- DH 辅助输出、loss 和 decoder。
- DH calibration 到 bundle 的连接。

**验收条件：**

- DH 专家可独立训练、保存、恢复和评估。
- 角度、叶瓣间距和 z 映射有明确单位与有效范围。
- calibration 缺失或 hash 不匹配时拒绝推理。
- 合成 DH 数据能完成小规模 overfit，固定标定数据能完成 round trip。

**验证命令：**

```bash
pytest -q tests/models/test_double_helix_expert.py tests/optics/test_double_helix_calibration.py
```

**依赖：** 验收门 E。

**预计涉及文件：**

- `src/unity_psf/models/psf_moe/experts/double_helix.py`
- DH optics 公共接口
- registry/config
- 两个聚焦测试文件

**预计规模：** 每个实施批次不超过 5 个文件；必要时拆成“标定接口”和“专家训练”两个任务。

### 验收门 F：三个完整专家

- 三个专家都从图像输入开始拥有完整 backbone 和 FiLM。
- 三个专家都有可独立加载的 prototype checkpoint。
- 三个专家遵守公共坐标、photon、置信度和 checkpoint contract。
- 轻量 adapter/head 不被计为完整专家。

## 11. 阶段 6：接入统一硬路由 MoE

### 任务 16：实现统一 PSF 模态解析与硬路由

**目标：** 用统一入口确定 PSF 模态，并且只构造和执行一个规范专家。

**实施细节：**

1. 新建 `ModalityResolver`，确定性来源优先级固定为：

```text
显式 config
    -> TIFF metadata
    -> calibration manifest
    -> 图像模态检测器
```

2. 多个确定性来源同时存在时必须一致；冲突时停止并报告冲突字段。
3. 路由结果至少包含 `modality`、`source`、`confidence` 和 `calibration_ref`。
4. 训练入口先解析模态，再从 expert registry 构造唯一专家。
5. 推理入口先解析模态和 channel，再从 bundle 选择唯一实例。
6. 首版不支持同一 batch 内不同模态，不计算专家权重加权和。
7. 使用 forward hook 测试未选专家没有被调用；使用参数统计确认未选专家没有 optimizer state。
8. 保留显式 `--modality` 参数作为科学实验和故障恢复入口。

**产出物：**

- `ModalityResolver`。
- 三专家工厂 registry。
- 训练和推理的硬路由接线。

**验收条件：**

- 三种模态均能路由到正确专家。
- 未选专家没有 forward、gradient 或 optimizer state。
- 冲突 metadata 和未知模态明确失败。
- 确定性路由不依赖图像检测器即可工作。

**验证命令：**

```bash
pytest -q tests/models/test_modality_router.py tests/integration/test_hard_routing.py
```

**依赖：** 任务 14、任务 15。

**预计涉及文件：**

- `src/unity_psf/models/psf_moe/router.py`
- `src/unity_psf/models/psf_moe/__init__.py`
- 一个 modality resolver 模块
- 一个或两个测试文件

**预计规模：** 中，4-5 个文件。

### 任务 17：训练 raw TIFF 模态检测器

**目标：** 当 config、TIFF metadata 和 calibration manifest 都无法确定模态时，使用图像检测器判断 PSF family，并支持拒识。

**数据准备：**

1. 收集三种模态的 raw TIFF 或代表性 patch。
2. 标签只表示 `emitter_2d`、`astigmatism`、`double_helix`，不混入 channel 标签。
3. 按显微镜、实验日期和 acquisition batch 划分 train/validation/test，禁止同一采集序列跨集合泄漏。
4. 增加低信号、过曝、离焦异常和未知 PSF 作为拒识/OOD 样本。
5. 保存数据清单和来源 hash，不把大型原始数据提交到 Git。

**模型与训练细节：**

1. 检测器只做模态分类，不参与 localization loss。
2. 输入可以是固定数量的代表性 patch 或 TIFF 统计摘要，具体选择通过小型实验确定。
3. 输出 `modality probabilities`、最终 modality、confidence 和 `accepted/rejected`。
4. 在 validation set 上选择温度校准和拒识阈值，禁止使用 test set 调参。
5. 低置信度时要求显式 `--modality`，不自动选择最高概率类别。
6. detector checkpoint 和训练数据版本写入 bundle，但 detector 不与三个定位专家共享 optimizer。

**评估指标：**

- accuracy
- macro-F1
- 混淆矩阵
- expected calibration error
- 拒识覆盖率与拒识准确率
- 错误路由后 localization 性能下降
- 不同显微镜和不同采集批次的泛化

**产出物：**

- 模态数据 manifest。
- detector 训练和评估入口。
- 带校准阈值的 detector checkpoint。
- router fallback 接线。

**验收条件：**

- 数据划分无采集序列泄漏。
- detector 在低置信度和 OOD 样本上能拒识。
- detector 失败不会绕过显式模态配置。
- 检测器指标和阈值可由固定 config 复现。

**验证命令：**

```bash
pytest -q tests/modality_detection
```

长时间 detector 训练和评估必须通过 SLURM 执行。

**依赖：** 任务 16，以及具有代表性的三模态数据。

**预计规模：** 这是独立研究子项目，必须进一步拆成“数据 contract”“模型训练”“置信度校准”“路由接入”四个中型任务。

</details>

### 任务 18：完成最终科学评估与消融

**目标：** 证明 UnityPSF 的价值来自统一的多模态条件计算，而不是只完成工程拼接。

**必须比较的基线：**

1. 三个完全独立训练和部署的单专家模型。
2. 当前 SoftMoE/FiLM domain 方案。
3. UnityPSF 显式 metadata 硬路由。
4. UnityPSF 图像 detector 硬路由。
5. 故意错误路由，用于量化 router 错误代价。
6. 可选 shared-backbone、soft routing 或 top-k routing 消融。
7. 像散单通道与双通道实例方案对比。

**科学指标：**

- localization precision/recall、Jaccard 或当前项目采用的等价指标
- x/y/z RMSE 或偏差
- photon 和 background 误差
- DH 角度、叶瓣间距和 z calibration 误差
- 每种模态的训练稳定性
- router accuracy、校准和拒识表现
- 推理吞吐、延迟、峰值 GPU 显存
- prototype、instance 和完整 bundle 大小
- 跨显微镜、跨采集批次和跨通道泛化

**实验纪律：**

1. 固定数据划分、seed、代码版本和 config。
2. 每个主要结论至少运行多个 seed。
3. 训练、验证和测试指标分别保存，不用 test set 选择 checkpoint。
4. 所有实验输出写入 `output/`，日志写入 `logs/`，不在项目根目录落文件。
5. 生成机器可读 metrics JSON 和论文表格来源文件。
6. 明确区分工程成功、科学等价和科学提升三种结论。

**验收条件：**

- 一个 `unitypsf_joint.ckpt` 可实例化一个处理三种 PSF 的 `UnityPSF` 模型。
- 双通道像散实例保持参数、物理状态和 checkpoint 独立。
- 所有主要结论有基线和消融支持。
- 性能、显存和存储开销被完整报告。
- 论文中只能使用实验实际支持的“硬路由多专家”表述，不把未实现的联合学习或软路由写成贡献。

**验证方式：** 通过固定 SLURM 实验矩阵运行，检查每个作业退出码、配置快照、指标文件和 artifact hash，再生成汇总报告。

**依赖：** 任务 16、任务 17。

**预计规模：** 大型实验阶段，按模态和消融拆成多个独立批次。

## 12. 最终 joint checkpoint 结构

```text
unitypsf_joint.ckpt
    schema: unity_psf.joint_checkpoint.v1
    model_identity: UnityPSF
    supported_modalities: [...]
    contracts:
        input / output / coordinate / photon / reconstruction
    router:
        deterministic resolver / optional detector
    experts:
        emitter_2d:
            main: model / FiLM / metadata
        astigmatism:
            left: model / FiLM / peak zmap / gamma / physical state
            right: model / FiLM / peak zmap / gamma / physical state
        double_helix: optional until scientifically validated
    calibration:
        versioned inference-required payloads
    provenance:
        data manifest hashes / code / config / seed
    training_state: optional for resume role
    integrity:
        nested payload hashes / whole-file SHA-256
```

第一个双模态 + 多通道里程碑的 `supported_modalities` 只包含 `emitter_2d` 和
`astigmatism`，checkpoint 内包含 Emitter2D main 与 Astigmatism left/right。没有经过
训练与科学验收的 Double Helix 不写入空状态，也不标记为 supported。

joint checkpoint 至少记录：

- checkpoint schema、UnityPSF 版本和唯一模型身份。
- 支持的 PSF 模态。
- measurement channel layout。
- 所有已训练实例的嵌套 state、channel binding 和 payload hash。
- 每个实例的 `parent_checkpoint_hash`。
- peak zmap、physical state 和 DH calibration 的内嵌推理状态与 hash。
- modality detector 的版本、阈值和训练数据 manifest hash。
- 公共坐标、z、photon 和重建 contract 版本。
- release/resume role；resume 额外记录 optimizer、scheduler、AMP、RNG 和 step。

## 13. 风险与处理方式

| 风险 | 影响 | 处理方式 |
| --- | --- | --- |
| 输入帧和测量通道继续混用 | 高 | 先完成任务 2，并让所有后续配置只依赖新 contract |
| 复制后的实例仍共享参数 | 高 | 使用重新构造加 strict state load，并检查 Parameter 身份和 `data_ptr()` |
| 新 checkpoint 破坏旧模型加载 | 高 | 新格式只写 v2，旧格式保留只读 loader，等价性通过后再弃用 |
| `run_high_fidelity.py` 继续膨胀 | 高 | 新逻辑进入聚焦模块，大文件只保留薄接线；结构调整按小批次执行 |
| left/right 物理状态串写 | 高 | 每个 channel 使用独立 run layout、condition store 和原子写入 |
| router 错误选择 PSF | 高 | 确定性 metadata 优先，检测器做置信度校准和拒识 |
| 三个完整专家显存过高 | 中 | 在构造前完成硬路由，只将选中专家放到 GPU |
| 三种模态输出单位不一致 | 高 | 公共输出 contract 版本化，并在每种 expert 验收中检查 |
| joint checkpoint 不完整或部分写入 | 高 | barrier 后由 rank 0 原子提交，保存后立即重载并校验所有 required instance |
| 单一总 loss 掩盖某个模态失败 | 高 | 指标和图按 modality/channel 分开，发布门要求每一项分别达标 |
| Origami spool 被误当作配准多通道 | 高 | 第一版只用 `Emitter2D(main)`；无配准证据不构造虚假 left/right 配对 |
| 实验数据泄漏导致结果虚高 | 高 | 按显微镜和 acquisition batch 分组划分，并保存数据 manifest |
| “真正 MoE”论文表述过度 | 高 | 明确称为 modality-routed hard MoE，并用消融证明条件计算价值 |

## 14. 首个可用版本明确不做的内容

- 不修改 Neptune v0.3。
- 不定义固定的 `LeftAstigmatismExpert` 或 `RightAstigmatismExpert`。
- 不让实例共享 Parameter、optimizer、peak zmap、gamma state 或 checkpoint。
- 不在通道训练结束后自动更新 `astigmatism_base.ckpt`。
- 不在首版合并 left/right batch、loss 或梯度。
- 不在首版增加跨通道 consistency loss。
- 不在像散等价性通过前删除 SoftMoE 和 legacy checkpoint 路径。
- 不在首版支持同一 batch 内混合多种 PSF。
- 不把 shared-backbone、soft routing 或 top-k routing 作为首版生产架构。
- 不要求图像 detector 阻塞前三种确定性路由的交付。
- 不把 per-expert checkpoint 作为正式发布接口；它们只允许作为迁移输入或内部恢复材料。
- 不在 DH 数据与科学验收完成前写入空 DH expert 或宣称三模态支持。
- 不等待 DH 数据才交付第一版；先冻结双模态 + 多通道里程碑。

## 15. 实施过程约束

1. 每个任务开始前重新读取将修改的文件及其直接调用者。
2. 每个实现批次最多修改 5 个文件；特殊情况不得超过 7 个文件。
3. 每个任务先增加能表达业务意图的测试，再修改实现。
4. 每完成 2-3 个任务必须通过对应验收门，并由人工确认是否继续。
5. GPU 训练、长评估和 CUDA smoke 统一通过 SLURM 执行。
6. 运行产物写入 `output/`，日志写入 `logs/`，缓存写入 `.local/cache/` 或 `.local/tmp/`。
7. 新路径稳定前保留旧路径；删除兼容代码必须单独审计所有调用面。
8. 每次验收记录命令、退出码、config、seed、指标和 diff 范围。
9. 每个正式训练 run 必须生成 `report.html`、固定图组、`summary.json` 和 checkpoint hash。
10. 总指标不能替代 modality/channel 分项；视觉结果不合理时即使测试通过也必须停止。

## 16. 完成标准

只有满足以下全部条件，才能认为 UnityPSF 已完成目标架构：

- 顶层恰好存在三个规范完整专家类型。
- 单通道像散从 `astigmatism_base.ckpt` 创建一个 `main` 实例。
- 多通道像散从同一原型创建多个初值相同、训练后完全独立的实例。
- 每个实例独立拥有 FiLM、optimizer、scheduler、peak zmap、gamma state 和 checkpoint。
- 硬路由只构造和执行被选中的 PSF 专家与 channel 实例。
- 当前像散 left/right crop、过滤、重建和 union 行为通过等价性验收。
- 二维发射体和双螺旋分别通过自己的科学验证。
- 模态检测器具有校准置信度和拒识能力。
- 一个可移动且经过 hash 校验的 `unitypsf_joint.ckpt` 可以实例化一个顶层
  `UnityPSF` 模型，并路由全部已声明支持的模态和通道。
- 第一里程碑完成 Origami Emitter2D main 与 Astigmatism left/right 的双模态 +
  多通道训练、推理、可视化报告和 joint checkpoint。
- DH 数据到位前，两模态 checkpoint 明确拒绝 DH；DH 验收后在同一 schema 下升级三模态。
- 每个正式结论都能追溯到 GPU run、固定 config、结果图、指标和 checkpoint hash。
- Neptune v0.3 保持独立可运行。
