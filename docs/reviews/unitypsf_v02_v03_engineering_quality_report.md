# UnityPSF v0.2 与 v0.3 工程质量对比报告

> 用途：阶段性工作汇报与工程质量复盘
>
> 对比对象：`unity_v0.2` 与 `unity_v0.3`
>
> 复核日期：2026-10-12

## 一、结论摘要

从“能够运行研究实验”到“能够被复现、审计和持续演进的研究工程”，v0.3 已完成一次明显的工程化升级。

最重要的变化不是新增了多少模型代码，而是建立了统一的训练语义和可验证边界：

1. 用 `unity-psf-train` 统一正式训练入口，避免多个 launcher 各自解释配置。
2. 用 `ResolvedTrainingContract` 固化模态、通道、seed、step budget、配置、外部输入和 GPU 数量，并用 SHA256 contract hash 连接 plan、run、resume 和 checkpoint。
3. 用 v2 joint checkpoint 和 physical provenance 记录模型身份、路由、校准输入及其哈希，提升结果可追溯性。
4. 将 2D、Astigmatism、Double Helix 的路由、通道和物理初始化显式化，拒绝不合法的 route、map、PCA/PPG 输入和 GPU 数量。
5. 增加正式 GPU smoke gate、短跑 gate、held-out 检查、optimizer update 检查、resume shard 检查和可见 validation report。
6. 清理旧 Neptune/archive/legacy 训练入口，把仍有消费者的兼容面收敛成薄 shim，并补充兼容性文档。
7. 把训练仿真主路径从旧 LUT simulation 迁移到 CUDA-GEMM，并配套缓存窗口、FP16 AMP、性能配置和 CUDA 专用 smoke。
8. 形成 ADR、架构文档、质量脚本和冗余代码清单，使设计决策和清理依据能够被后来者复核。

综合判断：v0.2 更像“功能较多、实验路线较多的研究工作树”；v0.3 已接近“有明确 release contract 的研究软件包”。不过 v0.3 仍不是完全无债务的生产级工程：它仍依赖外部校准输入和 GPU/SLURM 环境，兼容层与历史配置仍需持续收敛，科学质量也不能由 smoke test 单独证明。

## 二、可量化的结构变化

以下统计排除了 `.local/`、缓存目录、`__pycache__` 和 `.pyc`，用于比较源码树形态；它们不是 Git 提交统计。

| 指标 | v0.2 | v0.3 | 变化说明 |
| --- | ---: | ---: | --- |
| 可见文件数 | 2460 | 1576 | 减少 884 个，主要来自历史脚本、缓存和旧运行入口收敛/移除 |
| Python 文件 | 289 | 269 | 代码规模减少，但公共边界和契约测试增加 |
| Python 代码行 | 65049 | 57072 | 减少约 12.3%，说明清理不是单纯堆功能 |
| 测试文件 | 57 | 58 | 数量基本持平，但测试主题从流程覆盖扩展到 contract/gate/兼容面 |
| 测试代码行 | 9010 | 7701 | 测试更集中于行为和边界，减少部分历史流程耦合 |
| ADR 文档 | 3 | 6 | 增加 channel-linked、resolved contract、formal entry lock 三类关键决策 |
| SLURM 脚本 | 50 | 22 | 旧 archive 和重复 launcher 被收敛为当前正式路线及兼容路径 |
| 开发依赖 | 仅 pytest | pytest + ruff + mypy | 增加 lint 和类型检查工具链 |

### 版本标识方面的改进

v0.2 目录名是 `unity_v0.2`，但 `README.md` 和 `pyproject.toml` 的包版本写成 `0.4.0`，存在报告、目录和发布版本不一致的问题。v0.3 的 README、包描述和 `pyproject.toml` 统一为 `0.3.0`，并明确 `unity_psf` 为公共命名空间，删除无消费者的 `neptune_v04` 发布别名。

## 三、具体完成的工程改进

### 1. 训练配置从“多处解释”升级为“单一解析合同”

v0.2 的训练选择、joint YAML、modality config、runtime builder、resume 和 checkpoint provenance 分别解释训练参数，存在 plan 与实际运行漂移的风险。

v0.3 新增 `ResolvedTrainingContract`：

- 记录 run id、run directory、epoch、所需 GPU、执行模式和 rank 分配方式。
- 记录每个 `modality:channel` 实例、model seed、data seed、step budget 和 resolved config。
- 记录实际使用的外部输入及 SHA256。
- 对语义内容做稳定序列化并计算 contract hash。
- `joint.yaml` 与 `training_contract.json` 从同一个对象生成。
- resume 和 checkpoint 使用同一 contract hash；修改 route、seed、budget、optimizer、anchor、PCA 输入或 resolved YAML 时会得到不同 hash。

证据：`unity_v0.3/src/unity_psf/contracts/training.py`；`unity_v0.3/src/unity_psf/cli/train_selection.py:127-248`；ADR 0005；`tests/contracts/test_resolved_training_contract.py`。

### 2. 正式训练入口收敛，阻止旁路发布 checkpoint

v0.3 将 `unity-psf-train plan|smoke|run|resume` 定义为唯一正式入口。内部 Expert Parallel executor 必须收到 canonical CLI 标记和匹配的 contract hash，直接启动内部模块会在训练前失败。正式 SLURM 脚本调用 CLI，不再各自直接调用内部 trainer。

这解决了 v0.2 中多个 joint/per-channel/三模态 launcher 并存带来的入口分裂问题，也让“什么运行可以产生正式 release checkpoint”变得明确。

证据：`unity_v0.3/docs/adr/0006-formal-training-entry-lock.md`；`unity_v0.3/docs/compatibility.md`；`unity_v0.3/tests/cli/test_train_selection.py:98-129`。

### 3. 路由、通道和输入校验前移到 CLI

v0.3 在训练启动前校验：

- route 格式、模态是否重复、channel 是否允许、channel 是否重复。
- map 是否对应已选择的单通道 2D/Astigmatism route。
- Double Helix 是否提供合法 PPG/PCA map 和 approved beads manifest。
- PPG schema、数组有限性、维度、held-out NCC gate。
- run id 是否为安全的单层目录名。
- 所需 GPU 数与可见 CUDA 设备是否匹配。

这些校验将错误从“训练跑起来后才发现”提前为“启动前失败”，减少昂贵 GPU 运行和不可解释实验结果。

证据：`unity_v0.3/src/unity_psf/cli/train_selection.py:47-124`；`unity_v0.3/tests/cli/test_train_selection.py`。

### 4. Checkpoint 从权重文件升级为可验证的模型交付物

v0.3 的 checkpoint metadata 增加并校验：

- schema version、model family、model name 和 checkpoint role。
- expert type、instance id、channel spec 和 condition schema。
- parent checkpoint hash、code version、experts、router mode。
- instance checkpoint 必须有 channel spec 和 parent checkpoint hash。
- release checkpoint 记录 resolved contract schema/hash 与 physical provenance。

同时保留旧 v1 reader 作为只读兼容面，新正式 release 要求 v2 contract provenance。这使 checkpoint 不再只是“某次训练生成的二进制权重”，而是包含身份、来源和完整性信息的可审计产物。

证据：`unity_v0.3/src/unity_psf/contracts/checkpoint_metadata.py`；`unity_v0.3/src/unity_psf/contracts/joint_checkpoint.py`；`unity_v0.3/docs/compatibility.md`。

### 5. 物理初始化和校准输入显式化

v0.3 将不同模态的物理初始化写入配置和 provenance：

- 2D 默认使用显式 `circular_2d_50nm` anchor。
- Astigmatism 默认使用 `astigmatism_660nm`、100 nm anchor。
- 选用样本 coefficient map 时要求文件存在、字段完整、值有效，并记录 SHA256。
- Double Helix 使用经过 manifest 校验的 PPG/PCA map，并记录输入哈希。
- 禁止把 coefficient map 与额外固定 Astigmatism anchor 叠加造成物理语义重复。
- 记录每个 modality/channel 的 anchor profile、anchor value、map 或 DH 输入来源。

这一步把“实验者脑中的初始化假设”变成了配置、代码和 checkpoint 中可检查的事实。

### 6. 增加 GPU smoke gate 和短跑质量 gate

v0.3 新增正式 GPU gate：

- 通过 `--require-cuda` 禁止 CPU fallback。
- 按所选模态数匹配 GPU/rank 数。
- smoke 要求每个 channel 有 finite loss、optimizer update、held-out loss。
- 检查 checkpoint route、resume shard progress 和 report 是否一致。
- formal gate 对外部输入生成 manifest 和 SHA256。

这让“程序能启动”与“每个实例确实更新过并输出正确 checkpoint”区分开来。文档也明确 smoke 只证明执行和接口健康，不等价于 localization accuracy。

证据：`unity_v0.3/docs/formal-gpu-gate.md`；`unity_v0.3/tests/configs/test_default_three_expert_scheme.py:13-83`；`unity_v0.3/tests/cli/test_train_selection.py:81-129`。

### 7. 训练性能路线统一到 CUDA-GEMM

v0.3 正式 Double Helix 配置将训练仿真主路径切换为 `cuda_gemm`，并配套：

- cached window 数据策略。
- vector batch size 1024。
- FP16 AMP。
- CUDA 专用 smoke 和 launcher。
- 缓存按 physical model version 失效并重新生成。

与此同时，旧 LUT simulation 分支、训练 LUT patch bank/cache 和历史 replay 入口按冗余审查清单处理；Double Helix 标定所需 LUT 仍保留在物理模块中，没有把“训练路径清理”误做成“删除所有 LUT 代码”。

证据：`unity_v0.3/docs/reviews/unitypsf-v03-redundancy-inventory-20261010.md:88-90`；`unity_v0.3/docs/performance/unitypsf-v03-liteloc-training.md`；`unity_v0.3/configs/modalities/double_helix/double_helix_raw_tiff_300epoch.yaml`。

### 8. 兼容面和源码边界得到治理

v0.3 完成了以下边界治理：

- 内部代码迁移到 canonical module path。
- 旧 `runtime_config.py`、`online.py` 等路径收敛成只转发 API 的薄 shim。
- 保留仍有真实消费者的 `legacy_decode.py`、`InstanceRouter` 和 `double_helix` wrapper。
- 删除无源码、无脚本、无 checkpoint 消费者的 `neptune_v04` 包别名。
- 移除历史 Neptune standard/archive、旧 fast-route 和 DH replay 入口。
- 用 `docs/compatibility.md` 记录 retained shims、正式入口和 removed surface。

这体现了“有消费者才保留兼容层，有替代路径才删除旧入口”的工程判断，而不是简单删目录。

### 9. 文档和审查机制从说明性文字升级为工程记录

相较 v0.2 的 3 个 ADR，v0.3 增加了：

- channel-linked refinement 的架构边界和验证指标。
- resolved training contract 的决策和后果。
- formal training entry lock。
- formal GPU gate 操作说明。
- compatibility surface 迁移说明。
- redundancy inventory，记录已清理、暂时保留和仍需统一的项目。

冗余审查还明确区分了无用 import/局部变量、仍有消费者的兼容代码、正式训练仍依赖的 ROI/gamma 路径，以及配置语义漂移。这比“把旧代码全部删掉”更符合科研软件的复现需求。

## 四、工程质量维度对比

| 维度 | v0.2 | v0.3 | 评价 |
| --- | --- | --- | --- |
| 正确性边界 | 主要由流程测试和配置约定保证 | route、contract、checkpoint、输入和 entrypoint 均有显式校验 | 明显提升 |
| 可复现性 | 运行参数分散在多个 YAML/launcher | resolved contract + hash + input hashes + seed/budget | 明显提升 |
| 可维护性 | 历史 archive、兼容 namespace、多个 trainer 并存 | canonical package、单一正式入口、薄 shim、旧路线隔离 | 明显提升 |
| 可测试性 | 以流程和数值模块为主 | 增加 contract、CLI、GPU gate、checkpoint provenance、兼容面测试 | 明显提升 |
| 可观测性 | 有 logs/output，但语义约束较弱 | manifest、rank status、resume shard、validation report、physical provenance | 明显提升 |
| 性能工程 | 多条旧训练仿真路径并存 | CUDA-GEMM、cached window、AMP、专门性能文档 | 明显提升，但仍需真实 GPU benchmark 支撑 |
| 文档治理 | 有架构文档，但版本和历史路径较混杂 | ADR、compatibility、formal gate、redundancy inventory 更系统 | 明显提升 |
| 发布一致性 | v0.2 目录名与包版本存在 v0.2/v0.4 混用 | v0.3 README/包版本/入口命名一致 | 明显提升 |
| 科学结论保证 | 测试不能替代真实数据验证 | smoke 明确声明不等于 localization accuracy | 认识更严谨，但科学验证仍需实验结果 |

## 五、验证结果

### 本次复核实测

在当前工作区和当前外部数据状态下执行：

```text
unity_v0.3: python -m pytest -q
285 passed, 1 skipped in 180.28s

unity_v0.2: python -m pytest -q
301 passed, 11 failed in 189.54s

unity_v0.3: mypy
Success: no issues found in 6 source files

unity_v0.3: ruff check src tests
未执行成功：当前环境未安装 ruff（command not found）

unity_v0.2 / unity_v0.3: python -m compileall -q src
命令退出 0；解释器输出了 home directory 不可写提示，但未报告编译错误
```

### v0.2 测试失败的解释

v0.2 的 11 个失败集中在 Double Helix/Zeiss/T-cell 物理初始化测试。失败原因是测试引用的外部产物不存在，例如：

```text
output/double_helix/calibration/tcell_dh_bead_phase_refresh_20260817/
independent_pixel_pupils/arrays/bead_01_complex_pupil.npz
```

因此不能把 v0.2 的结果写成“全部测试失败”，也不能把它写成“代码逻辑全部通过”。更准确的表述是：301 个测试通过，11 个测试因本地缺少外部校准产物而失败；这同时暴露了 v0.2 对工作区外部 artifact 的隐式依赖。

### v0.3 质量门禁的边界

v0.3 的 pytest 和 mypy 结果较好，但当前复核没有运行真实三 GPU SLURM smoke，也没有重新生成或验证完整外部校准输入。因此：

- `285 passed, 1 skipped` 证明当前 Python contract/test surface 通过。
- 它不证明真实三 GPU 训练性能已经达标。
- 它不证明外部 PPG、coefficient map、pupil 文件科学上正确。
- 它不替代 held-out localization、跨数据集泛化和真实 raw TIFF 结果。

## 六、仍需继续改进的地方

### 1. 把外部校准输入从隐式路径变成可交付输入包

v0.2 的失败说明仅在配置中写绝对路径仍不够。v0.3 已有 formal input staging 和 manifest，但正式复现还需要：

- 固定的 input bundle 目录结构。
- manifest 中记录来源、SHA256、尺寸和 schema。
- 在测试/CI 中对缺失输入进行明确 skip 或 fixture 注入，而不是让路径错误表现为深层 `FileNotFoundError`。

### 2. 连接真实 GPU CI

v0.3 文档已明确说明当前嵌套 `.github/workflows/quality.yml` 在父仓库根部并不自动生效。下一步应接入真实 GPU runner，自动执行 `--require-cuda` smoke，并保存 gate receipt。

### 3. 继续拆分当前配置中的历史字段

冗余清单指出，base config 仍保留部分旧 zmap 和兼容字段，例如 `dual_domain_coeff_maps`、`peak_zmap_bootstrap`、`regen_train_dataset_each_epoch`。这些字段需要逐项证明仍有消费者或移动到明确的 compatibility config，避免新成员误以为它们属于当前默认路线。

### 4. 处理兼容层的生命周期

当前兼容 shim 是合理的，但应为每个 shim 增加：

- 迁移目标。
- 保留原因。
- 最低支持版本。
- 删除条件。

这样可以防止“临时兼容代码”永久化。

### 5. 补充真实实验级质量指标

工程 contract 已经较完整，下一阶段需要把工程门禁和科学门禁连接起来：held-out localization error、precision/recall、uncertainty coverage、跨模态 registration residual、运行时间、显存和 checkpoint size，并将阈值写入正式 benchmark report。

## 七、汇报时可以直接使用的工作总结

本阶段我对 UnityPSF 做的核心工作，是把原先依赖个人经验和多入口脚本维持的研究代码，收敛成具有统一训练契约、明确模块边界和可复核产物的工程。具体包括：统一 `unity-psf-train` 正式入口；建立覆盖路由、seed、预算、配置和外部校准输入的 resolved training contract；用 contract hash 约束 resume 和 checkpoint；引入 v2 joint checkpoint、physical provenance 和输入哈希；增加 CUDA GPU gate、短跑验证、held-out 检查和 resume 检查；将训练仿真迁移到 CUDA-GEMM 并配套 cached-window/AMP 性能路线；清理旧 Neptune/archive/legacy 入口；保留必要兼容层并记录迁移规则；补充 ADR、架构说明、质量脚本和冗余代码审查清单；同时通过测试和类型检查验证关键边界。

可以将工程质量的阶段性结论概括为：**v0.2 已具备较完整的科研功能，但配置、入口、外部 artifact 和历史代码之间耦合较重；v0.3 在不牺牲科研路线的前提下，完成了训练契约化、发布物可追溯化、入口收敛化、兼容面显式化和质量门禁化，代码从“能跑的研究工作树”提升为“可以被复现和审计的研究软件包”。**

## 八、证据索引

- [v0.3 README](../README.md)
- [v0.3 pyproject.toml](../../pyproject.toml)
- [ResolvedTrainingContract](../../src/unity_psf/contracts/training.py)
- [训练选择 CLI](../../src/unity_psf/cli/train_selection.py)
- [Checkpoint metadata](../../src/unity_psf/contracts/checkpoint_metadata.py)
- [兼容性说明](../compatibility.md)
- [Formal GPU gate](../formal-gpu-gate.md)
- [ADR 0005: resolved training contract](../adr/0005-resolved-training-contract.md)
- [ADR 0006: formal training entry lock](../adr/0006-formal-training-entry-lock.md)
- [v0.3 冗余代码审查清单](./unitypsf-v03-redundancy-inventory-20261010.md)

<!-- dod-receipt cmd="python -m pytest -q (unity_v0.3)" exit=0 diff="+275/-0 (1 file)" -->
