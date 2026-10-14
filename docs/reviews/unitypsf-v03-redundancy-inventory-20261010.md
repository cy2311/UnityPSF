# UnityPSF v0.3 冗余代码审查清单

- 审查日期：2026-10-10
- 审查范围：`unity_v0.3/src`、`unity_v0.3/scripts`、`unity_v0.3/configs`、`unity_v0.3/tests`
- 目的：识别已经没有用途的代码、配置和运行文件，供后续逐项处理。
- 当前状态：无用 import 已清理；训练用 LUT simulation 已迁移到 CUDA-GEMM。

## 验证基线

当前代码基线通过：

```text
pytest -q                  257 passed, 1 skipped in 60.27s
python -m compileall -q    exit 0
```

后续每一项清理都应在删除前后运行相关测试；涉及训练入口、配置或 checkpoint contract 的改动，应额外运行完整测试集。

## 结论摘要

### 可以直接清理

这些项目已经有静态证据表明当前文件内没有使用，删除风险低：

1. 无用 import。
2. 无用局部变量。
3. 已更名但残留的类型注解名称。
4. 缺失的类型导入。

### 暂时不能删除

这些代码虽然不是当前 `5705` CUDA-GEMM 训练路线的全部组成部分，但仍被其他配置、测试、历史复现入口或兼容 API 使用：

1. `legacy_decode.py` 的 LiteLoc 风格评估逻辑。
2. `runtime_config.py`、`online.py` 等兼容导出。
3. ROI library、posterior sampling、gamma physical update。
4. 旧 checkpoint schema 和 compatibility wrappers。

### 需要先统一语义再处理

以下内容不是简单的死代码，而是当前默认路线和历史路线混在一起：

1. README 已统一写成 Soft-MoE FiLM；正式 checkpoint 仍使用显式 deterministic routing。
2. base config 已统一使用 `100 nm` anchor。
3. base config 已关闭 peak-zmap bootstrap；正式 CUDA-GEMM 路线按物理状态生成流程处理初始模型。
4. base config 仍保留旧 zmap 和 `dual_domain_coeff_maps` 字段。

## A. 第一批：可直接清理的源码项

### A1. 无用 import（已处理）

| 文件 | 位置 | 说明 |
| --- | --- | --- |
| `src/unity_psf/gamma_update/objective.py` | 第 9 行 | `torch.nn.functional as F` 未使用。 |
| `src/unity_psf/infer_recon/filter/filter.py` | 第 5 行 | `json` 未使用。 |
| `src/unity_psf/training/multichannel.py` | 第 12 行 | `typing.Any` 未使用。 |
| `src/unity_psf/localization/materialized_eval.py` | 第 11 行 | `TrainingBatch` 未使用。 |
| `src/unity_psf/localization/data/online.py` | 第 45 行 | `_normalize_pxyz_target_order` 未使用。 |
| `src/unity_psf/localization/runtime/config.py` | 第 17 行 | `_is_soft_moe` 未使用。 |
| `src/unity_psf/models/psf_moe/__init__.py` | 第 5、20 行 | `Mapping` 和 `Emitter2DExpert` 未使用。 |
| `src/unity_psf/training/entrypoints/train_modality_expert_parallel.py` | 第 23 行 | `ModalityTrainingRuntime` 未使用。 |
| `src/unity_psf/training/entrypoints/train_modality_joint.py` | 第 19 行 | `ModalityTrainingBatch` 未使用。 |

上述项目已完成清理。没有列入表格的兼容 import 仍然保留：`run_high_fidelity.py`、`high_fidelity/engine.py` 和三个 `cli/train_*.py` adapter 中的私有 helper 会被测试或兼容调用方直接导入，不能按普通死代码删除。

### A2. 无用局部变量（已处理）

| 文件 | 位置 | 变量 |
| --- | --- | --- |
| `src/unity_psf/training/run_roi_library_smoke.py` | 第 181-182 行 | `roi_origin_xy_px`、`domain_names` |
| `src/unity_psf/localization/posterior.py` | 第 119 行 | `height` |
| `src/unity_psf/localization/data/online.py` | 第 2847 行 | `feature_dim` |
| `src/ppg/diagnose_bg_mean_sweep.py` | 第 69 行 | `windows` |

上述变量已删除，没有改变计算路径。

### A3. 过时类型名和缺失导入（已处理）

| 文件 | 位置 | 问题 | 处理建议 |
| --- | --- | --- | --- |
| `src/unity_psf/optics/vector_psf.py` | 第 268 行 | `VectorPSFTorchFit` 已不存在，实际类是 `LocalVectorPSFTorchFit`。 | 修正类型注解。 |
| `src/unity_psf/training/high_fidelity/engine.py` | 第 1957、2061 行 | 使用 `Sequence`，但没有导入。 | 已补充 `from typing import Sequence`。 |

`vector_psf.py` 的类型注解和 `Sequence` 导入均已修正；同时删除了该文件重复的 `math` import。

## B. 暂时保留：仍有真实消费者的代码

### B1. LUT simulation（已处理）

上述正式配置和两个训练 launcher 已改用 `cuda_gemm`，并显式打开相机读出以保留原 LUT 路径的 ADU 输入域。5705 的默认光子域训练配置不受该开关影响。`localization/data/online.py` 的训练 LUT 分支、patch bank/cache、预热及专用参数已删除。runtime config 对旧 `lut` backend 和 `lut_simulation` 字段明确报错。DH 标定 LUT 仍保留在当前 Double Helix 物理模块中；历史 replay 脚本已按 F2 删除。

### B2. LiteLoc 风格 decode 和 metrics

`src/unity_psf/localization/legacy_decode.py` 仍被：

- `training/localizer_eval.py`
- PPG 训练与诊断模块
- 多个 localization/training 测试

使用。它是评估 contract 的一部分，不是废弃代码。

### B3. 兼容 facade

以下模块虽然代码很薄，但仍是现有 import surface：

- `src/unity_psf/localization/runtime_config.py`
- `src/unity_psf/localization/online.py`
- `src/unity_psf/models/psf_moe/router.py` 中的 `InstanceRouter`
- `src/double_helix/` 顶层 wrapper

本轮已完成：

1. 内部训练、评估、测试和 launcher 已迁移到 `localization.runtime.config` 与
   `localization.data.online`。
2. 两个旧模块路径收敛为只转发 canonical API 的薄 shim。
3. `neptune_v04` 无源码、脚本或 checkpoint 消费者，已从源码和打包配置删除。
4. `InstanceRouter`、`legacy_decode.py` 和 `double_helix` 顶层包仍有真实消费者，继续保留。
5. 兼容面和迁移规则记录在 `docs/compatibility.md`。

### B4. ROI library 和 physical update

5705 的训练结果依赖：

```text
raw TIFF
-> ROI library
-> posterior sampling
-> gamma update
-> physical state/cache refresh
```

以下模块仍属于当前正式训练路径：

- `src/unity_psf/roi_library/`
- `src/unity_psf/localization/roi_posterior_sampling.py`
- `src/unity_psf/gamma_update/`
- `src/unity_psf/training/high_fidelity/engine.py`
- `src/unity_psf/training/high_fidelity/physical_state.py`

这些不能作为冗余代码删除，只能继续做局部重构或性能优化。

### B5. Double Helix legacy 和 archive（已处理）

历史 Double Helix replay、Neptune standard/archive 和旧 fast-route 入口已经从最终 v0.3 树中删除。当前保留的是 `src/double_helix/` 顶层的正式校准与物理模块，以及 `unity_psf` 的 CUDA-GEMM 训练入口。

## C. 配置和文档漂移

### C1. README 的 MoE 描述（已统一）

README 已明确区分 deterministic checkpoint routing 和当前训练使用的
Soft-MoE FiLM experts。

### C2. anchor 默认值（已统一）

`configs/base/microtube.yaml` 和当前正式 CUDA-GEMM 训练路线均使用
`gamma_init_anchor_nm: 100.0`。

### C3. peak zmap bootstrap 默认值（已统一）

`configs/base/microtube.yaml` 和 5705 正式路线均关闭 `peak_zmap_bootstrap`：

```yaml
peak_zmap_bootstrap:
  enabled: false
```

这不是死代码，而是默认实验定义不统一。建议拆分：

- 当前 microtube CUDA-GEMM 默认配置。
- 旧 zmap/peak bootstrap compatibility 配置。

### C4. 旧 zmap 字段仍在 base config

以下字段在当前 5705 路线上不生效或被覆盖：

- `online_generation.dual_domain_coeff_maps`
- `peak_zmap_bootstrap`
- `regen_train_dataset_each_epoch`

`online_generation.lut_simulation` 已删除。其余字段待分别核对用途后处理。

## D. 运行产物：不是源码，应单独清理

当前工作区存在以下本地运行文件：

| 路径 | 约占用 |
| --- | ---: |
| `unity_v0.3/.local/` | 113 MB |
| `unity_v0.3/output/` | 759 MB |
| `unity_v0.3/logs/` | 98 MB |
| `unity_v0.3/.mypy_cache/` | 103 MB |
| `unity_v0.3/.pytest_cache/` | 64 KB |
| `__pycache__/`、`*.pyc` | 分散存在 |

这些路径已经在 [`.gitignore`](/home/guest/Others/main/race/unity_v0.3/.gitignore) 中排除。清理时应区分：

- 可直接重建的 cache、`__pycache__`、pytest/mypy cache。
- 仍需保留的 checkpoint、metrics、diagnostic figures 和训练日志。

不要用一次性递归删除覆盖整个 `output/`，因为其中包含 5705 的可复核结果。

## E. 建议处理顺序

### Phase 1：低风险源码清理

- 删除 A1 的无用 import。
- 删除 A2 的无用局部变量。
- 修正 A3 的类型名和 `Sequence` 导入。
- 运行完整测试：`pytest -q`。

### Phase 2：统一当前默认路线

- 修正 README 的 hard/soft MoE 描述。
- 决定 base config 是否切换到 `100 nm` anchor。
- 把 CUDA-GEMM 默认配置和 LUT/旧 zmap compatibility 配置分开。
- 为配置添加测试，防止 anchor、backend 和 bootstrap 状态再次漂移。

### Phase 3：旧路线清理（已完成）

`scripts/archive`、`src/double_helix/legacy`、两个旧 fast-route launcher 和无消费者的
`neptune_v04` 别名已确认不属于最终主线并删除。仍需维护的是旧 checkpoint 只读加载合同、
LiteLoc 风格评估合同和 DH 校准 API。

## 当前不应删除的清单

在没有新证据前，保留以下模块：

- `localization/data/online.py` 的 native/CUDA-GEMM/cached-window 实现。
- `training/high_fidelity/engine.py`。
- `training/loop.py`。
- `localization/legacy_decode.py`。
- `localization/runtime/config.py`。
- `roi_library/`。
- `gamma_update/`。
- `peak/`。
- 旧 checkpoint contract 和 compatibility wrapper。

## 工作区状态说明

当前 `unity_v0.3` 在父仓库中整体显示为未跟踪目录。后续整理源码前，应先确认 v0.3 的 canonical Git 边界，否则普通 `git diff` 无法准确反映单个文件的清理变化。

## F. 可删除性审计（2026-10-10）

本轮审计目标是区分真正无消费者的代码、兼容层和仍属于正式训练路径的模块。结论如下。

### F1. 已完成：旧版共享 Stem PSFMoE 脚手架

当前正式模型路线是：

```text
UnityPSF
-> ModalityRouter / InstanceRouter
-> AstigmatismExpert / Emitter2DExpert / DoubleHelixExpert
-> Soft-MoE FiLM
```

该早期共享特征 Stem 架构已从正式代码中移除。删除范围包括：

- `PSFMoE`、`SharedPSFStem`、`AdaptedPSFExpert`、`PSFExpertHead`。
- `PSFRouter`、`RoutingResult` 和旧 shared-stem dispatch contract。
- `LegacyAstigmatismExpert`、`LegacyEmitter2DExpert`、`LegacyDoubleHelixExpert`。
- `PSFExpertOutput` contract 及其公开导出。

同时删除了只承载旧架构的 `models/psf_moe/base.py`，并将模型公开入口收敛到当前完整 Expert 实现。预计清理约 300--400 行源码和兼容导出。

迁移和验证已完成：

1. 删除旧 `PSFMoE` 测试，当前 `UnityPSF`/完整 Expert contract 测试继续覆盖路由、输出和 checkpoint 行为。
2. 在 `unity_v0.3/src`、`tests`、`scripts` 和 `configs` 内搜索确认不再存在 `PSFExpertOutput`、`SharedPSFStem`、`PSFRouter` 和 `Legacy*Expert` 引用；并存的 `unity`、`unity_v0.2` 历史版本不属于本次清理范围。
3. 完整测试通过：`257 passed, 1 skipped in 60.27s`。
4. checkpoint 加载路径仍由 `UnityPSF._build_expert` 构造当前 `Emitter2DExpert`、`AstigmatismExpert` 和 `DoubleHelixExpert`，不依赖旧类名反序列化。

### F2. 已完成：历史训练路线

根据最终版本决策，以下无 active consumer 的历史入口已删除：

- `scripts/train/fast_route_3052_strength_oldzmap.sbatch`。
- `scripts/train/fast_route_3367_peakbootstrap.sbatch`。
- `scripts/archive/` 下 42 个历史训练、校准和复现脚本。
- `src/double_helix/legacy/` 下 16 个历史 replay/诊断模块。
- 只依赖上述入口的 `scripts/gui/submit_training_gui.py` 和 `submit_training_web.py`。

当前 peak bootstrap、zmap 和 physical update 代码仍保留在正式 `unity_psf` 训练路径中；删除的是旧 launcher 和旧 replay 实现，不影响当前 CUDA-GEMM 训练。

### F3. 已完成：配置 owner 和薄 launcher

配置已收敛为 `base + modality/experiment contract + named override + runtime override`，
其中 runtime override 使用 `--set`。字段没有被直接删除；重复项从正式 override 移除后由 base 继承，
历史实验仍保留独立的命名 override。

已完成的改动：

1. `configs/base/microtube.yaml` 显式声明 `cached_window_cache_epochs: 8`，不再依赖 Python 默认值。
2. 新增 `configs/overrides/cuda_gemm_from_scratch.yaml`，承接从零训练的 CUDA-GEMM、zmap 路径和评估参数。
3. `configs/overrides/standard_roi_gamma.yaml` 成为正式 ROI-Gamma 路线的 canonical 差异配置；重复 base 默认值已移除，保留 batch/step、ROI geometry、100 nm anchor、gamma schedule 和兼容别名。
4. `unitypsf_cuda_gemm_from_scratch.sbatch`、`unitypsf_cuda_gemm_roi_gamma.sbatch` 和 batch-budget launcher 删除 inline YAML，统一通过 `unity_psf.config.materialize` 加载 canonical 文件。
5. `materialize` 新增重复可用的 `--set dotted.path=value`，profile 和 batch budget 不再通过 Python 内联改写 resolved YAML。
6. active smoke override 的 TIFF 环境变量统一为 `UNITYPSF_RAW_TIFF`；旧 `NEPTUNE_V04_RAW_TIFF_PATH` 不再写入新配置。
7. `configs/README.md` 增加目录 owner、覆盖优先级、正式字段登记和历史兼容状态。

验证结果：

- 完整测试：`260 passed, 1 skipped`。
- 配置测试：`22 passed`；F3 layering contract：`3 passed`。
- 所有 `scripts/train/*.sbatch` 通过 `bash -n`，所有 `configs/**/*.yaml` 可解析。

剩余的 `gamma_start25/50*`、`standard_roi_gamma_smoke3.yaml` 和 batch-budget
override 是有明确历史用途的变体，不与正式 launcher 混用，也未删除字段。

### F4. 已完成：配置入口和 launcher 静态清理

本轮完成了 F4 的前三项低风险修复：

1. `config/load.py` 的 `materialize_config()` 现在复用 `resolve_config_reference()` 正确解析 override 中的 `project://` coefficient map 路径。
2. 两个 CUDA-GEMM launcher 删除硬编码的 resolved-config 断言；这些 contract 检查集中到 `tests/configs/test_config_layering.py`。
3. 删除 batch-budget launcher 中没有消费者的 `APPEND_DOMAIN_ONEHOT` 和 `DOMAIN_BALANCE_MODE` 环境变量。
4. `materialize_config()` 对未展开的 `${VAR}` 直接报错，避免生成不可运行的 resolved config。
5. 移除 standard ROI-Gamma metadata 中重复的训练控制字段，只保留描述性 metadata。
6. README 明确区分单任务 `base + override` 和联合训练 `experiment -> modality` 两条入口，并修正 ROI owner。

新增测试覆盖 `project://` 物化、未展开环境变量和 launcher 无重复配置逻辑；完整测试通过：`262 passed, 1 skipped`。

### F4. 其余静态清理项

静态检查还发现：

- 若干 CLI adapter 导入私有 helper；这些 import 看似未使用，但属于兼容导出，不应按普通死代码删除。
- `localization/online.py`、`localization/runtime_config.py`、`localization/runtime/paths.py` 仍有真实调用方，应保留。

### F5. 当前判断

UnityPSF v0.3 的主要剩余整理空间集中在兼容导出和大模块的局部拆分，正式配置主路径已经完成 owner 收敛。

推荐清理顺序：

1. 清理兼容导出和已确认的无用 import。
2. 在新增实验中沿用 `configs/README.md` 的 owner 规则。
3. 再评估 `engine.py`、`online.py` 等大模块的局部拆分。
