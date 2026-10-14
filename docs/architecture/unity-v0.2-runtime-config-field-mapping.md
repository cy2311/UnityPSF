# Unity v0.2 Runtime Config Field Mapping

状态：Phase 3 正式契约

本文记录 localization runtime 配置从输入 YAML 到 materialized runtime 的实际字段流。它是边界说明，不是新的 schema framework，也不引入第二个 config facade。

## 1. 唯一入口与责任边界

唯一 public builder 是 `build_localization_runtime_config`，位于 `src/unity_psf/localization/runtime/config.py`。它只编排以下 owner 并组装最终 runtime：

| Owner | 责任 | 不负责 |
| --- | --- | --- |
| `model_config.py` | model/expert route、model 参数和 model 语义校验 | provider 或 loss 默认值 |
| `loss_config.py` | loss name、legacy loss 字段迁移、active SMLM/GMM 参数和 loss 语义校验 | model route、provider 路径 |
| `contracts.py` | input frame、channel layout、expert instance、resolved contract | 训练对象实例化 |
| `provider_config.py` | online/microtube provider 参数、物理 ranges、camera/provider materialization、路径解析 | model/loss contract |
| `optimizer_config.py` | optimizer、legacy optimizer/scheduler 和 training runtime contract | 模型或 provider 参数 |
| `config.py` | schema 边界检查、调用各 owner、组装 runtime | 再实现任何字段级解析 |

训练入口、`training/runtime.py` 和 evaluator 只消费 materialized runtime；不从 CLI 或另一个 config facade 反向读取原始字段。

## 2. Schema 版本策略

`build_localization_runtime_config` 是 materialization boundary，不替代 joint schema loader。它显式接受：

| `schema_version` | 状态 | 说明 |
| --- | --- | --- |
| `0.4` | 保留 | 仍有 astigmatism smoke、baseline/replay 测试消费者；只做字段级迁移，不改写输入 mapping。 |
| `unitypsf.instance_training.v1` | 正式 | emitter-2D、astigmatism 和 DH instance training 配置。 |
| 缺省 | 保留 | 内存构造和旧内部调用没有根版本字段；由具体 owner 的字段校验负责语义边界。 |
| 其他显式版本 | 拒绝 | `config.py` 抛出 `ValueError`，不把未知版本静默当作 v1，也不改写 source config。 |

`unitypsf.joint_training.v1` 由 `training/joint_config.py::load_joint_config` 在 joint loader 边界校验；joint 文件先通过 `bind_instance` 变成 instance config，再进入本 builder。

## 3. 字段映射

| 输入 schema 字段 | legacy alias / migration | canonical contract | materialized runtime | owner / validation |
| --- | --- | --- | --- | --- |
| `schema_version` | 无 | accepted root version | 不复制到 runtime | `config.py`; unsupported explicit version -> `ValueError` |
| `train.device` | 无 | device selection | `device: str` | `config.py`; string materialization |
| `train.epochs` | 无 | epoch interval | `epochs.start=1`, `epochs.stop=int(...)` | `config.py`; integer conversion |
| `train.max_batches` | 无 | optional training limit | `max_batches`；同时进入 `resolved_contract.training_runtime` | `config.py` / `optimizer_config.py` |
| `train.model.name` | `model.name` 可与 `model.params` 同层读取 | resolved model name | `model.name`、`resolved_contract.model.name` | `model_config.py`; modality/model mismatch -> `ValueError` |
| `train.model.params` | legacy flat params 保留在 model mapping | resolved model params | `model.params` | `model_config.py`; condition dimension、Emitter2D z-disabled 校验 |
| `train.expert.name` / `expert_type` | `astigmatism_expert`、`emitter` 等 modality aliases | `expert_instance.expert_type` | model route 和 `expert_instance` | `conditioning_config.py` / `contracts.py` |
| `train.input_frame_spec.input_frame_channels` | `channels` -> canonical；发 `DeprecationWarning` | `input_frame_spec.input_frame_channels` | top-level `input_frame_spec` 与 provider `channels` | `InputFrameSpec.from_value`; conflict -> `ValueError` |
| `train.input_frame_spec.frame_order` | `order` -> canonical；冲突不覆盖 | `input_frame_spec.frame_order` | top-level `input_frame_spec` | `InputFrameSpec.from_value`; invalid shape -> `ValueError` |
| `train.channel_layout` | channel `id` -> `channel_id` | channel list、crop、frame size | top-level `channel_layout`、resolved contract | `ChannelLayout.from_value`; duplicate/invalid crop -> `ValueError` |
| `train.expert.instance_id/channel_id` | 单通道默认 `main`；joint binding 写入实际 channel | `expert_instance` | top-level 和 resolved contract | `ExpertInstanceSpec.from_value`; channel 不在 layout -> `ValueError` |
| `train.online_generation.channels` | 与 input frame canonical 字段保持一致 | input width/provider channel count | provider `params.channels` | `contracts.py` / `provider_config.py`; mismatch -> `ValueError` |
| `train.online_generation.condition_*` | legacy soft-MoE fields 保留 | condition dimensions/fields | model params、provider params、resolved provider contract | `model_config.py`, `conditioning_config.py`, `provider_config.py` |
| `train.online_generation.domain_count` | 单 channel formal route 强制为 1 | domain count | provider `domain_count`、resolved contract | `provider_config.py`; coefficient map count mismatch -> `ValueError` |
| `train.online_generation.dual_domain_coeff_maps` | legacy `path` / `alternating_coeff_maps_npz` 仍按字段解析 | resolved coefficient-map entries | provider params、resolved contract | `provider_config.py`; mapping/path type and count validation |
| `train.online_generation.pupil_carrier_complex_npz` | 无 | carrier array | provider `pupil_carrier_complex` | `provider_config.py`; path/shape validation |
| `train.online_generation` physical fields | `optical.*`、`simulation.*`、`scaling.*` 作为 fallback source | normalized ranges and optical values | provider params (`z_range`, `photon_range`, `na`, pixel size, LUT settings) | `provider_config.py`; range/number validation |
| root `camera.*` | online camera fields remain fallback | camera semantics | provider `camera_*` params | `provider_config.py`; numeric materialization |
| `train.loss.name` | omitted name + flat legacy GMM keys -> `active_smlm_gmm_loss` | loss name | `loss.name`、resolved loss | `loss_config.py`; modality incompatibility -> `ValueError` |
| `train.loss.params` | flat `gmm_*` keys are collected as `legacy_params` when applicable | canonical loss params | `loss.params`、resolved contract loss | `loss_config.py`; unknown params/order/z-disable checks |
| `train.scaling.photon_max/z_max/bg_max` | `normalization.photon_scale` and `train_params.z_max` are legacy fallbacks | physical scales | provider/loss photon and z scale; background range | `loss_config.py` / `provider_config.py` |
| `train.optimizer` | `learning_rate` and `smlm_overrides` are legacy sources | optimizer and scheduler contract | `optimizer`、`resolved_contract.training_runtime` | `optimizer_config.py`; no second optimizer facade |
| `smlm_overrides.lr_scheduler/lr_step_*` | retained only when an active scheduler is configured | scheduler contract | `training_runtime.scheduler` | `optimizer_config.py`; unsupported wiring is marked inactive, not silently activated |
| `train.feedback.map_path` | 无 | feedback artifact path | optional top-level `feedback.map_path` | `config.py`; string materialization |

## 4. Formal contract 与 materialized runtime

`resolved_contract` 是对实际 materialized objects 的可审计摘要，不是第二份输入配置。其字段来自已构造的 `model`、`batch_provider`、`loss`、`optimizer` 和 modality contract：

```text
runtime.model                 <- model_config
runtime.batch_provider        <- provider_config
runtime.loss                 <- loss_config
runtime.optimizer             <- optimizer_config
runtime.input_frame_spec     <- contracts / InputFrameSpec
runtime.channel_layout        <- contracts / ChannelLayout
runtime.expert_instance       <- contracts / ExpertInstanceSpec
runtime.resolved_contract     <- above materialized values + training runtime metadata
```

formal channel 集合由 `training/modality_runtime.py` 审计：Emitter2D 为 `left + right`，Astigmatism 为 `left + right`，DH 为 `main`。`bind_instance` 只把 joint instance 绑定为单实例输入，不改变 modality 的正式 channel 集合。

## 5. 兼容与清理清单

| Surface | 当前决定 | 迁移/删除条件 |
| --- | --- | --- |
| `0.4` astigmatism smoke 与 baseline fixture | 保留 | 至少一个 replay/training consumer 仍使用；替换为 v1 fixture 并完成 snapshot parity 后再评估删除 |
| `InputFrameSpec.channels/order` | 保留兼容读取，canonical 输出 | 所有 fixture 和外部 replay 改为 canonical 后，经过一个 release cycle 再删除 |
| flat legacy GMM loss fields | 保留字段级读取 | 无 replay/checkpoint/config consumer 且 v1 loss params 已完成 parity 后删除 |
| `resolve_localization_model_config` public export | 已删除 | 内部 `_resolve_localization_model_config` 只由唯一 builder 使用 |
| 旧 runtime facade / defaults wrapper | 不保留 | 不得重新添加；任何新字段必须进入现有 owner |

## 6. 验收证据

`tests/training/test_runtime_config_contract.py` 固定：

- `0.4` astigmatism 的完整 model/provider/loss/training/modality resolved snapshot；
- v1 emitter-2D `left` bound instance 的完整 resolved snapshot；绝对 coefficient-map 路径只归一为文件名；
- `channels/order` legacy alias 的 warning、canonical output 和 source immutability；
- canonical/legacy input-frame 冲突继续抛出 `ValueError`；
- 未知显式 runtime schema 继续抛出 `ValueError`，且不改写输入。

这些测试验证的是 materialized state，而不是 helper 调用顺序。任何 snapshot 漂移必须先说明是正式默认值、schema migration 或物理 contract 的有意变化，再更新测试与本文映射。
