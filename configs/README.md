# 配置目录

## Canonical 结构

正式配置分为两条入口，不能把联合实验 YAML 当作普通训练 override：

```text
单任务：base -> named override -> runtime --set / environment
联合训练：experiment -> project:// modality contract
```

同一个训练任务只允许有一个 YAML 来源负责一个参数。launcher 不再内联 YAML，
只负责 GPU/CPU 资源、输入路径、输出路径和启动命令。

| 目录或入口 | canonical owner | 可以覆盖 | 历史兼容状态 |
| --- | --- | --- | --- |
| `base/` | 共享光学、相机、训练循环和通用 provider 默认值 | 被命名 override 局部覆盖 | 保留旧字段，直到所有历史配置迁移 |
| `modalities/` | 单一 PSF 模态的模型、loss、channel layout 和模态物理合同 | 不从 launcher 覆盖 | 正式模态合同，仍被 joint experiment 使用 |
| `experiments/` | 联合执行方式、实例、seed、rank 和 step budget | 不复制模态字段 | 正式 joint contract |
| `calibration/` | 物理系统、载波、zmap 和校准输入 | 由具体实验选择 | 仍被 DH/物理更新流程消费 |
| `overrides/` | 明确命名的实验变体 | 只覆盖实际变化的字段 | 历史路线保留，但必须标明用途 |
| `scripts/train/*.sbatch` | 资源、路径、环境和启动顺序 | 通过 `--set` 传临时 budget | 不允许重新定义训练 YAML |

高保真单任务使用 `base + overrides`，由 `materialize_config()` 物化。
联合训练从 `experiments/` 读取执行计划，再通过 `project://` 加载每个
`modalities/` contract；联合实验不是普通的 `base + override` YAML。

## 正式字段登记

下面的登记覆盖当前 UnityPSF v0.3 正式 CUDA-GEMM 路线中容易漂移的字段。
未列出的字段遵循其所在 YAML 文件的 owner，不在 launcher 中重复定义。

| 字段 | canonical owner | 当前正式值/覆盖关系 | 历史兼容状态 |
| --- | --- | --- | --- |
| `train.online_generation.simulation_backend` | `base/`，正式路线由 `overrides/standard_roi_gamma.yaml` 固定 | `cuda_gemm` | LUT 值只允许历史配置读取，运行时已拒绝 |
| `batch_strategy`、`sequence_count`、`sequence_window_chunks` | `base/` | cached-window 通用默认；模态可按合同覆盖 | 保留旧 provider 字段 |
| `cached_window_cache_epochs`、`cached_window_prefetch_batches`、`cached_window_cache_storage` | `base/` | `8`、`1`、`cpu`；正式 override 只改变 cache frame budget | 新缓存合同；旧配置缺省时回退 base |
| `condition_feature_dim`、`condition_dim`、`domain_count` | `modalities/`；标准 microtube 变体由 `standard_roi_gamma.yaml` 覆盖 | `13`、`15`、`2`，`condition_dim = feature_dim + domain_count` | 两个维度字段都保留以兼容旧 checkpoint |
| `expert_mode`、`domain_balance_mode` | `modalities/` 或 named override | `soft_moe`、`alternate_step` | sparse/hard MoE 不属于当前默认路线 |
| `fixed_astig_anchor_nm`、`nat_wake.gamma_init_anchor_nm` | named override 与物理校准合同 | `100.0` | 旧 99 nm zmap 结果只作历史结果，不再是默认初始化 |
| `roi_bank_gamma.start_epoch`、`update_interval_epochs` | named ROI-Gamma override | `30`、`10` | `gamma_start25/50*` 是可复现历史变体 |
| ROI library size / emitters | `roi_library_max_rois` 由 `base/` canonical | `2048` / `5000` | `roi_bank_max_rois` 仅在主字段缺失时作为旧配置回退 |
| `train.epochs`、`train.batch_size`、`steps_per_epoch` | base/experiment；正式 microtube override 覆盖 batch/step；临时 profile 用 launcher `--set` | 默认 300 / 64 / 157 | batch-budget 通过独立 override 保留 |
| raw TIFF path | launcher 环境 `UNITYPSF_RAW_TIFF`，由 override 消费 | `${UNITYPSF_RAW_TIFF}` | `NEPTUNE_V04_RAW_TIFF_PATH` 仅作旧环境名，不写入新 YAML |
| coefficient map paths | launcher 环境 `UNITYPSF_LEFT_MAP` / `UNITYPSF_RIGHT_MAP` | 仅 from-scratch override 使用 | 历史绝对路径不再写入 launcher heredoc |

`--set` 只用于一次运行的资源或预算参数，例如 profile 的 epoch 数和最大
batch 数；它不会产生新的长期配置文件。所有可复现实验参数必须进入
`base/`、`modalities/`、`experiments/` 或命名 override。

联合实验使用 `project://` 引用模态配置，路径始终相对于 Unity 项目根目录，
不随当前 YAML 所在层级变化。例如：

```yaml
config: project://configs/modalities/emitter_2d/emitter_2d_single_channel_300epoch.yaml
```

## 迁移规则

1. 新实验先选择一个 canonical base 或 modality contract。
2. 只把与 base 不同的字段放入有明确名称的 override。
3. launcher 禁止 `cat <<YAML`、Python 内联改写 resolved YAML 或复制完整训练配置。
4. 删除字段前，先在本表标记 owner、覆盖关系和兼容状态，并通过全仓库搜索确认没有消费者。
