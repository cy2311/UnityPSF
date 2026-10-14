# unity_v0.2 开发复杂度 Review 与后续路线图

- Review date: 2026-08-21
- Scope: `/home/guest/Others/main/race/unity_v0.2`
- Status: current review, debt inventory, and execution roadmap
- Last local verification: `253 passed, 4 warnings` (external Neptune comparison removed)

## 1. Executive Summary

Unity v0.2 的主要结构性问题已经在 Phase 1-7 收敛。当前核心训练/runtime 代码不再存在多条 formal 实现路径、入口私有 API 泄漏或 runtime config 双 facade。Phase 8 已完成 GPU/Slurm 300 epoch end-to-end parity 验收。

剩余债务主要属于四类：

1. `unity_v0.2` 仍是父仓库未跟踪副本，而 GitHub `UnityPSF/main` 已有已发布版本；二者需要收敛为一个 canonical checkout；
2. archive/compatibility 的 `42 + 19 + 16` 条记录已分类，但 6 个 `UNKNOWN` 仍缺外部 artifact/provenance 证据；
3. DH runtime parity 已完成，但没有 heldout metric，科学性能结论不能外推；
4. gamma runtime 与 training loop 仍然较大，但没有出现新的独立消费者边界，不应为行数继续拆分。

当前不应再按“文件够不够小”推进重构。只有出现真实消费者边界、行为测试和明确删除收益时，才继续移动或删除代码。

## 2. Current Architecture

| Responsibility | Formal owner | Current decision |
| --- | --- | --- |
| UnityPSF route | `models/unity_psf.py`, `models/psf_moe/router.py` | `ModalityRouter` 是正式 route；旧 `InstanceRouter` 仅支持只读 checkpoint route。 |
| DH image model | `DoubleHelixImageExpert` | DH formal path 不依赖 shared stem。 |
| Modality runtime | `training/modality_runtime.py` | runtime construction、formal audit、channel metadata 和 provider adaptation。 |
| Runtime config | `localization/runtime/config.py` | `build_localization_runtime_config` 是唯一 public builder。 |
| Online provider | `localization/data/online.py` | `OnlineBatchProviderConfig` 与 `build_online_batch_provider` 是唯一 public surface。 |
| High-fidelity orchestration | `training/high_fidelity/engine.py` | 只保留 orchestration、runtime assembly、resume 和 artifact summary。 |
| High-fidelity condition runtime | `training/high_fidelity/condition_runtime.py` | condition-store materialization 与 provider override。 |
| High-fidelity peak bootstrap | `training/high_fidelity/peak_bootstrap.py` | raw-TIFF peak/z-map bootstrap、domain 选择和 path materialization。 |
| Physical state | `training/high_fidelity/physical_state.py` | physical state、coefficient map 和 checkpoint extra。 |
| Inference quality fields | `infer_recon/filter/filter.py` | 三个 quality field helper 的 canonical owner。 |
| Gamma/checkpoint | `training/high_fidelity/gamma_runtime.py`, `training/loop.py` | 当前保留大文件；先 characterization，暂无新 facade。 |

Formal channel contract remains:

| Modality | Required channels |
| --- | --- |
| `double_helix` | `main` |
| `emitter_2d` | `left`, `right` |
| `astigmatism` | `left`, `right` |

## 3. Completed Work

### Phase 1-6

- DH 已从 PSFMoE/shared-stem scaffold 迁移为独立 `DoubleHelixImageExpert`。
- modality runtime、joint config、instance initialization、validation record 和 CLI ownership 已分开。
- `high_fidelity/engine.py` 的 diagnostics、physical state、raw-TIFF inference、ROI source、posterior、gamma 责任已收敛。
- online provider 的 camera、coordinates、conditioning、rendering、target contract 已拥有明确 owner；未为了行数拆出单消费者模块。
- runtime config 已按 schema、semantic validation、provider materialization 和 final builder 分层。
- 数值 helper 已按 backend、dtype、device、gradient 和 normalization 证明后去重。
- compatibility/archive lifecycle 已建立；仍承担校准、replay、物理诊断或复现的 legacy 代码保留。
- 测试 fixture 审计确认没有值得新增 global factory 的重复；外部 `neptune_iwae/Normalization.py` 对照已从 UnityPSF 测试中移除，保留 UnityPSF 自有 raw-TIFF gain contract 测试。

### Phase 7: ownership and hot-path convergence

- P1a 完成：`run_high_fidelity.py` 只导出 `main`、`parse_args`、`resume_epoch_training_config`；condition-store、peak bootstrap、physical state 的消费者已迁移到真实 owner。
- P1b 完成：filter quality helper 的 canonical 实现只保留在 `filter.py`；alias、缺失值、非有限值和各向异性 PSF 有测试。
- P1c 完成：native、LUT、cached-window 的 batch contract、seed、order、cache generation、dtype/device 和 failure semantics 已固定；不新增 provider facade。
- P2 runtime alias 完成：`localization/runtime_config.py` 已删除，消费者改用 `localization.runtime`。
- P2 gamma/checkpoint characterization 完成：多个真实消费者仍存在，继续拆分暂不产生清晰 ownership 收益。
- P3 完成：旧 review 已标记 superseded，当前路线图与测试基线统一。

## 4. Debt Classification

### D0: No immediate structural debt

以下区域已有明确 owner 和 contract，不应继续机械重构：

- formal modality/runtime assembly；
- DH independent image expert path；
- online provider public API；
- runtime config public builder；
- high-fidelity entrypoint boundary；
- inference filter quality helper ownership。

### D1: Must resolve before further engineering work

| Debt | Evidence | Exit condition |
| --- | --- | --- |
| Canonical repository boundary | `unity_v0.2` 是未跟踪副本，而 GitHub `UnityPSF/main` 已发布 v0.2 内容 | 选择并保留一个 canonical checkout；后续源码改动、CI receipt 和 GitHub history 必须落在同一 Git worktree。 |
| Six lifecycle `UNKNOWN` records | 4 个 ROI/gamma overrides 可能由动态配置合并消费；2 个 DH 入口依赖外部数据集/历史 artifact | 为每项写入具体 consumer 或 artifact/provenance；无法取得证据时保持 `UNKNOWN`，不删除。 |

### D2: Conditional cleanup, not scheduled deletion

Phase 9 当前没有 `DELETE_CANDIDATE`。以下类别只有在新的可验证证据出现后，才可重新审计：

- `scripts/archive` 中无近期 artifact、无论文复现引用且已有 canonical replacement 的脚本；
- `double_helix/legacy` 中同时满足无 active consumer、无 replay、无 calibration、无复现要求的模块；
- 没有正式消费者的 package `__all__` 导出和 compatibility wrapper；
- 已被新实现替代但仍残留的 test-only helper；
- 已失效的旧 config/schema fixture。

此前“严格可删除 3%-7%”和“广义低价值 10%-18%”只是审计前估计，现已失效，不得作为删除目标。实际可删除量当前为 **0 行已批准**；未来只按 inventory 的逐项证据决定。

### D3: Keep unless a real boundary appears

- `localization/data/online.py` 的 native/LUT/cached-window sequence hot path；
- `training/high_fidelity/gamma_runtime.py`；
- `training/loop.py` 的训练、resume、weights-only、legacy codec 和 RNG 逻辑；
- 仍承担物理校准、checkpoint replay、z-bin evaluation 或论文复现的 legacy/archive 代码；
- module-local characterization fixtures。

## 5. Replanned Roadmap

### Phase 8: GPU/Slurm and formal parity gate

目标：证明优化后的 Unity 可以按当前正式训练任务工作，而不仅是本地测试通过。

当前状态：**accepted**。

已验证：

- 旧正式基线 `three-modality-physical-update-300-4989` 和优化后 run `phase8-parity-20260821-r5` 都在三 GPU expert-parallel 上完成 300 epoch，并生成 joint checkpoint、summary、report 与五个 instance artifact。
- 两次 run 的 `training_signature`、world size、rank assignment 和每个 instance 的 optimizer step count 均一致：`3 GPUs`、`125100 steps/instance`。
- 优化后 run 生成三个 modality 的 simulation preview PNG；cached-window Film conditioning 的 feature-dimension 回归已修复，focused provider contracts `10 passed`。
- 优化后最终生成 `metadata/stage_status.json`（`completed`）、`metrics/joint_training_summary.json`、`checkpoints/unitypsf_joint.ckpt` 和 `report/report.html`。

最终 per-instance 对照（优化后 / 4989）：

| Instance | Final loss | Jaccard | RMSE XY (nm) |
| --- | ---: | ---: | ---: |
| emitter 2D left | `-255.15 / -248.52` | `0.8762 / 0.8604` | `7.74 / 8.53` |
| emitter 2D right | `-236.40 / -235.55` | `0.8755 / 0.8614` | `7.70 / 8.55` |
| astigmatism left | `-102.88 / -113.08` | `0.9186 / 0.9421` | `18.03 / 19.69` |
| astigmatism right | `-69.14 / -76.59` | `0.8651 / 0.8764` | `20.54 / 20.28` |
| double helix main | `0.0088 / 0.0171` | no heldout metric | no heldout metric |

结论：Phase 8 的功能一致性、route/channel 完整性、GPU execution、checkpoint schema 和训练步数一致性均通过。最终权重不要求逐 bit 相同；指标差异已记录，且没有发现结构性退化。DH 当前仍使用 `trained-smoke` 状态，没有 heldout metric，因此后续若需要科学性能结论，应另行增加 DH heldout evaluation，不阻塞本 Phase 8 runtime parity 验收。

执行顺序：

1. 已完成：以相同 seed、rank mapping、provider 选择和训练步数完成 v0.2 parity run。
2. 已完成：记录 resolved batch metadata、seed/order、cache generation、dtype/device 和正式 channel contract。
3. 已完成：验证多 rank barrier、rank/GPU mapping、run manifest、completed stage status 与五个 expert instance artifact。
4. 已完成：生成 joint checkpoint、resume checkpoints、summary 和 report；checkpoint reload/resume contract 有测试覆盖。
5. 已完成：比较旧正式入口与 v0.2 的 training signature、第一批/训练步数契约、checkpoint schema 和最终 per-instance metrics。

验收：已满足。Phase 8 完成后允许进入 Phase 9，但不得把 Phase 8 的“功能一致”误写成“最终权重逐 bit 相同”。

### Phase 9: Machine-readable dead-code and lifecycle inventory

前置条件：Phase 8 accepted。

目标：把“垃圾代码比例”从估计变成可审计清单。

当前状态：**accepted, audit complete**。

机器可读清单：

`docs/architecture/unity-v0.2-lifecycle-inventory.json`

审计范围覆盖 `src/**/*.py`、`scripts/**/*.{py,sh,sbatch}`、
`configs/**/*.{yaml,yml,json}` 和 `tests/**/*.py`，共 `353` 条记录：

| Lifecycle | Count | Decision |
| --- | ---: | --- |
| `ACTIVE` | 270 | 当前正式代码、测试或配置，不进入删除流程 |
| `ARCHIVE_REQUIRED` | 42 | archive/历史标准流程仍需保留，等待 owner 和 provenance 决策 |
| `MIGRATE` | 19 | compatibility namespace，迁移完并完成 parity 后再考虑删除 |
| `LEGACY_REPLAY` | 16 | DH 物理、校准、checkpoint replay 或历史实验用途，暂不删除 |
| `UNKNOWN` | 6 | 证据不足，禁止删除 |
| `DELETE_CANDIDATE` | 0 | 当前没有满足安全删除闸门的条目 |

六个 `UNKNOWN` 已复核：两个 DH legacy 入口依赖外部数据集/历史 artifact，四个 ROI/gamma override 通过 CLI 或配置合并动态消费。它们不能因为没有静态文本引用就被判定为死代码。

Phase 9 的验收结论是“清单完成、删除候选为零”，而不是强行制造删除项。后续 Phase 10 只有在获得 owner、artifact/provenance、canonical replacement parity 和 release-cycle 证据后，才可把单个条目改成 `DELETE_CANDIDATE`。

为每个 Python、shell/Slurm、config 和 archive entrypoint 建立记录：

```text
path
lines
static_consumers
dynamic_or_config_consumers
test_consumers
artifact_or_paper_references
canonical_replacement
owner
lifecycle
deletion_gate
```

`lifecycle` 只允许：`ACTIVE`、`ARCHIVE_REQUIRED`、`LEGACY_REPLAY`、`MIGRATE`、`DELETE_CANDIDATE`、`UNKNOWN`。

验收：所有 `DELETE_CANDIDATE` 都有零 active consumer、零 replay/physics/reproduction requirement、replacement parity 和删除日期；`UNKNOWN` 不得被删除。

### Phase 10: Canonical repository and verification boundary

目标：结束 `unity/`、`unity_v0.2/` 与 GitHub `UnityPSF` 之间的双工作树歧义。

执行：

1. 将 GitHub `UnityPSF/main` 指定为唯一发布历史；不要再向父仓库的无关 remote 提交 UnityPSF。
2. 选择一个目录作为唯一可编辑 checkout。推荐直接使用 GitHub checkout，不再把 `unity_v0.2` 作为未跟踪源码副本继续演化。
3. 将 `docs/architecture/unity-v0.2-lifecycle-inventory.json`、Phase 8 receipt 和 review 纳入该 checkout；不提交 `output/`、`logs/`、`.local/`、cache 或 checkpoint。
4. 固定 CI baseline：完整 pytest、compileall、relevant `bash -n`、architecture/compatibility boundary tests；GPU parity 作为已归档 release receipt，不在每次 CI 重跑。

验收：一个 Git worktree、一条 remote history、一个确定的测试命令集合；本地源码改动可以被 diff、commit、push 与 rollback。

### Phase 11: Provenance closure for `UNKNOWN`, no migration by default

目标：收窄 6 个 `UNKNOWN` 的证据缺口，而不是为了清单好看而删除或迁移。

范围：

1. 四个 ROI/gamma override：确定每个由哪个 CLI、config composition 或实验 artifact 消费；若没有消费者，先标为 `MIGRATE` 或 `ARCHIVE_REQUIRED`，不直接删除。
2. `prepare_real_calibration.py` 与 `run_fd_zmap.py`：关联最后一个已验证数据集、artifact 和负责复现的人；无法确认时保持 `UNKNOWN`。
3. 只有对单个条目证明“零 active consumer、零 physics/replay/reproduction requirement、已有 canonical replacement parity、经过一个 release cycle”后，才将其改为 `DELETE_CANDIDATE`。

验收：`UNKNOWN=0` 或每个剩余 `UNKNOWN` 都有明确外部阻塞和 owner；任何状态变化都更新 JSON inventory。

### Phase 12: Conditional archive retirement

前置条件：Phase 11 产生至少一个 `DELETE_CANDIDATE`。

每次只处理一个条目：迁移实际消费者、执行 artifact parity、删除代码/脚本/测试/文档引用、运行 focused regression，再更新 inventory。若 Phase 11 不产生候选，本阶段不执行。

禁止恢复 alias、新建 facade，或把 archive 目录名当作删除证据。

### Phase 13: Optional DH scientific evaluation

这不是结构重构任务。只有需要对外宣称 DH 最终定位性能时，才为正式 DH route 添加 heldout evaluation，并与 `trained-smoke` 分开报告。它不影响当前 Phase 8 的 runtime parity 结论。

## 6. Non-goals

当前路线不包括：

- 继续为了减少行数拆 online/gamma/training loop；
- 新增 provider/config/checkpoint facade；
- 恢复 shared-stem DH compatibility；
- 删除没有完成生命周期验证的 legacy/archive 代码；
- 把所有测试 fixture 提取到全局工厂；
- 用 CPU 结果替代 GPU/Slurm scientific validation。

## 7. Verification Baseline

最近完整本地基线（排除缺少仓库外 `neptune_iwae/Normalization.py` 的历史对照测试）：

```text
253 passed, 4 warnings in 257.11s
```

验证命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=unity_v0.2/src \
  pytest -q -p no:cacheprovider unity_v0.2/tests

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=unity_v0.2/src \
  python -m compileall -q unity_v0.2/src

bash -n \
  unity_v0.2/scripts/train/unitypsf_dual_modality_dual_channel_2gpu_300epoch.sbatch \
  unity_v0.2/scripts/train/unitypsf_dual_modality_dual_channel_2gpu_smoke.sbatch \
  unity_v0.2/scripts/train/unitypsf_three_modality_raw_tiff_300epoch.sbatch
```

Warnings 来自无 CUDA 环境下的 PyTorch 初始化和 tifffile deprecation；没有测试失败。

## 8. Working Boundary and Closeout

- `output/`、`logs/`、`.local/`、cache 和 checkpoint 不属于源码清理对象。
- `unity_v0.2` 当前在父仓库中整体未跟踪；在 Phase 10 完成 canonical checkout 收敛前，不应继续在多个副本上编辑源码。
- 本文是当前路线图的唯一依据；2026-08-20 旧 review 已 superseded。
- 任何新删除必须先更新 lifecycle inventory，再提供 focused regression receipt；没有 `DELETE_CANDIDATE` 就不执行删除。
