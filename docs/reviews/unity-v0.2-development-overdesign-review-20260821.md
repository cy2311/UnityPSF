# unity_v0.2 开发复杂度 Review 与后续路线图

- Review date: 2026-08-21
- Scope: `/home/guest/Others/main/race/unity_v0.2`
- Status: current review, debt inventory, and execution roadmap
- Last local verification: `255 passed, 4 warnings`

## 1. Executive Summary

Unity v0.2 的主要结构性问题已经在 Phase 1-7 收敛。当前核心训练/runtime 代码不再存在多条 formal 实现路径、入口私有 API 泄漏或 runtime config 双 facade。

剩余债务主要属于四类：

1. GPU/Slurm 与长训练 parity 尚未在目标节点完成；
2. legacy/archive 生命周期仍需要逐项 owner 和删除证据；
3. gamma runtime 与 training loop 仍然较大，但目前没有足够独立消费者支持继续拆分；
4. `unity_v0.2` 在父仓库中整体未跟踪，提交边界和 CI 审计边界仍不清晰。

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
- 测试 fixture 审计确认没有值得新增 global factory 的重复。

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

### D1: Must resolve before formal training submission

| Debt | Evidence | Exit condition |
| --- | --- | --- |
| GPU/Slurm parity | 当前完整回归仅在本地 CPU 环境完成 | 目标节点 smoke 覆盖三个 modality、五个 expert instance、native/LUT/cached-window、checkpoint reload/resume、rank/GPU manifest。 |
| Formal artifact parity | resolved snapshot 和 checkpoint contract 有本地测试，但缺目标节点产物 | 新旧入口比较 resolved runtime、首批 batch metadata、checkpoint schema 和 resume 后首步结果。 |
| Repository tracking boundary | `unity_v0.2` 在父仓库中整体显示为未跟踪 | 明确独立仓库、父仓库纳管或其他正式 tracking 策略，确保 diff/CI/rollback 可审计。 |

### D2: Evidence-driven cleanup candidates

这些不是立即删除项，必须逐个建立消费者和生命周期证据：

- `scripts/archive` 中无近期 artifact、无论文复现引用且已有 canonical replacement 的脚本；
- `double_helix/legacy` 中同时满足无 active consumer、无 replay、无 calibration、无复现要求的模块；
- 没有正式消费者的 package `__all__` 导出和 compatibility wrapper；
- 已被新实现替代但仍残留的 test-only helper；
- 已失效的旧 config/schema fixture。

预计严格可删除代码约为总源码的 **3%-7%**；广义低价值候选约 **10%-18%**。这只是审计优先级估计，不是删除授权。最终比例必须由 machine-readable inventory 计算，而不能按行数或目录名推断。

### D3: Keep unless a real boundary appears

- `localization/data/online.py` 的 native/LUT/cached-window sequence hot path；
- `training/high_fidelity/gamma_runtime.py`；
- `training/loop.py` 的训练、resume、weights-only、legacy codec 和 RNG 逻辑；
- 仍承担物理校准、checkpoint replay、z-bin evaluation 或论文复现的 legacy/archive 代码；
- module-local characterization fixtures。

## 5. Next Roadmap

### Phase 8: GPU/Slurm and formal parity gate

目标：证明优化后的 Unity 可以按当前正式训练任务工作，而不仅是本地测试通过。

执行顺序：

1. 提交 1-5 epoch GPU smoke，覆盖 DH `main`、emitter 2D `left + right`、astigmatism `left + right`。
2. 分别验证 native、LUT、cached-window provider 的 resolved batch metadata、seed/order、cache generation、dtype/device。
3. 验证多 rank barrier、rank/GPU mapping、run manifest 和 stage status。
4. 保存 checkpoint，执行 reload、full resume 和 weights-only initialization。
5. 比较旧正式入口与 Unity 入口的 resolved runtime、第一批 batch、首步 loss、checkpoint schema 和 resume continuity。

验收：GPU smoke 全部成功；无未解释 snapshot 漂移；checkpoint 可 reload/resume；三个 modality、五个 expert instance 都生成完整 artifact。

### Phase 9: Machine-readable dead-code and lifecycle inventory

目标：把“垃圾代码比例”从估计变成可审计清单。

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

### Phase 10: Archive and compatibility retirement

只处理 Phase 9 中已经证明可退役的条目：

1. 先删除无消费者的 exports/wrappers；
2. 再删除已经有 parity 证据的 archive scripts；
3. 最后处理旧 config/fixture；
4. 每个 slice 都运行相关 lifecycle、import、archive shell 和 regression tests。

禁止恢复 alias 或新建第二个 facade。

### Phase 11: Submission and repository hygiene

目标：让后续改动可以被正常审查、提交和回滚。

需要明确 `unity_v0.2` 的 git tracking 边界，清理未分类的 root-level artifact，固定 CI 命令和 GPU smoke 入口，并将最终 baseline 写入 README/review 文档。

## 6. Non-goals

当前路线不包括：

- 继续为了减少行数拆 online/gamma/training loop；
- 新增 provider/config/checkpoint facade；
- 恢复 shared-stem DH compatibility；
- 删除没有完成生命周期验证的 legacy/archive 代码；
- 把所有测试 fixture 提取到全局工厂；
- 用 CPU 结果替代 GPU/Slurm scientific validation。

## 7. Verification Baseline

最近完整本地基线：

```text
255 passed, 4 warnings in 308.34s
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
- `unity_v0.2` 当前在父仓库中整体未跟踪；这必须在正式提交前解决或明确记录。
- 本文是当前路线图的唯一依据；2026-08-20 旧 review 已 superseded。
- 任何新删除必须先更新 lifecycle inventory，再提供 focused regression receipt。
