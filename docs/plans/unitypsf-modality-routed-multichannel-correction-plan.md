# UnityPSF 按模态路由的多通道修正规划

- 状态：步骤 1、2、3 已完成；步骤 4 修复和双 GPU gate 已通过，300 epochs 正式训练 job `4530` 运行中
- 日期：2026-08-05
- 目标：对齐 Neptune v0.3 的双通道联合训练语义，同时保留 UnityPSF 的多模态 MoE 和单文件 joint checkpoint

## 0. 实施状态

2026-08-05 已完成步骤 1：

- 新增 `unity_psf.joint_checkpoint.v2`，`experts` 只按 modality 保存一套完整网络；
- `channel_states[modality][channel_id]` 分别保存 physical state、calibration 和 provenance；
- 新增 `ModalityRouter`，正式 `UnityPSF` registry 和 activation audit 均只按 modality；
- `UnityPSF.localize(...)` 继续显式接收并校验 `channel_id`；
- v1 checkpoint 继续通过 legacy 路径只读校验和加载，读取不会改写原文件；
- 全量 CPU 工程回归通过，尚未进行新架构 GPU 训练或科学性能验收。

2026-08-05 已完成步骤 2：

- 新增 `ModalityTrainingRuntime`，每个 modality 只持有一个 model、optimizer 和 scheduler；
- 新增 `ModalityChannelStream`，left/right 各自持有 batch provider、loss、physical state、
  calibration 和 provenance，不再持有完整网络或 optimizer；
- 采用确定性 round-robin 平衡调度；例如 left/right 预算为 2/1 时，顺序固定为
  `left, right, left`；
- 每个 batch 显式携带 `channel_id` 和 FiLM physical condition；left/right loss 共同更新
  同一个 expert state；
- 新增正式入口 `unity-psf-train-modality-joint`，可从现有四条 channel 配置构造两个
  modality runtime，并生成 v2 joint checkpoint；
- 双模态、双通道 CPU smoke 已通过：四条 channel 数据流只生成 `emitter_2d` 和
  `astigmatism` 两套完整网络，checkpoint 回载和四条 channel route 均通过。

2026-08-05 已完成步骤 3：

- Expert Parallel ownership 已改为一个 rank 负责一个 modality expert，并在 rank 内联合训练
  left/right；
- 训练进程只使用 torchrun 提供的 rank 身份，不初始化 distributed process group，也不使用
  NCCL/Gloo collective 或 barrier；
- 每个模态每 epoch 原子保存 v2 resume shard，其中包含 model、optimizer、scheduler、累计
  channel progress、epoch、完成状态和 RNG state；
- `--resume` 可按模态恢复网络、优化器、调度器、RNG、通道状态快照和训练进度，并用
  joint/config 内容签名拒绝配置漂移；
- 每次 torchrun attempt 与 rank status、validation artifact 绑定；旧 attempt 不会参与当前
  汇合，rank 失败会写原子失败状态并使 rank 0 立即终止；
- rank 0 通过共享文件系统等待必需模态完成，再原子组装不含 training state 的 v2 release
  `unitypsf_joint.ckpt`；
- joint checkpoint 已完成回载和所有 modality/channel route smoke，并生成可见验证报告；
- 旧 `unity-psf-train-joint-expert-parallel` 命令名保留为薄兼容入口，实际转发到新的模态级实现；
  旧 `one_instance_per_rank` 配置语义不兼容，正式配置必须显式使用
  `rank_assignment: one_modality_per_rank`。

步骤 3 已通过双 rank CPU 工程 smoke；尚未提交正式 GPU 训练，也未完成 held-out 科学指标验收。
当前入口按原 YAML 和确定性 epoch seed 重建静态 online provider，尚未启用训练中动态
gamma/peak-zmap 更新。因此本步骤只恢复 checkpoint 中的通道状态快照；在步骤 4 接入动态物理
更新前，必须先增加显式 physical-context restore hook，并验证恢复后下一批 FiLM condition。
下一步进入步骤 4。

2026-08-05 已完成步骤 4 的工程实现与 GPU smoke：

- 每个 channel runtime 已保留真实 `ChannelTrainingContext`，训练 provider 和 held-out provider
  共用该通道独立的 condition store；
- resume shard 保存当前 physical state，恢复时先调用 channel restore hook，再强制校验
  `condition_store_version` 与 checkpoint 一致；
- online held-out provider 使用固定、与训练 seed 不同的 1-based 数据流，分别按 Emitter2D
  二维匹配和 Astigmatism 三维匹配；
- 每个 epoch 记录 left、right 和模态聚合的 `eval_loss`、precision、recall、Jaccard、
  `RMSE_XY_nm`、`RMSE_Z_nm`、photon relative error、样本数、route count 和 optimizer steps；
- 模态聚合从 TP/FP/FN、平方误差和 photon 误差累加量重新计算，不平均通道指标；
- 固定 held-out 样本生成 GT/prediction overlay、z 误差和 reconstruction，Emitter2D 的
  `RMSE_Z_nm` 明确为不适用；
- 初始双 GPU smoke job `4526` 验证了两张 RTX 3090、两套模态 expert、四条 channel route、
  joint checkpoint 回载和静态报告链路；
- 原正式 job `4527` 已停止，不再恢复。检查确认其 Astigmatism 使用了错误 loss/target 组合，
  且显式 AdamW 配置未被 runtime 采用，旧指标还把 AMP 跳步误记为 optimizer update；
- 修复后 Astigmatism 使用 `active_smlm_gmm_loss`、`legacy_iwae` target order、AdamW、
  StepLR、float16 AMP 和独立 left/right zmap。正式启动会审计实际 runtime 类和物理文件 SHA256；
- AMP 现在只统计真实参数更新，溢出跳步不推进 scheduler；GradScaler 跨 epoch 复用并写入
  resume shard，NaN/Inf loss 直接失败；
- job `4528` 暴露并保留了旧 AMP 计数问题，不作为有效 gate；修复后的 job `4529` 已通过，
  生成 joint checkpoint、`summary.json`、静态报告和分模态/分通道指标；
- 新正式 job `4530` 已从统一初始化启动，rank 0 训练 Emitter2D left/right，rank 1 训练
  Astigmatism left/right。未使用 `--resume`，不读取旧正式任务或 smoke checkpoint。
- job `4530` 的第 1 epoch 已完成：两个模态均为 834 次尝试、824 次真实更新、10 次初始 AMP
  warm-up 跳步。Astigmatism left/right 分别预测 57/53 个 emitter，precision 均为 1.0，
  recall 为 0.460/0.427，`RMSE_XY_nm` 为 21.56/25.72，`RMSE_Z_nm` 为 31.35/34.95；
  已通过“非零预测、真实参数更新、分通道三维指标有效”的早期科学放行检查。

job `4529` 只证明 CUDA、运行契约、真实 optimizer update、路由、恢复、指标和报告链路正确；
其每通道仅 8 个 batch，不构成科学结果。job `4530` 完成并通过人工图像与指标检查前，
不冻结新的科学 baseline。

## 1. 修正结论

UnityPSF 顶层 MoE 只按 `PSF modality` 路由：

```text
modality router
    +-- Emitter2DExpert
    +-- AstigmatismExpert
    +-- DoubleHelixExpert
```

`channel_id` 不再参与选择完整网络。它在模态 expert 内部用于选择通道物理状态、构造
FiLM condition，并保留输出所属通道。

因此，正式训练单元是“一个模态 expert”，不是 `(modality, channel_id)` 实例。

本文件取代以下旧设计：

- left/right 分别拥有完整网络、optimizer 和训练 checkpoint；
- 顶层 router 使用 `(modality, channel_id)` 选择完整网络；
- Expert Parallel 将每个通道视为独立 expert instance。

ADR 0003 中“一个 UnityPSF、一个 `unitypsf_joint.ckpt`”继续有效；其中通道网络完全独立的
部分由本文件修正。

## 2. 目标架构

```text
UnityPSF
    |
    +-- ModalityRouter(modality)
            |
            +-- Emitter2DExpert
            |      +-- one localization network
            |      +-- one optimizer
            |      +-- left/right channel conditions
            |      +-- left/right physical and calibration states
            |
            +-- AstigmatismExpert
            |      +-- one localization network
            |      +-- one optimizer
            |      +-- left/right channel conditions
            |      +-- independent peak-zmap/gamma/calibration per channel
            |
            +-- DoubleHelixExpert
                   +-- one localization network
                   +-- one optimizer
                   +-- left/right channel conditions
                   +-- independent DH calibration/physics per channel
```

网络参数在同一模态内共享，物理状态在不同测量通道间独立。

## 3. 训练语义

对齐 Neptune v0.3：

1. 每个模态只创建一个 model、optimizer、scheduler 和训练 checkpoint。
2. left/right 数据进入同一个训练循环，按 step 或 sequence 平衡调度。
3. 不要求每个 batch 严格包含成对的 left/right，但每个 epoch 必须覆盖所有启用通道。
4. 每个样本携带 `channel_id`、通道物理 condition 和可选 channel one-hot。
5. left/right loss 共同更新该模态的同一套网络参数。
6. left/right 的 crop、peak-zmap、gamma、calibration、condition provider 和 physical version
   继续独立保存和更新。

Expert Parallel 使用“一张卡负责一个模态”：

```text
rank 0 / GPU 0: Emitter2DExpert，联合训练 left + right
rank 1 / GPU 1: AstigmatismExpert，联合训练 left + right
rank 2 / GPU 2: DoubleHelixExpert，数据就绪后加入
```

专家训练期间不依赖 NCCL collective。各模态完成后再同步并原子生成 joint checkpoint，
避免不同模态耗时不一致导致 barrier timeout。

## 4. Checkpoint 结构

正式 checkpoint 仍是一个物理文件：

```text
unitypsf_joint.ckpt
    +-- experts
    |      +-- emitter_2d.model_state_dict
    |      +-- astigmatism.model_state_dict
    |      +-- double_helix.model_state_dict      # 数据验收后加入
    |
    +-- channel_states
    |      +-- emitter_2d.left/right
    |      +-- astigmatism.left/right
    |      +-- double_helix.left/right
    |
    +-- router
    |      +-- key: modality
    |
    +-- training_state                     # resume checkpoint only
    +-- provenance
    +-- integrity
```

旧的 `(modality, channel_id)` 四实例 checkpoint 只作为兼容导入和消融基线，不作为新的
正式 UnityPSF baseline。

## 5. Eval 要求

SLURM job `4525` 没有独立 held-out eval，只记录了 training loss，不能作为科学性能
baseline。修正后的正式训练必须为每个模态和通道建立固定且与训练集不重叠的验证集。

至少记录：

- `eval_loss`；
- precision、recall、Jaccard；
- `RMSE_XY_nm`；
- `RMSE_Z_nm`，Emitter2D 标记为不适用；
- photon relative error；
- 每通道样本数、route count 和 optimizer step count。

报告必须分别展示 left、right 和模态聚合结果，并生成固定样本的 GT/prediction overlay、
误差随 z 分布和 reconstruction。没有 GT 的真实数据不得伪造定位精度指标。

## 6. 实施顺序

### 步骤 1：修正路由与 checkpoint contract

- 将 expert registry key 从 `(modality, channel_id)` 改为 `modality`。
- `UnityPSF.localize(...)` 继续接收 `channel_id`，但 router 只解析 modality。
- 将 channel physical/calibration state 嵌套到对应模态 expert state。
- 提供旧 joint checkpoint 的只读兼容导入，不原地改写旧文件。

### 步骤 2：实现模态内多通道联合训练（已完成）

- 每个模态只构造一个 runtime 和 optimizer。
- 合并同模态 channel batch provider，加入可审计的平衡调度。
- 保留每通道独立 physical context 和指标，不再创建 per-channel model runtime。
- 验证 left/right batch 都能更新同一个 model state。

正式 CPU smoke 命令：

```bash
unity-psf-train-modality-joint \
  --config <双模态双通道配置.yaml> \
  --run-root <输出目录> \
  --run-id <运行编号>
```

当前 smoke 验收范围是训练 ownership、平衡调度、分通道指标、v2 checkpoint 和回载路由。
它不是 held-out scientific eval，也没有替代步骤 4 的真实数据训练和可视化验收。

### 步骤 3：修正 Expert Parallel 与恢复（已完成）

- rank ownership 从 channel instance 改为 modality expert。
- 去除专家训练阶段的短时 NCCL barrier。
- 支持模态级 checkpoint、完成状态和断点恢复。
- 使用 attempt ID 和训练配置签名隔离不同启动，并让失败 rank 快速传播。
- 所有必需模态完成后再生成、回载并路由验收 `unitypsf_joint.ckpt`。

正式入口保持不变：

```bash
torchrun --standalone --nproc-per-node=2 \
  -m unity_psf.cli.train_joint_expert_parallel \
  --config <双模态双通道配置.yaml> \
  --run-root <输出目录> \
  --run-id <运行编号>
```

中断后使用相同的配置、输出目录和运行编号，并增加 `--resume`。当前验收只证明工程契约、
模态级恢复、无 collective 协调、v2 release 组装和回载路由正确，不替代步骤 4 的 GPU 与
held-out scientific eval。

### 步骤 4：补齐 eval 后重新训练（工程实现和 GPU smoke 已完成）

- 接入动态 gamma/peak-zmap 时，先实现 channel physical-context restore hook，并验证恢复后的
  provider condition/version 与 checkpoint 一致。
- 增加固定 held-out eval provider 和 per-channel/per-modality metrics。
- 先运行双 GPU、双模态、双通道短 smoke。
- smoke 通过后从统一初始化重新训练 300 epochs。
- `4525` 产物仅保留为独立通道消融结果，不从中恢复正式 baseline。

执行状态：

- 修复后 GPU gate：job `4529`，已完成；
- 300 epochs：job `4530`，运行中；
- 新 baseline：待 job `4530` 完成并人工验收后冻结。

## 7. 验收条件

- [x] 顶层 router 的唯一专家选择维度是 modality。
- [x] 同一模态的 left/right 使用同一个模型对象和 optimizer。
- [x] left/right 数据都对该共享模型产生梯度更新。
- [x] left/right 的 physical state、calibration 和 provenance 在共享 runtime 内独立保存。
- [x] 双模态 checkpoint 只包含两套完整定位网络，而不是四套通道网络。
- [x] Expert Parallel 每个 rank 负责一个模态，并在内部联合处理该模态的所有通道。
- [x] 专家训练与最终汇合不使用 NCCL collective，不受不同模态训练时长差导致的 barrier timeout 影响。
- [x] GPU smoke 生成 held-out eval metrics、可视化报告、joint checkpoint 回载与 route smoke。
- [ ] 300-epoch 正式 run 生成完整 held-out 指标和可视化报告。
- [ ] 人工检查图像和科学指标通过后，冻结新的双模态 baseline。

## 8. 暂不纳入

- 未取得 Double Helix 数据前，不宣称三模态科学 baseline。
- 第一版不增加跨通道 consistency loss。
- 第一版不使用 left/right 各自的完整网络作为正式模型。
- 第一版不训练 raw TIFF 自动模态 detector；先使用显式 modality metadata 路由。
