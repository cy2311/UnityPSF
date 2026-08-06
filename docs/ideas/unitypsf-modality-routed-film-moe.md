# UnityPSF Multimodal PSF MoE

- Status: Accepted target architecture
- Date: 2026-08-04
- Modalities: 2D emitter, astigmatism, double helix

## Goal

UnityPSF 的目标是用一个 joint checkpoint 和一个顶层模型处理多种 PSF，同时保留
现有 localization、FiLM、peak zmap、gamma update 和 reconstruction 流程。

第一版采用硬路由：先确定 PSF 模态，再激活对应 expert。未选中的 expert 不参与 forward、反向传播和参数更新。

## Core Model

UnityPSF 对外只有一个 `UnityPSF` 模型身份。模型内部定义三个 canonical experts：

1. `Emitter2DExpert`
2. `AstigmatismExpert`
3. `DoubleHelixExpert`

`AstigmatismExpert` 最初只有一套网络和一个基础 checkpoint。通道数量是运行时配置，不写死在 expert 结构中。

```text
unitypsf_joint.ckpt
   -> UnityPSF
        +-- hard router(modality, channel_id)
        +-- Emitter2DExpert
   |
        +-- AstigmatismExpert
   |      +-- preprocessing
   |      +-- full localization backbone
   |      +-- FiLM
   |      +-- heads/loss/decoder
   |      +-- astigmatism_base.ckpt
   |
        +-- DoubleHelixExpert
          +-- preprocessing
          +-- full localization backbone
          +-- FiLM
          +-- DH heads/loss/decoder
          +-- DH calibration/physics
```

## Routing And FiLM

外层 router 只负责选择 PSF expert：

```text
emitter_2d   -> Emitter2DExpert
astigmatism  -> AstigmatismExpert
double_helix -> DoubleHelixExpert
```

第一版优先使用 config、TIFF metadata 或 calibration manifest 确定模态。只有在这些信息缺失时，才使用图像 detector。低置信度时应停止并要求显式配置。

FiLM 不负责选择 PSF 模态。它在每个 expert 内部表达：

- Zernike 条件。
- field position。
- calibration 参数。
- acquisition/domain 差异。

## Astigmatism Channel Instances

### Single Channel

单通道时，只从 `astigmatism_base.ckpt` 创建一个 `AstigmatismExpert` 实例：

```text
astigmatism_base.ckpt
    -> AstigmatismExpert(channel=main)
    -> main peak zmap/gamma state
    -> astigmatism_main.ckpt
```

这是 UnityPSF 最基础的 astigmatism 运行方式，不需要 left/right 结构。

### Multiple Channels

多通道时，从同一个 `astigmatism_base.ckpt` 复制多份初始状态：

```text
                         astigmatism_base.ckpt
                                  |
                      copy the same state_dict
                                  |
                 +----------------+----------------+
                 |                                 |
                 v                                 v
AstigmatismExpert(channel=left)   AstigmatismExpert(channel=right)
                 |                                 |
        left peak zmap                   right peak zmap
        left gamma state                 right gamma state
        left optimizer                   right optimizer
                 |                                 |
      astigmatism_left.ckpt            astigmatism_right.ckpt
```

left/right 是同一个 `AstigmatismExpert` 类的两个运行时实例，不是两种预先写死的 expert 类型。

两个实例在初始时可以拥有完全相同的权重，但复制完成后必须彻底独立：

- 不共享 Parameter 对象。
- 不共享 FiLM 参数。
- 不共享 optimizer state。
- 不共享 peak zmap 和 gamma state。
- 不共享训练 checkpoint。

### Peak Zmap

left/right 可以使用相同 anchor 策略，但必须从各自 raw crop 独立构建 peak zmap：

```text
left raw crop  -> anchor initialization -> left peak zmap
right raw crop -> anchor initialization -> right peak zmap
```

当前 660 nm astigmatism profile 可以继续使用 99 nm anchor。但 99 nm 应是 profile 配置，而不是所有波长和 PSF 通用的常量。

## Training And Inference

双通道训练保持当前独立流程：

```text
left crop
   -> left peak zmap
   -> AstigmatismExpert(channel=left)
   -> left loss/gamma update
   -> astigmatism_left.ckpt

right crop
   -> right peak zmap
   -> AstigmatismExpert(channel=right)
   -> right loss/gamma update
   -> astigmatism_right.ckpt
```

left/right 使用各自的 optimizer，两个 loss 互不反向传播。第一版不合并 batch，也不添加跨通道 consistency loss。

推理时用户只加载 joint checkpoint，顶层模型在内部选择已训练的通道实例：

```text
model = UnityPSF.from_checkpoint("unitypsf_joint.ckpt")

left image  -> model.localize(modality=astigmatism, channel=left)
right image -> model.localize(modality=astigmatism, channel=right)

left localizations + right localizations
    -> current union reconstruction
```

各通道网络虽然独立，但必须使用一致的坐标、z 轴、photon 和 reconstruction contract。

## Joint Checkpoint

正式交付物是一个包含 router、expert instances 和 calibration state 的物理文件：

```text
unitypsf_joint.ckpt
    +-- model / router / contracts
    +-- Emitter2D instances
    +-- Astigmatism main/left/right instances
    +-- per-channel peak zmap / gamma / physical state
    +-- optional Double Helix instances and calibration
    +-- provenance / hashes / optional resume state
```

每个保存时间点可以有 `latest`、milestone 或 release 文件，但每个文件都表示完整的
UnityPSF 模型，而不是某一个 expert。旧的 per-channel checkpoint 只作为迁移输入和内部
恢复材料。两模态版本不创建空 DH expert；DH 通过验收后再扩展
`supported_modalities`。

第一个正式版本先完成 Origami `Emitter2D(main)` 与 Astigmatism `left/right`，即双模态
+ 多通道；Double Helix 数据到位后沿同一模型和 checkpoint schema 扩展三模态。

## Implementation Order

1. 定义三个 canonical expert interfaces 和 modality router。
2. 先实现单一 `AstigmatismExpert` 及 `astigmatism_base.ckpt`。
3. 实现根据 channel layout 复制独立 expert 实例的机制。
4. 迁移各通道独立 peak-zmap、training 和 checkpoint 流程。
5. 完成单通道和双通道 inference/reconstruction parity 与可视化报告。
6. 实现 joint checkpoint 和顶层 `UnityPSF` 的保存、加载和版本校验。
7. 接入 Origami `Emitter2DExpert`，完成双模态 + 多通道联合训练。
8. Double Helix 数据就绪后接入 `DoubleHelixExpert` 并升级三模态。

## Acceptance Criteria

- 顶层只有一个 `AstigmatismExpert` 定义和一个基础 checkpoint。
- 单通道时只创建一个 astigmatism 实例。
- 多通道时，所有实例从同一基础状态复制，之后独立训练。
- 每个通道各自拥有 FiLM、optimizer、peak zmap、gamma state 和 checkpoint。
- hard routing 时只运行被选中的 PSF expert 和通道实例。
- 当前 left/right crop、filter、reconstruction 和 union 行为保持不变。
- 每个通道状态在 joint checkpoint 内可独立验证、恢复和复现。
- 用户只需指定一个 `unitypsf_joint.ckpt` 即可完成路由。
- 第一里程碑的一个 checkpoint 同时包含 Emitter2D main 与 Astigmatism left/right。
- 每次正式训练都产生按 modality/channel 分开的结果图和 `report.html`。

## Summary

UnityPSF 对外是一个 joint checkpoint、一个顶层模型和一个 localization API；内部包含
`Emitter2DExpert`、`AstigmatismExpert` 和 `DoubleHelixExpert` 三个 canonical expert
类型。运行时根据 modality 和 channel layout 激活独立实例。这些实例可以共用初始来源，
但不共享训练参数、peak zmap 或物理状态。第一阶段先完成 Origami 2D 与 Astigmatism
left/right 的双模态 + 多通道模型，再在 DH 数据到位后升级为三模态。
