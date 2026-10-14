# UnityPSF v0.3 与 LiteLoc 训练速度对比

## 结论

在同一类 RTX 3090 训练条件下，当前固定 cache 方案的 UnityPSF 30 epoch
测试为 15,000 optimizer steps、总耗时 1,279.26 s，即 **85.28 ms/step**。
LiteLoc 正式 10,000-frame cache run 记录为 **69.98 ms/update**。因此当前差距
约 **15.3 ms/step（约 22%）**。

这不是 simulation 速度造成的主要差距。UnityPSF 之前的分段 profiling 量级约为：

| 阶段 | UnityPSF 当前量级 | 判断 |
|---|---:|---|
| simulation / cache 取 batch | 约 6--10 ms | 固定 cache 后已不是主瓶颈 |
| model forward + loss | 约 26--28 ms | 主要由网络和 conditioning 结构决定 |
| backward | 约 54--57 ms | 当前最大开销，包含两 expert 的梯度图 |
| GMM loss 内核 | 约 4--10 ms | 有优化空间，但不是全部 50 ms 的来源 |
| optimizer step | 数 ms 量级 | 不是主要差距来源 |

这些是不同 profiling run 的量级，受 CUDA 同步和 cache warm-up 影响，不能机械地
逐项相加；它们用于判断瓶颈优先级，不替代 30 epoch wall-time 基准。

## 为什么同样是 GMM，速度仍然不同

LiteLoc 和 UnityPSF 都计算“每个像素预测分布对真实 emitter 的 GMM
log-likelihood”，但 GMM 只决定 loss 的一部分，不能代表整个 training step。

LiteLoc 使用一个 `LiteLoc` 网络和一条 forward/backward 路径，训练时对每个
三帧局部上下文只输出中间帧。UnityPSF 当前默认模型是
`active_smlm_soft_moe_double_unet`：它包含 shared/union U-Net、FiLM
conditioning 和两个 expert；默认 `soft` routing 会对同一个 batch 计算两个
完整 expert，再按 gate 权重融合。反向传播需要保留并回传两条 expert 图，因而
forward/backward 比 LiteLoc 重得多。

GMM 组件数也不是当前主因：本次 UnityPSF 速度测试是 64 x 64 输出，即约
4,096 个 component；LiteLoc 对应的 128 x 128 输出约 16,384 个 component。
UnityPSF 的 component 数反而更少，但网络仍然更慢，说明差距主要来自模型结构。

两者的 loss 实现也不完全相同。LiteLoc 在 `loss_utils.py` 中手写 dense
`logsumexp`；UnityPSF 当前默认使用 PyTorch
`MixtureSameFamily`。两者都需要 component-target 配对张量和反向传播，但
PyTorch distribution 封装可能带来额外 kernel/临时张量开销。这个部分值得做
独立 GPU microbenchmark，不能只凭“都是 GMM”推断速度相同。

优化器方面，LiteLoc 使用 NAdam，UnityPSF 当前使用 AdamW。AdamW 通常不是
UnityPSF 的主要慢点；在没有 profile 证明前，不应先改 optimizer 破坏训练超参
对齐。

## v0.3 默认训练方案

```text
physical model version
        |
        v
CUDA GEMM simulation
        |
        v
生成当前 physical version 的 10,000 个中心帧
        |
        v
CPU pinned cache（1 batch GPU asynchronous prefetch）
        |
        v
按 temporal cursor 顺序循环取 batch
        |
        v
三帧 local context -> 中间帧监督 -> GMM + background loss
```

默认语义如下：

- cache 生成使用 CUDA GEMM GPU renderer，不是 CPU simulation，也不是 LUT 查表。
- 10,000 个中心帧在同一个 physical-model version 内复用；epoch 之间只移动
  cursor，不重新仿真，不打乱时间顺序。
- 每个样本仍然是连续三帧 local context，监督中间帧。
- cache 存在 pinned CPU memory，训练前预取一个 batch 到 GPU，以便 H2D 与计算
  重叠。
- physical model update 改变 condition version 后，旧 cache、sequence cache 和
  renderer-side cache 一起失效；下一次 provider 请求按新物理版本重新生成。
- 默认配置为 `batch_size: 10`、`steps_per_epoch: 500`、64 x 64、
  `vector_batch_size: 4096`、FP16 AMP。

## 优化优先级

1. **先优化 model/backward。** 测试 `torch.compile`、channels-last 和更窄的
   feature width。
2. **再做 GMM microbenchmark。** 固定输出、target 数、component 数和 dtype，
   对比 `MixtureSameFamily` 与手写 fused/chunked `logsumexp`，同时检查峰值显存
   和梯度误差。只有实测稳定收益后才改默认 backend。
3. **最后考虑 optimizer。** fused AdamW 或 capturable AdamW 可能节省少量 kernel
   launch，但预期收益小于网络结构优化。

单纯继续扩大 cache 只能减少 simulation/搬运部分，不能消除当前约 50 ms 级别的
 backward 开销；因此不应把 cache size 当作解决 UnityPSF 与 LiteLoc 差距的主要
 手段。

## 可复现实测记录

- UnityPSF run: `output/unitypsf_liteloc_cache_speed_5700/`
- UnityPSF metrics:
  `output/unitypsf_liteloc_cache_speed_5700/metrics/training_metrics.jsonl`
- UnityPSF stage receipt:
  `output/unitypsf_liteloc_cache_speed_5700/metadata/stage_status.json`
- UnityPSF: 30 epochs, 15,000 steps, 1,279.26 s wall time。
- LiteLoc: RTX 3090, batch 10, 10,000-frame cache, 69.98 ms/update。
- Soft-MoE speed baseline: `output/unitypsf_moe_soft_30ep_5701/`。

这些数字是吞吐对比，不是定位精度对比；两套网络的模型容量、输出分辨率和
conditioning contract 仍不同。
