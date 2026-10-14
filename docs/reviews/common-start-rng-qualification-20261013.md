# 第一批：共同训练起点与分段可复现验收

## 范围与实现

本轮统一 canonical runtime 构建前的 Python、NumPy、Torch CPU/CUDA seed；保留各 channel 的独立 data seed。未改变 loss、BN、阈值、CUDA kernel 或 checkpoint schema。

qualification 的 continuous 路径在任何训练 batch 前同步保存每模态 epoch-0 resume shard。两条路径随后都加载同一份 shard；initial manifest 记录 SHA256，interrupted 加载前检查 hash。父进程还校验两组科学配置和输入清单完全一致，普通 resume 的合同 provenance 校验没有放宽。初始 scaler 按生产训练同样的 init_scale=128 创建。

启动固定 PYTHONHASHSEED、CUBLAS_WORKSPACE_CONFIG 与既有确定性设置。最终脚本记录 harness/trace 文件 hash，每次 worker 启动核对，防止跨进程混用实现。

每 epoch 的前 3 个实际训练 attempt 记录输入 images、conditions、targets、forward output、loss、梯度、model/optimizer/scheduler/scaler、逐源 RNG 和物理状态指纹。没有额外调用 provider/forward/backward。梯度是 unscale/clip/optimizer step 返回后保留的梯度；step after 状态早于该步 scheduler advance，epoch checkpoint 则在 held-out 与 epoch hook 之后。

before 指纹是在实际 batch 已生成之后、forward 之前采集，不是 provider 调用之前。每 epoch 只采样前三步，因此“首次采样差异”不能解释成整条训练首次分叉。最终门禁同时要求共同起点、trace 覆盖与一致性、所有 epoch 和最终状态通过；严格字节比较与容差比较分开报告。epoch 与终态包含物理状态/calibration，不比较允许不同的输出路径 provenance。

## GPU 证据

| 作业 | 定位 |
| --- | --- |
| 5727 | trace target 解包错误，已修复；排除 |
| 5728 | 初版小 GPU 验证；共同起点通过，DH 全程一致；2D/astig 分叉 |
| 5729 | 正式对齐预算：3 GPUs、64×64、batch64、3 epoch ×157 steps/channel；完整运行/中断/恢复 |
| 5730 | 脚本修订前后窗口内的 probe，不作为冻结最终脚本证据 |
| 5731 | 冻结最终脚本的小 GPU 验证：3 GPUs、64×64、batch4、3 epoch ×8 steps/channel |

正式 5729 证据：`output/unitypsf/common-start-aligned-1791901249/`。
最终脚本 5731 证据：`output/unitypsf/common-start-frozen-probe-1791902216/`。
日志：`logs/slurm/unitypsf_gpu_aligned-5729.*` 与 `unitypsf_gpu_aligned-5731.*`。两次都使用 CUDA，没有 CPU fallback。外层 failed 是等价性门禁失败，并非训练 worker 崩溃。

## 结果与判断

- 三模态的共同初始 checkpoint、模型/optimizer/scheduler/scaler/全局 RNG/物理状态均一致。
- 三模态即时 epoch-1 restore 均通过。
- **DH：5729 与5731的采样 trace、全部 epoch checkpoint、最终状态均严格相同；Python/NumPy RNG 的原有差异消失。**
- **5729 astigmatism：epoch1第1个被观察的训练输入 images 不同，但 conditions、targets 和 before 的逐源 RNG 相同。**
- **5729 emitter_2d：epoch1第3个被观察的训练输入 images 不同，但 conditions、targets 和 before 的逐源 RNG 相同。**
- 因而2D/astig正式基线已经在中断前分叉，不能将终态差异归因于checkpoint恢复；下一步应隔离vector CUDA renderer/数据缓存生成。
- 5731 中，astig epoch1模型tensor相同，但held-out浮点统计不完全相同，故完整epoch严格门禁仍失败；采样前三训练步并不能覆盖held-out渲染或后续训练步。

这轮完成的是“统一起点、消除全局RNG初始差异、可定位分叉”的第一批；**不是全部CUDA确定性问题解决**。自定义浮点atomicAdd是渲染非确定性候选，仍需后续A/B隔离；本轮没有据此改kernel，也没有放宽门禁。

## 验证与边界

主要实现后全量 pytest：441 passed、1 skipped。最后门禁修订后相关回归：27 passed；最终trace测试5 passed。修改范围Ruff通过。测试覆盖seed复现、字节指纹/负零、指纹不消耗RNG、恢复首批差异定位、终态相同不能掩盖trace失败。

新文件位于scripts/train、tests/training、docs/reviews，GPU输出位于output/logs。父Git下项目仍整体未跟踪；既有根缓存保留，没有提交或删除历史产物。

下一批只做固定物理参数和固定emitter events的重复渲染比较，以及连续/恢复cached-window输入重建对照。先区分renderer浮点聚合与provider缓存，不再泛化为“所有resume实现有问题”。
