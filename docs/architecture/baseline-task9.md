# UnityPSF 任务 9 基线冻结

基线标识：`unitypsf_task9_baseline_20260802`

冻结日期：2026-08-02

冻结范围：三种 PSF 模态的独立 left/right 编排、每通道配置物化、本地顺序调度、SLURM 独立脚本、父级 manifest 状态汇总。

## 已冻结行为

- `PSFModality` 的 `emitter_2d`、`astigmatism` 和 `double_helix` 都可以通过同一套 `MultichannelTrainingPlan` 生成独立 channel 运行。
- 每个 channel 有自己的 `ChannelRunSpec`，包括 `instance_id`、`channel_id`、seed、crop、run root、run name、config 和可选 prototype 引用。
- 每个 channel 的输出目录都是独立单通道 run：

  ```text
  <run-root>/<modality>/channels/left/
  <run-root>/<modality>/channels/right/
  ```

- `unity-psf-train-multichannel --mode plan` 会写出：

  ```text
  channels/left/config.yaml
  channels/right/config.yaml
  ```

  每个配置只保留自己的 crop 和 channel，并把 `train.expert.instance_id`、`channel_id` 绑定到当前 channel。
- `--mode local` 按 channel 顺序启动独立 Python 子进程；一个 channel 失败时，默认保留已完成 sibling 并继续执行。
- `--mode slurm` 为每个 channel 生成独立脚本、日志目录和输出目录。脚本不共享 checkpoint 或 physical-state 路径。
- 父编排器只保存 `metadata/multichannel_manifest.json`，不创建联合模型，不持有联合 optimizer/scheduler，不执行联合 backward。
- 原型 checkpoint 只作为 channel 初始化引用；任务 9 不回写 prototype，也不合并实例 checkpoint。

## 验证凭据

```text
tests/training/test_multichannel_orchestrator.py: 8 passed
tests/cli/test_multichannel.py: 2 passed
合计：10 passed
全量 UnityPSF 测试：59 passed, 5 warnings
CLI plan/slurm smoke：exit code 0
```

## 冻结边界

- 任务 9 只冻结编排和实例目录边界。left/right 的 peak zmap、gamma optimizer、coefficient map、physical-state 和 checkpoint 内嵌引用仍由任务 10 实现。
- 任务 10 不得把两个 channel 合并为一个 optimizer、scheduler、physical-state 文件或 checkpoint。
- 任何修改 `ChannelRunSpec` 路径语义、父 manifest schema 或 CLI channel config 物化规则的变更，都应新建后续 baseline。
