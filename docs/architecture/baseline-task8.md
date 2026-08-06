# UnityPSF 任务 8 基线冻结

基线标识：`unitypsf_task8_baseline_20260802`

冻结日期：2026-08-02

冻结范围：实例级 v2 checkpoint 保存、完整续训、prototype 权重初始化、实例 lineage 校验、RNG 恢复、legacy checkpoint 兼容和高保真 manifest 来源标记。

## 已冻结行为

- 训练调用方通过 `checkpoint_metadata` 显式提供 v2 `CheckpointMetadata`；训练 loop 不从模型名称或路径猜测专家和通道。
- v2 instance checkpoint 保留顶层 `epoch`、`step_count`、`global_step`、`model_state_dict`、`optimizer_state_dict` 和可选 `scheduler_state_dict`，并写入 metadata、Python/NumPy/Torch RNG、AMP scaler 和 physical-state extra。
- `resume_training_checkpoint(...)` 是完整续训入口。v2 instance 必须匹配 `expert_type`、`instance_id`、`channel_id` 和 `parent_checkpoint_hash`，并恢复 optimizer、scheduler、计数器和 RNG。
- `initialize_model_from_checkpoint(...)` 只接受 prototype v2 checkpoint 的权重初始化，不加载旧实例 optimizer、scheduler、计数器或 RNG。
- `load_training_checkpoint(...)` 继续读取 legacy training、legacy v1 和 raw state-dict 载荷；高保真运行 manifest 记录 `checkpoint_format`。
- 普通实例训练只写自己的 run checkpoint，不修改 `astigmatism_base.ckpt`。
- checkpoint 文件使用同目录临时文件加原子替换，physical-state 文件继续使用任务 7 的原子发布机制。

## 验证凭据

任务 8 定向测试：

```text
7 passed, 1 warning
```

任务 7 physical-state、baseline runtime 和 checkpoint contract 回归：

```text
14 passed, 1 warning
```

warning 来自当前环境的 CUDA 探测，不是任务 8 新增行为失败。

全量 UnityPSF 测试结果：`49 passed, 5 warnings`。high-fidelity 单通道 smoke 和 legacy resume smoke 均以退出码 0 完成。

## 冻结边界

- 当前恢复点是 epoch 边界；epoch 内中断仍需要额外的 batch provider cursor/state contract。
- 任务 9 可以在此基础上实现 left/right 独立编排，但不得把两个实例合并为一个 optimizer 或一个 checkpoint。
- 若改变 v2 metadata、identity 校验或 legacy 分支，应新建后续 baseline，不覆盖本文件。
