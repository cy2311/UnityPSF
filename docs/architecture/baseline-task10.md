# UnityPSF 任务 10 基线冻结

基线标识：`unitypsf_task10_baseline_20260804`

冻结日期：2026-08-04

冻结范围：多通道像散实例的 raw domain、peak zmap、coefficient map、gamma、
physical state 和 checkpoint 引用隔离。

## 已冻结行为

- left 和 right 只使用自己的 channel run 目录：

  ```text
  <run>/channels/left/
  <run>/channels/right/
  ```

- 每个 channel 都使用自己的
  `metadata/current_physical_state.json`，并独立保存 checkpoint、peak stage、
  coefficient map、gamma 更新计数和 condition-store 状态。
- 有 expert instance 的三种 PSF runtime 都输出统一的 `expert_instance`、
  `channel_layout` 和 `input_frame_spec` contract；单 channel 的 model/provider
  domain 数和 condition 维度保持一致。
- 两个 channel 可以从同一个 anchor profile 或 prototype 初始化，但不共享
  `Parameter`、optimizer、scheduler、physical state、peak zmap 或训练 checkpoint。
- physical state 身份必须匹配当前 `expert_type`、`instance_id` 和 `channel_id`。
  把 right 的 state 替换到 left 的位置会被 checkpoint extra 和 resume 拒绝。
- coefficient map 的名称必须等于当前 channel；legacy resume 也不能把 right map
  载入 left。每个 channel 的 checkpoint physical state 最多包含一张 map。
- physical state 记录 peak zmap 的 SHA-256；恢复和 checkpoint 写入前都会重新计算
  并校验该 hash。zmap 被替换或损坏时直接失败。
- 单 channel peak bootstrap 不会在多个 domain 中静默选第一个；找不到当前
  channel 的唯一 domain 时显式报错。
- 单 channel runtime 只加载当前 channel 的 coefficient map。gamma 的单 domain
  ROI split 绑定到当前 channel，父编排器不合并不同 channel 的 gamma 更新。
- 多通道 CLI 在写出 `channels/<id>/config.yaml` 前会过滤另一 channel 的：
  `real_tiff_wake.domains`、`dual_domain_coeff_maps`、LUT `dual_domain_zmaps`、
  ROI-bank `base_coeff_maps`、ROI-bank source `domains` 和 `auto_build_domains`。
- 多个候选 domain/map 没有当前 channel 的唯一匹配时，配置物化失败；只有一个
  候选时绑定并将其名称规范化为当前 channel。
- 任务 10 不组装最终 UnityPSF bundle；父 manifest 只记录独立 channel 运行状态。

## 冻结入口

- `src/unity_psf/training/channel_context.py`
- `src/unity_psf/training/run_high_fidelity.py`
- `src/unity_psf/localization/runtime_config.py`
- `src/unity_psf/cli/multichannel.py`
- `tests/training/test_multichannel_physical_isolation.py`
- `tests/cli/test_multichannel.py`

## 验证凭据

任务 10 定向测试：

```text
tests/cli/test_multichannel.py
tests/training/test_multichannel_physical_isolation.py
tests/training/test_channel_physical_context.py
tests/training/test_astigmatism_runtime.py
tests/training/test_multichannel_orchestrator.py
结果：37 passed, 5 warnings
```

全量测试命令：

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/home/guest/Others/main/race/unity/.local/cache/matplotlib \
XDG_CACHE_HOME=/home/guest/Others/main/race/unity/.local/cache \
/home/guest/anaconda3/bin/pytest -q \
-o cache_dir=/home/guest/Others/main/race/unity/.local/cache/pytest
```

全量结果：

```text
73 passed, 5 warnings in 42.52s
exit code: 0
```

warning 来自当前环境的 CUDA 探测和已有 `vector_psf.py` tensor 构造，不是
任务 10 新增行为失败。

## 冻结边界

- `neptune_v0.3` 不属于本基线，必须保持不变。
- 任务 10 没有把 left/right 合并为一个模型、optimizer、scheduler、batch、loss、
  gradient 或 checkpoint。
- 任务 11 尚未负责 bundle builder、可移动 manifest、完整 artifact 清单和最终
  bundle hash 验收；在任务 11 完成前不能宣称最终模型包交付完成。
- 若改变 channel run 路径、physical-state identity、zmap hash 或 CLI 过滤规则，
  应新增后续 baseline，不覆盖本文件。
