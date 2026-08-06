# UnityPSF 任务 7 基线冻结

基线标识：`unitypsf_task7_baseline_20260802`

冻结日期：2026-08-02

冻结范围：单实例物理状态上下文、单通道 peak bootstrap、gamma/conditioning 状态、physical-state manifest 和现有 checkpoint 物理引用。

## 已冻结行为

- 单通道默认实例为 `astigmatism / main / main`。
- 每个实例独立拥有 raw crop、peak zmap、coefficient map、`ConditioningProviderStore` 和 physical-state 路径。
- 单通道 peak bootstrap 只使用当前 channel crop，并写入当前 run layout。
- 660 nm 像散的 99 nm anchor 只属于 `astigmatism_660nm` profile。
- gamma 更新只更新当前实例的 coefficient map 和 condition-store version。
- physical-state JSON 和 gamma coefficient `.npz` 均通过同目录临时文件和原子替换发布。
- manifest 记录 `initial_physical_state_hash` 和 `latest_physical_state_hash`。
- checkpoint 写入并校验 physical artifact 的存在性和 hash；resume 会恢复物理状态、实例绑定和 store version。
- 单通道 ROI-bank domain 无法明确绑定时直接失败，不静默使用其它 channel。

## 冻结入口

- `src/unity_psf/training/channel_context.py`
- `src/unity_psf/training/run_high_fidelity.py`
- `src/unity_psf/optics/profiles.py`
- `src/unity_psf/localization/conditioning.py`
- `src/unity_psf/training/loop.py`
- `tests/training/test_channel_physical_context.py`

## 验证凭据

任务 7 定向测试：

```text
8 passed
```

像散 runtime 与任务 7 定向回归：

```text
13 passed, 5 warnings
```

全量测试命令：

```bash
timeout 120s env CUDA_VISIBLE_DEVICES='' \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR=/home/guest/Others/main/race/unity/.local/cache/matplotlib \
XDG_CACHE_HOME=/home/guest/Others/main/race/unity/.local/cache \
/home/guest/anaconda3/bin/pytest -q \
-o cache_dir=/home/guest/Others/main/race/unity/.local/cache/pytest
```

全量结果：

```text
42 passed, 5 warnings in 45.66s
exit code: 0
```

warning 来自当前环境的 CUDA 探测和已有 `vector_psf.py` tensor 构造，不是任务 7 新增失败。

单通道高保真 smoke 也已通过。`checkpoint_latest.pt` 中的 `physical_state_hash` 与 run manifest 的 `latest_physical_state_hash` 一致，`initial_physical_state_hash` 同时存在于 checkpoint 和 manifest。

## 冻结边界

- `neptune_v0.3` 不属于本基线，必须保持不变。
- 任务 8 的完整 checkpoint v2 contract、随机数状态恢复和“完整续训/仅权重初始化”双 API 尚未实现。
- 三专家 PSF MoE 的训练实验、模态检测器和科学消融不属于本基线。
- 当前父仓库仍将 `main/race/unity/` 作为整体未跟踪迁移目录显示；本冻结记录不改变该仓库状态。

任务 8 及后续 MoE 工作必须以本文件记录的行为和验证结果为起点；若改变上述冻结行为，应新增基线文件，不覆盖本文件。
