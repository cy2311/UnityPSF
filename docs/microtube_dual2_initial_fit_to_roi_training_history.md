# `microtube_dual_2`：Peak/Initial fit、Z-map、ROI 到训练的过程记录

记录日期：2026-10-19  
证据范围：`results/microtube/microtube_dual_2` 及其配置、日志、run manifest、metrics 和 artifacts。本文按该目录当前留存证据重建实验链路；SLURM job 的先后以 job ID 和日志/产物为线索，不把目录名或配置注释当作运行成功证明。

## 结论摘要

- 训练由 `submit_neptune_v03_retrain.sbatch` 从 Anchor-99 / peak-bootstrap 系列配置派生，但本次 resolved config 的 `peak_zmap_bootstrap.enabled` 是 `false`，且 `run/stages/peak` 没有留存文件。因此本目录不能证明本次 run 执行过独立的 peak-bootstrap stage；可追溯的是它使用了预先配置的双通道 Z-map。
- 初始训练 job `5814` 使用 `3392` Z-map，训练至 epoch 30、global step 12,510；在首次 ROI/TIFF hook 读入错误路径后失败。训练阶段本身已完成前 30 epochs。
- `5815` 用本次 800 mW raw TIFF 重新完成左右 Z-map initial fit。后来有两条不同续训：`5817` 修复 TIFF 路径、但继续用 `3392` map；`5818` 尝试换成 `5815` 新 map，但配置构建阶段报 `KeyError`；`5819` 修正后换新 map，从 `5814` epoch 30 checkpoint 续训。
- 到本次核验的最新 artifacts 快照（2026-10-19 22:03 +08），旧 map 分支 `5817` metrics 到 epoch 184；新 map 分支 `5819` 已更新到 epoch 60、global step 25,020。两条均未在本目录证实完成目标 300 epochs。它们都是模拟 held-out 指标，不是 raw-TIFF 检测精度。
- 因此，“最后训练”不是一条线性续训：`5817` 和 `5819` 是从共同 epoch-30 checkpoint 分叉、采用不同 Z-map 的对照续训。新 map 是否最终改善训练，还不能由当前快照下结论。

## 数据、代码与初始化来源

| 项目 | 记录 |
|---|---|
| 本次 raw TIFF 输入别名 | `datasets/training_sets/microtubule_3d_800mw_7_1` |
| TIFF 实际目标 | `/share/home/guest/race/datasets/samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif` |
| 训练代码 | `neptune_v0.3/src/neptune_v03/training/run_high_fidelity.py` |
| 配置派生源 | `results/microtube_right_benchmark_20260711/runs/neptune_v03_fd_nat/training_dual_anchor99_gamma10_conditionfix/resolved_config.yaml` |
| 所有本文列出的训练 job | Neptune v0.3；不是 Unity v0.3/v0.4 训练 |

提交脚本将 `phase_retrieval.input.tiff_path` 改为本次 raw TIFF，但原始 `train.real_tiff_wake.tiff_path` 仍是 `neptune_iwae/test_data/microtube/raw/...`。这正是 `5814` ROI hook 失败的路径来源。`5817`/`5819` 的修复配置把两处 TIFF path 都指到 `datasets/training_sets/microtubule_3d_800mw_7_1`。

配置元数据写有 `initialization_family: microtube_peak_bootstrap_anchor99`，且 `initial_zmap_tag` 写作 `..._3421`；但训练配置中的实际 `dual_domain_coeff_maps` 文件路径指向 `3392` 目录。配置同时关闭 inline `peak_zmap_bootstrap`。所以应区分“Anchor-99/peak-bootstrap 的配置 lineage”与“本次实际运行的 peak stage / Z-map 文件”：后者由 manifest 中 map 路径判定，不能仅凭 tag 或 initialization 名称推断。

## 按时间顺序

### 1. 旧的 initial Z-map：job `3392`

旧 map 位于：

```text
neptune_v0.3/output/high_quality_initial_zmap/
  microtube_formal_order2_frame1000_emit500_round20_p1baseline_recomputed_3392/
```

虽命名为 microtube，`3392` 的 SLURM 输出和左右 diagnostics 记录的输入 TIFF 是 PAINT 数据：

```text
/share/home/guest/race/datasets/PAINT/spool_200mW_30ms_3D_-lh1_1_MMStack_Default.ome.tif
```

该 initial fit 使用 order-2 NAT、每侧最多 500 个 emitter、20 轮 alternating fit（每轮 local/global 各 100 steps）、每通道第 1 percentile baseline。左右结果不对称：

| Channel | Baseline | Selected emitters | 状态 | Patch NCC mean |
|---|---:|---:|---|---:|
| Left | 112 ADU | 0 | `insufficient_emitters` | 0 |
| Right | 107 ADU | 500 | `ok` | 0.8816 |

因此这套旧 map 不能视为从本次 `microtube_dual_2` TIFF 完成的左右 initial fit；left 尤其没有足够 emitter 支撑拟合。

### 2. 本次 raw TIFF 的左右 initial Z-map：job `5815`

脚本：`results/microtube/microtube_dual_2/rebuild_and_compare_microtube_zmap.sbatch`  
日志：`zmap-rebuild-5815.out` / `zmap-rebuild-5815.err`  
输出：`zmap_microtube_raw/{left,right}/`

左右 pipeline 使用同一份本次 800 mW TIFF，order-2 NAT、最多 500 个 selected emitters、approximate fit 60 steps、20 轮 alternating fit（每轮 local/global 各 100 steps）、baseline percentile 1，baseline 取帧 `[0,100)`。

| Channel | Baseline | Harvest emitters | Selected / fit emitters | Alternating patch NCC mean |
|---|---:|---:|---:|---:|
| Left | 189 ADU | 65,466 | 500 | 0.8741 |
| Right | 171 ADU | 1,016,014 | 500 | 0.8983 |

两侧均输出 `preferred_full_roi_zernike_maps_nm.npz`。map shape 为 `[13,1200,600]`。与旧 `3392` 导出图的 RMS 差异：left `6.373 nm`，right `8.288 nm`。差异只说明 maps 不同，不等同于 accuracy 验证。此阶段产出的新 map 是后续 `5819` 配置实际引用的 map。

#### `5815` 的代码与 map 生成链

可复现的调度入口是 `results/microtube/microtube_dual_2/rebuild_and_compare_microtube_zmap.sbatch`：它检查 raw TIFF、旧 map 存在且目标 map 尚不存在，设定 Neptune Python path，然后依次调用 left 和 right 两个独立 pipeline。参数包含 GPU `cuda:0`、每侧最多 500 emitters、20 轮 alternating、每轮 local/global 各 100 steps、selection pool multiplier 50，以及第 0–99 帧的第 1 percentile baseline。

实际 Python 入口为 `neptune_v0.3/scripts/train/run_high_quality_initial_zmap.py::main`。它调用 `build_zmap_preset("microtube", side)`（`neptune_iwae/zmap_main/pipeline_presets.py`），再将 preset 的 TIFF 覆盖为脚本传入的 `datasets/training_sets/microtubule_3d_800mw_7_1`。pipeline summary 记录该别名解析到 `/share/home/guest/race/datasets/samples/sample/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif`，模型 checkpoint 为 `neptune_iwae/zmap_main/assets/microtube/checkpoint_best_jaccard.pt`，输入 pipeline config 为同目录下的 `pipeline_smlm_uni_decode_aligned_fixedscale_depthwise_mu20000_sigma1000.json`；preset 的左右 ROI 分别是 x `[0,600)` 和 `[600,1200)`，y `[0,1200)`。

每侧处理顺序：

1. `_estimate_channel_baseline` 从 TIFF 第 0–99 帧的对应 channel crop 计算第 1 percentile baseline，得到 left `189 ADU`、right `171 ADU`。
2. `run_clean_reharvest`（`neptune_iwae/zmap_main/clean_reharvest.py`）加载 pretrained detector，在 TIFF 第 0–999 帧做 emitter harvest，再生成 clean patch library。报告记录 left candidate/accepted 为 `89,893/65,466`，right 为 `1,044,041/1,016,014`，两侧 clean patch library 均为 256。
3. `run_real_nat_diagnostics`（`neptune_iwae/zmap_main/real_nat_diagnostics.py`）读取 harvest，按空间分布挑选 500 emitters，执行 approximate fit 60 steps 和 order-2 NAT alternating fit 20 轮，并记录拟合诊断。patch NCC mean：left approximate/alternating `0.8742/0.8741`；right `0.8884/0.8983`。
4. `run_export_nat_zmap`（`neptune_iwae/zmap_main/export_nat_zmap.py`）将 13 个 Zernike modes 导出为全 ROI coefficient stack，并写 preferred map。导出配置默认 `preferred_stage="alternating"`，因此训练用的 `preferred_full_roi_zernike_maps_nm.npz` 是 alternating map；left diagnostics 的 `preferred_recon_stage=approximate` 是 raw/recon 诊断选择，不改变此导出默认值。

供 `5819` 使用的精确文件：

```text
results/microtube/microtube_dual_2/zmap_microtube_raw/left/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz
results/microtube/microtube_dual_2/zmap_microtube_raw/right/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz
```

中间证据分别在各侧 `clean_reharvest/clean_reharvest_summary.json`、`real_nat_diagnostics/real_nat_diagnostics_summary.json`、`real_nat_diagnostics_payload.pt`、`export_nat_zmap/export_nat_zmap_summary.json` 和 `pipeline_summary.json`。`zmap_comparison.md/.json` 是新 map 与旧 `3392` map 的比较报告，不是 map 本身。

### 3. 首次训练：job `5814`，旧 map

配置：`resolved_config.yaml`  
输出：`run/`  
训练设置：300 epochs、417 steps/epoch、dual-channel Soft-MoE、GMM loss、cached temporal-window simulation。

`5814` 的 manifest/config 实际引用 `3392/{left,right}/...preferred_full_roi_zernike_maps_nm.npz`。训练到 epoch 30、global step 12,510 后，首次建立 raw-TIFF ROI bank 的 epoch-end hook 报 `FileNotFoundError`，路径为：

```text
neptune_iwae/test_data/microtube/raw/
spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif
```

epoch 30 的 checkpoint 已生成，故这是 ROI hook 输入路径失败，不是前 30 epochs 的训练失败。epoch 30 模拟 held-out 指标：Jaccard `0.8551`、precision `0.9864`、recall `0.8653`、XY RMSE `24.47 nm`、Z RMSE `41.10 nm`、eval loss `229.87`。

### 4. 路径修复续训：job `5817`，仍用旧 map

配置和运行目录：`retry_5814_pathfix/resolved_config.yaml`、`retry_5814_pathfix/run/`。manifest 证实从 `5814` epoch 30 / global step 12,510 checkpoint 恢复，下一 epoch 为 31。它把 phase TIFF 与 ROI raw TIFF 路径都改为本次数据，但 `dual_domain_coeff_maps` 仍指向旧 `3392`。

恢复时先执行 epoch 30 ROI hook，左右各构建 78 个 ROI 并执行 100-step physical update，随后继续训练；配置按每 10 epochs 更新。当前物理 map 从 epoch 30 开始由 gamma feedback 更新，但初始 map provenance 仍为 `3392`。

ROI 抽样 raw/reconstruction montage 的 5 个 patch NCC：left `0.293, 0.011, 0.208, -0.080, 0.505`；right `0.791, 0.507, 0.509, 0.157, 0.718`。样本质量波动，left 尤其偏低。该诊断不是全帧 detection overlay。

本目录快照中 `5817` 的 held-out metrics 到 epoch 184 / global step 76,728：Jaccard `0.8597`、precision `0.9302`、recall `0.9190`、XY RMSE `27.33 nm`、Z RMSE `48.70 nm`。epoch 183 的 Jaccard 为 `0.8831`。这些是模拟 held-out 指标；metrics/checkpoint 未达到 300 epochs，目录中也未见训练正常完成的证据。

### 5. 新 map 续训尝试：job `5818` 失败、job `5819` 使用新 map

`retry_5815_zmap/resolved_config.yaml` 将左右 `dual_domain_coeff_maps` 改为 `zmap_microtube_raw` 的新 map，但仍从共同的 `5814` epoch 30 / global step 12,510 checkpoint 恢复。也就是说，这是换初始 map 的分支续训，不是从头训练。

- `5818` 在准备/合并配置时失败，stderr 报 `KeyError: 'dual_domain_coeff_maps'`；该目录没有可用 eval metrics。
- 实际留存的 `5819` 提交脚本为 `.local/scripts/retrain_microtube_dual2_5815_zmap.sbatch`。脚本从 `retry_5814_pathfix/resolved_config.yaml` 复制配置，重写 phase TIFF 和 `real_tiff_wake.tiff_path`，把左右 `dual_domain_coeff_maps` 改到上述两个 `preferred_full_roi_zernike_maps_nm.npz`，并把 metadata 标为 `zmap_rebuild_job_5815`。它检查旧 checkpoint、raw TIFF、左右 map 都存在，再以 job name `mtR_5815_zmap` 提交到 `cpu1` 分区（1 GPU、12 CPU、96 GB RAM、最长 120 小时）。
- 训练命令入口为 `sbatch .local/scripts/retrain_microtube_dual2_5815_zmap.sbatch`；脚本最终执行 `python -u -m neptune_v03.training.run_high_fidelity --config .../retry_5815_zmap/resolved_config.yaml --run-root .../retry_5815_zmap --run-name run --seed 20260711 --resume-checkpoint .../run/checkpoints/checkpoint_latest.pt`。源 checkpoint 是 `5814` epoch 30 / global step 12,510；代码 `resume_epoch_training_config` 将下一 epoch 设为 31。配置目标 300 epochs、417 steps/epoch，双域 `active_smlm_soft_moe_double_unet` + GMM loss；AdamW lr `0.0006`、weight decay `0.1`，StepLR 每 10 epoch 乘 `0.9`。Neptune v0.3 的 `run_high_fidelity.py::main` 载入配置、构建 runtime、恢复 checkpoint、写 manifest，然后调用 `neptune_v0.3/src/neptune_v03/training/loop.py::train_epochs`；ROI gamma hook 在恢复点先运行，之后按配置每 10 epochs 更新 physical map。
- resolved run manifest（不是只看 YAML tag）确认 `5819` 的训练 runtime 左右 maps 都指向 `zmap_microtube_raw`；raw ROI bank path 也是本次 TIFF。`run_high_fidelity.py::_build_roi_bank_gamma_hook` / `_resolve_roi_bank_source` 读取该 TIFF 配置并接入自动 ROI projection update。epoch 30 ROI hook 自动构建左右 ROI，各域执行 100-step physical update；后续更新输出到 `run/artifacts/roi_bank_gamma/` 并更新 current physical state。初始 map provenance 记在 run manifest；当前反馈 map 另记在 `metadata/current_physical_state.json`，二者是不同阶段的文件。
- 本次核验的最新快照已到 epoch 60 / global step 25,020；epoch 60 模拟 held-out Jaccard `0.8486`、precision `0.9711`、recall `0.8706`、XY RMSE `25.73 nm`、Z RMSE `44.57 nm`、eval loss `360.22`。epoch 50 gamma update 两域 held-out loss 分别上升约 `1.86%`、`1.17%`；两次均记录 accepted，但 policy 是 `monitor`，accepted 不代表 loss 改善。
- 训练主日志 `.out` 当前为空，`.err` 主要是 vector PSF 中 Tensor copy 的 warning；进度以 `run/metrics/` 与 checkpoints 核验。快照未达到 300 epochs，不能称为最终训练结果。

#### `5819` 文件与入口索引

| 用途 | 文件 / 目录 |
|---|---|
| SLURM 启动与配置改写 | `.local/scripts/retrain_microtube_dual2_5815_zmap.sbatch` |
| 运行时 resolved config | `results/microtube/microtube_dual_2/retry_5815_zmap/resolved_config.yaml` |
| 代码入口 | `neptune_v0.3/src/neptune_v03/training/run_high_fidelity.py`（`main`、resume、ROI gamma hook） |
| manifest / 初始 map provenance | `retry_5815_zmap/run/metadata/run_manifest.json` |
| 当前 gamma feedback map | `retry_5815_zmap/run/metadata/current_physical_state.json` |
| 最新 / 最佳模型 checkpoint | `retry_5815_zmap/run/checkpoints/checkpoint_latest.pt`、`checkpoint_best.pt` |
| 评估、训练、ROI 更新指标 | `retry_5815_zmap/run/metrics/eval_metrics.jsonl`、`training_metrics.jsonl`、`gamma_update_metrics.jsonl`、`training_timing.jsonl` |
| stdout/stderr | `retry_5815_zmap/logs/neptune-v03-5819.out`、`neptune-v03-5819.err` |
| ROI/gamma/可视诊断产物 | `retry_5815_zmap/run/artifacts/roi_bank_gamma/`（按 `step_*/` 与 `latest/` 组织） |

## 训练分支与判断

```text
Anchor-99 / peak-bootstrap 配置 lineage
  └─ inline peak_zmap_bootstrap=false；本目录没有独立 peak-stage 产物
      └─ 5814: 以旧 3392 map 训练到 epoch 30
          ├─ 5817: 修 TIFF 路径，继续旧 3392 map → metrics 至 epoch 184
          └─ 5819: 改用 5815 新 raw-TIFF map → metrics 至 epoch 60（本次快照）

独立 initial map rebuild：5815
  └─ 本次 800 mW TIFF 左右各 fit 500 emitters
      └─ 被 5819 分支引用；未被 5817 使用

5818：新 map 分支的配置准备失败（KeyError），没有进入有效训练
```

主要判断：

- `5817` 与 `5819` 从相同 epoch-30 权重恢复，但采用不同 initial Z-map，是 map provenance 不同的两条训练分支；不能把 `5817` 的 epoch 184 结果说成使用新 map 的训练结果。
- `5819` 当前 epoch 60 的 held-out 指标还不足以证明新 map 优于旧 map。即使两条分支 epoch 相同，也要考虑训练随机性、gamma update 等因素；最终比较应使用一致 checkpoint/评估设置，并以完成的 runs 为准。
- 两条分支的 held-out Jaccard/precision/recall/RMSE 均来自模拟验证集，不代表 raw TIFF 的 emitter recall 或真实重建质量。
- physical update 的 `accepted` 不是性能改善声明。两条分支均采用 `heldout_accept_policy: monitor`。
- 本目录没有完整带 emitter circles 的 raw-TIFF held-out detection overlay，无法从现有 ROI montage 量化真实 TIFF 漏检率。

## 关键证据

- 初始配置与提交入口：`submit_neptune_v03_retrain.sbatch`、`resolved_config.yaml`
- `5814`：`neptune-v03-5814.err`、`run/metadata/stage_status.json`、`run/metadata/run_manifest.json`、`run/metrics/eval_metrics.jsonl`
- 新 map rebuild：`rebuild_and_compare_microtube_zmap.sbatch`、`zmap-rebuild-5815.out`、`zmap_microtube_raw/{left,right}/real_nat_diagnostics/real_nat_diagnostics_summary.json`
- Z-map 代码链：`neptune_v0.3/scripts/train/run_high_quality_initial_zmap.py`、`neptune_iwae/zmap_main/pipeline_presets.py`、`clean_reharvest.py`、`real_nat_diagnostics.py`、`export_nat_zmap.py`
- map 比较：`zmap_comparison.md` / `zmap_comparison.json`
- `5817` 分支：`retry_5814_pathfix/resolved_config.yaml`、`retry_5814_pathfix/run/metadata/run_manifest.json`、`retry_5814_pathfix/run/metrics/eval_metrics.jsonl`、`retry_5814_pathfix/run/metrics/gamma_update_metrics.jsonl`
- `5818/5819` 分支：`.local/scripts/retrain_microtube_dual2_5815_zmap.sbatch`、`retry_5815_zmap/logs/neptune-v03-5818.err`、`retry_5815_zmap/resolved_config.yaml`、`retry_5815_zmap/run/metadata/run_manifest.json`、`retry_5815_zmap/run/metadata/current_physical_state.json`、`retry_5815_zmap/run/metrics/`、`retry_5815_zmap/run/checkpoints/`、`retry_5815_zmap/run/artifacts/roi_bank_gamma/`
- 旧 `3392` map 来源核验：`neptune_v0.3/logs/slurm/nv03_3367hq-3392.out` 及旧 map 左右的 `real_nat_diagnostics_summary.json`

## v0.4 独立复刻顺序

目标调整为：**所有训练从 epoch 0 开始**，使用 v0.4 重建且与 5815 完全一致的左右 Z-map，训练至 epoch 300；ROI library 与 gamma/physical feedback 也由 v0.4 内训练代码运行。

### v0.4 代码与输入边界

- 两个执行入口均在 `unity_v0.4/scripts/train/`；训练通过 `python -m neptune_v03.training.run_high_fidelity` 调用 `unity_v0.4/src/neptune_v03/`。作业脚本先 `cd unity_v0.4` 并设置 `PYTHONPATH=unity_v0.4/src`，没有把 Neptune v0.3/v0.4 以外的源码目录加入 Python 搜索路径。
- Z-map 所需的旧实现副本在 `unity_v0.4/src/neptune_iwae/zmap_main/`；SMLM 兼容包在 `unity_v0.4/src/neptune/smlm_v2a/`。为满足副本内模板的历史导入路径，`unity_v0.4/src/smlm_v2a` 是指向 v0.4 内 `neptune/smlm_v2a` 的兼容链接。原 Neptune 源码未修改。
- raw TIFF 和从 5819 配置派生的训练超参数仍是外部数据/配置输入，不是执行代码。当前从零训练不加载 5814/5819 checkpoint；模型从随机初始化开始。`seed=20260711`，目标 `train.epochs=300`。
- ROI library 在 `roi_bank_gamma` 第一次触发时由 v0.4 的 `run_high_fidelity.py` 根据同一 raw TIFF 的 frame `[0,100)` 自动建立；gamma 更新间隔为 10 epoch，配置范围 epoch 30–300。

### 作业与验证记录（2026-10-20）

1. 作业 `5820` 首次重建失败：v0.4 starter loader 预期的模板导入路径未解析到 v0.4 副本。修复为 v0.4 内部兼容链接后重提；没有改原始源文件。
2. 作业 `5821` 在 GPU 上成功重建左右 Z-map。参数为每侧最多 500 emitters、alternating 20 rounds、每轮 local/global 各 100 steps、selection pool multiplier 50、baseline percentile 1%、baseline frames `[0,100)`。运行日志确认 `cuda:0`，左右 `pipeline_summary.json` 均记录选择 500 emitters。
3. 重建 map 与 5815 的 `zmap_microtube_raw/{left,right}/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz` 同形状 `[13,1200,600]`、mode order 相同，逐元素差异 RMS/mean absolute/P95/max 全部为 `0 nm`。两侧 SHA-256 也分别完全匹配：left `6087ae71f84a6534c72f3b152c4119d86f373d540348c40941065ecb7a5cb7ec`，right `187d87ec3e3e7347f22253d2333bd170c6bdaac13d2b8b74c52b0547d0d7bfa5`。
4. 作业 `5822` 曾错误地从 epoch 30 checkpoint 恢复；发现后取消。其部分产物在旧路径 `results/microtube/microtube_dual_2/v04_5819_repro/`，不得与从零训练结果混用。
5. 正确的从零训练作业为 `5823`，启动时 manifest `resume=null`，使用新 map，SLURM 请求并分配 1 GPU。截止本记录核验，作业仍在运行：训练 metrics 到 epoch 81 / global step 33,397，eval metrics 到 epoch 80（Jaccard `0.8756`）；gamma updates 已到 epoch 80，左右 feedback map 已产出。该值是中途指标，不能作为 300 epoch 最终比较。

### v0.4/results 产物位置

- 实体复制的左右 map、harvest、diagnostics、export 与 `pipeline_summary.json`：`unity_v0.4/results/microtube_dual2_v04_from0/zmap_microtube_raw_v04/{left,right}/`。preferred map 的 SHA-256 已与原始重建目录核验一致。
- 训练运行目录入口：`unity_v0.4/results/microtube_dual2_v04_from0/training/`。训练还在运行期间，此处是指向 canonical run 的**实时符号链接**，使 checkpoint/metrics/ROI artifacts 可在 v0.4/results 下查看；canonical 运行数据目前由 SLURM 作业写入 `results/microtube/microtube_dual_2/v04_5819_repro_from0/`。这样避免复制正在变化的 checkpoint 并产生不一致快照。
- stdout/stderr 实时入口：`unity_v0.4/results/microtube_dual2_v04_from0/logs/train-5823.out`、`train-5823.err`，链接到 canonical SLURM logs。
- 训练完成后如需完全脱离外部结果根目录，需在作业结束后将实时链接目标归档为 v0.4/results 内实体文件；作业仍运行时不要移动或复制覆盖正在写入的目录。

原计划“从 epoch-30 checkpoint 续训”的描述已废止。本次训练唯一有效分支是作业 `5823` 的 epoch 0–300 从零训练；须待 job 完成后再核验 completed status、最终 checkpoint/metrics、ROI/gamma artifacts，并与 `retry_5815_zmap` 的最终结果做数值比较。此前较早的“两个作业尚未提交”状态也已由上述实际作业记录取代。
