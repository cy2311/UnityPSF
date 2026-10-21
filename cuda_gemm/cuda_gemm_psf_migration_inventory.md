# CUDA GEMM Identical-PSF 迁移清单

## 1. 目标与结论

目标是将当前 `unity_v0.4` 在线训练中的 LUT PSF simulation，替换为
`unity_v0.3` 已使用的 CUDA GEMM vector-PSF simulation。这里的
“identical PSF”应理解为：同一个 simulation batch 使用同一套 vector-PSF
optical context（pupil、polarization、CZT transform、校正场/载波等），但每个
emitter 仍单独输入自己的 `coeffs_rad` 和 `z_nm`，由 CUDA extension 生成独立
PSF patch，之后再按 frame/index 做投影和累加；不是把多个 emitter 合并成一个
LUT patch，也不是继续使用预计算的 LUT patch bank。

初步结论：

1. v0.4 已经有 CUDA GEMM renderer 和 CUDA extension 文件，不能简单按文件名
   全量覆盖。
2. v0.4 的 CUDA `.cu/.cpp` 实现与 v0.3 基本一致，优先保留当前文件，先做
   接口和调用链核对。
3. 主要迁移工作在 Python simulation/provider 层：v0.4 当前把 LUT cache、LUT
   插值、sub-pixel shift、patch projection 和 sequence 生成集中在
   `src/neptune_v03/localization/online.py`；v0.3 已将这些职责拆成
   `data/online_*` 模块，并让 `cuda_gemm` 走统一的 per-emitter renderer。
4. 迁移时需要把 v0.3 的 `unity_psf` import 改成 v0.4 的 `neptune_v03`，不能
   直接复制后运行。

## 2. 两版当前状态

### v0.3：目标实现

- 入口配置：`../unity_v0.3/configs/overrides/cuda_gemm_from_scratch.yaml`
- 训练 launcher：`../unity_v0.3/scripts/train/unitypsf_cuda_gemm_from_scratch.sbatch`
- simulation 主实现：`../unity_v0.3/src/unity_psf/localization/simulator.py`
- CUDA GEMM wrapper：`../unity_v0.3/src/unity_psf/localization/cuda_gemm_renderer.py`
- online provider：`../unity_v0.3/src/unity_psf/localization/data/online_provider.py`
- native/sequence 生成：`../unity_v0.3/src/unity_psf/localization/data/online_native.py`
- renderer cache 和 simulator config：
  `../unity_v0.3/src/unity_psf/localization/data/online_physical.py`
- provider schema：`../unity_v0.3/src/unity_psf/localization/data/online_types.py`
- camera readout、target contract、geometry、validation：
  `../unity_v0.3/src/unity_psf/localization/data/online_camera.py`、
  `online_targets.py`、`online_geometry.py`、`online_validation.py`

v0.3 的 provider 只接受 `native`、`cuda_gemm`、`ppg_direct`，没有 LUT
backend。CUDA GEMM renderer 对大量 emitter 按 4096 个一批切分后调用
`render_psf_bank`，再使用 `reduce_frames` 完成 frame projection。

### v0.4：当前实现与替换点

- 当前 simulation：`src/neptune_v03/localization/simulator.py`
- 当前 online orchestration 与 LUT 实现：
  `src/neptune_v03/localization/online.py`
- 当前 CUDA wrapper：`src/neptune_v03/localization/cuda_gemm_renderer.py`
- 当前 CUDA extension：
  `src/neptune_v03/localization/cuda_extension/bindings.cpp`、
  `kernels.cu`、`setup.py`
- 当前默认 LUT 配置：`configs/microtube_base.yaml`
- 仍显式使用 LUT 的 launcher：
  `scripts/train/fast_route_3052_strength_oldzmap.sbatch`、
  `scripts/train/fast_route_3367_peakbootstrap.sbatch`

v0.4 的 `online.py` 仍包含以下 LUT 专用代码：

- `_LUTPatchBank`、`_LUTPatchBankCache`
- `_build_lut_patch_bank`、`_lookup_lut_patches`
- `_linear_lut_indices_and_weight`
- `_apply_lut_subpixel_shift` 及 Triton shift kernel
- `_project_patches_to_frames` 及 Triton projection kernel
- `_PhysicalVersionedLUTLifecycle`
- `_build_lut_like_sequence`、`_build_lut_like_sequence_batch`
- `_simulate_lut_like_sequence_frames`
- LUT field/storage/epoch prewarm/backend validation 相关 helper

这些代码不是 CUDA GEMM 路径的依赖，应在切换完成并通过兼容性检查后删除，
或先隔离为明确的 legacy 模块；不要让新 `cuda_gemm` 路径继续调用它们。

## 3. 文件迁移清单

### A. 必须迁移或重构的运行代码

| 优先级 | v0.3 来源 | v0.4 目标/动作 | 说明 |
|---|---|---|---|
| P0 | `src/unity_psf/localization/simulator.py` | `src/neptune_v03/localization/simulator.py` | 迁移 `renderer` 注入、`_build_emitter_renderer`、per-emitter `render_many/render_frames` 分发；保留 v0.4 已有的包名和现有额外 PSF 类型时需人工合并。 |
| P0 | `src/unity_psf/localization/cuda_gemm_renderer.py` | 当前同名文件 | 当前仅有关键差异：v0.3 对 `render_psf_bank` 按 4096 emitter 分块；需确认这是 v0.4 需要的显存保护，然后保留/迁移该行为。 |
| P0 | `src/unity_psf/localization/data/online_types.py` | `src/neptune_v03/localization/online.py` 或新 `localization/data/online_types.py` | 统一 provider schema，明确默认 `simulation_backend="cuda_gemm"`，删除 `lut_simulation` 专用字段。 |
| P0 | `src/unity_psf/localization/data/online_physical.py` | 新 `src/neptune_v03/localization/data/online_physical.py`，或先合并进 `online.py` | 管理 vector renderer cache、physical version lifecycle、`LocalizationSimulatorConfig` 构造。 |
| P0 | `src/unity_psf/localization/data/online_native.py` | 新 `src/neptune_v03/localization/data/online_native.py` | 用统一 simulator 产生 sequence/frame，完成 camera readout、targets、conditioning；这是替代 `_simulate_lut_like_sequence*` 的核心。 |
| P0 | `src/unity_psf/localization/data/online_provider.py` | 新 `src/neptune_v03/localization/data/online_provider.py`，并更新包出口 | provider 只分发 `native/cuda_gemm`，不能再把 LUT 作为有效 backend。 |
| P1 | `src/unity_psf/localization/data/online_camera.py` | 同路径新文件 | 迁移相机 Poisson/readout contract，避免 LUT route 的旧 ADU 处理重复执行。 |
| P1 | `src/unity_psf/localization/data/online_targets.py` | 同路径新文件 | 迁移 target order、detect target、padding/finalization；需适配 v0.4 当前 target contract。 |
| P1 | `src/unity_psf/localization/data/online_geometry.py` | 同路径新文件 | 迁移 cached/sliding/random field origin；仅保留 CUDA GEMM 所需的 conditioning geometry。 |
| P1 | `src/unity_psf/localization/data/online_validation.py` | 同路径新文件 | 把 `lut` 拒绝逻辑改成明确错误，并保留 empirical PSF/native 等约束。 |
| P1 | `src/unity_psf/localization/data/online_conditioning.py` | 对照 v0.4 现有 conditioning 实现合并 | CUDA GEMM 需要将 absolute coefficient maps/conditioning provider 转成 renderer 使用的 per-emitter coefficients。不要无差别覆盖 v0.4 的 conditioning contract。 |

建议采用“先拆分、后删除 LUT”的方式：先把 v0.4 的 `online.py` 拆出上述
provider/native/physical 模块并保持行为，再切换 backend；不要直接把 v0.3
整个 `online` 包覆盖到 v0.4，因为两版 target、conditioning 和 package API
已经不同。

### B. CUDA extension：需要核对，不应默认搬运

以下文件在两版目前内容基本一致：

- `src/unity_psf/localization/cuda_extension/bindings.cpp`
- `src/unity_psf/localization/cuda_extension/kernels.cu`
- `src/unity_psf/localization/cuda_extension/setup.py`
- `src/unity_psf/localization/cuda_extension/__init__.py`

v0.4 已有对应文件：

- `src/neptune_v03/localization/cuda_extension/bindings.cpp`
- `src/neptune_v03/localization/cuda_extension/kernels.cu`
- `src/neptune_v03/localization/cuda_extension/setup.py`
- `src/neptune_v03/localization/cuda_extension/__init__.py`

因此这里只需确认以下事项，不建议重复复制：

1. extension module 名称 `neptune_v03_cuda_gemm_v1` 与 wrapper 一致；
2. `render_psf_bank` 与 `reduce_frames` 的 tensor shape、dtype、device contract
   一致；
3. 编译缓存落在 `.local/cache/torch_extensions`，不能写入源码目录；
4. CUDA 不可用时仍给出明确错误，不能静默退回 LUT。

### C. 已存在但需要接通的 v0.4 文件

以下文件不应搬运，但要修改调用或配置：

- `src/neptune_v03/localization/__init__.py`：导出新的 provider/data 模块，
  保持外部 import 兼容。
- `src/neptune_v03/localization/runtime_config.py`：读取
  `simulation_backend: cuda_gemm`，停止从 `lut_simulation` 推导 vector 参数；
  保留必要的通用 `vector_psf_size`、`npupil`、z-range、photon/background
  参数。
- `src/neptune_v03/localization/training/runtime.py` 与
  `src/neptune_v03/training/run_high_fidelity.py`：确认 provider 构造仍使用
  新 schema。
- `src/neptune_v03/optics/vector_psf.py`：当前与 v0.3 已基本对齐，不需要整文件
  替换；只确认 `build_vector_psf_context`、CZT matrix 和 coefficient order。
- `src/neptune_v03/field_origin.py`、conditioning 相关模块：保留 v0.4 API，
  仅确保结果能传给 CUDA GEMM renderer。

### D. 配置和 launcher

需要新增或修改：

- 将 `configs/microtube_base.yaml` 的 `simulation_backend: lut` 改为
  `cuda_gemm`，并删除/停用 `lut_simulation` 配置块。
- 修改两个 `scripts/train/fast_route_*.sbatch`，移除 LUT 专用参数，补齐
  `npupil`、`vector_psf_size`、`vector_batch_size`、`simulation_output_device`
  以及 CUDA 可用性要求。
- 参考但不要直接复制 v0.3 的
  `configs/overrides/cuda_gemm_from_scratch.yaml`；v0.4 的 training/runtime
  schema 不同，需要重新生成适配配置。
- 如保留旧 LUT 配置用于历史复现，应移到显式 legacy 配置，并禁止默认训练路由
  读取。

## 4. 测试与验收清单

### 必须补齐/迁移的测试

- v0.3 `tests/localization/test_simulator_density.py`：确认 emitter density、
  z/photon/background sampling 在新 simulator 中保持一致。
- v0.3 `tests/localization/test_online_module_boundaries.py`：确认 provider
  schema 和模块边界，不再暴露 LUT-only 字段。
- v0.3 `tests/localization/test_conditioning_profile.py` 中 CUDA GEMM 相关用例：
  确认同一 seed 可复现、不同 frame 不相同、metadata 标记为 `cuda_gemm`。
- v0.3 配置测试中关于 `simulation_backend == "cuda_gemm"` 的断言：迁移到
  v0.4 的配置测试，不要继续断言默认 `lut`。
- 新增 renderer contract 测试：`render_psf_bank` 输出 shape、归一化、每个
  emitter 独立性、`reduce_frames` 的 frame index/边界裁剪。

### 验收顺序

1. CPU/static：import、config materialization、provider schema、LUT backend
   被拒绝。
2. CUDA unit：小 `npupil`、小 `vector_psf_size` 下验证 extension output 和
   Python vector renderer 的数值接近性。
3. CUDA integration：同一 optical context 下比较单 emitter、多 emitter、
   多 frame 的 `model_input/detect_tar/bkg_tar/pxyz_tar`。
4. memory regression：验证 4096 emitter chunk 不产生显存峰值回归。
5. training smoke：确认 launcher 不再初始化 LUT patch bank/cache/warmup。

## 5. 暂不搬运的内容

- `../unity_v0.3/src/double_helix/**`、`../unity_v0.3/src/ppg/**`：属于其他 PSF/校准路线，和
  identical vector PSF CUDA GEMM simulation 无直接依赖。
- `../unity_v0.3/src/unity_psf/optics/empirical_psf.py`、
  `../unity_v0.3/src/unity_psf/optics/zmap_renderer.py`：只有在 v0.4 明确需要 empirical/astigmatism route 时才
  单独迁移。
- `.local/`、`logs/`、`output/`、编译生成的 `.so/.o`、缓存和 checkpoint：
  不属于源码迁移，不应复制。
- v0.3 的完整配置、训练 launcher、README：只能作为参数和 contract 参考，
  不能覆盖 v0.4 的训练入口。

## 6. 推荐实施顺序

1. 先为 v0.4 建立 `localization/data/` 模块边界，保留当前行为跑通静态测试。
2. 将 simulator 的 renderer 创建改成可注入，并接入
   `CudaGemmRenderer.render()` / `reduce_frames()`。
3. 将 online provider 的默认 backend 和 sequence path 切到 `cuda_gemm`。
4. 删除或隔离 LUT patch bank、LUT interpolation、Triton LUT shift/projection
   和 LUT lifecycle 代码。
5. 更新配置与 launcher，最后运行 CUDA integration/training smoke。

本清单只描述迁移范围和依赖，尚未修改 v0.4 的运行代码。
