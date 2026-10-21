# CUDA GEMM 迁移代码包

本目录是根据 `cuda_gemm_psf_migration_inventory.md` 整理出的迁移工作区。
它保存了 v0.3 的 CUDA GEMM 参考实现和 v0.4 当前实现的对照快照，供后续
逐文件合并使用。

## 目录

### `reference_v03/`

来源：`../unity_v0.3`。

- `src/unity_psf/localization/simulator.py`
  - v0.3 的 per-emitter simulation 入口。
- `src/unity_psf/localization/cuda_gemm_renderer.py`
  - CUDA GEMM wrapper；包含大 batch 按 4096 emitter 分块调用逻辑。
- `src/unity_psf/localization/cuda_extension/`
  - `bindings.cpp`、`kernels.cu`、`setup.py` 和 package marker。
- `src/unity_psf/localization/data/`
  - provider schema、native/sequence 生成、renderer lifecycle、camera、target、
    conditioning、cached-window 相关模块。
- `src/unity_psf/optics/vector_psf.py`
  - CUDA GEMM 依赖的 vector-PSF optical context 参考实现。
- `configs/overrides/cuda_gemm_from_scratch.yaml`
  - v0.3 CUDA GEMM 配置参考，不可直接覆盖 v0.4 配置。
- `scripts/train/unitypsf_cuda_gemm_from_scratch.sbatch`
  - v0.3 launcher 参考。
- `tests/`
  - simulator、provider、conditioning、configuration contract 参考测试。

### `current_v04/`

来源：当前 `unity_v0.4` 工作区；其中 CUDA GEMM 相关文件已同步本次实现。

- `src/neptune_v03/localization/online.py`
  - 当前 LUT orchestration 和 LUT patch-bank 实现；后续替换的主要对象。
- `src/neptune_v03/localization/simulator.py`
  - 当前 simulator；需与 v0.3 reference 合并，而不是直接覆盖。
- `src/neptune_v03/localization/cuda_gemm_renderer.py`
  - 当前已经存在的 CUDA GEMM wrapper。
- `src/neptune_v03/localization/cuda_extension/`
  - 当前已经存在的 CUDA extension；与 v0.3 reference 对照核验后优先保留。
- `src/neptune_v03/optics/vector_psf.py`
  - 当前 optical implementation，用于核对 v0.3 依赖是否已经具备。
- `runtime_config.py`、training runtime、配置、launcher、diagnostic 和测试
  - 当前需要接入 `simulation_backend: cuda_gemm` 的集成面。
- `scripts/train/prepare_microtube_dual2_5819_config.py` 和
  `reproduce_microtube_dual2_v04_from0.sbatch`
  - 复现配置强制切到 CUDA GEMM；新的日志、Z-map、训练配置、checkpoint 和
    report 输出都放在
    `results/microtube_dual2_v04_from0_cuda_gemm/`，不覆盖 LUT 基线目录。

## 使用规则

1. 本目录是迁移 staging area，不是新的可安装 Python package。
2. `reference_v03` 内的 import 仍然是 `unity_psf.*`；迁移到 v0.4 时必须改为
   `neptune_v03.*`，并逐项处理 target/conditioning API 差异。
3. 不要把 `reference_v03/src/unity_psf/localization/data/` 整体直接覆盖到
   `src/neptune_v03/localization/`；v0.4 的 package、target contract 和训练
   入口不同。
4. CUDA `.cu/.cpp` 文件仅做接口和行为核对；当前两版内容基本一致，不需要
   重复引入编译产物。
5. 本目录不包含 `.so`、`.o`、torch extension cache、日志、checkpoint 或
   output 数据。

## 推荐下一步

1. 以 `reference_v03/src/.../data/` 为模板，在 v0.4 中建立对应模块边界。
2. 先接通 `simulator.py` 的 renderer 注入和 `CudaGemmRenderer.reduce_frames`。
3. 将当前 `online.py` 的 provider/backend 分支切换到新模块。
4. 再删除或隔离 LUT patch bank、LUT interpolation 和 Triton LUT projection。
5. 运行 `reference_v03/tests/` 中适配后的 contract tests，再进行 CUDA smoke。
