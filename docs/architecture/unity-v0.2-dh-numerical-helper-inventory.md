# Unity v0.2 DH Numerical Helper Inventory

状态：Phase 4 验收记录

本清单按实际调用边界记录 Double-Helix 数值 helper。合并标准是输入 shape、dtype、device、gradient、归一化和边界行为全部一致；函数名相似不是合并理由。

## 已合并

| 原实现 | 统一实现 | 实际 contract | 证据 |
| --- | --- | --- | --- |
| `field_gamma._legendre` | `double_helix.numerics.legendre_polynomial` | Torch tensor；保留输入 dtype/device；degree 0 返回 `ones_like` 常量，degree >= 1 保留 autograd；负 degree 抛 `ValueError`。 | `tests/optics/test_double_helix_numerics.py`；field-gamma、physical-update、direct-gamma paths 回归通过。 |
| `physical_update._legendre` | 同上 | 与 field-gamma recurrence 逐项一致。 | 同上。 |
| `gamma_field._legendre` | 同上 | 与 DirectGammaZernikeField 的坐标 basis 计算一致。 | 同上。 |
| 六处 DH `_ncc` | `double_helix.numerics.normalized_cross_correlation` | 输入 shape 相同且至少二维；在最后两维执行 per-sample center/sum；`clamp_min(1e-12)` 保留零方差和空 spatial 输入返回零的行为，NaN 继续传播；Torch gradient/device/dtype 透传。 | `tests/optics/test_double_helix_numerics.py`；calibration、field-gamma、localization、LG、pixel-pupil、shared-carrier consumers 回归通过。 |

## 明确保留

| Helper | 保留原因 |
| --- | --- |
| `field_fit._legendre` | NumPy array backend；返回 NumPy dtype/array，不能用 Torch helper 替代。 |
| `field_gamma.spatial_gamma_terms` | 只枚举一阶及以上空间项，degree 必须为正；用于 global `(0,0)` 与 spatial residual 分离。 |
| `physical_update.spatial_gamma_terms` | 包含 `(0,0)` 常数项，允许 degree=0；完整 FOV physical update 的参数布局不同。 |
| `lut._fourier_shift` | 单张二维 NumPy、固定 float32 输出、非 differentiable LUT preprocessing。 |
| `local_fit._fourier_shift_batch` | NumPy batch shift，shift 数组按 batch 广播并固定 float32 输出。 |
| `vector_model.fourier_shift` | Torch differentiable、任意 leading batch shape、输入 dtype/device 保留；正式 renderer/gradient 路径。 |

## 边界与不变量

- NCC 的统一实现只承担最后两维 spatial reduction；调用者如有额外语义（例如 bead/plane 维）必须先显式 reshape，再恢复原 shape。
- 不把 NumPy 与 Torch helper 合并为动态 backend facade；这样会增加 dtype/device 分支并模糊物理路径。
- 不合并两个 `spatial_gamma_terms`，因为常数项是否存在直接改变 gamma 参数列数和物理解释。
- 不改变 Fourier shift 的周期边界、FFT normalization 或输出 dtype；三条路径继续独立测试和命名。

## 验收

Phase 4 本轮仅合并语义完全一致的 Torch recurrence 和 NCC，删除 9 处重复实现，新增一个无状态 `numerics.py`。没有改变 NumPy、FFT、gamma term enumeration 或物理模型接口。

验收命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  pytest -q -p no:cacheprovider tests

env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  python -m compileall -q src
```

完整 suite、compileall 和 focused numerical tests 必须通过；GPU/Slurm scientific run 仍是训练提交前的独立验证，不在本次 CPU helper refactor 的宣称范围内。
