# 配置目录

- `base/`：可被覆盖的基础流程配置。
- `modalities/`：单个 PSF 模态及其通道训练合同。
- `experiments/`：组合多个模态实例的联合实验配置。
- `calibration/`：PSF 校准和 NAT 初始化配置。
- `overrides/`：对基础流程的局部覆盖。

联合实验使用 `project://` 引用模态配置，路径始终相对于 Unity 项目根目录，
不随当前 YAML 所在层级变化。例如：

```yaml
config: project://configs/modalities/emitter_2d/emitter_2d_single_channel_300epoch.yaml
```
