# UnityPSF 冻结实验基线

以下实验资产用于验证项目整理前后的行为一致性。整理期间不得移动、重命名或删除这些目录。

## 正式联合训练基线

- 运行：`output/unitypsf/dual-modality-mixed-channel-300epoch-4545/`
- 日志：`logs/slurm/unitypsf_2gpu_300-4545.out`
- 联合 checkpoint SHA-256：`9d11461df9b7330724db252b6ac54708ae0fde67fd0a2eeb755664c92b61ccf2`
- 汇总指标 SHA-256：`8b3b154d9c64cdbb757ecf97a83bd0d04cc64e15b157379cf4b976aa191b2743`
- 联合训练明细 SHA-256：`d802202259a667f071f25d8782d8a0446a2c2827896394c272d0f880ca41bd03`

该运行已生成联合 checkpoint、训练汇总、最终指标、科研图和 HTML 报告。

## 保留的对照资产

- `output/unitypsf/dual-modality-mixed-channel-300epoch-4544/`
- `output/unitypsf/dual-modality-dual-channel-300epoch-4525/`
- `output/unitypsf/calibration/`
- `output/unitypsf/diagnostics/`

在完成独立安装、训练、推理和 checkpoint 加载验收之前，不处理上述资产。
