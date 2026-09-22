# UnityPSF v2 Direct CUDA Zernike-map Benchmark

This folder is the reproducible benchmark bundle for the formal 10,000-frame
microtube simulation. The renderer uses the calibration-derived Zernike maps
at `results/microtube_right_benchmark_20260711/calibration/right_fd_zmap/export_nat_zmap/preferred_full_roi_zernike_maps_nm.npz`.
It does not use a PSF LUT.

## Contract

| Parameter | Value |
|---|---:|
| Frames | 10,000 |
| ROI | 64 x 64 pixels |
| Density | 1.0 / um2 |
| Pupil sampling | 64 x 64 |
| PSF patch | 21 x 21 |
| GPU | NVIDIA GeForce RTX 3090 |
| Zernike map | 13 x 1200 x 600, cropped to 13 x 64 x 64 |
| Emitters | 641,714 total |

The benchmark samples the 13 Zernike coefficients at each emitter's rounded
map coordinate, converts them to phase coefficients, and sends the coefficient
vector and emitter z position to the direct CUDA renderer. Fractional x/y
position is applied afterward by Fourier subpixel shifting.

## Timing result

| Component | Seconds |
|---|---:|
| Raw CUDA render kernel | 2.752725 |
| Fourier subpixel shift | 0.119238 |
| Patch postprocess | 0.012516 |
| Frame reduction | 0.009876 |
| Host copy | 0.029846 |
| Measured simulation wall | 3.033462 |

The detailed figure is `figures/timing_breakdown_detailed.png`. It contains
only the steady-state simulation path and component totals. One-time CUDA
build and TIFF write costs are intentionally excluded from this performance
figure.

## Files

- `benchmark_unitypsf_v2_cuda_zmap.py`: formal benchmark runner.
- `benchmark_10k.sbatch`: Slurm GPU job script.
- `plot_timing_breakdown.py`: reproducible bar-chart generator.
- `cuda_gemm_renderer.py`: Python CUDA extension wrapper.
- `cuda_extension/kernels.cu`: phase-factored CUDA kernels.
- `cuda_extension/bindings.cpp`: PyTorch extension bindings.
- `cuda_extension/setup.py`: standalone extension build definition.
- `results/summary.json`: benchmark metadata and timing summary.
- `results/timing_blocks.csv`: timing for every 64-frame block.
- `figures/timing_breakdown_detailed.png`: detailed timing chart.

The 10,000-frame TIFF is intentionally not copied into this GitHub bundle
because it is about 158 MB. The generated local file is:
`results/microtube/unitypsf_v2_cuda_zmap_density1_roi64_npupil64_10k_phasefactored/frames.tif`.

## Reproduce the figure

```bash
python cuda/unitypsf_v2_zmap_benchmark/plot_timing_breakdown.py \
  --summary cuda/unitypsf_v2_zmap_benchmark/results/summary.json \
  --blocks cuda/unitypsf_v2_zmap_benchmark/results/timing_blocks.csv \
  --output cuda/unitypsf_v2_zmap_benchmark/figures/timing_breakdown_detailed.png
```

## Re-run the GPU benchmark

```bash
sbatch cuda/unitypsf_v2_zmap_benchmark/benchmark_10k.sbatch
```

The job requires a CUDA-enabled Slurm node, PyTorch with CUDA support, and the
existing `neptune_v0.3` Python package in this repository.
