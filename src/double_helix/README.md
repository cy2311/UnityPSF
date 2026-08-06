# Double-Helix Microscope 1 FD Z Map

This module adapts the Neptune v0.4 local/global fitting pattern to the
DHPSFU Microscope 1 simulation. It does not use Neptune's single-lobe peak
detector or vector-PSF renderer. Instead, it uses the supplied `31x31x119`
calibration stack as an empirical PSF lookup table.

Official simulation calibration:

- lateral pixel size: 200 nm
- axial calibration step: 33.3 nm
- GT frame IDs: 1-based; TIFF pages: 0-based
- image mapping: `x_px=x_nm/200+15`, `y_px=y_nm/200+15`

Run the complete oracle-supervised fit from the `unity` project directory:

```bash
python -m double_helix.run_fd_zmap \
  --dataset-root /home/guest/Others/main/race/datasets/training_sets/double_helix/Simulated_datasets_Microscope1 \
  --output-dir output/double_helix_microscope1_zmap
```

The selected map is written as `fd_z_offset_nm`, with the convention:

```text
z_corrected = z_lut - fd_z_offset_nm
```

The publication describes a single global simulated PSF and does not describe
field-dependent aberrations. The pipeline therefore enables the selected map
only when both frame-held-out and spatial-block-held-out bootstrap confidence
intervals support improvement over the global calibration-LUT baseline. The
ungated candidate and its block-bootstrap uncertainty are always exported for
audit.
