# Neptune ZMap Main

`neptune/zmap_main` is the canonical microtubule raw TIFF to NAT zmap pipeline.

It isolates the working local-vector route that fixed the bad ring-like morphology seen in the older scalar forward path.

The microtubule pipeline now uses its own self-contained backed-up assets:

- `neptune/zmap_main/assets/microtube/pipeline_smlm_uni_decode_aligned_fixedscale_depthwise_mu20000_sigma1000.json`
- `neptune/zmap_main/assets/microtube/checkpoint_best_jaccard.pt`

## Canonical flow

`neptune/zmap_main` owns the NAT field model, local/global MLE optimizer,
diagnostics, and export contract. `neptune/peak` is the current standard
starter-free harvesting wrapper. `neptune/nat_maps` is output/example storage,
not a place for new solver logic. `neptune/neptune_core` is a legacy/training
mirror and should not define a separate in-situ zmap standard.

1. `neptune.peak` Gaussian peak harvest
2. `real_nat_diagnostics`
3. `export_nat_zmap`
4. promote standard artifacts into `record/situ/<sample>/zmap_standard/<side>/`

The current in-situ standard uses an order-1 shared NAT gamma field, Poisson
negative log-likelihood local/global alternating LM, latent per-emitter z,
`freeze_defocus_zero_gauge=true`, `freeze_initial_astig_standard=false`, and
`vectorfit_astig_anchor_mode=init_only`. The astig anchor initializes the
z/astig gauge but is released for global gamma fitting; `hard_fixed` is only a
diagnostic fallback.

## One-command entry

```bash
python -m neptune.zmap_main.pipeline_cli --side left
python -m neptune.zmap_main.pipeline_cli --side right
```

## Presets

- `left`
  - microtubule TIFF
  - depthwise microtubule checkpoint from `zmap_main/assets/microtube/`
  - full-frame harvest over first 100 frames
  - left ROI `x in [0, 600)`, `y in [0, 1200)`
  - order-2 NAT, 1000 emitters, local-vector renderer

- `right`
  - same microtubule TIFF
  - same checkpoint from `zmap_main/assets/microtube/`
  - right-half harvest over full 8000 frames
  - harvest crop `x in [600, 1200)`
  - right ROI `x in [600, 1200)`, `y in [0, 1200)`
  - right-half normalization `input_offset=257`, `input_scale=1800`
  - order-2 NAT, 1000 emitters, local-vector renderer

## Standard outputs

After promotion, training should consume:

- `param/zmap/microtube/left/alternating_full_roi_zernike_maps_nm.npz`
- `param/zmap/microtube/right/alternating_full_roi_zernike_maps_nm.npz`

Each side also keeps:

- `provisional_non_astig_rms_nm.npz`
- `real_nat_diagnostics_summary.json`
- `export_nat_zmap_summary.json`
- `standard_metadata.json`
