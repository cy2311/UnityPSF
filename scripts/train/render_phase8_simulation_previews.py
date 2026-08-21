"""Render one formal provider batch per modality for Phase 8 visual inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unity_psf.config import load_config
from unity_psf.localization.runtime import build_localization_runtime_config
from unity_psf.training.runtime import build_runtime_batch_provider


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output/unitypsf/phase8-parity-20260821-r5/diagnostics/phase8_simulation_preview"
CONFIGS = {
    "double_helix": PROJECT_ROOT / "configs/modalities/double_helix/double_helix_raw_tiff_300epoch.yaml",
    "emitter_2d": PROJECT_ROOT / "configs/modalities/emitter_2d/emitter_2d_dual_channel_300epoch.yaml",
    "astigmatism": PROJECT_ROOT / "configs/modalities/astigmatism/astigmatism_dual_channel_300epoch.yaml",
}


def _single_channel_config(config: dict[str, object]) -> dict[str, object]:
    train = dict(config["train"])
    layout = dict(train["channel_layout"])
    layout["channels"] = [dict(layout["channels"][0])]
    train["channel_layout"] = layout
    expert = dict(train["expert"])
    expert["channel_id"] = "left"
    train["expert"] = expert
    return {**config, "train": train}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for modality, config_path in CONFIGS.items():
        config = load_config(config_path)
        if modality != "double_helix":
            config = _single_channel_config(config)
        runtime = build_localization_runtime_config(config, config_base_dir=config_path.parent, seed=4311)
        params = {**dict(runtime["batch_provider"]["params"]), "batch_size": 1, "steps_per_epoch": 1}
        provider_config = {**runtime["batch_provider"], "params": params}
        batch = next(iter(build_runtime_batch_provider({**runtime, "batch_provider": provider_config})(1)))
        inputs = batch.inputs.model_input if hasattr(batch.inputs, "model_input") else batch.inputs
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        image = inputs.detach().cpu().numpy()[0]
        frame = image[image.shape[0] // 2]
        figure, axis = plt.subplots(figsize=(5, 5), constrained_layout=True)
        rendered = axis.imshow(frame, cmap="magma", interpolation="nearest")
        axis.set_title(f"{modality} simulation | seed=4311 | shape={tuple(image.shape)}")
        axis.set_axis_off()
        figure.colorbar(rendered, ax=axis, fraction=0.046, pad=0.04)
        path = OUTPUT_DIR / f"{modality}_simulation.png"
        figure.savefig(path, dpi=180, facecolor="white")
        plt.close(figure)
        print({"modality": modality, "path": str(path), "shape": tuple(image.shape)}, flush=True)


if __name__ == "__main__":
    main()
