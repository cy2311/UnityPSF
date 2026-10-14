from __future__ import annotations

from pathlib import Path

from unity_psf.config import load_config
from unity_psf.localization import build_localization_model_registry, build_localization_runtime_config
from unity_psf.models.psf_moe.experts.emitter_2d import Emitter2DExpert
from unity_psf.runtime import ensure_run_layout
from unity_psf.training import build_trainer_runtime, train_epochs


CONFIG_PATH = Path(__file__).parents[2] / "configs" / "modalities" / "emitter_2d" / "emitter_2d_single_channel_smoke.yaml"


def test_emitter_2d_runtime_uses_complete_expert_film_and_z_mask(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)

    runtime_config = build_localization_runtime_config(
        config,
        config_base_dir=CONFIG_PATH.parent,
        seed=19,
    )

    assert runtime_config["model"]["name"] == "emitter_2d_expert"
    assert runtime_config["model"]["params"]["disabled_attr"] == [3]
    assert runtime_config["loss"]["name"] == "active_smlm_gmm_loss"
    assert runtime_config["loss"]["params"]["disable_attr"] == 3
    assert runtime_config["batch_provider"]["params"]["conditioning_mode"] == "film"
    assert runtime_config["batch_provider"]["params"]["condition_fields"] == ("field_x", "field_y")
    assert runtime_config["expert_instance"] == {
        "expert_type": "emitter_2d",
        "instance_id": "main",
        "channel_id": "main",
        "prototype_ref": None,
    }

    runtime = build_trainer_runtime(
        runtime_config,
        layout=ensure_run_layout(tmp_path, "emitter-smoke"),
        model_registry=build_localization_model_registry(),
    )
    assert isinstance(runtime.model, Emitter2DExpert)
    batch = next(iter(runtime.batch_provider(1)))
    assert isinstance(batch.inputs.model_input, tuple)
    assert batch.inputs.model_input[1].shape == (1, 2)

    results = train_epochs(
        model=runtime.model,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        batch_provider=runtime.batch_provider,
        layout=runtime.layout,
        config=runtime.config,
        loss_fn=runtime.loss_fn,
    )

    assert results[0].step_count == 1
    assert results[0].checkpoint_path.is_file()
