from __future__ import annotations

from pathlib import Path

import pytest

from unity_psf.contracts.modality import ChannelLayout, MeasurementChannelSpec, PSFModality
from unity_psf.training.multichannel import (
    ChannelRunStatus,
    MultichannelTrainingPlan,
    build_multichannel_plan,
    build_multimodal_training_plans,
)


def _layout() -> ChannelLayout:
    return ChannelLayout(
        channels=(
            MeasurementChannelSpec("left", crop=(0, 0, 16, 16), anchor_profile="anchor"),
            MeasurementChannelSpec("right", crop=(16, 0, 16, 16), anchor_profile="anchor"),
        ),
        frame_size=(16, 32),
    )


def test_plan_creates_independent_left_right_specs_for_each_modality(tmp_path: Path) -> None:
    for modality in PSFModality:
        plan = build_multichannel_plan(
            channel_layout=_layout(),
            expert_type=modality,
            run_root=tmp_path,
            run_name=modality.value,
            prototype_ref=f"checkpoints/{modality.value}_base.ckpt",
            seed=11,
        )

        assert isinstance(plan, MultichannelTrainingPlan)
        assert plan.expert_type is modality
        assert plan.channel_ids == ("left", "right")
        assert plan.channel_specs[0].crop == (0, 0, 16, 16)
        assert plan.channel_specs[1].crop == (16, 0, 16, 16)
        assert plan.channel_specs[0].seed == plan.channel_specs[1].seed == 11
        assert plan.channel_specs[0].run_dir != plan.channel_specs[1].run_dir
        assert plan.channel_specs[0].run_dir == tmp_path / modality.value / "channels" / "left"
        assert plan.channel_specs[1].run_dir == tmp_path / modality.value / "channels" / "right"
        assert plan.channel_specs[0].instance_id == "left"
        assert plan.channel_specs[1].instance_id == "right"


def test_plan_can_override_channel_seeds_and_prototypes(tmp_path: Path) -> None:
    plan = build_multichannel_plan(
        channel_layout=_layout(),
        expert_type="double_helix",
        run_root=tmp_path,
        channel_seeds={"left": 3, "right": 4},
        prototype_ref={"left": "left.ckpt", "right": "right.ckpt"},
    )

    assert [item.seed for item in plan.channel_specs] == [3, 4]
    assert [item.prototype_ref for item in plan.channel_specs] == ["left.ckpt", "right.ckpt"]


def test_local_execution_is_sequential_and_preserves_completed_sibling(tmp_path: Path) -> None:
    plan = build_multichannel_plan(
        channel_layout=_layout(),
        expert_type="astigmatism",
        run_root=tmp_path,
        run_name="astig",
    )
    calls: list[str] = []

    def runner(spec):
        calls.append(spec.channel_id)
        if spec.channel_id == "left":
            return 0
        raise RuntimeError("right failed")

    result = plan.execute_local(runner)

    assert calls == ["left", "right"]
    assert result.status == ChannelRunStatus.FAILED
    assert result.channels["left"].status == ChannelRunStatus.COMPLETED
    assert result.channels["right"].status == ChannelRunStatus.FAILED
    assert result.channels["left"].run_dir.exists()
    assert result.channels["right"].error == "right failed"
    assert (plan.parent_run_dir / "metadata" / "multichannel_manifest.json").is_file()


def test_slurm_scripts_have_separate_channel_outputs_and_commands(tmp_path: Path) -> None:
    plan = build_multichannel_plan(
        channel_layout=_layout(),
        expert_type="emitter_2d",
        run_root=tmp_path,
        run_name="emitter",
        entrypoint="unity_psf.training.run_localization",
        config_path=tmp_path / "config.yaml",
    )

    scripts = plan.write_slurm_scripts(tmp_path / "submitted")

    assert tuple(item.channel_id for item in scripts) == ("left", "right")
    left_text = scripts[0].script_path.read_text(encoding="utf-8")
    right_text = scripts[1].script_path.read_text(encoding="utf-8")
    assert "--run-name left" in left_text
    assert "--run-name right" in right_text
    assert str(tmp_path / "emitter" / "channels" / "left") in left_text
    assert str(tmp_path / "emitter" / "channels" / "right") in right_text
    assert scripts[0] != scripts[1]


def test_multimodal_builder_keeps_one_plan_per_expert_type(tmp_path: Path) -> None:
    plans = build_multimodal_training_plans(
        channel_layouts={
            "astigmatism": _layout(),
            "double_helix": _layout(),
            "emitter_2d": _layout(),
        },
        run_root=tmp_path,
        seed=5,
    )

    assert tuple(plan.expert_type for plan in plans) == (
        PSFModality.ASTIGMATISM,
        PSFModality.DOUBLE_HELIX,
        PSFModality.EMITTER_2D,
    )
    assert all(plan.channel_ids == ("left", "right") for plan in plans)
    assert len({plan.parent_run_dir for plan in plans}) == 3


def test_plan_rejects_seed_for_unknown_channel() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        build_multichannel_plan(
            channel_layout=_layout(),
            expert_type="astigmatism",
            run_root="/tmp",
            channel_seeds={"center": 1},
        )


def test_plan_rejects_prototype_for_unknown_channel(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown channel prototype"):
        build_multichannel_plan(
            channel_layout=_layout(),
            expert_type="astigmatism",
            run_root=tmp_path,
            prototype_ref={"center": "center.ckpt"},
        )


def test_multimodal_builder_namespaced_run_names_and_channel_overrides(tmp_path: Path) -> None:
    plans = build_multimodal_training_plans(
        channel_layouts={"astigmatism": _layout(), "double_helix": _layout()},
        run_root=tmp_path,
        run_name="experiment",
        channel_seeds={"astigmatism": {"left": 3, "right": 4}},
        prototype_refs={"double_helix": {"left": "dh-left.ckpt", "right": "dh-right.ckpt"}},
    )

    assert [plan.parent_run_name for plan in plans] == ["experiment-astigmatism", "experiment-double_helix"]
    assert [spec.seed for spec in plans[0].channel_specs] == [3, 4]
    assert [spec.prototype_ref for spec in plans[1].channel_specs] == ["dh-left.ckpt", "dh-right.ckpt"]
