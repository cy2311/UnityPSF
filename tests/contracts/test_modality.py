from __future__ import annotations

import pytest
import torch

from unity_psf.contracts import (
    ChannelLayout,
    ExpertInstanceSpec,
    InputFrameSpec,
    MeasurementChannelSpec,
    ModalityBatch,
    PSFExpertOutput,
    PSFModality,
)


def test_input_frame_spec_records_positive_frame_count_and_order() -> None:
    spec = InputFrameSpec(input_frame_channels=3, frame_order="oldest_to_newest")

    assert spec.input_frame_channels == 3
    assert spec.frame_order == "oldest_to_newest"
    assert spec.channels == 3

    with pytest.raises(ValueError, match="positive"):
        InputFrameSpec(input_frame_channels=0)
    with pytest.raises(ValueError, match="frame_order"):
        InputFrameSpec(input_frame_channels=1, frame_order=" ")
    with pytest.raises(ValueError, match="length"):
        InputFrameSpec(input_frame_channels=3, frame_order=("previous", "current"))
    assert InputFrameSpec(channels=3, order="temporal").input_frame_channels == 3
    assert InputFrameSpec().input_frame_channels == 3


def test_input_frame_spec_accepts_new_config_mapping() -> None:
    spec = InputFrameSpec.from_value(
        {"input_frame_channels": 3, "frame_order": "temporal"}
    )

    assert spec == InputFrameSpec(input_frame_channels=3, frame_order="temporal")
    with pytest.raises(ValueError, match="agree"):
        InputFrameSpec.from_value({"input_frame_channels": 3, "channels": 2})
    with pytest.raises(ValueError, match="agree"):
        InputFrameSpec.from_value(
            {"input_frame_channels": 1, "frame_order": "temporal", "order": "reverse"}
        )


def test_measurement_channel_spec_validates_crop_and_keeps_physical_references() -> None:
    spec = MeasurementChannelSpec(
        channel_id="left",
        crop=(0, 0, 128, 128),
        anchor_profile="astigmatism_660nm_99nm",
        calibration_ref="calibration/left.yaml",
    )

    assert spec.channel_id == "left"
    assert spec.crop == (0, 0, 128, 128)
    assert spec.anchor_profile == "astigmatism_660nm_99nm"
    assert spec.calibration_ref == "calibration/left.yaml"

    with pytest.raises(ValueError, match="crop"):
        MeasurementChannelSpec(channel_id="left", crop=(-1, 0, 128, 128))
    with pytest.raises(ValueError, match="crop"):
        MeasurementChannelSpec(channel_id="left", crop=(0, 0, 0, 128))
    with pytest.raises(ValueError, match="channel_id"):
        MeasurementChannelSpec(channel_id=" ")


def test_channel_layout_supports_single_dual_and_custom_measurement_channels() -> None:
    left = MeasurementChannelSpec("left", crop=(0, 0, 128, 128))
    right = MeasurementChannelSpec("right", crop=(128, 0, 128, 128))
    dual = ChannelLayout(channels=(left, right), frame_size=(128, 256))
    assert ChannelLayout(measurement_channels=(left, right), frame_size=(128, 256)) == dual

    assert dual.channel_ids == ("left", "right")
    assert dual.measurement_channels == dual.channels
    assert dual["left"] is left
    assert dual["right"] is right
    assert dual.input_instances == 2

    single = ChannelLayout.from_value({"channels": [{"id": "main"}]})
    custom = ChannelLayout.from_value(
        [{"id": "aux_b"}, {"id": "aux_a"}]
    )
    assert single.channel_ids == ("main",)
    assert custom.channel_ids == ("aux_b", "aux_a")
    with pytest.raises(ValueError, match="agree"):
        ChannelLayout.from_value({"channels": [{"id": "left"}], "measurement_channels": [{"id": "right"}]})


def test_channel_layout_rejects_empty_duplicate_and_out_of_bounds_channels() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ChannelLayout(channels=())
    with pytest.raises(ValueError, match="unique"):
        ChannelLayout(
            channels=(MeasurementChannelSpec("left"), MeasurementChannelSpec("left"))
        )
    with pytest.raises(TypeError, match="sequence"):
        ChannelLayout(channels={MeasurementChannelSpec("left")})
    with pytest.raises(ValueError, match="frame_size"):
        ChannelLayout(channels=(MeasurementChannelSpec("left", crop=(0, 0, 8, 8)),))
    with pytest.raises(ValueError, match="bounds"):
        ChannelLayout(
            channels=(MeasurementChannelSpec("right", crop=(128, 0, 128, 128)),),
            frame_size=(128, 128),
        )


def test_expert_instance_spec_is_one_canonical_expert_bound_to_one_channel() -> None:
    spec = ExpertInstanceSpec(
        expert_type="astig",
        instance_id="astig-left",
        channel_id="left",
        prototype_ref="checkpoints/astigmatism_base.ckpt",
    )

    assert spec.expert_type is PSFModality.ASTIGMATISM
    assert spec.instance_id == "astig-left"
    assert spec.channel_id == "left"
    assert spec.prototype_ref == "checkpoints/astigmatism_base.ckpt"

    with pytest.raises(ValueError, match="instance_id"):
        ExpertInstanceSpec("astig", " ", "left")
    with pytest.raises(ValueError, match="channel_id"):
        ExpertInstanceSpec("astig", "astig-left", " ")


def test_psf_modality_does_not_define_measurement_channel_count() -> None:
    assert not hasattr(PSFModality.ASTIGMATISM, "required_channels")
    assert PSFModality.parse("astig") is PSFModality.ASTIGMATISM


def test_existing_batch_and_output_contracts_remain_usable() -> None:
    batch = ModalityBatch.from_value("double_helix", batch_size=2)
    assert batch.values == (PSFModality.DOUBLE_HELIX, PSFModality.DOUBLE_HELIX)

    output = PSFExpertOutput(
        detection_logits=torch.zeros(2, 4, 4),
        xy_offset=torch.zeros(2, 2, 4, 4),
        z=torch.zeros(2, 4, 4),
        photons=torch.ones(2, 4, 4),
    )
    assert output.validate(batch_size=2) is output


def test_psf_moe_default_accepts_three_temporal_input_frames() -> None:
    from unity_psf.models.psf_moe import PSFMoE
    from unity_psf.models.psf_moe.base import SharedPSFStem

    model = PSFMoE(feature_channels=8).eval()
    assert SharedPSFStem().in_channels == 3
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 8, 8), PSFModality.DOUBLE_HELIX)
    assert output.detection_logits.shape == (1, 8, 8)
