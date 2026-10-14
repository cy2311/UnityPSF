import numpy as np

from globloc_raw_detector import detect_channel


def test_detect_channel_returns_one_peak_with_deterministic_score() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[16, 17] = 100.0
    image[16, 16] = 50.0
    image[15, 17] = 50.0

    detections = detect_channel(
        image,
        sigma_signal=0.8,
        sigma_background=2.0,
        threshold_sigma=3.0,
        min_distance=2,
        exclude_border=4,
    )

    assert detections["x_px"].tolist() == [17.0]
    assert detections["y_px"].tolist() == [16.0]
    assert detections["score"].shape == (1,)
    assert detections["score"][0] > 0.0


def test_detect_channel_rejects_non_image_input() -> None:
    try:
        detect_channel(np.zeros((2, 3, 4), dtype=np.float32))
    except ValueError as exc:
        assert "2D" in str(exc)
    else:
        raise AssertionError("expected a 2D input validation error")
