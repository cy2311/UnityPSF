from neptune_v03.localization.runtime_config import build_localization_runtime_config


def test_online_provider_propagates_vector_zemit0() -> None:
    config = {
        "optical": {
            "pixel_size_nm_x": 100.0,
            "pixel_size_nm_y": 100.0,
            "wavelength_nm": 660.0,
            "NA": 1.4,
        },
        "simulation": {
            "psf": {
                "psf_type": "vector",
                "vector": {
                    "zemit0": 2.5e-7,
                },
            },
        },
        "train": {
            "batch_size": 1,
            "online_generation": {
                "enabled": True,
            },
        },
    }

    runtime = build_localization_runtime_config(config, seed=7)

    assert runtime["batch_provider"]["params"]["zemit0"] == 2.5e-7
