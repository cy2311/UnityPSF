from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from unity_psf.runtime.layout import ensure_run_layout

from .contract import PeakBootstrapConfig
from .pipeline import run_peak_bootstrap_pipeline


def load_peak_config(path: Path | str) -> PeakBootstrapConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return peak_config_from_mapping(payload)


def peak_config_from_mapping(payload: dict[str, Any]) -> PeakBootstrapConfig:
    values = dict(payload)
    frame_range = values.pop("frame_range")
    local_z_range = values.pop("local_z_range_nm", (-600.0, 600.0))
    initial_coefficients = values.pop("initial_zernike_coefficients_nm", {})
    return PeakBootstrapConfig(
        **values,
        frame_range=(int(frame_range[0]), int(frame_range[1])),
        local_z_range_nm=(float(local_z_range[0]), float(local_z_range[1])),
        initial_zernike_coefficients_nm={
            str(mode): float(value) for mode, value in dict(initial_coefficients).items()
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neptune v0.4 peak bootstrap pipeline.")
    parser.add_argument("--run-root", type=Path, required=True, help="Root directory for run outputs.")
    parser.add_argument("--run-name", required=True, help="Run name under --run-root.")
    parser.add_argument("--config-json", type=Path, required=True, help="PeakBootstrapConfig JSON file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = ensure_run_layout(args.run_root, args.run_name, stage_names=("peak",))
    config = load_peak_config(args.config_json)
    result = run_peak_bootstrap_pipeline(layout=layout, config=config)
    print(f"summary_path={result.artifacts.summary_path}")
    print(f"selected_emitters={result.summary.selected_emitters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
