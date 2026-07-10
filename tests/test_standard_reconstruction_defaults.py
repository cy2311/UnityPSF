import importlib.util
import sys
from pathlib import Path

from neptune_v03.infer_recon.filter import apply_filter_recon
from neptune_v03.infer_recon.recon import render_standard, render_subpixel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_formal_infer_module():
    path = PROJECT_ROOT / "scripts/infer/run_3371_full8000_infer_filter_recon.py"
    spec = importlib.util.spec_from_file_location("formal_infer_defaults", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_dual_renderer_module():
    path = PROJECT_ROOT / "scripts/infer/render_union_raw_ratio_bicolor.py"
    spec = importlib.util.spec_from_file_location("dual_renderer_defaults", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_python_entry_points_default_to_smap_style_reconstruction(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["render_standard", "--infer-dir", str(tmp_path), "--output-dir", str(tmp_path / "render")],
    )
    standard_args = render_standard._parse_args()
    assert standard_args.prob_threshold == 0.7
    assert standard_args.render_pixel_nm == 20.0
    assert standard_args.spot_radius_nm == 28.0
    assert standard_args.radius_mode == "fixed"
    assert standard_args.render_weight == "count"
    assert standard_args.display_mode == "quantile"
    assert standard_args.display_imax_min == -2.5228787452803374
    assert standard_args.gamma == 1.0

    monkeypatch.setattr(
        sys,
        "argv",
        ["render_subpixel", "--predictions", "p.h5", "--output-dir", "out", "--width-px", "1", "--height-px", "1", "--camera-pixel-nm-x", "100", "--camera-pixel-nm-y", "100"],
    )
    subpixel_args = render_subpixel._parse_args()
    assert subpixel_args.prob_threshold == 0.7
    assert subpixel_args.render_pixel_nm == 20.0
    assert subpixel_args.spot_radius_nm == 28.0
    assert subpixel_args.radius_mode == "fixed"
    assert subpixel_args.display_mode == "quantile"
    assert subpixel_args.display_imax_min == -2.5228787452803374
    assert subpixel_args.gamma == 1.0

    monkeypatch.setattr(
        sys,
        "argv",
        ["apply_filter_recon", "--infer-dir", str(tmp_path), "--output-dir", str(tmp_path / "filter")],
    )
    filter_args = apply_filter_recon._parse_args()
    assert filter_args.prob_min == 0.7
    assert filter_args.render_pixel_nm == 20.0
    assert filter_args.spot_radius_nm == 28.0
    assert filter_args.radius_mode == "fixed"
    assert filter_args.render_weight == "count"
    assert filter_args.display_mode == "quantile"
    assert filter_args.display_imax_min == -2.5228787452803374


def test_formal_infer_and_standard_wrappers_use_same_defaults(monkeypatch, tmp_path) -> None:
    module = _load_formal_infer_module()
    monkeypatch.setattr(sys, "argv", ["formal_infer", "--output-dir", str(tmp_path)])
    args = module.parse_args()
    assert args.filter_prob_min == 0.7
    assert args.render_pixel_nm == 20.0
    assert args.spot_radius_nm == 28.0
    assert args.radius_mode == "fixed"
    assert args.degrid is True
    assert args.rcc_drift is True
    assert args.rcc_frame_block_size == 500

    sbatch = (PROJECT_ROOT / "scripts/infer/standard_channel_infer_recon.sbatch").read_text(encoding="utf-8")
    assert "${FILTER_PROB_MIN:-0.70}" in sbatch
    assert "${RENDER_PIXEL_NM:-20.0}" in sbatch
    assert "${SPOT_RADIUS_NM:-28.0}" in sbatch

    pipeline = (PROJECT_ROOT / "run_standard_pipeline.sh").read_text(encoding="utf-8")
    assert 'FILTER_PROB_MIN="${FILTER_PROB_MIN:-0.70}"' in pipeline
    assert 'RENDER_PIXEL_NM="${RENDER_PIXEL_NM:-20.0}"' in pipeline
    assert 'SPOT_RADIUS_NM="${SPOT_RADIUS_NM:-28.0}"' in pipeline
    assert 'RADIUS_MODE="${RADIUS_MODE:-fixed}"' in pipeline
    assert 'LEFT_PREDICTIONS="$LEFT_OUTPUT_DIR/left/infer/predictions_degrid_rcc_corrected.h5"' in pipeline
    assert 'RIGHT_PREDICTIONS="$RIGHT_OUTPUT_DIR/right/infer/predictions_degrid_rcc_corrected.h5"' in pipeline

    web = (PROJECT_ROOT / "scripts/gui/submit_training_web.py").read_text(encoding="utf-8")
    assert 'form.get("filter_prob_min", "0.70")' in web


def test_dual_and_compatibility_entries_do_not_restore_old_defaults(monkeypatch, tmp_path) -> None:
    module = _load_dual_renderer_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dual_renderer",
            "--left-predictions",
            "left.h5",
            "--right-predictions",
            "right.h5",
            "--sample-tiff",
            "sample.tif",
            "--output-dir",
            str(tmp_path),
        ],
    )
    args = module.parse_args()
    assert args.render_pixel_nm == 20.0
    assert args.spot_radius_nm == 28.0
    assert args.radius_mode == "fixed"
    assert args.render_weight == "count"
    assert args.display_mode == "quantile"
    assert args.display_imax_min == -2.5228787452803374
    assert args.gamma == 1.0

    for relative_path in (
        "scripts/infer/run_3371_full8000_infer_filter_recon.sbatch",
        "scripts/infer/run_3371_left_root_infer_recon.sbatch",
    ):
        wrapper = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "${FILTER_PROB_MIN:-0.70}" in wrapper
