from __future__ import annotations

import base64
import html
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import tifffile

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - preview falls back to metadata-only mode.
    Image = None
    ImageDraw = None


ROOT = Path("/home/guest/Others/main/race")
UNITY_DIR = ROOT / "unity"
TRAINING_SETS_DIR = ROOT / "datasets/training_sets"
DEFAULT_RAW_TIFF = ROOT / "neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif"
LEGACY_ENTRY_ROOT = UNITY_DIR / "scripts/archive/neptune_standard"
TRAIN_SCRIPT = LEGACY_ENTRY_ROOT / "train_standard_3367_hqzmap.sh"
PIPELINE_SCRIPT = LEGACY_ENTRY_ROOT / "run_standard_pipeline.sh"
SUPPORTED_ZMAP_SAMPLES = ("microtube", "paint", "ncp", "dynamin", "membrane")
WORKFLOW_MODES = ("train_infer_recon", "train_only")
CHANNEL_MODES = ("dual", "left", "right")
KNOWN_SAMPLE_PATHS = {
    "microtube default": DEFAULT_RAW_TIFF,
    "dynamin_3d_5_1": TRAINING_SETS_DIR / "dynamin_3d_5_1",
    "membrane_3d_4_1": TRAINING_SETS_DIR / "membrane_3d_4_1",
    "ncp_3d_500mw_1_1": TRAINING_SETS_DIR / "ncp_3d_500mw_1_1",
    "paint_3d_lh1": TRAINING_SETS_DIR / "paint_3d_lh1",
}


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class SubmissionConfig:
    raw_tiff: str
    workflow_mode: str
    channel_mode: str
    crop_preset: str
    zmap_sample_kind: str
    left_crop: Crop
    right_crop: Crop
    epochs: int
    batch_size: int
    steps_per_epoch: int
    roi_size: int
    psf_size: int
    roi_stride: int
    start_epoch: int
    update_interval_epochs: int
    target_projected_emitters: int
    hq_max_emitters: int
    hq_alternating_rounds: int
    hq_spatial_balance_grid_px: str
    filter_prob_min: str
    run_tag: str


def _resolve_tiff_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidates: list[Path] = []
        for pattern in ("*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff"):
            candidates.extend(sorted(path.glob(pattern)))
        if candidates:
            return candidates[0].resolve()
    raise FileNotFoundError(f"Could not find TIFF file from: {path}")


def infer_zmap_sample_kind_from_path(path: Path) -> str | None:
    text = str(path).lower()
    if "dynamin" in text:
        return "dynamin"
    if "membrane" in text:
        return "membrane"
    if "ncp" in text:
        return "ncp"
    if "paint" in text or "lh1" in text:
        return "paint"
    if "microtube" in text or "spool_800mw" in text or "3d_7_1" in text:
        return "microtube"
    return None


def _read_tiff_info(path: Path) -> tuple[tuple[int, int, int], np.ndarray]:
    with tifffile.TiffFile(str(path)) as tif:
        if not tif.series:
            raise ValueError(f"TIFF has no image series: {path}")
        series = tif.series[0]
        shape = tuple(int(v) for v in series.shape)
        if len(shape) == 2:
            frames, height, width = 1, shape[0], shape[1]
        elif len(shape) >= 3:
            frames, height, width = int(np.prod(shape[:-2])), shape[-2], shape[-1]
        else:
            raise ValueError(f"Unsupported TIFF shape: {shape}")
        try:
            first = np.asarray(series.asarray(key=0))
        except Exception:
            first = np.asarray(tif.pages[0].asarray())
    while first.ndim > 2:
        first = first[0]
    return (frames, height, width), np.asarray(first, dtype=np.float32)


def _preview_png_data_uri(frame: np.ndarray, left: Crop, right: Crop, *, max_side: int = 900) -> str | None:
    if Image is None or ImageDraw is None:
        return None
    lo, hi = np.percentile(frame, [0.2, 99.8])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
    img = Image.fromarray((scaled * 255.0).astype(np.uint8), mode="L").convert("RGB")
    scale = min(float(max_side) / max(img.width, img.height), 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    draw = ImageDraw.Draw(img)

    def rect(crop: Crop, color: str, label: str) -> None:
        x0 = int(crop.left * scale)
        y0 = int(crop.top * scale)
        x1 = int((crop.left + crop.width) * scale)
        y1 = int((crop.top + crop.height) * scale)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 5, y0 + 5), label, fill=color)

    rect(left, "#1c8f49", "left")
    rect(right, "#b91c1c", "right")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_int(form: dict[str, str], key: str, minimum: int = 0) -> int:
    try:
        value = int(str(form.get(key, "")).strip())
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _crop_from_form(form: dict[str, str], prefix: str) -> Crop:
    return Crop(
        left=_safe_int(form, f"{prefix}_left"),
        top=_safe_int(form, f"{prefix}_top"),
        width=_safe_int(form, f"{prefix}_width", 1),
        height=_safe_int(form, f"{prefix}_height", 1),
    )


def _validate_crop(crop: Crop, *, frame_width: int, frame_height: int, label: str) -> None:
    if crop.left + crop.width > frame_width or crop.top + crop.height > frame_height:
        raise ValueError(
            f"{label} crop exceeds TIFF bounds: crop=({crop.left},{crop.top},{crop.width},{crop.height}), "
            f"frame=({frame_width},{frame_height})"
        )


def _default_crops(shape: tuple[int, int, int], preset: str) -> tuple[Crop, Crop]:
    _, height, width = shape
    if preset == "center400":
        return (
            Crop(max(0, min(100, width - 400)), max(0, min(400, height - 400)), 400, 400),
            Crop(max(0, min(700, width - 400)), max(0, min(400, height - 400)), 400, 400),
        )
    half = width // 2
    return Crop(0, 0, half, height), Crop(half, 0, width - half, height)


def _default_run_tag(config: SubmissionConfig) -> str:
    sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.zmap_sample_kind.strip() or "sample")
    return (
        f"gui_{sample}_{config.crop_preset}_hqzmap_emit{config.hq_max_emitters}_"
        f"roi{config.roi_size}_psf{config.psf_size}_e{config.epochs}_"
        f"start{config.start_epoch}_int{config.update_interval_epochs}_"
        f"bs{config.batch_size}_s{config.steps_per_epoch}"
    )


def _format_export(config: SubmissionConfig) -> dict[str, str]:
    valid_roi_size = max(1, int(config.roi_size) - 16)
    env = {
        "NEPTUNE_V04_RAW_TIFF_PATH": config.raw_tiff,
        "SAMPLE_TIFF": config.raw_tiff,
        "PIPELINE_MODE": "train_infer" if config.workflow_mode == "train_infer_recon" else "train_infer",
        "CHANNEL_MODE": config.channel_mode,
        "ZMAP_SAMPLE_KIND": config.zmap_sample_kind,
        "EPOCHS": str(config.epochs),
        "BATCH_SIZE": str(config.batch_size),
        "STEPS_PER_EPOCH": str(config.steps_per_epoch),
        "ROI_SIZE": str(config.roi_size),
        "VALID_ROI_SIZE": str(valid_roi_size),
        "PSF_SIZE": str(config.psf_size),
        "ROI_STRIDE": str(config.roi_stride),
        "START_EPOCH": str(config.start_epoch),
        "UPDATE_INTERVAL_EPOCHS": str(config.update_interval_epochs),
        "TARGET_PROJECTED_EMITTERS": str(config.target_projected_emitters),
        "HQ_MAX_EMITTERS": str(config.hq_max_emitters),
        "HQ_ALTERNATING_ROUNDS": str(config.hq_alternating_rounds),
        "LEFT_DOMAIN_CROP_LEFT": str(config.left_crop.left),
        "LEFT_DOMAIN_CROP_TOP": str(config.left_crop.top),
        "LEFT_DOMAIN_CROP_WIDTH": str(config.left_crop.width),
        "LEFT_DOMAIN_CROP_HEIGHT": str(config.left_crop.height),
        "RIGHT_DOMAIN_CROP_LEFT": str(config.right_crop.left),
        "RIGHT_DOMAIN_CROP_TOP": str(config.right_crop.top),
        "RIGHT_DOMAIN_CROP_WIDTH": str(config.right_crop.width),
        "RIGHT_DOMAIN_CROP_HEIGHT": str(config.right_crop.height),
        "HQ_LEFT_CROP_X0": str(config.left_crop.left),
        "HQ_LEFT_CROP_X1": str(config.left_crop.left + config.left_crop.width),
        "HQ_RIGHT_CROP_X0": str(config.right_crop.left),
        "HQ_RIGHT_CROP_X1": str(config.right_crop.left + config.right_crop.width),
        "HQ_CROP_Y0": str(config.left_crop.top),
        "HQ_CROP_Y1": str(config.left_crop.top + config.left_crop.height),
        "HQ_LEFT_ROI_X_MIN_PX": str(config.left_crop.left),
        "HQ_LEFT_ROI_X_MAX_PX": str(config.left_crop.left + config.left_crop.width),
        "HQ_RIGHT_ROI_X_MIN_PX": str(config.right_crop.left),
        "HQ_RIGHT_ROI_X_MAX_PX": str(config.right_crop.left + config.right_crop.width),
        "HQ_ROI_Y_MIN_PX": str(config.left_crop.top),
        "HQ_ROI_Y_MAX_PX": str(config.left_crop.top + config.left_crop.height),
        "FILTER_PROB_MIN": config.filter_prob_min,
        "DISPLAY_MODE": "quantile",
        "DISPLAY_IMAX_MIN": "-2.5228787452803374",
        "RCC_DRIFT_ENABLED": "true",
        "RCC_FRAME_BLOCK_SIZE": "500",
        "RCC_PIXEL_NM": "50.0",
        "RUN_TAG": config.run_tag,
    }
    if config.hq_spatial_balance_grid_px.strip():
        env["HQ_SPATIAL_BALANCE_GRID_PX"] = config.hq_spatial_balance_grid_px.strip()
    return env


def _submit_command(config: SubmissionConfig) -> tuple[list[str], dict[str, str] | None]:
    env = _format_export(config)
    for key, value in env.items():
        if "," in value:
            raise ValueError(f"Environment value for {key} contains comma, which sbatch --export cannot encode safely: {value}")
    if config.workflow_mode == "train_infer_recon":
        return ["bash", str(PIPELINE_SCRIPT)], env
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    return ["sbatch", f"--export={export_arg}", str(TRAIN_SCRIPT)], None


def _collect_config(form: dict[str, str]) -> SubmissionConfig:
    raw_tiff = _resolve_tiff_path(form.get("raw_tiff", ""))
    shape, _ = _read_tiff_info(raw_tiff)
    _, frame_height, frame_width = shape
    left_crop = _crop_from_form(form, "left")
    right_crop = _crop_from_form(form, "right")
    _validate_crop(left_crop, frame_width=frame_width, frame_height=frame_height, label="left")
    _validate_crop(right_crop, frame_width=frame_width, frame_height=frame_height, label="right")
    if left_crop.top != right_crop.top or left_crop.height != right_crop.height:
        raise ValueError("Current HQ bootstrap requires left/right crops with matching y and height.")
    zmap_sample_kind = form.get("zmap_sample_kind", "").strip().lower()
    if zmap_sample_kind not in SUPPORTED_ZMAP_SAMPLES:
        raise ValueError(f"Unsupported sample kind: {zmap_sample_kind}")
    workflow_mode = form.get("workflow_mode", "train_infer_recon").strip()
    if workflow_mode not in WORKFLOW_MODES:
        raise ValueError(f"Unsupported workflow mode: {workflow_mode}")
    channel_mode = form.get("channel_mode", "dual").strip()
    if channel_mode not in CHANNEL_MODES:
        raise ValueError(f"Unsupported channel mode: {channel_mode}")
    inferred = infer_zmap_sample_kind_from_path(raw_tiff)
    if inferred is not None and inferred != zmap_sample_kind:
        raise ValueError(f"Path looks like {inferred!r}, but sample kind is {zmap_sample_kind!r}.")
    config = SubmissionConfig(
        raw_tiff=str(raw_tiff),
        workflow_mode=workflow_mode,
        channel_mode=channel_mode,
        crop_preset=form.get("crop_preset", "full"),
        zmap_sample_kind=zmap_sample_kind,
        left_crop=left_crop,
        right_crop=right_crop,
        epochs=_safe_int(form, "epochs", 1),
        batch_size=_safe_int(form, "batch_size", 1),
        steps_per_epoch=_safe_int(form, "steps_per_epoch", 1),
        roi_size=_safe_int(form, "roi_size", 1),
        psf_size=_safe_int(form, "psf_size", 1),
        roi_stride=_safe_int(form, "roi_stride", 1),
        start_epoch=_safe_int(form, "start_epoch", 0),
        update_interval_epochs=_safe_int(form, "update_interval_epochs", 1),
        target_projected_emitters=_safe_int(form, "target_projected_emitters", 1),
        hq_max_emitters=_safe_int(form, "hq_max_emitters", 1),
        hq_alternating_rounds=_safe_int(form, "hq_alternating_rounds", 1),
        hq_spatial_balance_grid_px=form.get("hq_spatial_balance_grid_px", "").strip(),
        filter_prob_min=form.get("filter_prob_min", "0.70").strip() or "0.70",
        run_tag=form.get("run_tag", "").strip(),
    )
    if not config.run_tag:
        config = replace(config, run_tag=_default_run_tag(config))
    return config


def _flatten_form(post_body: bytes) -> dict[str, str]:
    parsed = parse_qs(post_body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in parsed.items()}


def _html_page(message: str = "", form: dict[str, str] | None = None) -> str:
    form = dict(form or {})
    sample_key = form.get("known_sample", "dynamin_3d_5_1")
    raw_path = form.get("raw_tiff") or str(KNOWN_SAMPLE_PATHS.get(sample_key, KNOWN_SAMPLE_PATHS["dynamin_3d_5_1"]))
    try:
        raw_tiff = _resolve_tiff_path(raw_path)
        shape, first_frame = _read_tiff_info(raw_tiff)
        inferred = infer_zmap_sample_kind_from_path(raw_tiff) or "microtube"
        crop_preset = form.get("crop_preset", "center400" if inferred == "ncp" else "full")
        default_left, default_right = _default_crops(shape, crop_preset)
        left = Crop(
            int(form.get("left_left", default_left.left)),
            int(form.get("left_top", default_left.top)),
            int(form.get("left_width", default_left.width)),
            int(form.get("left_height", default_left.height)),
        )
        right = Crop(
            int(form.get("right_left", default_right.left)),
            int(form.get("right_top", default_right.top)),
            int(form.get("right_width", default_right.width)),
            int(form.get("right_height", default_right.height)),
        )
        preview = _preview_png_data_uri(first_frame, left, right)
        info = f"{shape[0]} frames, {shape[1]} x {shape[2]} px"
        raw_path = str(raw_tiff)
    except Exception as exc:
        inferred = form.get("zmap_sample_kind", "dynamin")
        crop_preset = form.get("crop_preset", "full")
        left = Crop(0, 0, 600, 1200)
        right = Crop(600, 0, 600, 1200)
        preview = None
        info = f"TIFF load failed: {exc}"

    values = {
        "known_sample": sample_key,
        "raw_tiff": raw_path,
        "workflow_mode": form.get("workflow_mode", "train_infer_recon"),
        "channel_mode": form.get("channel_mode", "dual"),
        "zmap_sample_kind": form.get("zmap_sample_kind", inferred),
        "crop_preset": crop_preset,
        "left_left": form.get("left_left", str(left.left)),
        "left_top": form.get("left_top", str(left.top)),
        "left_width": form.get("left_width", str(left.width)),
        "left_height": form.get("left_height", str(left.height)),
        "right_left": form.get("right_left", str(right.left)),
        "right_top": form.get("right_top", str(right.top)),
        "right_width": form.get("right_width", str(right.width)),
        "right_height": form.get("right_height", str(right.height)),
        "epochs": form.get("epochs", "300"),
        "batch_size": form.get("batch_size", "24"),
        "steps_per_epoch": form.get("steps_per_epoch", "417"),
        "roi_size": form.get("roi_size", "96"),
        "psf_size": form.get("psf_size", "25"),
        "roi_stride": form.get("roi_stride", "88"),
        "start_epoch": form.get("start_epoch", "30"),
        "update_interval_epochs": form.get("update_interval_epochs", "5"),
        "target_projected_emitters": form.get("target_projected_emitters", "5000"),
        "hq_max_emitters": form.get("hq_max_emitters", "500"),
        "hq_alternating_rounds": form.get("hq_alternating_rounds", "20"),
        "hq_spatial_balance_grid_px": form.get("hq_spatial_balance_grid_px", ""),
        "filter_prob_min": form.get("filter_prob_min", "0.70"),
        "run_tag": form.get("run_tag", ""),
    }

    def field(name: str, label: str, width: int = 10) -> str:
        return (
            f'<label>{html.escape(label)}'
            f'<input name="{html.escape(name)}" value="{html.escape(values[name])}" style="width:{width}ch"></label>'
        )

    sample_options = "\n".join(
        f'<option value="{html.escape(name)}" {"selected" if name == sample_key else ""}>{html.escape(name)}</option>'
        for name in KNOWN_SAMPLE_PATHS
    )
    kind_options = "\n".join(
        f'<option value="{kind}" {"selected" if kind == values["zmap_sample_kind"] else ""}>{kind}</option>'
        for kind in SUPPORTED_ZMAP_SAMPLES
    )
    workflow_options = "\n".join(
        f'<option value="{mode}" {"selected" if mode == values["workflow_mode"] else ""}>{mode}</option>'
        for mode in WORKFLOW_MODES
    )
    channel_options = "\n".join(
        f'<option value="{mode}" {"selected" if mode == values["channel_mode"] else ""}>{mode}</option>'
        for mode in CHANNEL_MODES
    )
    preset_options = "\n".join(
        f'<option value="{preset}" {"selected" if preset == values["crop_preset"] else ""}>{preset}</option>'
        for preset in ("full", "center400", "custom")
    )
    message_html = f'<pre class="message">{html.escape(message)}</pre>' if message else ""
    top_status_html = (
        f'<section class="top-status"><strong>Last action</strong><pre>{html.escape(message)}</pre></section>'
        if message
        else '<section class="top-status muted"><strong>Status</strong><span>Ready to load a sample, dry run, or submit Slurm.</span></section>'
    )
    preview_html = f'<img class="preview" src="{preview}" alt="TIFF first-frame preview">' if preview else "<p>No preview available.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Neptune v0.4 Slurm Submitter</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202124; background: #f6f7f8; }}
    main {{ display: grid; grid-template-columns: minmax(420px, 560px) 1fr; gap: 16px; padding: 16px; }}
    section {{ background: #fff; border: 1px solid #d8dde3; border-radius: 8px; padding: 14px; margin-bottom: 12px; }}
    h1 {{ font-size: 20px; margin: 0 0 12px; }}
    h2 {{ font-size: 15px; margin: 0 0 10px; }}
    label {{ display: inline-flex; flex-direction: column; gap: 3px; margin: 0 8px 8px 0; color: #4a5563; }}
    input, select {{ box-sizing: border-box; padding: 5px 7px; border: 1px solid #b7c0cc; border-radius: 5px; background: #fff; color: #111827; }}
    input.path {{ width: 100%; }}
    button {{ padding: 7px 10px; border: 1px solid #1f5f99; border-radius: 5px; background: #2368a2; color: white; cursor: pointer; }}
    button.secondary {{ background: #fff; color: #2368a2; }}
    button:disabled {{ opacity: .55; cursor: wait; }}
    .row {{ display: flex; flex-wrap: wrap; align-items: end; gap: 4px; }}
    .preview {{ max-width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; background: #111; }}
    .message {{ white-space: pre-wrap; background: #101828; color: #e5e7eb; padding: 10px; border-radius: 6px; overflow-x: auto; }}
    .top-status {{ position: sticky; top: 0; z-index: 10; grid-column: 1 / -1; margin: 0; background: #ecfdf3; border-color: #7cce9a; }}
    .top-status.muted {{ background: #fff; border-color: #d8dde3; }}
    .top-status strong {{ display: block; margin-bottom: 5px; }}
    .top-status pre {{ margin: 0; white-space: pre-wrap; max-height: 170px; overflow: auto; }}
    .top-status span {{ color: #5f6b7a; }}
    .hint {{ color: #5f6b7a; margin: 4px 0 0; }}
  </style>
  <script>
    window.addEventListener('DOMContentLoaded', () => {{
      for (const form of document.querySelectorAll('form')) {{
        form.addEventListener('submit', (event) => {{
          const submitter = event.submitter;
          if (!submitter || submitter.formAction.endsWith('/submit')) {{
            for (const button of form.querySelectorAll('button')) button.disabled = true;
            if (submitter) submitter.textContent = 'Submitting...';
          }}
        }});
      }}
    }});
  </script>
</head>
<body>
<main>
  {top_status_html}
  <form method="post">
    <h1>Neptune v0.4 Slurm Submitter</h1>
    <section>
      <h2>Dataset</h2>
      <div class="row">
        <label>Known sample<select name="known_sample">{sample_options}</select></label>
        <button class="secondary" formaction="/load">Load Sample</button>
      </div>
      <label style="display:flex">Raw TIFF / symlink path
        <input class="path" name="raw_tiff" value="{html.escape(values["raw_tiff"])}">
      </label>
      <p class="hint">{html.escape(info)}</p>
      <div class="row">
        <label>Workflow<select name="workflow_mode">{workflow_options}</select></label>
        <label>Recon channel<select name="channel_mode">{channel_options}</select></label>
      </div>
      <div class="row">
        <label>Sample kind<select name="zmap_sample_kind">{kind_options}</select></label>
        <label>Crop preset<select name="crop_preset">{preset_options}</select></label>
        <button class="secondary" formaction="/load">Refresh Preview</button>
      </div>
    </section>
    <section>
      <h2>Crop</h2>
      <div class="row">{field("left_left", "left x")}{field("left_top", "left y")}{field("left_width", "left w")}{field("left_height", "left h")}</div>
      <div class="row">{field("right_left", "right x")}{field("right_top", "right y")}{field("right_width", "right w")}{field("right_height", "right h")}</div>
    </section>
    <section>
      <h2>Training</h2>
      <div class="row">{field("epochs", "epochs")}{field("batch_size", "batch")}{field("steps_per_epoch", "steps")}{field("roi_size", "roi")}{field("psf_size", "psf")}{field("roi_stride", "stride")}</div>
      <div class="row">{field("start_epoch", "start update")}{field("update_interval_epochs", "interval")}{field("target_projected_emitters", "target emitters", 12)}{field("hq_max_emitters", "HQ emitters", 12)}{field("hq_alternating_rounds", "HQ rounds", 12)}{field("hq_spatial_balance_grid_px", "HQ grid px", 12)}</div>
      <div class="row">{field("filter_prob_min", "recon prob", 10)}</div>
      <label style="display:flex">Run tag
        <input class="path" name="run_tag" value="{html.escape(values["run_tag"])}" placeholder="empty = auto">
      </label>
      <div class="row">
        <button class="secondary" formaction="/dry-run">Dry Run</button>
        <button formaction="/submit">Submit Slurm</button>
      </div>
    </section>
    {message_html}
  </form>
  <section>
    <h2>First Frame Preview</h2>
    {preview_html}
  </section>
</main>
</body>
</html>"""


class SubmitHandler(BaseHTTPRequestHandler):
    def _send_html(self, page: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = page.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query, keep_blank_values=True)
        form = {key: values[-1] if values else "" for key, values in query.items()}
        self._send_html(_html_page(form=form))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        form = _flatten_form(self.rfile.read(length))
        path = urlparse(self.path).path
        try:
            if path == "/load":
                sample_path = KNOWN_SAMPLE_PATHS.get(form.get("known_sample", ""), Path(form.get("raw_tiff", "")))
                form["raw_tiff"] = str(sample_path)
                self._send_html(_html_page(form=form))
                return
            config = _collect_config(form)
            command, command_env = _submit_command(config)
            command_text = " ".join(command)
            if command_env:
                env_text = "\n".join(f"{key}={value}" for key, value in sorted(command_env.items()))
                command_text = f"{env_text}\n\n{command_text}"
            if path == "/dry-run":
                message = f"Dry run OK.\nRun tag: {config.run_tag}\n\n{command_text}"
                self._send_html(_html_page(message=message, form={**form, "run_tag": config.run_tag}))
                return
            if path == "/submit":
                env = os.environ.copy()
                if command_env:
                    env.update(command_env)
                result = subprocess.run(command, check=True, text=True, capture_output=True, env=env)
                output = (result.stdout or "").strip()
                job_match = re.search(r"(?:Submitted batch job|train_job=)\s*(?:=)?\s*(\d+)", output)
                job_id = job_match.group(1) if job_match else None
                manifest_dir = UNITY_DIR / ".local/submissions"
                manifest_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = manifest_dir / f"web_submission_{job_id or int(time.time())}.json"
                manifest = {
                    "submitted_at_unix": time.time(),
                    "job_id": job_id,
                    "sbatch_stdout": output,
                    "command": command,
                    "command_env": command_env,
                    "config": asdict(config),
                }
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                message = f"{output}\nManifest: {manifest_path}\n\n{command_text}"
                self._send_html(_html_page(message=message, form={**form, "run_tag": config.run_tag}))
                return
            self._send_html(_html_page(message=f"Unknown action: {path}", form=form), HTTPStatus.NOT_FOUND)
        except subprocess.CalledProcessError as exc:
            self._send_html(_html_page(message=(exc.stderr or exc.stdout or str(exc)).strip(), form=form), HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_html(_html_page(message=f"Error: {exc}", form=form), HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[web-submit] {self.address_string()} - {fmt % args}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Start the Neptune v0.4 web Slurm submitter.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, int(args.port)), SubmitHandler)
    print(f"Neptune v0.4 web submitter: http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
