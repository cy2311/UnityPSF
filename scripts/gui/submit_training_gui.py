from __future__ import annotations

import json
import re
import subprocess
import time
import tkinter as tk
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tifffile

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError:  # pragma: no cover - GUI remains usable without preview.
    Image = None
    ImageDraw = None
    ImageTk = None


ROOT = Path("/home/guest/Others/main/race")
UNITY_DIR = ROOT / "unity"
DEFAULT_RAW_TIFF = ROOT / "neptune_iwae/test_data/microtube/raw/spool_800mW_30ms_3D_7_1_MMStack_Default.ome.tif"
TRAINING_SETS_DIR = ROOT / "datasets/training_sets"
TRAIN_SCRIPT = UNITY_DIR / "scripts/archive/neptune_standard/train_standard_3367_hqzmap.sh"
SUPPORTED_ZMAP_SAMPLES = ("microtube", "paint", "ncp", "dynamin", "membrane")
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


def validate_zmap_sample_kind_for_tiff(sample: str, raw_tiff: Path) -> None:
    sample_norm = str(sample).strip().lower()
    inferred = infer_zmap_sample_kind_from_path(raw_tiff)
    if sample_norm not in SUPPORTED_ZMAP_SAMPLES:
        raise ValueError(
            f"Unsupported sample kind {sample_norm!r}. The current high-quality zmap presets support: "
            f"{', '.join(SUPPORTED_ZMAP_SAMPLES)}."
        )
    if inferred is not None and inferred != sample_norm:
        raise ValueError(
            f"The raw TIFF path looks like {inferred!r}, but sample kind is {sample_norm!r}. "
            f"Refusing to submit with a mismatched initial-zmap preset."
        )


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


def _preview_image(frame: np.ndarray, *, max_side: int = 720):
    if Image is None or ImageTk is None:
        return None, None, 1.0
    lo, hi = np.percentile(frame, [0.2, 99.8])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((frame - lo) / (hi - lo), 0.0, 1.0)
    img = Image.fromarray((scaled * 255.0).astype(np.uint8), mode="L").convert("RGB")
    scale = min(float(max_side) / max(img.width, img.height), 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    return img, ImageTk.PhotoImage(img), scale


def _draw_crop_overlay(base_img, scale: float, left: Crop, right: Crop, mode: str):
    if ImageDraw is None:
        return base_img
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    def rect(crop: Crop, color: str, label: str) -> None:
        x0 = int(crop.left * scale)
        y0 = int(crop.top * scale)
        x1 = int((crop.left + crop.width) * scale)
        y1 = int((crop.top + crop.height) * scale)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 4, y0 + 4), label, fill=color)

    if mode in {"dual", "single_left"}:
        rect(left, "#35d06f", "left")
    if mode in {"dual", "single_right"}:
        rect(right, "#ff4d4d", "right")
    return img


def _safe_int(value: tk.StringVar, name: str, minimum: int = 0) -> int:
    try:
        parsed = int(value.get())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return parsed


def _crop_from_vars(prefix: str, vars_by_name: dict[str, tk.StringVar]) -> Crop:
    return Crop(
        left=_safe_int(vars_by_name[f"{prefix}_left"], f"{prefix} crop left"),
        top=_safe_int(vars_by_name[f"{prefix}_top"], f"{prefix} crop top"),
        width=_safe_int(vars_by_name[f"{prefix}_width"], f"{prefix} crop width", 1),
        height=_safe_int(vars_by_name[f"{prefix}_height"], f"{prefix} crop height", 1),
    )


def _validate_crop(crop: Crop, *, frame_width: int, frame_height: int, label: str) -> None:
    if crop.left + crop.width > frame_width or crop.top + crop.height > frame_height:
        raise ValueError(
            f"{label} crop exceeds TIFF bounds: crop=({crop.left},{crop.top},{crop.width},{crop.height}), "
            f"frame=({frame_width},{frame_height})"
        )


def _default_run_tag(config: SubmissionConfig) -> str:
    sample = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.zmap_sample_kind.strip() or "sample")
    return (
        f"gui_{sample}_{config.crop_preset}_hqzmap_emit{config.hq_max_emitters}_"
        f"roi{config.roi_size}_psf{config.psf_size}_e{config.epochs}_"
        f"start{config.start_epoch}_int{config.update_interval_epochs}_"
        f"bs{config.batch_size}_s{config.steps_per_epoch}"
    )


def _format_export(config: SubmissionConfig) -> dict[str, str]:
    env = {
        "NEPTUNE_V04_RAW_TIFF_PATH": config.raw_tiff,
        "ZMAP_SAMPLE_KIND": config.zmap_sample_kind,
        "EPOCHS": str(config.epochs),
        "BATCH_SIZE": str(config.batch_size),
        "STEPS_PER_EPOCH": str(config.steps_per_epoch),
        "ROI_SIZE": str(config.roi_size),
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
        "RUN_TAG": config.run_tag,
    }
    if config.hq_spatial_balance_grid_px.strip():
        env["HQ_SPATIAL_BALANCE_GRID_PX"] = config.hq_spatial_balance_grid_px.strip()
    return env


def _sbatch_command(config: SubmissionConfig) -> list[str]:
    env = _format_export(config)
    for key, value in env.items():
        if "," in value:
            raise ValueError(f"Environment value for {key} contains comma, which sbatch --export cannot encode safely: {value}")
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    return ["sbatch", f"--export={export_arg}", str(TRAIN_SCRIPT)]


class SubmitTrainingGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Neptune v0.4 Training Submitter")
        self.frame_shape: tuple[int, int, int] | None = None
        self.first_frame: np.ndarray | None = None
        self.base_preview = None
        self.tk_preview = None
        self.preview_scale = 1.0

        self.vars: dict[str, tk.StringVar] = {
            "raw_tiff": tk.StringVar(value=str(DEFAULT_RAW_TIFF)),
            "sample_path": tk.StringVar(value=str(DEFAULT_RAW_TIFF)),
            "known_sample": tk.StringVar(value="microtube default"),
            "channel_mode": tk.StringVar(value="dual"),
            "crop_preset": tk.StringVar(value="full"),
            "zmap_sample_kind": tk.StringVar(value="microtube"),
            "left_left": tk.StringVar(value="0"),
            "left_top": tk.StringVar(value="0"),
            "left_width": tk.StringVar(value="600"),
            "left_height": tk.StringVar(value="1200"),
            "right_left": tk.StringVar(value="600"),
            "right_top": tk.StringVar(value="0"),
            "right_width": tk.StringVar(value="600"),
            "right_height": tk.StringVar(value="1200"),
            "epochs": tk.StringVar(value="300"),
            "batch_size": tk.StringVar(value="24"),
            "steps_per_epoch": tk.StringVar(value="417"),
            "roi_size": tk.StringVar(value="96"),
            "psf_size": tk.StringVar(value="25"),
            "roi_stride": tk.StringVar(value="88"),
            "start_epoch": tk.StringVar(value="30"),
            "update_interval_epochs": tk.StringVar(value="5"),
            "target_projected_emitters": tk.StringVar(value="5000"),
            "hq_max_emitters": tk.StringVar(value="500"),
            "hq_alternating_rounds": tk.StringVar(value="20"),
            "hq_spatial_balance_grid_px": tk.StringVar(value=""),
            "run_tag": tk.StringVar(value=""),
        }
        self.info_var = tk.StringVar(value="Load a TIFF to inspect shape and preview.")
        self.status_var = tk.StringVar(value="")

        self._build()
        self._load_tiff()

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        preview = ttk.Frame(main)
        preview.grid(row=0, column=1, sticky="nsew")

        self._dataset_section(controls)
        self._channel_section(controls)
        self._crop_section(controls)
        self._training_section(controls)
        self._action_section(controls)

        ttk.Label(preview, textvariable=self.info_var).grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(preview)
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.command_text = tk.Text(preview, height=10, width=90)
        self.command_text.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(preview, textvariable=self.status_var, foreground="#006400").grid(row=3, column=0, sticky="w", pady=(6, 0))

    def _dataset_section(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Dataset / TIFF", padding=8)
        box.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Known sample").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            box,
            textvariable=self.vars["known_sample"],
            values=tuple(KNOWN_SAMPLE_PATHS.keys()),
            width=24,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(box, text="Use", command=self._use_known_sample).grid(row=0, column=2)
        ttk.Label(box, text="Sample path").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(box, textvariable=self.vars["sample_path"], width=58).grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Button(box, text="Select Path", command=self._browse_sample_path).grid(row=1, column=2, pady=(6, 0))
        ttk.Label(box, text="Raw TIFF").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(box, textvariable=self.vars["raw_tiff"], width=58).grid(row=2, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Button(box, text="Load Preview", command=self._load_tiff).grid(row=3, column=1, sticky="w", pady=(6, 0))

    def _channel_section(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Channel Mode", padding=8)
        box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for idx, (label, value) in enumerate(
            (("Dual channel", "dual"), ("Single left", "single_left"), ("Single right", "single_right"))
        ):
            ttk.Radiobutton(box, text=label, variable=self.vars["channel_mode"], value=value, command=self._refresh_preview).grid(
                row=0, column=idx, sticky="w", padx=(0, 8)
            )

    def _crop_section(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="ROI / Crop", padding=8)
        box.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for idx, (label, value) in enumerate((("Full ROI", "full"), ("Center 400x400", "center400"), ("Custom", "custom"))):
            ttk.Radiobutton(box, text=label, variable=self.vars["crop_preset"], value=value, command=self._apply_crop_preset).grid(
                row=0, column=idx, sticky="w", padx=(0, 8)
            )
        ttk.Label(box, text="sample kind").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            box,
            textvariable=self.vars["zmap_sample_kind"],
            values=SUPPORTED_ZMAP_SAMPLES,
            width=14,
            state="readonly",
        ).grid(row=1, column=1, sticky="w", pady=(8, 0))
        self._crop_row(box, 2, "left")
        self._crop_row(box, 3, "right")
        for name in ("left_left", "left_top", "left_width", "left_height", "right_left", "right_top", "right_width", "right_height"):
            self.vars[name].trace_add("write", lambda *_: self._refresh_preview())

    def _crop_row(self, parent: ttk.Frame, row: int, prefix: str) -> None:
        ttk.Label(parent, text=prefix).grid(row=row, column=0, sticky="w", pady=(8, 0))
        labels = (("x", "left"), ("y", "top"), ("w", "width"), ("h", "height"))
        for idx, (label, suffix) in enumerate(labels):
            ttk.Label(parent, text=label).grid(row=row, column=1 + idx * 2, sticky="e", pady=(8, 0))
            ttk.Entry(parent, textvariable=self.vars[f"{prefix}_{suffix}"], width=7).grid(
                row=row, column=2 + idx * 2, sticky="w", padx=(2, 6), pady=(8, 0)
            )

    def _training_section(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Training Preset", padding=8)
        box.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        fields = (
            ("epochs", "epochs"),
            ("batch", "batch_size"),
            ("steps", "steps_per_epoch"),
            ("roi", "roi_size"),
            ("psf", "psf_size"),
            ("stride", "roi_stride"),
            ("start update", "start_epoch"),
            ("interval", "update_interval_epochs"),
            ("target emitters", "target_projected_emitters"),
            ("HQ emitters", "hq_max_emitters"),
            ("HQ rounds", "hq_alternating_rounds"),
            ("HQ grid px", "hq_spatial_balance_grid_px"),
        )
        for idx, (label, key) in enumerate(fields):
            row = idx // 3
            col = (idx % 3) * 2
            ttk.Label(box, text=label).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=(2, 2))
            ttk.Entry(box, textvariable=self.vars[key], width=10).grid(row=row, column=col + 1, sticky="w", padx=(0, 10), pady=(2, 2))
        ttk.Label(box, text="run tag").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(box, textvariable=self.vars["run_tag"], width=52).grid(row=4, column=1, columnspan=5, sticky="ew", pady=(8, 0))

    def _action_section(self, parent: ttk.Frame) -> None:
        box = ttk.Frame(parent)
        box.grid(row=4, column=0, sticky="ew")
        ttk.Button(box, text="Dry Run", command=self._dry_run).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(box, text="OK Submit Slurm", command=self._submit).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(box, text="Cancel", command=self.root.destroy).grid(row=0, column=2)

    def _browse_tiff(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select raw TIFF",
            initialdir=str(ROOT),
            filetypes=(("TIFF", "*.tif *.tiff *.ome.tif *.ome.tiff"), ("All files", "*")),
        )
        if selected:
            self.vars["sample_path"].set(selected)
            self.vars["raw_tiff"].set(selected)
            self._load_tiff()

    def _browse_sample_path(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select sample TIFF or symlink",
            initialdir=str(TRAINING_SETS_DIR if TRAINING_SETS_DIR.exists() else ROOT),
            filetypes=(("TIFF / symlink", "*.tif *.tiff *.ome.tif *.ome.tiff *"), ("All files", "*")),
        )
        if not selected:
            selected_dir = filedialog.askdirectory(
                title="Select sample directory",
                initialdir=str(TRAINING_SETS_DIR if TRAINING_SETS_DIR.exists() else ROOT),
            )
            selected = selected_dir
        if selected:
            self.vars["sample_path"].set(selected)
            self.vars["raw_tiff"].set(selected)
            self._load_tiff()

    def _use_known_sample(self) -> None:
        path = KNOWN_SAMPLE_PATHS.get(self.vars["known_sample"].get())
        if path is None:
            return
        self.vars["sample_path"].set(str(path))
        self.vars["raw_tiff"].set(str(path))
        self._load_tiff()

    def _load_tiff(self) -> None:
        try:
            path = _resolve_tiff_path(self.vars["raw_tiff"].get())
            self.vars["raw_tiff"].set(str(path))
            self.vars["sample_path"].set(str(path))
            self.frame_shape, self.first_frame = _read_tiff_info(path)
            frames, height, width = self.frame_shape
            inferred = infer_zmap_sample_kind_from_path(path)
            if inferred in SUPPORTED_ZMAP_SAMPLES:
                self.vars["zmap_sample_kind"].set(inferred)
                if inferred == "ncp":
                    self.vars["crop_preset"].set("center400")
                elif self.vars["crop_preset"].get() != "custom":
                    self.vars["crop_preset"].set("full")
            elif inferred in {"dynamin", "membrane"}:
                self.vars["zmap_sample_kind"].set(inferred)
                if self.vars["crop_preset"].get() != "custom":
                    self.vars["crop_preset"].set("full")
            self.info_var.set(f"TIFF: {frames} frames, {height} x {width} px | {path}")
            if self.vars["crop_preset"].get() != "custom":
                self._apply_crop_preset()
            self._refresh_preview()
        except Exception as exc:
            self.info_var.set("Failed to load TIFF.")
            messagebox.showerror("TIFF load failed", str(exc))

    def _apply_crop_preset(self) -> None:
        if self.frame_shape is None:
            return
        _, height, width = self.frame_shape
        preset = self.vars["crop_preset"].get()
        if preset == "full":
            half = width // 2
            values = {
                "left_left": 0,
                "left_top": 0,
                "left_width": half,
                "left_height": height,
                "right_left": half,
                "right_top": 0,
                "right_width": width - half,
                "right_height": height,
            }
            self.vars["hq_max_emitters"].set("500")
            self.vars["hq_spatial_balance_grid_px"].set("")
        elif preset == "center400":
            values = {
                "left_left": max(0, min(100, width - 400)),
                "left_top": max(0, min(400, height - 400)),
                "left_width": 400,
                "left_height": 400,
                "right_left": max(0, min(700, width - 400)),
                "right_top": max(0, min(400, height - 400)),
                "right_width": 400,
                "right_height": 400,
            }
            self.vars["zmap_sample_kind"].set("ncp")
            self.vars["hq_max_emitters"].set("1500")
            self.vars["hq_spatial_balance_grid_px"].set("128")
        else:
            self._refresh_preview()
            return
        for key, value in values.items():
            self.vars[key].set(str(value))
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self.first_frame is None or self.frame_shape is None:
            return
        if Image is None or ImageTk is None:
            self.preview_label.configure(text="Pillow is not available; preview disabled.")
            return
        base, _, scale = _preview_image(self.first_frame)
        if base is None:
            return
        self.base_preview = base
        self.preview_scale = scale
        try:
            left = _crop_from_vars("left", self.vars)
            right = _crop_from_vars("right", self.vars)
        except ValueError:
            left = Crop(0, 0, 1, 1)
            right = Crop(0, 0, 1, 1)
        overlay = _draw_crop_overlay(base, scale, left, right, self.vars["channel_mode"].get())
        self.tk_preview = ImageTk.PhotoImage(overlay)
        self.preview_label.configure(image=self.tk_preview, text="")

    def _collect_config(self) -> SubmissionConfig:
        if self.frame_shape is None:
            self._load_tiff()
        if self.frame_shape is None:
            raise ValueError("No TIFF loaded")
        channel_mode = self.vars["channel_mode"].get()
        if channel_mode != "dual":
            raise ValueError("Current standard training backend is dual-channel only. Use dual for submission; single-channel GUI support is preview-only.")
        _, frame_height, frame_width = self.frame_shape
        left_crop = _crop_from_vars("left", self.vars)
        right_crop = _crop_from_vars("right", self.vars)
        _validate_crop(left_crop, frame_width=frame_width, frame_height=frame_height, label="left")
        _validate_crop(right_crop, frame_width=frame_width, frame_height=frame_height, label="right")
        if left_crop.top != right_crop.top or left_crop.height != right_crop.height:
            raise ValueError("Current HQ bootstrap requires left/right crops with matching y and height.")
        config = SubmissionConfig(
            raw_tiff=str(_resolve_tiff_path(self.vars["raw_tiff"].get())),
            channel_mode=channel_mode,
            crop_preset=self.vars["crop_preset"].get(),
            zmap_sample_kind=self.vars["zmap_sample_kind"].get(),
            left_crop=left_crop,
            right_crop=right_crop,
            epochs=_safe_int(self.vars["epochs"], "epochs", 1),
            batch_size=_safe_int(self.vars["batch_size"], "batch_size", 1),
            steps_per_epoch=_safe_int(self.vars["steps_per_epoch"], "steps_per_epoch", 1),
            roi_size=_safe_int(self.vars["roi_size"], "roi_size", 1),
            psf_size=_safe_int(self.vars["psf_size"], "psf_size", 1),
            roi_stride=_safe_int(self.vars["roi_stride"], "roi_stride", 1),
            start_epoch=_safe_int(self.vars["start_epoch"], "start_epoch", 0),
            update_interval_epochs=_safe_int(self.vars["update_interval_epochs"], "update_interval_epochs", 1),
            target_projected_emitters=_safe_int(self.vars["target_projected_emitters"], "target_projected_emitters", 1),
            hq_max_emitters=_safe_int(self.vars["hq_max_emitters"], "HQ emitters", 1),
            hq_alternating_rounds=_safe_int(self.vars["hq_alternating_rounds"], "HQ rounds", 1),
            hq_spatial_balance_grid_px=self.vars["hq_spatial_balance_grid_px"].get().strip(),
            run_tag=self.vars["run_tag"].get().strip(),
        )
        validate_zmap_sample_kind_for_tiff(config.zmap_sample_kind, Path(config.raw_tiff))
        if not config.run_tag:
            config = replace(config, run_tag=_default_run_tag(config))
            self.vars["run_tag"].set(config.run_tag)
        return config

    def _show_command(self, command: list[str]) -> None:
        self.command_text.delete("1.0", tk.END)
        self.command_text.insert(tk.END, " ".join(command))

    def _dry_run(self) -> None:
        try:
            config = self._collect_config()
            command = _sbatch_command(config)
            self._show_command(command)
            self.status_var.set("Dry run complete. Review command; no job submitted.")
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))

    def _submit(self) -> None:
        try:
            config = self._collect_config()
            command = _sbatch_command(config)
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return
        self._show_command(command)
        if not messagebox.askokcancel("Submit Slurm job", f"Submit training job?\n\nRun tag:\n{config.run_tag}"):
            return
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            messagebox.showerror("sbatch failed", (exc.stderr or exc.stdout or str(exc)).strip())
            return
        output = (result.stdout or "").strip()
        job_match = re.search(r"Submitted batch job\s+(\d+)", output)
        job_id = job_match.group(1) if job_match else None
        manifest_dir = UNITY_DIR / ".local/submissions"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"gui_submission_{job_id or int(time.time())}.json"
        manifest = {
            "submitted_at_unix": time.time(),
            "job_id": job_id,
            "sbatch_stdout": output,
            "command": command,
            "config": asdict(config),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self.status_var.set(f"{output} | manifest: {manifest_path}")
        messagebox.showinfo("Submitted", f"{output}\n\nManifest:\n{manifest_path}")


def main() -> int:
    root = tk.Tk()
    SubmitTrainingGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
