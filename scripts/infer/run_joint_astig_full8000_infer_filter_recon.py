#!/usr/bin/env python3
"""Run the joint UnityPSF astigmatism expert through the standard infer/recon path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
UNITY_DIR = Path(__file__).resolve().parents[2]
SRC_ROOT = UNITY_DIR / "src"
for path in (ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from unity_psf.config import load_config
from unity_psf.localization import build_localization_runtime_config
from unity_psf.models import UnityPSF

import run_3371_full8000_infer_filter_recon as standard


class JointAstigmatismAdapter(torch.nn.Module):
    """Expose one joint-checkpoint channel with the legacy infer/recon call shape."""

    def __init__(self, model: UnityPSF, *, channel_id: str) -> None:
        super().__init__()
        self.model = model
        self.channel_id = str(channel_id)
        self.condition_dim = int(getattr(model.experts["astigmatism"], "condition_dim"))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        images, conditions = inputs
        return self.model.localize(
            images,
            modality="astigmatism",
            channel_id=self.channel_id,
            conditions=conditions,
        ).raw


def channel_coeff_map(model: UnityPSF, *, side: str, override: Path | None = None) -> Path:
    if override is not None:
        path = override.resolve()
    else:
        state = model.channel_state("astigmatism", side)
        entries = state["physical_state"].get("coeff_maps", ())
        if not entries or not isinstance(entries[0], dict):
            raise ValueError(f"joint checkpoint has no coefficient map for astigmatism:{side}")
        path = Path(str(entries[0]["coeff_maps_npz"])).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"missing coefficient map for astigmatism:{side}: {path}")
    return path


def main() -> int:
    args = standard.parse_args()
    standard.assert_output_dir_available(args.output_dir, overwrite=bool(args.overwrite_output))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = (args.checkpoint or (args.run_dir / "checkpoints/unitypsf_joint.ckpt")).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing joint checkpoint: {checkpoint}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal joint UnityPSF inference requires CUDA; refusing CPU fallback.")

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = UnityPSF.from_checkpoint(checkpoint, device=device, load_mode="eager")
    model.eval()
    if "astigmatism" not in model.experts:
        raise ValueError("joint checkpoint does not contain an astigmatism expert")
    for side in ("left", "right"):
        model.channel_state("astigmatism", side)

    config = load_config(args.config)
    runtime = build_localization_runtime_config(config, config_base_dir=args.config.parent, seed=0)
    frame_proc = None
    if str(args.input_preprocess) == "fd_deeploc_recenter":
        from Normalization import build_inference_frame_normalizer

        frame_proc = build_inference_frame_normalizer(config)

    print(
        json.dumps(
            {
                "cuda_available": True,
                "device": str(device),
                "device_name": torch.cuda.get_device_name(0),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "model": model.describe(),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )

    sides = ("left", "right") if args.side == "both" else (args.side,)
    summaries = []
    coeff_maps: dict[str, str] = {}
    for side in sides:
        override = args.right_coeff_map if side == "right" else None
        coeff_map = channel_coeff_map(model, side=side, override=override)
        coeff_maps[side] = str(coeff_map)
        adapter = JointAstigmatismAdapter(model, channel_id=side)
        adapter.eval()
        if side == "left":
            crop_left = int(args.left_crop_left)
            crop_top = int(args.left_crop_top)
            crop_width = int(args.left_crop_width)
            crop_height = int(args.left_crop_height)
            domain_index = 0
        else:
            crop_left = int(args.right_crop_left)
            crop_top = int(args.right_crop_top)
            crop_width = int(args.right_crop_width)
            crop_height = int(args.right_crop_height)
            domain_index = int(args.right_domain_index)
        summaries.append(
            standard.run_side(
                side=side,
                args=args,
                model=adapter,
                runtime=runtime,
                device=device,
                coeff_map=coeff_map,
                crop_left=crop_left,
                crop_top=crop_top,
                crop_width=crop_width,
                crop_height=crop_height,
                domain_index=domain_index,
                domain_count=int(args.domain_count),
                frame_proc=frame_proc,
            )
        )

    manifest = {
        "standard": "unitypsf_joint_astigmatism_full8000_infer_filter_recon_v0.1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "source_train_dir": str(args.run_dir),
        "sample_tiff": str(args.sample_tiff),
        "model": model.describe(),
        "coeff_maps": coeff_maps,
        "sides": summaries,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
