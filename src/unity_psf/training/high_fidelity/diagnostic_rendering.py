from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def render_gamma_monitor_markdown(payload: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Gamma Update Monitor",
            "",
            f"- epoch: {payload.get('epoch')}",
            f"- best_step: {payload.get('best_step')}",
            f"- selected_poisson_nll: {payload.get('selected_poisson_nll')}",
            f"- selected_sampled_emitter_count: {payload.get('selected_sampled_emitter_count')}",
            f"- selected_projected_photons: {payload.get('selected_projected_photons')}",
            f"- selected_background_mean: {payload.get('selected_background_mean')}",
            f"- heldout_available: {payload.get('heldout_available')}",
            f"- heldout_monitor_mode: {payload.get('heldout_monitor_mode')}",
            f"- heldout_initial_loss: {payload.get('heldout_initial_loss')}",
            f"- heldout_final_loss: {payload.get('heldout_final_loss')}",
            f"- heldout_loss_delta: {payload.get('heldout_loss_delta')}",
            f"- diagnostic_png_path: {payload.get('diagnostic_png_path')}",
            "",
        ]
    )


def write_raw_vs_reconstruction_png(
    path: Path,
    *,
    raw_frame: torch.Tensor,
    reconstruction: torch.Tensor,
    background: torch.Tensor | None = None,
    raw_is_photon: bool = True,
) -> None:
    if background is None:
        raw = to_uint8(raw_frame.detach().cpu())
        recon = to_uint8(reconstruction.detach().cpu())
    else:
        scale_frames = [reconstruction]
        scale_backgrounds = [background]
        if raw_is_photon:
            scale_frames.append(raw_frame)
            scale_backgrounds.append(background)
        display_scale = background_anchored_display_scale(scale_frames, scale_backgrounds)
        raw = (
            to_uint8_background_anchored(raw_frame.detach().cpu(), background, display_scale)
            if raw_is_photon
            else to_uint8(raw_frame.detach().cpu())
        )
        recon = to_uint8_background_anchored(reconstruction.detach().cpu(), background, display_scale)
    canvas = np.concatenate([raw, recon], axis=1)
    write_grayscale_png(path, canvas)


def to_uint8(frame: torch.Tensor) -> np.ndarray:
    array = frame.numpy().astype(np.float32)
    low = float(np.min(array))
    high = float(np.max(array))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    return np.clip((array - low) / (high - low) * 255.0, 0.0, 255.0).astype(np.uint8)


def background_anchored_display_scale(
    frames: list[torch.Tensor],
    backgrounds: list[torch.Tensor],
    *,
    bg_gray: int = 24,
) -> dict[str, float | int | str]:
    positive_signal: list[np.ndarray] = []
    bg_values: list[float] = []
    for frame, background in zip(frames, backgrounds, strict=False):
        frame_array = torch.as_tensor(frame).detach().cpu().numpy().astype(np.float32)
        bg_array = torch.as_tensor(background).detach().cpu().numpy().astype(np.float32)
        if bg_array.shape != frame_array.shape:
            bg_array = np.broadcast_to(bg_array, frame_array.shape)
        bg_values.append(float(np.median(bg_array)))
        signal = frame_array - bg_array
        positive_signal.append(signal[signal > 0.0])
    merged = (
        np.concatenate([item for item in positive_signal if item.size > 0])
        if any(item.size > 0 for item in positive_signal)
        else np.array([], dtype=np.float32)
    )
    signal_high = float(np.percentile(merged, 99.5)) if merged.size else 1.0
    signal_high = max(signal_high, 1.0)
    return {
        "mode": "background_anchored",
        "background_gray_uint8": int(bg_gray),
        "background_reference_photon": float(np.median(bg_values)) if bg_values else 0.0,
        "signal_high_photon": signal_high,
    }


def to_uint8_background_anchored(
    frame: torch.Tensor,
    background: torch.Tensor,
    display_scale: Mapping[str, object],
) -> np.ndarray:
    array = torch.as_tensor(frame).detach().cpu().numpy().astype(np.float32)
    bg = torch.as_tensor(background).detach().cpu().numpy().astype(np.float32)
    if bg.shape != array.shape:
        bg = np.broadcast_to(bg, array.shape)
    bg_gray = float(display_scale.get("background_gray_uint8", 24))
    signal_high = max(float(display_scale.get("signal_high_photon", 1.0)), 1.0)
    scaled = bg_gray + (array - bg) / signal_high * (255.0 - bg_gray)
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def tile_frames_uint8(frames: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate([to_uint8(frame.detach().cpu()) for frame in frames], axis=1)


def poisson_nll_value(raw_frame: torch.Tensor, reconstruction: torch.Tensor) -> float:
    raw = raw_frame.detach().cpu().to(dtype=torch.float32).clamp_min(0.0)
    recon = reconstruction.detach().cpu().to(dtype=torch.float32).clamp_min(1e-6)
    return float((recon - raw * torch.log(recon)).mean().item())


def ncc_value(raw_frame: torch.Tensor, reconstruction: torch.Tensor) -> float:
    raw = raw_frame.detach().cpu().to(dtype=torch.float32)
    recon = reconstruction.detach().cpu().to(dtype=torch.float32)
    raw_centered = raw - raw.mean()
    recon_centered = recon - recon.mean()
    denom = torch.sqrt(
        raw_centered.square().sum().clamp_min(1e-8)
        * recon_centered.square().sum().clamp_min(1e-8)
    )
    return float(((raw_centered * recon_centered).sum() / denom).item())


def write_grayscale_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    height, width = int(image.shape[0]), int(image.shape[1])
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(raw_rows)),
            _png_chunk(b"IEND", b""),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


__all__ = [
    "background_anchored_display_scale",
    "ncc_value",
    "poisson_nll_value",
    "render_gamma_monitor_markdown",
    "tile_frames_uint8",
    "to_uint8",
    "to_uint8_background_anchored",
    "write_grayscale_png",
    "write_raw_vs_reconstruction_png",
]
