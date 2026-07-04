from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neptune_v03.field_origin import build_sliding_window_origin_bank, sliding_window_origin_index_for_xy

from .types import FOVSelection, ROICandidate


def clamp_roi_origin(
    *,
    center_xy_px: tuple[float, float],
    domain_width_px: int,
    domain_height_px: int,
    roi_size_px: int,
) -> tuple[int, int]:
    roi = int(roi_size_px)
    width = int(domain_width_px)
    height = int(domain_height_px)
    if roi <= 0:
        raise ValueError("roi_size_px must be positive")
    if width < roi or height < roi:
        raise ValueError("domain dimensions must be at least roi_size_px")
    x0 = int(round(float(center_xy_px[0]) - roi / 2.0))
    y0 = int(round(float(center_xy_px[1]) - roi / 2.0))
    return (max(0, min(x0, width - roi)), max(0, min(y0, height - roi)))


def grid_cell_id_for_xy(
    xy_px: tuple[float, float],
    *,
    domain_width_px: int,
    domain_height_px: int,
    grid_shape: tuple[int, int],
) -> int:
    rows, cols = int(grid_shape[0]), int(grid_shape[1])
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape must contain positive row/col counts")
    width = float(domain_width_px)
    height = float(domain_height_px)
    if width <= 0 or height <= 0:
        raise ValueError("domain dimensions must be positive")
    x = max(0.0, min(float(xy_px[0]), np.nextafter(width, 0.0)))
    y = max(0.0, min(float(xy_px[1]), np.nextafter(height, 0.0)))
    col = min(cols - 1, int(x / width * cols))
    row = min(rows - 1, int(y / height * rows))
    return row * cols + col


def roi_overlap_fraction(
    origin_a_xy_px: tuple[float, float] | tuple[int, int],
    origin_b_xy_px: tuple[float, float] | tuple[int, int],
    *,
    roi_size_px: int,
) -> float:
    roi = float(roi_size_px)
    if roi <= 0:
        raise ValueError("roi_size_px must be positive")
    ax0, ay0 = float(origin_a_xy_px[0]), float(origin_a_xy_px[1])
    bx0, by0 = float(origin_b_xy_px[0]), float(origin_b_xy_px[1])
    overlap_w = max(0.0, min(ax0 + roi, bx0 + roi) - max(ax0, bx0))
    overlap_h = max(0.0, min(ay0 + roi, by0 + roi) - max(ay0, by0))
    return float((overlap_w * overlap_h) / (roi * roi))


def build_roi_candidate(
    *,
    candidate_id: int,
    center_xy_px: tuple[float, float],
    emitter_indices: Sequence[int],
    emitter_probabilities: Sequence[float],
    domain_width_px: int,
    domain_height_px: int,
    roi_size_px: int,
    grid_shape: tuple[int, int],
) -> ROICandidate:
    origin = clamp_roi_origin(
        center_xy_px=center_xy_px,
        domain_width_px=domain_width_px,
        domain_height_px=domain_height_px,
        roi_size_px=roi_size_px,
    )
    probabilities = np.asarray(tuple(emitter_probabilities), dtype=np.float32)
    indices = tuple(int(v) for v in emitter_indices)
    return ROICandidate(
        candidate_id=int(candidate_id),
        origin_xy_px=origin,
        center_xy_px=(float(center_xy_px[0]), float(center_xy_px[1])),
        grid_cell_id=grid_cell_id_for_xy(
            center_xy_px,
            domain_width_px=domain_width_px,
            domain_height_px=domain_height_px,
            grid_shape=grid_shape,
        ),
        emitter_indices=indices,
        emitter_count=len(indices),
        quality_score=_mean_probability_score(probabilities),
    )


def build_sliding_window_guided_candidates(
    *,
    xy_px: np.ndarray,
    probabilities: np.ndarray,
    domain_width_px: int,
    domain_height_px: int,
    roi_size_px: int,
    stride_px: int,
    grid_shape: tuple[int, int],
    valid_core_size_px: int | None = None,
) -> tuple[ROICandidate, ...]:
    xy = np.asarray(xy_px, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy_px must have shape [N, 2]")
    if probs.shape[0] != xy.shape[0]:
        raise ValueError("probabilities must have length N")
    core_size = int(valid_core_size_px or roi_size_px)
    if core_size <= 0 or core_size > int(roi_size_px):
        raise ValueError("valid_core_size_px must be positive and no larger than roi_size_px")
    origins = build_sliding_window_origin_bank(
        field_width_px=int(domain_width_px),
        field_height_px=int(domain_height_px),
        roi_width_px=core_size,
        roi_height_px=core_size,
        stride_px=int(stride_px),
    )
    by_origin: dict[int, list[int]] = {}
    for emitter_idx, center in enumerate(xy):
        origin_idx = sliding_window_origin_index_for_xy(
            (float(center[0]), float(center[1])),
            origins=origins,
            roi_width_px=core_size,
            roi_height_px=core_size,
        )
        by_origin.setdefault(int(origin_idx), []).append(int(emitter_idx))

    candidates: list[ROICandidate] = []
    for candidate_id, origin_idx in enumerate(sorted(by_origin)):
        core_origin = origins[int(origin_idx)]
        origin = _context_origin_from_core(
            core_origin,
            domain_width_px=int(domain_width_px),
            domain_height_px=int(domain_height_px),
            roi_size_px=int(roi_size_px),
            valid_core_size_px=core_size,
        )
        x0, y0 = origin
        x1, y1 = x0 + int(roi_size_px), y0 + int(roi_size_px)
        inside = (
            (xy[:, 0] >= float(x0))
            & (xy[:, 0] < float(x1))
            & (xy[:, 1] >= float(y0))
            & (xy[:, 1] < float(y1))
        )
        emitter_indices = tuple(int(v) for v in np.nonzero(inside)[0].tolist())
        emitter_probs = probs[np.asarray(emitter_indices, dtype=np.int64)] if emitter_indices else np.empty((0,), dtype=np.float32)
        center_xy = _candidate_center_from_origin(core_origin, roi_size_px=core_size)
        candidates.append(
            ROICandidate(
                candidate_id=int(candidate_id),
                origin_xy_px=origin,
                center_xy_px=center_xy,
                grid_cell_id=grid_cell_id_for_xy(
                    center_xy,
                    domain_width_px=domain_width_px,
                    domain_height_px=domain_height_px,
                    grid_shape=grid_shape,
                ),
                emitter_indices=emitter_indices,
                emitter_count=len(emitter_indices),
                quality_score=_mean_probability_score(emitter_probs),
                origin_bank_index=int(origin_idx),
                valid_core_origin_xy_px=core_origin,
                valid_core_offset_xy_px=(int(core_origin[0]) - int(origin[0]), int(core_origin[1]) - int(origin[1])),
                valid_core_size_px=core_size,
            )
        )
    return tuple(candidates)


def build_emitter_centered_candidates(
    *,
    xy_px: np.ndarray,
    probabilities: np.ndarray,
    domain_width_px: int,
    domain_height_px: int,
    roi_size_px: int,
    grid_shape: tuple[int, int],
) -> tuple[ROICandidate, ...]:
    xy = np.asarray(xy_px, dtype=np.float32)
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy_px must have shape [N, 2]")
    if probs.shape[0] != xy.shape[0]:
        raise ValueError("probabilities must have length N")

    candidates: list[ROICandidate] = []
    for candidate_id, center in enumerate(xy):
        origin = clamp_roi_origin(
            center_xy_px=(float(center[0]), float(center[1])),
            domain_width_px=domain_width_px,
            domain_height_px=domain_height_px,
            roi_size_px=roi_size_px,
        )
        x0, y0 = origin
        x1, y1 = x0 + int(roi_size_px), y0 + int(roi_size_px)
        inside = (
            (xy[:, 0] >= float(x0))
            & (xy[:, 0] < float(x1))
            & (xy[:, 1] >= float(y0))
            & (xy[:, 1] < float(y1))
        )
        emitter_indices = tuple(int(v) for v in np.nonzero(inside)[0].tolist())
        emitter_probs = probs[np.asarray(emitter_indices, dtype=np.int64)] if emitter_indices else np.empty((0,), dtype=np.float32)
        candidates.append(
            build_roi_candidate(
                candidate_id=candidate_id,
                center_xy_px=(float(center[0]), float(center[1])),
                emitter_indices=emitter_indices,
                emitter_probabilities=emitter_probs,
                domain_width_px=domain_width_px,
                domain_height_px=domain_height_px,
                roi_size_px=roi_size_px,
                grid_shape=grid_shape,
            )
        )
    return tuple(candidates)


def select_fov_balanced_candidates(
    candidates: Sequence[ROICandidate],
    *,
    max_rois: int,
    target_emitters: int,
    grid_cell_count: int,
    roi_size_px: int,
    max_overlap_fraction: float,
) -> FOVSelection:
    max_rois = int(max_rois)
    grid_cell_count = int(grid_cell_count)
    if max_rois <= 0:
        return _selection_summary((), candidates, grid_cell_count=grid_cell_count, target_emitters=target_emitters)
    by_cell: dict[int, list[ROICandidate]] = {cell: [] for cell in range(grid_cell_count)}
    for candidate in candidates:
        by_cell.setdefault(int(candidate.grid_cell_id), []).append(candidate)
    for cell_candidates in by_cell.values():
        cell_candidates.sort(key=_candidate_sort_key)

    selected: list[ROICandidate] = []
    selected_ids: set[int] = set()

    for cell in range(grid_cell_count):
        if len(selected) >= max_rois:
            break
        for candidate in by_cell.get(cell, []):
            if _try_select(
                candidate,
                selected=selected,
                selected_ids=selected_ids,
                roi_size_px=roi_size_px,
                max_overlap_fraction=max_overlap_fraction,
            ):
                break

    remaining = sorted((candidate for candidate in candidates if candidate.candidate_id not in selected_ids), key=_candidate_sort_key)
    for candidate in remaining:
        if len(selected) >= max_rois or _selected_emitter_count(selected) >= int(target_emitters):
            break
        _try_select(
            candidate,
            selected=selected,
            selected_ids=selected_ids,
            roi_size_px=roi_size_px,
            max_overlap_fraction=max_overlap_fraction,
        )

    return _selection_summary(tuple(selected), candidates, grid_cell_count=grid_cell_count, target_emitters=target_emitters)


def _try_select(
    candidate: ROICandidate,
    *,
    selected: list[ROICandidate],
    selected_ids: set[int],
    roi_size_px: int,
    max_overlap_fraction: float,
) -> bool:
    if int(candidate.candidate_id) in selected_ids:
        return False
    if any(
        roi_overlap_fraction(candidate.origin_xy_px, other.origin_xy_px, roi_size_px=roi_size_px) > float(max_overlap_fraction)
        for other in selected
    ):
        return False
    selected.append(candidate)
    selected_ids.add(int(candidate.candidate_id))
    return True


def _candidate_sort_key(candidate: ROICandidate) -> tuple[float, int, int]:
    return (-float(candidate.quality_score), -int(candidate.emitter_count), int(candidate.candidate_id))


def _mean_probability_score(probabilities: np.ndarray) -> float:
    if probabilities.size == 0:
        return 0.0
    return float(probabilities.mean(dtype=np.float64))


def _candidate_center_from_origin(origin_xy_px: tuple[int, int], *, roi_size_px: int) -> tuple[float, float]:
    return (float(origin_xy_px[0]) + float(roi_size_px) / 2.0, float(origin_xy_px[1]) + float(roi_size_px) / 2.0)


def _context_origin_from_core(
    core_origin_xy_px: tuple[int, int],
    *,
    domain_width_px: int,
    domain_height_px: int,
    roi_size_px: int,
    valid_core_size_px: int,
) -> tuple[int, int]:
    margin_x = max(0, (int(roi_size_px) - int(valid_core_size_px)) // 2)
    margin_y = margin_x
    max_x0 = max(0, int(domain_width_px) - int(roi_size_px))
    max_y0 = max(0, int(domain_height_px) - int(roi_size_px))
    x0 = max(0, min(int(core_origin_xy_px[0]) - margin_x, max_x0))
    y0 = max(0, min(int(core_origin_xy_px[1]) - margin_y, max_y0))
    return int(x0), int(y0)


def _selected_emitter_count(candidates: Sequence[ROICandidate]) -> int:
    return int(sum(int(candidate.emitter_count) for candidate in candidates))


def _selection_summary(
    selected: tuple[ROICandidate, ...],
    candidates: Sequence[ROICandidate],
    *,
    grid_cell_count: int,
    target_emitters: int,
) -> FOVSelection:
    selected_cells = {int(candidate.grid_cell_id) for candidate in selected if 0 <= int(candidate.grid_cell_id) < int(grid_cell_count)}
    nonempty_cells = {int(candidate.grid_cell_id) for candidate in candidates if 0 <= int(candidate.grid_cell_id) < int(grid_cell_count)}
    selected_emitters = _selected_emitter_count(selected)
    summary = {
        "selected_roi_count": int(len(selected)),
        "selected_emitter_count": int(selected_emitters),
        "selected_grid_cell_count": int(len(selected_cells)),
        "grid_cell_count": int(grid_cell_count),
        "skipped_empty_grid_cell_count": int(max(0, int(grid_cell_count) - len(nonempty_cells))),
        "target_emitters": int(target_emitters),
        "target_emitters_reached": bool(selected_emitters >= int(target_emitters)),
        "coverage_fraction": float(len(selected_cells) / max(1, int(grid_cell_count))),
    }
    return FOVSelection(candidates=selected, summary=summary)
