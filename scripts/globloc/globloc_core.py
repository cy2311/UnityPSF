from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def roi_start_xy(x: float, y: float, roi_size: int) -> tuple[int, int]:
    half = int(roi_size) // 2
    return int(np.floor(float(x))) - half, int(np.floor(float(y))) - half


def build_dT_all(left_xy: np.ndarray, right_xy: np.ndarray) -> np.ndarray:
    left = np.asarray(left_xy, dtype=np.float32)
    right = np.asarray(right_xy, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError(f"expected matching (N, 2) arrays, got {left.shape} and {right.shape}")
    dts = np.zeros((left.shape[0], 4, 5), dtype=np.float32)
    dts[:, 1, :2] = left - right
    dts[:, 2:, :] = 1.0
    return dts


def match_frame_candidates(
    left_xy: np.ndarray, right_xy: np.ndarray, max_distance: float
) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(left_xy, dtype=np.float32)
    right = np.asarray(right_xy, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 2 or right.shape[1] != 2:
        raise ValueError(f"expected (N, 2) arrays, got {left.shape} and {right.shape}")
    if not len(left) or not len(right):
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)

    tree = cKDTree(right)
    candidate_pairs: list[tuple[float, int, int]] = []
    for left_index, neighbors in enumerate(tree.query_ball_point(left, float(max_distance))):
        for right_index in neighbors:
            distance = float(np.linalg.norm(left[left_index] - right[right_index]))
            candidate_pairs.append((distance, left_index, right_index))

    candidate_pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    accepted: list[tuple[int, int, float]] = []
    for distance, left_index, right_index in candidate_pairs:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        accepted.append((left_index, right_index, distance))

    accepted.sort(key=lambda item: item[0])
    if not accepted:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)
    pairs = np.asarray([(left_index, right_index) for left_index, right_index, _ in accepted], dtype=np.int64)
    distances = np.asarray([distance for _, _, distance in accepted], dtype=np.float32)
    return pairs, distances
