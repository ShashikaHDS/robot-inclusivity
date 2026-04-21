"""Accurate robot-reachability: rotation-aware footprint fitting with
differential-drive motion.

The classical inflation-based approach (dilate obstacles by the robot's
half-footprint and flood-fill) treats the robot as axis-aligned and lets
it move holonomically. That over-reports unreachable area for rectangular
robots: a 40×70 cm robot looks 70×70 to the algorithm, so it fails to
squeeze through any gap narrower than 70 cm — even diagonally.

This module computes reachability with:

  * **Rotation**: the footprint is rasterised at N discrete orientations
    (default 8, every 45°). For each orientation we precompute the set of
    cells where placing the robot's center would collide with any
    obstacle. That's an (N × H × W) collision cube.
  * **Differential-drive motion**: state = (row, col, θ_idx). Allowed
    transitions are:
        - rotate in place by ±Δθ (one quantisation step)
        - translate one cell forward along the current heading
    No instant-omnidirectional strafing. Robot must face the direction
    it wants to move.
  * **BFS** over (N × H × W) states from the seed mask. A cell is marked
    "reachable" if *any* orientation was reachable at that cell.

All pure numpy + stdlib deque — no new dependencies.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.ndimage import binary_dilation as _scipy_dilate
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# 8 discrete orientations aligned with the 8-neighbour grid directions.
# θ_k = 2π k / 8. Each orientation has an integer (dy, dx) forward step.
_ORIENT_MOVES = [
    ( 0,  1),   # 0°   — east
    ( 1,  1),   # 45°  — north-east
    ( 1,  0),   # 90°  — north
    ( 1, -1),   # 135° — north-west
    ( 0, -1),   # 180° — west
    (-1, -1),   # 225° — south-west
    (-1,  0),   # 270° — south
    (-1,  1),   # 315° — south-east
]
_N_ORIENTS = len(_ORIENT_MOVES)


def _rasterise_rect_footprint(
    half_w_m: float,
    half_l_m: float,
    theta_rad: float,
    resolution: float,
) -> np.ndarray:
    """Return a small uint8 mask (minimum bounding box) where 1 = footprint
    occupies that cell when placed at the origin with heading theta_rad.

    half_w_m: half the short-side dimension (perpendicular to heading)
    half_l_m: half the long-side dimension  (along heading)
    """
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    # Rectangle corners in robot frame: (±half_l, ±half_w)
    corners = np.array([
        (+half_l_m, +half_w_m),
        (+half_l_m, -half_w_m),
        (-half_l_m, -half_w_m),
        (-half_l_m, +half_w_m),
    ], dtype=np.float32)
    # Rotate into world frame
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    world = corners @ rot.T  # (4, 2) — x, y world-frame offsets from center

    # AABB of rotated rect
    min_xy = world.min(axis=0)
    max_xy = world.max(axis=0)

    # Rasterise on a local grid that spans the AABB with `resolution`-sized cells
    nx = int(math.ceil((max_xy[0] - min_xy[0]) / resolution)) + 2
    ny = int(math.ceil((max_xy[1] - min_xy[1]) / resolution)) + 2
    nx = max(nx, 1); ny = max(ny, 1)

    mask = np.zeros((ny, nx), dtype=np.uint8)
    # For each cell in the local grid, test if its center lies inside the rotated rect
    # Rotated-rect point-inside test: transform cell center back to robot frame
    inv_rot = rot.T  # inverse of rotation = transpose
    # Local grid -> world offsets from center
    xs = (np.arange(nx) + 0.5) * resolution + min_xy[0]  # (nx,)
    ys = (np.arange(ny) + 0.5) * resolution + min_xy[1]  # (ny,)
    XX, YY = np.meshgrid(xs, ys)  # (ny, nx)
    # Stack into (ny, nx, 2) and multiply by inv rotation
    world_pts = np.stack([XX, YY], axis=-1)  # (ny, nx, 2)
    robot_pts = world_pts @ inv_rot.T  # (ny, nx, 2)
    rx = robot_pts[..., 0]
    ry = robot_pts[..., 1]
    inside = (np.abs(rx) <= half_l_m) & (np.abs(ry) <= half_w_m)
    mask[inside] = 1
    return mask


def _rasterise_circle_footprint(radius_m: float, resolution: float) -> np.ndarray:
    r_px = max(1, int(math.ceil(radius_m / resolution)))
    size = 2 * r_px + 1
    y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    mask = (x * x + y * y <= r_px * r_px).astype(np.uint8)
    return mask


def _dilate_with_structure(blocked: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Binary dilation of blocked by structure. Uses scipy if available,
    else a numpy convolution-equivalent."""
    if _HAS_SCIPY:
        # scipy.ndimage.binary_dilation treats non-zero = structuring element
        return _scipy_dilate(blocked.astype(bool), structure=structure.astype(bool)).astype(np.uint8)
    # Fallback — manual stamping
    h, w = blocked.shape
    sh, sw = structure.shape
    out = np.zeros_like(blocked, dtype=np.uint8)
    rows_b, cols_b = np.where(blocked)
    half_h, half_w_s = sh // 2, sw // 2
    for r, c in zip(rows_b.tolist(), cols_b.tolist()):
        y0 = r - half_h; y1 = y0 + sh
        x0 = c - half_w_s; x1 = x0 + sw
        ly0 = max(0, -y0); ly1 = sh - max(0, y1 - h)
        lx0 = max(0, -x0); lx1 = sw - max(0, x1 - w)
        y0c = max(0, y0); y1c = min(h, y1)
        x0c = max(0, x0); x1c = min(w, x1)
        out[y0c:y1c, x0c:x1c] |= structure[ly0:ly1, lx0:lx1]
    return out


def compute_footprint_reachable(
    blocked: np.ndarray,         # (H, W) uint8, 1 = obstacle
    half_w_m: float,             # half short-side
    half_l_m: float,             # half long-side
    footprint_shape: str,        # 'rectangular' | 'circular'
    resolution: float,           # meters per pixel
    start_mask: np.ndarray,      # (H, W) uint8, 1 = seed region (anchor)
    motion_model: str = "differential",  # "differential" or "holonomic"
    logf=None,
) -> Tuple[np.ndarray, dict]:
    """Compute the reachable set considering rotation and diff-drive motion.

    Returns:
        reachable: (H, W) uint8 mask — 1 = robot footprint can visit this cell
        meta: dict with timing/stats.
    """
    log = logf if logf else (lambda m, c="": None)
    H, W = blocked.shape
    assert start_mask.shape == (H, W), "start_mask must match blocked shape"

    # 1. Precompute per-orientation collision masks
    t0 = time.time()
    if footprint_shape == "circular":
        # Circle is rotation-invariant — one dilation suffices. But we still
        # need the orientation dimension for the BFS state; all layers share it.
        structure = _rasterise_circle_footprint(max(half_w_m, half_l_m), resolution)
        circle_collide = _dilate_with_structure(blocked, structure)
        collide_cube = np.broadcast_to(circle_collide[None, :, :], (_N_ORIENTS, H, W))
        # Make a writeable copy (broadcast is read-only)
        collide_cube = np.ascontiguousarray(collide_cube)
    else:  # rectangular
        collide_cube = np.empty((_N_ORIENTS, H, W), dtype=np.uint8)
        for k in range(_N_ORIENTS):
            theta = 2.0 * math.pi * k / _N_ORIENTS
            struct = _rasterise_rect_footprint(half_w_m, half_l_m, theta, resolution)
            collide_cube[k] = _dilate_with_structure(blocked, struct)
    log(f"[footprint] Precomputed {_N_ORIENTS} collision masks in {time.time() - t0:.2f}s", "info")

    # 2. Seed BFS from start cells × compatible orientations
    t0 = time.time()
    visited = np.zeros((_N_ORIENTS, H, W), dtype=bool)
    q = deque()
    seeds_r, seeds_c = np.where(start_mask > 0)
    for r, c in zip(seeds_r.tolist(), seeds_c.tolist()):
        for k in range(_N_ORIENTS):
            if not collide_cube[k, r, c] and not visited[k, r, c]:
                visited[k, r, c] = True
                q.append((r, c, k))

    if not q:
        log("[footprint] No collision-free seed — every orientation collides at the anchor.", "warn")
        return np.zeros((H, W), dtype=np.uint8), {"seeds": 0, "states_expanded": 0}

    log(f"[footprint] Seeded {len(q)} (cell,orient) states from {seeds_r.size} start cells.", "info")

    # 3. BFS
    is_diff = (motion_model == "differential")
    states_expanded = 0
    while q:
        r, c, k = q.popleft()
        states_expanded += 1

        # Rotate ±1
        for dk in (-1, 1):
            nk = (k + dk) % _N_ORIENTS
            if not visited[nk, r, c] and not collide_cube[nk, r, c]:
                visited[nk, r, c] = True
                q.append((r, c, nk))

        # Move along heading
        if is_diff:
            # Forward only — backwards would double the branching and the
            # BFS-of-reachability result is symmetric anyway.
            dy, dx = _ORIENT_MOVES[k]
            nr, nc = r + dy, c + dx
            if 0 <= nr < H and 0 <= nc < W:
                if not visited[k, nr, nc] and not collide_cube[k, nr, nc]:
                    visited[k, nr, nc] = True
                    q.append((nr, nc, k))
        else:
            # Holonomic fallback: any of the 8 neighbours at any orientation
            for (dy, dx) in _ORIENT_MOVES:
                nr, nc = r + dy, c + dx
                if 0 <= nr < H and 0 <= nc < W:
                    if not visited[k, nr, nc] and not collide_cube[k, nr, nc]:
                        visited[k, nr, nc] = True
                        q.append((nr, nc, k))

    log(f"[footprint] BFS visited {states_expanded} states in {time.time() - t0:.2f}s", "info")

    # 4. Project: cell reachable if any orientation reached it
    reachable = visited.any(axis=0).astype(np.uint8)
    meta = {
        "states_expanded": states_expanded,
        "orientations": _N_ORIENTS,
        "reachable_cells": int(reachable.sum()),
        "seed_cells": int(seeds_r.size),
    }
    return reachable, meta
