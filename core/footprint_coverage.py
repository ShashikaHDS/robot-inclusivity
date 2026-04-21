"""Accurate robot-reachability: rotation-aware footprint fitting with
differential-drive motion.

Why the default inflation path over-reports unreachable area
-----------------------------------------------------------
The classical approach dilates obstacles by the robot's half-footprint
(axis-aligned) and flood-fills. For a rectangular robot that's rotatable
(e.g. 40×70 cm), this treats it as a 70×70 axis-aligned box — so any gap
narrower than 70 cm is reported impassable, even though a rotated robot
could squeeze through.

This module recovers those passages with:

  * **Conservative, supersampled rasterisation** of the rotated
    rectangle. Sub-cell coverage prevents the Bresenham-style
    center-point rasteriser from missing edge cells and producing a
    collision mask that "leaks" the robot through walls.
  * **16 discrete orientations** (every 22.5°). Finer rotation means
    that the "both endpoints clear ⇒ sweep clear" approximation is much
    more reliable; intermediate body poses change only slightly between
    adjacent indices.
  * **Differential-drive motion**: rotate in place (±22.5°) at any
    orientation, forward translate by one cell only from the 8
    orientations aligned with the 8-neighbour grid (0°, 45°, 90°, …).
    Non-aligned orientations (22.5°, 67.5°, …) can only rotate; the
    robot must pivot to an aligned heading before moving.

Pure numpy + stdlib deque — no new dependencies.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Tuple

import numpy as np

try:
    from scipy.ndimage import binary_dilation as _scipy_dilate
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# 16 orientations, every 22.5°. Every other index aligns with the
# 8-neighbour grid directions — only those can take a forward move.
_N_ORIENTS = 16
_MOVES_STRIDE = 2

_ORIENT_MOVES = {}
for _i in range(0, _N_ORIENTS, _MOVES_STRIDE):
    _th = 2.0 * math.pi * _i / _N_ORIENTS
    _ORIENT_MOVES[_i] = (int(round(math.sin(_th))), int(round(math.cos(_th))))


def _rasterise_rect_footprint(
    half_w_m: float,
    half_l_m: float,
    theta_rad: float,
    resolution: float,
    oversample: int = 3,
) -> np.ndarray:
    """Rasterise a rotated rectangle *conservatively* — any cell overlapping
    the rectangle is marked. Uses integer supersampling so the boundary
    is never under-sampled into gaps the robot shouldn't fit through.

    half_w_m: half the short-side dimension (robot-frame Y)
    half_l_m: half the long-side dimension  (robot-frame X, along heading)
    """
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    # Rectangle corners in robot frame: (±half_l, ±half_w)
    corners = np.array([
        (+half_l_m, +half_w_m),
        (+half_l_m, -half_w_m),
        (-half_l_m, -half_w_m),
        (-half_l_m, +half_w_m),
    ], dtype=np.float32)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    world = corners @ rot.T  # (4, 2) world-frame offsets

    # Extra 1-cell margin makes the target grid big enough that the
    # supersampled cells on the far edges don't spill out.
    margin = resolution
    min_xy = world.min(axis=0) - margin
    max_xy = world.max(axis=0) + margin

    nx = max(1, int(math.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1)
    ny = max(1, int(math.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1)

    # Supersample
    sub_res = resolution / oversample
    n_sub_x = nx * oversample
    n_sub_y = ny * oversample
    xs_fine = (np.arange(n_sub_x) + 0.5) * sub_res + min_xy[0]
    ys_fine = (np.arange(n_sub_y) + 0.5) * sub_res + min_xy[1]
    XX, YY = np.meshgrid(xs_fine, ys_fine)
    world_pts = np.stack([XX, YY], axis=-1)  # (n_sub_y, n_sub_x, 2)

    # Transform world points into robot frame: robot_row = world_row @ R
    robot_pts = world_pts @ rot
    rx = robot_pts[..., 0]
    ry = robot_pts[..., 1]
    inside_fine = (np.abs(rx) <= half_l_m) & (np.abs(ry) <= half_w_m)
    # A coarse cell is marked if ANY of its sub-cells is inside
    inside = inside_fine.reshape(ny, oversample, nx, oversample).any(axis=(1, 3))
    return inside.astype(np.uint8)


def _rasterise_circle_footprint(radius_m: float, resolution: float) -> np.ndarray:
    # Add half a cell of margin so we never cut the corner cell in two.
    r_px = max(1, int(math.ceil(radius_m / resolution)) + 1)
    size = 2 * r_px + 1
    y, x = np.ogrid[-r_px:r_px + 1, -r_px:r_px + 1]
    # Compare squared-radius in *meters* so a partial-cell boundary is
    # included (conservative).
    rr = (radius_m + 0.5 * resolution) ** 2
    dist2 = (x * resolution) ** 2 + (y * resolution) ** 2
    return (dist2 <= rr).astype(np.uint8)


def _dilate_with_structure(blocked: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """Binary dilation of `blocked` by `structure`. scipy if available,
    else a numpy stamp loop."""
    if _HAS_SCIPY:
        return _scipy_dilate(blocked.astype(bool), structure=structure.astype(bool)).astype(np.uint8)
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
    blocked: np.ndarray,          # (H, W) uint8, 1 = obstacle
    half_w_m: float,              # half short-side
    half_l_m: float,              # half long-side
    footprint_shape: str,         # 'rectangular' | 'circular'
    resolution: float,            # meters per pixel
    start_mask: np.ndarray,       # (H, W) uint8 or 1D h*w seed
    motion_model: str = "differential",  # 'differential' or 'holonomic'
    logf=None,
) -> Tuple[np.ndarray, dict]:
    """Return (reachable_mask, meta).

    reachable_mask is a (H, W) uint8 where 1 marks cells the robot's
    center can visit, respecting rotation and diff-drive motion.
    """
    log = logf if logf else (lambda m, c="": None)
    H, W = blocked.shape

    # Accept 1D or 2D start mask
    start_mask = np.asarray(start_mask)
    if start_mask.ndim == 1 and start_mask.size == H * W:
        start_mask = start_mask.reshape(H, W)
    if start_mask.shape != (H, W):
        raise ValueError(
            f"start_mask shape {start_mask.shape} != blocked shape {(H, W)}"
        )

    t0 = time.time()
    if footprint_shape == "circular":
        struct = _rasterise_circle_footprint(max(half_w_m, half_l_m), resolution)
        collide_single = _dilate_with_structure(blocked, struct)
        # Broadcast: orientation doesn't matter for a circle. Writeable copy.
        collide_cube = np.broadcast_to(collide_single[None, :, :], (_N_ORIENTS, H, W))
        collide_cube = np.ascontiguousarray(collide_cube)
    else:  # rectangular
        collide_cube = np.empty((_N_ORIENTS, H, W), dtype=np.uint8)
        for k in range(_N_ORIENTS):
            theta = 2.0 * math.pi * k / _N_ORIENTS
            s_elem = _rasterise_rect_footprint(half_w_m, half_l_m, theta, resolution)
            collide_cube[k] = _dilate_with_structure(blocked, s_elem)
    log(f"[footprint] Precomputed {_N_ORIENTS} collision masks in {time.time() - t0:.2f}s", "info")

    # Seed
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
        return np.zeros((H, W), dtype=np.uint8), {
            "seeds": 0, "states_expanded": 0,
            "orientations": _N_ORIENTS,
            "reachable_cells": 0,
            "seed_cells": int(seeds_r.size),
        }

    log(f"[footprint] Seeded {len(q)} (cell,orient) states from {seeds_r.size} start cells.", "info")

    is_diff = (motion_model == "differential")
    circular = (footprint_shape == "circular")
    states_expanded = 0
    while q:
        r, c, k = q.popleft()
        states_expanded += 1

        # Rotate ±1 step (±22.5°). Both endpoints must be collision-free at
        # (r, c) — we're already collision-free at (r, c, k) by invariant,
        # and we explicitly test (r, c, nk).
        for dk in (-1, 1):
            nk = (k + dk) % _N_ORIENTS
            if not visited[nk, r, c] and not collide_cube[nk, r, c]:
                visited[nk, r, c] = True
                q.append((r, c, nk))

        # Forward move
        if is_diff:
            # Only from orientations aligned with 8-neighbour grid
            # (0°, 45°, 90°, …, 315°). The 22.5°-offset orientations
            # (odd k) must pivot before moving.
            if (k % _MOVES_STRIDE) == 0 or circular:
                dy_dx = _ORIENT_MOVES.get(k) if (k in _ORIENT_MOVES) else None
                if dy_dx is None and circular:
                    # Circular: pick nearest aligned move for this orientation
                    al = (k // _MOVES_STRIDE) * _MOVES_STRIDE
                    dy_dx = _ORIENT_MOVES.get(al)
                if dy_dx is not None:
                    dy, dx = dy_dx
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < H and 0 <= nc < W:
                        if not visited[k, nr, nc] and not collide_cube[k, nr, nc]:
                            visited[k, nr, nc] = True
                            q.append((nr, nc, k))
        else:
            # Holonomic: any 8-neighbour, any orientation
            for (dy, dx) in _ORIENT_MOVES.values():
                nr, nc = r + dy, c + dx
                if 0 <= nr < H and 0 <= nc < W:
                    if not visited[k, nr, nc] and not collide_cube[k, nr, nc]:
                        visited[k, nr, nc] = True
                        q.append((nr, nc, k))

    log(f"[footprint] BFS visited {states_expanded} states in {time.time() - t0:.2f}s", "info")

    reachable = visited.any(axis=0).astype(np.uint8)
    meta = {
        "states_expanded": states_expanded,
        "orientations": _N_ORIENTS,
        "reachable_cells": int(reachable.sum()),
        "seed_cells": int(seeds_r.size),
    }
    return reachable, meta
