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


def _rasterise_rect_union(
    half_w_m: float,
    half_l_m: float,
    theta_start: float,
    theta_end: float,
    n_samples: int,
    resolution: float,
    oversample: int = 3,
) -> np.ndarray:
    """Union of rectangle rasters at several angles sampled in [start, end].
    Used for building rotation-sweep masks that approximate the footprint
    traced while the robot pivots between adjacent orientations."""
    sampled = np.linspace(theta_start, theta_end, n_samples)
    rasters = [
        _rasterise_rect_footprint(half_w_m, half_l_m, float(t), resolution, oversample=oversample)
        for t in sampled
    ]
    # Align into a shared bounding box, then OR
    max_h = max(r.shape[0] for r in rasters)
    max_w = max(r.shape[1] for r in rasters)
    merged = np.zeros((max_h, max_w), dtype=np.uint8)
    for r in rasters:
        pad_y = (max_h - r.shape[0]) // 2
        pad_x = (max_w - r.shape[1]) // 2
        merged[pad_y:pad_y + r.shape[0], pad_x:pad_x + r.shape[1]] |= r
    return merged


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
    wall_safety_cells: int = 1,   # thicken obstacles by this many cells
    logf=None,
) -> Tuple[np.ndarray, dict]:
    """Return (reachable_mask, meta).

    reachable_mask is a (H, W) uint8 where 1 marks cells the robot BODY
    sweeps, respecting rotation and diff-drive motion.

    wall_safety_cells: pre-dilate obstacles by this radius before
    collision checks. 1 (default, 5 cm at 0.05 m/px) closes 1-pixel
    gaps and sub-pixel scan artifacts so the rotated footprint can't
    slip through thin walls. Set to 0 to disable.
    """
    log = logf if logf else (lambda m, c="": None)
    H, W = blocked.shape

    # ── Wall safety dilation ───────────────────────────────────────────
    if wall_safety_cells and wall_safety_cells > 0:
        safety_struct = np.ones(
            (2 * wall_safety_cells + 1, 2 * wall_safety_cells + 1), dtype=np.uint8
        )
        blocked = _dilate_with_structure(blocked, safety_struct)
        log(f"[footprint] Wall safety dilation: obstacles thickened by "
            f"{wall_safety_cells} px ({wall_safety_cells * resolution * 100:.1f} cm).", "info")

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
        # Orientation doesn't matter for a circle. Writeable copy.
        collide_cube = np.broadcast_to(collide_single[None, :, :], (_N_ORIENTS, H, W))
        collide_cube = np.ascontiguousarray(collide_cube)
        # Sweep masks collapse to the static mask (no rotation effect)
        sweep_cube = collide_cube.copy()
    else:  # rectangular
        collide_cube = np.empty((_N_ORIENTS, H, W), dtype=np.uint8)
        for k in range(_N_ORIENTS):
            theta = 2.0 * math.pi * k / _N_ORIENTS
            s_elem = _rasterise_rect_footprint(half_w_m, half_l_m, theta, resolution)
            collide_cube[k] = _dilate_with_structure(blocked, s_elem)
        # Sweep mask at index k represents the union of footprints at angles
        # between θ_k and θ_{k+1}, covering a 22.5° rotation. Used to gate
        # rotation transitions from k to k+1 (and in reverse for k to k-1).
        step_rad = 2.0 * math.pi / _N_ORIENTS
        sweep_cube = np.empty_like(collide_cube)
        for k in range(_N_ORIENTS):
            theta_k = 2.0 * math.pi * k / _N_ORIENTS
            swept = _rasterise_rect_union(
                half_w_m, half_l_m, theta_k, theta_k + step_rad,
                n_samples=5, resolution=resolution,
            )
            sweep_cube[k] = _dilate_with_structure(blocked, swept)
    log(f"[footprint] Precomputed {_N_ORIENTS} static + sweep collision masks "
        f"in {time.time() - t0:.2f}s", "info")

    # Seed
    t0 = time.time()
    visited = np.zeros((_N_ORIENTS, H, W), dtype=bool)
    q = deque()
    seeds_r, seeds_c = np.where(start_mask > 0)

    # If none of the requested start cells have a collision-free orientation,
    # nudge the seed to the nearest free cell within a small search radius.
    # This mimics the inflation path's "snap to largest component" behaviour
    # without silently jumping across an obstacle into a sealed room.
    def _has_any_fit(r, c):
        return bool((collide_cube[:, r, c] == 0).any())

    any_fit = any(_has_any_fit(int(r), int(c))
                  for r, c in zip(seeds_r.tolist()[:256], seeds_c.tolist()[:256]))
    if not any_fit and seeds_r.size > 0:
        snapped = False
        for max_r in (3, 6, 12, 24):
            r0 = int(seeds_r[0]); c0 = int(seeds_c[0])
            r_lo = max(0, r0 - max_r); r_hi = min(H, r0 + max_r + 1)
            c_lo = max(0, c0 - max_r); c_hi = min(W, c0 + max_r + 1)
            free_any = (collide_cube[:, r_lo:r_hi, c_lo:c_hi] == 0).any(axis=0)
            if free_any.any():
                # Nearest free cell within window (by Manhattan distance)
                local_r, local_c = np.where(free_any)
                best = np.argmin(np.abs(local_r - (r0 - r_lo)) + np.abs(local_c - (c0 - c_lo)))
                nr = int(r_lo + local_r[best]); nc = int(c_lo + local_c[best])
                seeds_r = np.array([nr]); seeds_c = np.array([nc])
                log(f"[footprint] Start cell snapped from ({r0},{c0}) to ({nr},{nc}) within r≤{max_r}.", "warn")
                snapped = True
                break
        if not snapped:
            log("[footprint] Start cell has no fit and no free cell within 24-px radius.", "warn")

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

        # Rotate ±1 step (±22.5°). Both endpoints must be collision-free AND
        # the swept area between them must be obstacle-free, so the robot
        # can actually perform the rotation in place.
        for dk in (-1, 1):
            nk = (k + dk) % _N_ORIENTS
            # sweep_idx is the "lower-angle" index in the (k, k+1) pair
            sweep_idx = k if dk == 1 else (k - 1) % _N_ORIENTS
            if (not visited[nk, r, c]
                    and not collide_cube[nk, r, c]
                    and not sweep_cube[sweep_idx, r, c]):
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
                    al = (k // _MOVES_STRIDE) * _MOVES_STRIDE
                    dy_dx = _ORIENT_MOVES.get(al)
                if dy_dx is not None:
                    dy, dx = dy_dx
                    nr, nc = r + dy, c + dx
                    if 0 <= nr < H and 0 <= nc < W:
                        ok = not visited[k, nr, nc] and not collide_cube[k, nr, nc]
                        # Corner-cut guard: a diagonal move is a simultaneous
                        # axial motion, so both L-corner cells must also clear.
                        if ok and dy != 0 and dx != 0:
                            if collide_cube[k, r + dy, c] or collide_cube[k, r, c + dx]:
                                ok = False
                        if ok:
                            visited[k, nr, nc] = True
                            q.append((nr, nc, k))
        else:
            # Holonomic: any 8-neighbour, any orientation
            for (dy, dx) in _ORIENT_MOVES.values():
                nr, nc = r + dy, c + dx
                if 0 <= nr < H and 0 <= nc < W:
                    ok = not visited[k, nr, nc] and not collide_cube[k, nr, nc]
                    if ok and dy != 0 and dx != 0:
                        if collide_cube[k, r + dy, c] or collide_cube[k, r, c + dx]:
                            ok = False
                    if ok:
                        visited[k, nr, nc] = True
                        q.append((nr, nc, k))

    log(f"[footprint] BFS visited {states_expanded} states in {time.time() - t0:.2f}s", "info")

    # ── Project to swept area ──────────────────────────────────────────
    # visited[k] marks cells where the robot CENTER can rest at orientation k.
    # The "coverage" a user cares about is the set of cells the robot BODY
    # sweeps across — every footprint cell at every reachable pose.
    # We dilate each visited[k] by the footprint at θ_k and union the results,
    # then intersect with the obstacle-free mask (the footprint at a
    # reachable pose is clear of obstacles by invariant, but this is a
    # safety belt against off-by-one rasterization).
    t0 = time.time()
    swept = np.zeros((H, W), dtype=np.uint8)
    for k in range(_N_ORIENTS):
        if not visited[k].any():
            continue
        if footprint_shape == "circular":
            struct = _rasterise_circle_footprint(max(half_w_m, half_l_m), resolution)
        else:
            theta = 2.0 * math.pi * k / _N_ORIENTS
            struct = _rasterise_rect_footprint(half_w_m, half_l_m, theta, resolution)
        swept |= _dilate_with_structure(visited[k].astype(np.uint8), struct)
    swept &= (blocked == 0).astype(np.uint8)
    log(f"[footprint] Swept-area projection: {time.time() - t0:.2f}s  "
        f"center_cells={int(visited.any(axis=0).sum())} → swept_cells={int(swept.sum())}", "info")

    meta = {
        "states_expanded": states_expanded,
        "orientations": _N_ORIENTS,
        "reachable_cells": int(swept.sum()),
        "center_cells": int(visited.any(axis=0).sum()),
        "seed_cells": int(seeds_r.size),
        "visited_cube": visited,  # (N_orient, H, W) bool — center reachability per orientation
    }
    return swept, meta


def simulate_coverage_path(
    blocked: np.ndarray,
    half_w_m: float,
    half_l_m: float,
    footprint_shape: str,
    resolution: float,
    start_mask: np.ndarray,
    wall_safety_cells: int = 1,
    row_overlap: float = 0.10,
    logf=None,
) -> Tuple[np.ndarray, dict]:
    """Simulate a boustrophedon (zig-zag) coverage sweep and return the
    swept body area plus the traced path.

    The robot runs the accurate footprint BFS first to find the set of
    centers it can actually park the body at; then rows of that mask are
    visited in lawnmower order, striped at (body width * (1 - overlap)).
    Output is the union of the body footprint stamped along the path —
    matches what a real coverage sweep would physically cover.

    row_overlap: fractional overlap between consecutive stripes (0.10 =
    10% overlap to keep coverage continuous against small reach errors).
    """
    log = logf if logf else (lambda m, c="": None)
    H, W = blocked.shape

    reachable, meta = compute_footprint_reachable(
        blocked, half_w_m, half_l_m, footprint_shape, resolution, start_mask,
        motion_model="differential",
        wall_safety_cells=wall_safety_cells,
        logf=log,
    )
    visited_cube = meta.get("visited_cube")
    # visited_cube is (_N_ORIENTS, H, W) — we want centers reachable at EAST (k=0)
    # or WEST (k=8) orientations so the robot can drive horizontally.
    if visited_cube is not None:
        horiz_centers = (visited_cube[0] | visited_cube[_N_ORIENTS // 2]).astype(np.uint8)
    else:
        horiz_centers = reachable

    # Body height in pixels along the perpendicular-to-heading axis (for east/west, that's Y = half_w_m)
    stripe_step_m = max(resolution, 2.0 * half_w_m * (1.0 - max(0.0, min(0.9, row_overlap))))
    stripe_step_px = max(1, int(round(stripe_step_m / resolution)))
    log(f"[coverage] Boustrophedon stripe step = {stripe_step_px} px "
        f"({stripe_step_px * resolution * 100:.1f} cm, {row_overlap * 100:.0f}% overlap).", "info")

    # Pre-compute east-facing footprint to stamp along the path
    if footprint_shape == "circular":
        fp = _rasterise_circle_footprint(max(half_w_m, half_l_m), resolution)
    else:
        fp = _rasterise_rect_footprint(half_w_m, half_l_m, 0.0, resolution)
    fh, fw = fp.shape

    # Find rows with any horizontal-facing reachable center
    row_has_center = horiz_centers.any(axis=1)
    active_rows = np.where(row_has_center)[0]
    if active_rows.size == 0:
        log("[coverage] No horizontally-reachable centers — coverage is empty.", "warn")
        return np.zeros((H, W), dtype=np.uint8), {
            "states_expanded": 0, "orientations": _N_ORIENTS,
            "reachable_cells": 0, "center_cells": 0, "seed_cells": int(start_mask.sum()),
            "path_length": 0, "mode": "coverage_path",
        }

    # Stripe rows, starting from the seed row and expanding outwards
    seed_rows = np.where(start_mask.any(axis=1))[0]
    anchor_row = int(seed_rows[0]) if seed_rows.size > 0 else int(active_rows[0])
    min_r, max_r = int(active_rows[0]), int(active_rows[-1])

    # Build a list of stripe rows centred near the anchor
    stripes = []
    r = anchor_row
    while r <= max_r:
        if r in active_rows:
            stripes.append(r)
        r += stripe_step_px
    r = anchor_row - stripe_step_px
    while r >= min_r:
        if r in active_rows:
            stripes.append(r)
        r -= stripe_step_px
    stripes = sorted(set(stripes))

    # Trace zig-zag: left-to-right, then right-to-left, alternating
    swept = np.zeros((H, W), dtype=np.uint8)
    path = []
    direction = 1
    for row in stripes:
        cols = np.where(horiz_centers[row])[0]
        if cols.size == 0:
            continue
        ordered = cols if direction == 1 else cols[::-1]
        for c in ordered:
            path.append((int(row), int(c)))
            y0 = row - fh // 2; y1 = y0 + fh
            x0 = c - fw // 2;   x1 = x0 + fw
            y0c = max(0, y0); y1c = min(H, y1)
            x0c = max(0, x0); x1c = min(W, x1)
            ly0 = y0c - y0;   lx0 = x0c - x0
            ly1 = ly0 + (y1c - y0c); lx1 = lx0 + (x1c - x0c)
            swept[y0c:y1c, x0c:x1c] |= fp[ly0:ly1, lx0:lx1]
        direction = -direction

    # Clip to obstacle-free (safety belt; should already be clean)
    swept &= (blocked == 0).astype(np.uint8)

    log(f"[coverage] Stripes={len(stripes)}, path={len(path)} waypoints, "
        f"swept_cells={int(swept.sum())}.", "success")
    return swept, {
        "states_expanded": meta.get("states_expanded", 0),
        "orientations": _N_ORIENTS,
        "reachable_cells": int(swept.sum()),
        "center_cells": int(horiz_centers.sum()),
        "seed_cells": int(start_mask.sum()),
        "path_length": len(path),
        "path": path,
        "stripe_step_px": stripe_step_px,
        "mode": "coverage_path",
    }
