"""Identify narrow corridor gaps that the robot's inflated footprint cannot
cross, and propose minimal wall-trim actions to widen them just enough.

A *narrow gap* is a passage that's physically wide enough but gets closed by
the half-footprint inflation. The two masks in every coverage result tell us
exactly where:

    inflation_only = (blocked == 1) AND (sourceBlocked == 0)

Connected components of `inflation_only` that touch BOTH a covered region
and an uncovered floor region are candidate narrow gaps — the robot would
get through if the wall on each side were trimmed slightly.

For each candidate we measure the throat width (distance transform from
the raw obstacle map), compute the required trim amount, locate the two
walls on either side via PCA-derived perpendicular rays, simulate the
widening (flip the trim cells in `sourceBlocked` and re-inflate), and
estimate the area unlocked.

Used by Step 4's *Optimize Layout* — corridor-widening moves compete with
object-relocation moves in the greedy loop, ranked by estimated unlock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


try:
    from scipy.ndimage import (
        label as _scipy_label,
        distance_transform_edt as _dte,
        binary_dilation as _bd,
    )
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class NarrowGap:
    """A diagnosed corridor that's too narrow for the inflated robot footprint."""
    id: int
    throat_rc: Tuple[int, int]
    throat_world: Tuple[float, float]
    current_width_m: float
    required_width_m: float
    needed_widening_m: float
    trim_cells: np.ndarray            # (N, 2) [row, col] cells to flip free
    estimated_gain_m2: float
    description: str


def find_narrow_gaps(
    act_result,
    yaml_data=None,
    *,
    min_unlock_m2: float = 0.5,
    max_widening_m: float = 1.0,
    min_widening_m: float = 0.05,
    top_n: int = 10,
    component_dilation: int = 2,
) -> List[NarrowGap]:
    """Return narrow-gap candidates ranked by estimated unlocked area.

    Parameters
    ----------
    act_result : Step-3 actual coverage result (must include 'blocked',
        'sourceBlocked', 'covPx', 'floorPx', 'params', 'resolution', w/h).
    yaml_data : optional, used for world-coordinate origin. Falls back to
        act_result['origin'] when None.
    min_unlock_m2 : drop gaps whose simulated unlock area is below this
        (filters out the harmless wall-edge halo that surrounds every wall).
    max_widening_m : drop gaps that need more than this much widening —
        beyond that it's not a "tweak the wall" fix, it's "knock the wall
        down"; emit `add_access` instead.
    min_widening_m : drop gaps already wider than required by less than this
        margin (rounding noise from pixelation).
    top_n : return at most this many candidates.
    component_dilation : expand each connected component by N cells when
        checking whether it touches a covered and an uncovered region.
    """
    if not _HAS_SCIPY:
        return []

    w = int(act_result["w"])
    h = int(act_result["h"])
    res = float(act_result.get("resolution", 0.05))
    cell_area = res * res
    params = act_result.get("params", {}) or {}

    blocked = np.asarray(act_result["blocked"], dtype=np.uint8).reshape(h, w)
    source_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]), dtype=np.uint8
    ).reshape(h, w)
    cov_px = np.asarray(act_result["covPx"], dtype=np.uint8).reshape(h, w)
    floor_px_arr = act_result.get("floorPx")
    if floor_px_arr is None:
        floor_px = (source_blocked == 0).astype(np.uint8)
    else:
        floor_px = np.asarray(floor_px_arr, dtype=np.uint8).reshape(h, w)

    # 1. Inflation-only halo: cells the robot can't enter, but the physical
    #    obstacle map says they're free.
    inflation_only = ((blocked == 1) & (source_blocked == 0)).astype(np.uint8)
    if not inflation_only.any():
        return []

    labels, n_comp = _scipy_label(inflation_only, structure=np.ones((3, 3)))
    if n_comp == 0:
        return []

    # 2. Distance from each cell to the nearest physical wall (in cells).
    dist_to_wall = _dte(source_blocked == 0)

    # 3. Robot's effective passage width — short side decides whether a gap
    #    is passable in any orientation.
    half_w = float(params.get("halfW", params.get("radius", 0.35)))
    half_l = float(params.get("halfL", params.get("radius", 0.35)))
    required_full_width_m = 2.0 * min(half_w, half_l)

    # 4. Map origin for world coords
    if yaml_data is not None:
        origin = yaml_data.get("origin", [0.0, 0.0, 0.0])
    else:
        origin = act_result.get("origin", (0.0, 0.0))
    ox = float(origin[0])
    oy = float(origin[1])

    # Avoid scoring tiny inflation-only specks by precomputing component sizes
    comp_sizes = np.bincount(labels.ravel())

    gaps: List[NarrowGap] = []
    for comp_id in range(1, n_comp + 1):
        if comp_sizes[comp_id] < 4:
            continue
        comp_mask = (labels == comp_id)

        # Bridge test: the component must separate covered floor from
        # uncovered floor — otherwise it's an inert edge halo, not a gap.
        dilated = _bd(comp_mask, iterations=component_dilation)
        touches_covered = bool((dilated & (cov_px == 1)).any())
        touches_uncovered = bool((dilated & (cov_px == 0) & (floor_px == 1)).any())
        if not (touches_covered and touches_uncovered):
            continue

        # Throat = narrowest point along this gap
        comp_distances = np.where(comp_mask, dist_to_wall, np.inf)
        throat_flat = int(np.argmin(comp_distances))
        throat_r = throat_flat // w
        throat_c = throat_flat % w
        throat_dist_cells = float(comp_distances[throat_r, throat_c])
        current_full_width_m = 2.0 * throat_dist_cells * res
        needed_widening_m = required_full_width_m - current_full_width_m

        if needed_widening_m < min_widening_m:
            continue
        if needed_widening_m > max_widening_m:
            continue  # too wide a job for a "trim the wall" fix

        # Estimate gap axis via PCA over component cells; perpendicular is
        # the wall-trim direction.
        comp_rows, comp_cols = np.where(comp_mask)
        pts = np.column_stack([comp_rows.astype(np.float32),
                                comp_cols.astype(np.float32)])
        pts_centered = pts - pts.mean(axis=0)
        if pts_centered.shape[0] >= 2:
            cov_mat = np.cov(pts_centered.T)
            if cov_mat.ndim == 0:
                perp = np.array([1.0, 0.0])
            else:
                eigvals, eigvecs = np.linalg.eigh(cov_mat)
                gap_axis = eigvecs[:, int(np.argmax(eigvals))]
                # Perpendicular in 2D
                perp = np.array([-gap_axis[1], gap_axis[0]])
        else:
            perp = np.array([1.0, 0.0])

        trim_per_side_cells = max(1, int(np.ceil(needed_widening_m / 2.0 / res)))

        # Helper: trace from (sr, sc) in `direction` until we hit a wall;
        # return the wall cell (first source_blocked==1 cell on the ray).
        def _trace_to_wall(sr: int, sc: int, direction, max_steps: int = 40):
            dr_step = float(direction[0])
            dc_step = float(direction[1])
            for step in range(1, max_steps + 1):
                rr = int(round(sr + dr_step * step))
                cc = int(round(sc + dc_step * step))
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    return None
                if source_blocked[rr, cc] == 1:
                    return (rr, cc)
            return None

        wall_a = _trace_to_wall(throat_r, throat_c, perp)
        wall_b = _trace_to_wall(throat_r, throat_c, -perp)

        # Collect cells to trim: from each wall, walk INTO the wall by
        # `trim_per_side_cells` cells along the same perpendicular.
        # Also widen along the gap axis by a small radius for a clean opening.
        # (`gap_radius_cells` is half the trim_per_side to keep the opening
        # localised; otherwise we'd carve the whole wall.)
        gap_radius_cells = max(1, trim_per_side_cells // 2 + 1)
        trim_set: set[Tuple[int, int]] = set()
        for wall_pos, direction in (
            (wall_a, perp), (wall_b, -perp),
        ):
            if wall_pos is None:
                continue
            wr, wc = wall_pos
            dr_step, dc_step = direction[0], direction[1]
            for d_perp in range(0, trim_per_side_cells):
                base_r = int(round(wr + dr_step * d_perp))
                base_c = int(round(wc + dc_step * d_perp))
                # Carve along the gap axis as well so the opening matches
                # the corridor's natural width.
                if pts_centered.shape[0] >= 2:
                    g_dr = float(gap_axis[0])
                    g_dc = float(gap_axis[1])
                else:
                    g_dr, g_dc = 0.0, 1.0
                for d_along in range(-gap_radius_cells, gap_radius_cells + 1):
                    tr = int(round(base_r + g_dr * d_along))
                    tc = int(round(base_c + g_dc * d_along))
                    if 0 <= tr < h and 0 <= tc < w and source_blocked[tr, tc] == 1:
                        trim_set.add((tr, tc))

        if not trim_set:
            continue

        trim_cells_arr = np.array(sorted(trim_set), dtype=np.int32)

        # Simulate the widening: flip trim cells in source_blocked, re-inflate,
        # score with the same _quick_reachable_area the optimiser uses.
        sim_sb = source_blocked.copy()
        sim_sb[trim_cells_arr[:, 0], trim_cells_arr[:, 1]] = 0
        from core.semantic_analysis import _quick_reachable_area
        post_cells, _ = _quick_reachable_area(sim_sb, floor_px, params, res)
        pre_cells = int(cov_px.sum())
        estimated_gain = float(post_cells - pre_cells) * cell_area
        if estimated_gain < min_unlock_m2:
            continue

        throat_wx = ox + throat_c * res
        throat_wy = oy + (h - 1 - throat_r) * res

        description = (
            f"Widen {current_full_width_m:.2f} m corridor at "
            f"({throat_wx:.1f}, {throat_wy:.1f}) m to ≥ "
            f"{required_full_width_m:.2f} m  "
            f"(trim {needed_widening_m / 2:.2f} m from each side, "
            f"{trim_cells_arr.shape[0]} cells)"
        )

        gaps.append(NarrowGap(
            id=len(gaps),
            throat_rc=(int(throat_r), int(throat_c)),
            throat_world=(float(throat_wx), float(throat_wy)),
            current_width_m=float(current_full_width_m),
            required_width_m=float(required_full_width_m),
            needed_widening_m=float(needed_widening_m),
            trim_cells=trim_cells_arr,
            estimated_gain_m2=float(estimated_gain),
            description=description,
        ))

    gaps.sort(key=lambda g: -g.estimated_gain_m2)
    return gaps[:top_n]
