"""Ground segmentation, ramp detection, and step detection via iterative RANSAC.

This module implements the v2 traversability approach:
  1. Segment ground into multiple planes (levels) using iterative RANSAC
  2. Detect transitions between levels (ramps and steps)
  3. Assess robot accessibility per transition (slope / step height)
  4. Merge results into the clean obstacle map
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class GroundLevel:
    """A detected ground plane from RANSAC."""
    level_id: int
    plane_normal: np.ndarray       # (3,) unit normal vector
    plane_offset: float            # d in ax+by+cz+d=0
    height_z: float                # average Z of inlier points
    inlier_points: np.ndarray      # (M, 3) ground points
    label: str = ""                # "Level 0", "Level 1", etc.

    def __post_init__(self):
        if not self.label:
            self.label = f"Level {self.level_id}"


@dataclass
class TransitionInfo:
    """A detected transition (ramp or step) between two ground levels."""
    transition_id: int
    type: str                       # "ramp" or "step"
    level_from: int                 # source level index
    level_to: int                   # destination level index
    start_xy: Tuple[float, float]   # world coords (low end)
    end_xy: Tuple[float, float]     # world coords (high end)
    angle_deg: float                # slope angle for ramps
    width_m: float                  # perpendicular extent
    length_m: float                 # along principal direction
    step_height_m: float            # |Z difference| between levels
    height_from: float              # Z of lower level
    height_to: float                # Z of upper level
    cells: np.ndarray               # (N, 2) grid row/col coordinates
    traversable: bool = False       # True if robot can pass


@dataclass
class GroundAnalysisResult:
    """Full result of ground segmentation + transition detection."""
    levels: List[GroundLevel]
    transitions: List[TransitionInfo]
    cell_size: float
    grid_origin: Tuple[float, float]  # (min_x, min_y) of the analysis grid
    grid_shape: Tuple[int, int]       # (height, width) of the analysis grid
    level_grid: Optional[np.ndarray] = None  # 2D grid with level_id per cell (-1 = unassigned)


# ── RANSAC Plane Fitting ──────────────────────────────────────────────────────

def _fit_plane_3pts(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
    """Fit a plane through 3 points. Returns (normal, offset) or None if degenerate."""
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-10:
        return None
    normal = normal / norm
    offset = -np.dot(normal, p1)
    return normal.astype(np.float32), float(offset)


def _ransac_one_plane(
    points: np.ndarray,
    max_iterations: int = 200,
    inlier_threshold: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """Run RANSAC to find the best-fit plane in the point cloud.

    Returns (normal, offset, inlier_mask) or (None, None, None) if no plane found.
    """
    n = points.shape[0]
    if n < 3:
        return None, None, None

    if rng is None:
        rng = np.random.default_rng(42)

    best_count = 0
    best_normal = None
    best_offset = None
    best_mask = None

    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        result = _fit_plane_3pts(points[idx[0]], points[idx[1]], points[idx[2]])
        if result is None:
            continue
        normal, offset = result

        # Distance from all points to the plane
        dists = np.abs(points @ normal + offset)
        inlier_mask = dists <= inlier_threshold
        count = int(inlier_mask.sum())

        if count > best_count:
            best_count = count
            best_normal = normal
            best_offset = offset
            best_mask = inlier_mask

    return best_normal, best_offset, best_mask


def segment_ground_ransac(
    points: np.ndarray,
    max_planes: int = 5,
    inlier_threshold: float = 0.05,
    min_inliers: int = 500,
    max_tilt_deg: float = 30.0,
    iterations_per_plane: int = 300,
    z_band: float = 2.0,
) -> List[GroundLevel]:
    """Segment ground into multiple horizontal planes using iterative RANSAC.

    Parameters
    ----------
    points : (N, 3) float array
    max_planes : maximum number of ground levels to detect
    inlier_threshold : max distance from plane to count as inlier (meters)
    min_inliers : minimum inlier count for a valid plane
    max_tilt_deg : reject planes tilted more than this from horizontal
    iterations_per_plane : RANSAC iterations per plane search
    z_band : pre-filter points to floor_anchor ± z_band meters

    Returns
    -------
    List of GroundLevel, sorted by ascending height_z.
    """
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < min_inliers:
        return []

    # Pre-filter to a reasonable Z-band around the floor
    z_vals = pts[:, 2]
    floor_anchor = float(np.percentile(z_vals, 5.0))
    z_mask = (z_vals >= floor_anchor - z_band) & (z_vals <= floor_anchor + z_band)
    pts_filtered = pts[z_mask]

    if pts_filtered.shape[0] < min_inliers:
        return []

    rng = np.random.default_rng(42)
    remaining = pts_filtered.copy()
    levels: List[GroundLevel] = []
    max_tilt_cos = math.cos(math.radians(max_tilt_deg))

    for level_id in range(max_planes):
        if remaining.shape[0] < min_inliers:
            break

        normal, offset, inlier_mask = _ransac_one_plane(
            remaining,
            max_iterations=iterations_per_plane,
            inlier_threshold=inlier_threshold,
            rng=rng,
        )

        if normal is None or inlier_mask is None:
            break

        inlier_count = int(inlier_mask.sum())
        if inlier_count < min_inliers:
            break

        # Check the plane is near-horizontal (normal Z-component close to ±1)
        if abs(float(normal[2])) < max_tilt_cos:
            # Too tilted — skip and remove these points to avoid re-finding
            remaining = remaining[~inlier_mask]
            continue

        # Ensure normal points upward
        if normal[2] < 0:
            normal = -normal
            offset = -offset

        inlier_pts = remaining[inlier_mask]
        height_z = float(np.mean(inlier_pts[:, 2]))

        levels.append(GroundLevel(
            level_id=len(levels),
            plane_normal=normal,
            plane_offset=offset,
            height_z=height_z,
            inlier_points=inlier_pts,
        ))

        # Remove inliers from remaining for next iteration
        remaining = remaining[~inlier_mask]

    # Sort by ascending height
    levels.sort(key=lambda lv: lv.height_z)
    for i, lv in enumerate(levels):
        lv.level_id = i
        lv.label = f"Level {i}"

    return levels


# ── 2D Grid Helpers ───────────────────────────────────────────────────────────

def _build_level_grid(
    levels: List[GroundLevel],
    cell_size: float,
    padding_m: float = 0.5,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Project all level points onto a 2D grid, assigning each cell to a level.

    Returns (level_grid, (min_x, min_y), (height, width)).
    level_grid has shape (height, width) with values = level_id or -1 for unassigned.
    """
    # Gather all points to determine grid bounds
    all_pts = np.concatenate([lv.inlier_points for lv in levels], axis=0)
    xy = all_pts[:, :2]
    min_xy = xy.min(axis=0) - padding_m
    max_xy = xy.max(axis=0) + padding_m
    width = max(1, int(math.ceil((max_xy[0] - min_xy[0]) / cell_size)) + 1)
    height = max(1, int(math.ceil((max_xy[1] - min_xy[1]) / cell_size)) + 1)

    # Initialize with -1 (unassigned)
    level_grid = np.full((height, width), -1, dtype=np.int32)
    height_grid = np.full((height, width), np.nan, dtype=np.float32)
    count_grid = np.zeros((height, width), dtype=np.int32)

    for lv in levels:
        pts = lv.inlier_points
        gx = np.floor((pts[:, 0] - min_xy[0]) / cell_size).astype(np.int32)
        gy = np.floor((pts[:, 1] - min_xy[1]) / cell_size).astype(np.int32)
        valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
        gx = gx[valid]
        gy = gy[valid]

        # For each cell, assign to the level with most points
        for i in range(len(gx)):
            r, c = int(gy[i]), int(gx[i])
            count_grid[r, c] += 1
            # Simple: last-wins per level, but we track counts to pick majority
            if level_grid[r, c] == -1 or level_grid[r, c] == lv.level_id:
                level_grid[r, c] = lv.level_id
                height_grid[r, c] = lv.height_z

    return level_grid, (float(min_xy[0]), float(min_xy[1])), (height, width)


def _build_height_grid(
    levels: List[GroundLevel],
    cell_size: float,
    origin: Tuple[float, float],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Build a per-cell ground height grid from level inlier points.

    Returns (height, width) float32 array with NaN for empty cells.
    """
    h, w = shape
    min_x, min_y = origin
    height_grid = np.full((h, w), np.nan, dtype=np.float32)
    count_grid = np.zeros((h, w), dtype=np.int32)
    sum_grid = np.zeros((h, w), dtype=np.float64)

    for lv in levels:
        pts = lv.inlier_points
        gx = np.floor((pts[:, 0] - min_x) / cell_size).astype(np.int32)
        gy = np.floor((pts[:, 1] - min_y) / cell_size).astype(np.int32)
        valid = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        gx, gy, zz = gx[valid], gy[valid], pts[valid, 2]
        np.add.at(sum_grid, (gy, gx), zz)
        np.add.at(count_grid, (gy, gx), 1)

    good = count_grid > 0
    height_grid[good] = (sum_grid[good] / count_grid[good]).astype(np.float32)
    return height_grid


# ── Transition Detection ──────────────────────────────────────────────────────

def _find_adjacent_level_pairs(
    level_grid: np.ndarray,
) -> List[Tuple[int, int]]:
    """Find pairs of level IDs that are spatially adjacent (4-connected)."""
    h, w = level_grid.shape
    pairs = set()
    for dr, dc in ((0, 1), (1, 0)):
        r1 = level_grid[:h - dr if dr else h, :w - dc if dc else w]
        r2 = level_grid[dr:, dc:]
        mask = (r1 >= 0) & (r2 >= 0) & (r1 != r2)
        if mask.any():
            a = r1[mask]
            b = r2[mask]
            for ai, bi in zip(a.tolist(), b.tolist()):
                pairs.add((min(ai, bi), max(ai, bi)))
    return sorted(pairs)


def _find_transition_zone(
    level_grid: np.ndarray,
    height_grid: np.ndarray,
    level_a: GroundLevel,
    level_b: GroundLevel,
    cell_size: float,
    origin: Tuple[float, float],
    min_cells: int = 5,
) -> List[TransitionInfo]:
    """Find ramp and step zones between two adjacent levels.

    Instead of looking at the entire plane boundary, this finds connected clusters
    of cells where the actual height transitions between the two level heights.
    Each cluster becomes a separate transition (ramp or step).
    """
    h, w = level_grid.shape
    lo_z = min(level_a.height_z, level_b.height_z)
    hi_z = max(level_a.height_z, level_b.height_z)
    height_diff = hi_z - lo_z
    margin = 0.03  # 3cm margin around level heights
    min_x, min_y = origin

    if height_diff < 0.02:
        return []  # Levels too close — no meaningful transition

    # ── Find transition cells ──
    # Method: cells adjacent to BOTH levels, or cells with height between the two levels.
    is_a = level_grid == level_a.level_id
    is_b = level_grid == level_b.level_id

    # Dilate both levels by 2 cells to find the overlap zone
    def dilate(mask, radius=2):
        m = mask.astype(np.uint8)
        for _ in range(radius):
            padded = np.pad(m, 1, mode="constant", constant_values=0)
            m = (
                padded[1:h+1, 1:w+1] | padded[0:h, 1:w+1] | padded[2:h+2, 1:w+1] |
                padded[1:h+1, 0:w] | padded[1:h+1, 2:w+2]
            ).astype(np.uint8)
        return m.astype(bool)

    near_a = dilate(is_a, radius=3)
    near_b = dilate(is_b, radius=3)
    overlap_zone = near_a & near_b  # Zone between both levels

    # Also include cells with intermediate heights (actual ramp surface)
    valid_h = ~np.isnan(height_grid)
    intermediate = valid_h & (height_grid > lo_z + margin) & (height_grid < hi_z - margin)

    # Transition zone = overlap between levels OR intermediate-height cells near both levels
    transition_mask = (overlap_zone & valid_h) | (intermediate & (near_a | near_b))

    # Exclude cells already solidly assigned to a level (height very close to level height)
    on_level_a = valid_h & (np.abs(height_grid - level_a.height_z) < margin)
    on_level_b = valid_h & (np.abs(height_grid - level_b.height_z) < margin)
    transition_mask = transition_mask & ~on_level_a & ~on_level_b

    # Also include the narrow boundary strip (1-cell thick border between levels)
    padded_b = np.pad(is_b.astype(np.uint8), 1, mode="constant", constant_values=0)
    border_ab = is_a & (
        padded_b[1:h+1, 1:w+1] | padded_b[0:h, 1:w+1] | padded_b[2:h+2, 1:w+1] |
        padded_b[1:h+1, 0:w] | padded_b[1:h+1, 2:w+2]
    ).astype(bool)
    padded_a = np.pad(is_a.astype(np.uint8), 1, mode="constant", constant_values=0)
    border_ba = is_b & (
        padded_a[1:h+1, 1:w+1] | padded_a[0:h, 1:w+1] | padded_a[2:h+2, 1:w+1] |
        padded_a[1:h+1, 0:w] | padded_a[1:h+1, 2:w+2]
    ).astype(bool)
    transition_mask = transition_mask | border_ab | border_ba

    if transition_mask.sum() < min_cells:
        return []

    # ── Connected component analysis on the transition zone ──
    rows_all, cols_all = np.where(transition_mask)
    if len(rows_all) == 0:
        return []

    # BFS to find connected clusters
    seen = np.zeros((h, w), dtype=bool)
    clusters: List[List[Tuple[int, int]]] = []
    for i in range(len(rows_all)):
        r0, c0 = int(rows_all[i]), int(cols_all[i])
        if seen[r0, c0]:
            continue
        queue = deque([(r0, c0)])
        seen[r0, c0] = True
        comp = [(r0, c0)]
        while queue:
            rr, cc = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and transition_mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
                    comp.append((nr, nc))
        if len(comp) >= min_cells:
            clusters.append(comp)

    # ── Classify each cluster as ramp or step ──
    transitions: List[TransitionInfo] = []
    for comp in clusters:
        cells = np.array(comp, dtype=np.int32)
        rows = cells[:, 0]
        cols = cells[:, 1]
        world_x = cols.astype(np.float32) * cell_size + min_x + cell_size * 0.5
        world_y = rows.astype(np.float32) * cell_size + min_y + cell_size * 0.5
        cell_heights = height_grid[rows, cols]
        valid = ~np.isnan(cell_heights)

        if valid.sum() < 3:
            continue

        ch = cell_heights[valid]
        z_range = float(ch.max() - ch.min())

        # Check for gradual height variation (ramp) vs abrupt (step)
        # Ramp: height varies significantly across the cluster
        is_ramp = z_range > 0.05

        if is_ramp:
            xy = np.column_stack((world_x[valid], world_y[valid]))
            center = xy.mean(axis=0)
            centered = xy - center

            # PCA for principal direction
            if centered.shape[0] >= 2 and np.ptp(centered, axis=0).max() > 0.01:
                cov = np.cov(centered.T)
                if cov.ndim == 2 and np.isfinite(cov).all():
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    principal = eigvecs[:, -1]
                else:
                    principal = np.array([1.0, 0.0])
            else:
                principal = np.array([1.0, 0.0])

            proj = centered @ principal
            min_idx = int(np.argmin(proj))
            max_idx = int(np.argmax(proj))

            start_xy = (float(xy[min_idx, 0]), float(xy[min_idx, 1]))
            end_xy = (float(xy[max_idx, 0]), float(xy[max_idx, 1]))

            length_m = max(float(np.ptp(proj)), cell_size)
            perp = np.array([-principal[1], principal[0]])
            width_m = max(float(np.ptp(centered @ perp)), cell_size)

            # Ramp angle from height range over horizontal length
            angle_deg = float(np.degrees(np.arctan2(z_range, length_m)))

            # Real ramps are longer than 1m and wider than 0.8m — skip noise
            if length_m < 1.0 or width_m < 0.8:
                continue

            # Ensure start is the low end
            if ch[min_idx] > ch[max_idx]:
                start_xy, end_xy = end_xy, start_xy

            transitions.append(TransitionInfo(
                transition_id=0,
                type="ramp",
                level_from=level_a.level_id,
                level_to=level_b.level_id,
                start_xy=start_xy,
                end_xy=end_xy,
                angle_deg=angle_deg,
                width_m=width_m,
                length_m=length_m,
                step_height_m=height_diff,
                height_from=lo_z,
                height_to=hi_z,
                cells=cells,
            ))
        else:
            # Step: abrupt change — must be at least 0.8m wide to matter
            edge_w = float(len(comp)) * cell_size
            if edge_w < 0.8:
                continue
            cx = float(np.mean(world_x[valid]))
            cy = float(np.mean(world_y[valid]))
            transitions.append(TransitionInfo(
                transition_id=0,
                type="step",
                level_from=level_a.level_id,
                level_to=level_b.level_id,
                start_xy=(cx, cy),
                end_xy=(cx, cy),
                angle_deg=90.0,
                width_m=float(len(comp)) * cell_size,
                length_m=0.0,
                step_height_m=height_diff,
                height_from=lo_z,
                height_to=hi_z,
                cells=cells,
            ))

    return transitions


def detect_transitions(
    levels: List[GroundLevel],
    cell_size: float = 0.20,
    min_ramp_slope_deg: float = 3.0,
    min_boundary_cells: int = 25,
) -> GroundAnalysisResult:
    """Detect ramp and step transitions between ground levels.

    Parameters
    ----------
    levels : list of GroundLevel from segment_ground_ransac()
    cell_size : grid cell size in meters
    min_ramp_slope_deg : minimum slope to classify as ramp (below = flat)
    min_boundary_cells : minimum cells for a valid transition cluster

    Returns
    -------
    GroundAnalysisResult with levels, transitions, and grid metadata.
    """
    if len(levels) == 0:
        return GroundAnalysisResult(
            levels=[], transitions=[], cell_size=cell_size,
            grid_origin=(0.0, 0.0), grid_shape=(0, 0),
        )

    # Build grids
    level_grid, origin, shape = _build_level_grid(levels, cell_size)
    height_grid = _build_height_grid(levels, cell_size, origin, shape)

    # Find adjacent level pairs and detect transitions for each
    pairs = _find_adjacent_level_pairs(level_grid)

    all_transitions: List[TransitionInfo] = []
    tid = 0

    for level_a_id, level_b_id in pairs:
        lv_a = levels[level_a_id]
        lv_b = levels[level_b_id]

        cluster_transitions = _find_transition_zone(
            level_grid, height_grid,
            lv_a, lv_b,
            cell_size, origin,
            min_cells=max(min_boundary_cells, 25),
        )

        for t in cluster_transitions:
            t.transition_id = tid
            tid += 1
            all_transitions.append(t)

    return GroundAnalysisResult(
        levels=levels,
        transitions=all_transitions,
        cell_size=cell_size,
        grid_origin=origin,
        grid_shape=shape,
        level_grid=level_grid,
    )


# ── Accessibility Assessment ──────────────────────────────────────────────────

def assess_transitions(
    result: GroundAnalysisResult,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
) -> GroundAnalysisResult:
    """Mark each transition as traversable or not based on robot capabilities."""
    for t in result.transitions:
        if t.type == "ramp":
            t.traversable = t.angle_deg <= max_slope_deg
        elif t.type == "step":
            t.traversable = t.step_height_m <= max_step_m
        else:
            t.traversable = False
    return result


def apply_transitions_to_obstacle_map(
    obstacle_grid: np.ndarray,
    result: GroundAnalysisResult,
    map_resolution: float,
    map_origin: Tuple[float, float],
) -> np.ndarray:
    """Block non-traversable transitions on the obstacle map.

    Parameters
    ----------
    obstacle_grid : 2D uint8 array (0=obstacle, 254=free, 205=unknown)
    result : GroundAnalysisResult with assessed transitions
    map_resolution : obstacle map meters per pixel
    map_origin : obstacle map origin (x, y) in world coords

    Returns
    -------
    Modified obstacle grid (copy) with non-traversable transitions blocked.
    """
    grid = obstacle_grid.copy()
    h, w = grid.shape
    ox, oy = map_origin

    for t in result.transitions:
        if t.traversable:
            continue  # Robot can pass — leave as-is

        # Convert transition cells from analysis grid to obstacle map pixels
        analysis_origin = result.grid_origin
        analysis_cell = result.cell_size

        for ci in range(t.cells.shape[0]):
            row_a, col_a = int(t.cells[ci, 0]), int(t.cells[ci, 1])
            # World coords of this analysis cell center
            wx = analysis_origin[0] + (col_a + 0.5) * analysis_cell
            wy = analysis_origin[1] + (row_a + 0.5) * analysis_cell
            # Convert to obstacle map pixel
            px = int((wx - ox) / map_resolution)
            py = int((wy - oy) / map_resolution)
            if 0 <= px < w and 0 <= py < h:
                grid[py, px] = 0  # Mark as obstacle

    return grid


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_accessibility_report(result: GroundAnalysisResult) -> str:
    """Generate a human-readable accessibility report."""
    lines = []
    lines.append(f"Ground Levels Detected: {len(result.levels)}")
    for lv in result.levels:
        n = lv.inlier_points.shape[0]
        lines.append(f"  {lv.label}: height={lv.height_z:.3f}m ({n:,} points)")

    lines.append(f"\nTransitions Detected: {len(result.transitions)}")
    for t in result.transitions:
        status = "PASS" if t.traversable else "FAIL"
        if t.type == "ramp":
            lines.append(
                f"  [{status}] Ramp (Level {t.level_from} -> Level {t.level_to}): "
                f"angle={t.angle_deg:.1f} deg, length={t.length_m:.2f}m, "
                f"width={t.width_m:.2f}m, height_diff={t.step_height_m:.3f}m"
            )
        else:
            lines.append(
                f"  [{status}] Step (Level {t.level_from} -> Level {t.level_to}): "
                f"height={t.step_height_m:.3f}m, edge_width={t.width_m:.2f}m"
            )

    pass_count = sum(1 for t in result.transitions if t.traversable)
    total = len(result.transitions)
    lines.append(f"\nAccessibility: {pass_count}/{total} transitions passable")

    return "\n".join(lines)


# ── Convenience: Full Pipeline ────────────────────────────────────────────────

def run_ground_analysis(
    points: np.ndarray,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
    cell_size: float = 0.20,
    max_planes: int = 5,
    inlier_threshold: float = 0.05,
    min_inliers: int = 500,
    log: callable = None,
) -> GroundAnalysisResult:
    """Run the full ground analysis pipeline: segment → detect → assess.

    Parameters
    ----------
    points : (N, 3) point cloud
    max_slope_deg : robot maximum traversable slope
    max_step_m : robot maximum climbable step height
    cell_size : analysis grid cell size
    max_planes : max ground levels to detect
    inlier_threshold : RANSAC inlier distance
    min_inliers : minimum points for a valid plane
    log : optional logging callback(message, level)
    """
    if log is None:
        log = lambda msg, lvl="info": None

    log(f"[Ground] Starting RANSAC segmentation (max_planes={max_planes})...", "info")
    levels = segment_ground_ransac(
        points,
        max_planes=max_planes,
        inlier_threshold=inlier_threshold,
        min_inliers=min_inliers,
    )
    log(f"[Ground] Found {len(levels)} ground level(s)", "info")
    for lv in levels:
        log(f"[Ground]   {lv.label}: z={lv.height_z:.3f}m, {lv.inlier_points.shape[0]:,} pts", "info")

    log("[Ground] Detecting transitions...", "info")
    result = detect_transitions(levels, cell_size=cell_size)
    log(f"[Ground] Found {len(result.transitions)} transition(s)", "info")

    log(f"[Ground] Assessing accessibility (max_slope={max_slope_deg} deg, max_step={max_step_m}m)...", "info")
    result = assess_transitions(result, max_slope_deg=max_slope_deg, max_step_m=max_step_m)

    for t in result.transitions:
        status = "PASS" if t.traversable else "FAIL"
        if t.type == "ramp":
            log(f"[Ground]   [{status}] Ramp: {t.angle_deg:.1f} deg, {t.length_m:.1f}m long", "info")
        else:
            log(f"[Ground]   [{status}] Step: {t.step_height_m:.3f}m", "info")

    return result
