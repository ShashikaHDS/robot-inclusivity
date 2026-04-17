"""Slope-based ramp detection for robot accessibility assessment.

v2.1 approach — directly detect ramp regions from ground slope:
  1. Build a smoothed ground heightmap from the point cloud
  2. Compute per-cell slope
  3. Find connected regions where slope > threshold → ramp candidates
  4. Filter by size (area, length, width) to remove noise
  5. Measure each ramp's angle, block on obstacle map if too steep
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class TransitionInfo:
    """A detected ramp region."""
    transition_id: int
    type: str                       # "ramp"
    level_from: int                 # unused (kept for compatibility)
    level_to: int                   # unused (kept for compatibility)
    start_xy: Tuple[float, float]   # world coords (low end)
    end_xy: Tuple[float, float]     # world coords (high end)
    angle_deg: float                # measured slope angle
    width_m: float                  # perpendicular extent
    length_m: float                 # along principal direction
    step_height_m: float            # height difference across the ramp
    height_from: float              # Z at low end
    height_to: float                # Z at high end
    cells: np.ndarray               # (N, 2) grid row/col coordinates
    traversable: bool = False       # True if robot can pass


@dataclass
class GroundAnalysisResult:
    """Result of slope-based ramp detection."""
    levels: list                    # empty (kept for compatibility)
    transitions: List[TransitionInfo]
    cell_size: float
    grid_origin: Tuple[float, float]  # (min_x, min_y)
    grid_shape: Tuple[int, int]       # (height, width)
    level_grid: Optional[np.ndarray] = None


# ── Ground Heightmap ──────────────────────────────────────────────────────────

def _build_ground_heightmap(
    points: np.ndarray,
    cell_size: float = 0.20,
    padding_m: float = 0.5,
    ground_percentile: float = 10.0,
    min_points_per_cell: int = 3,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Build a 2D ground heightmap from a point cloud.

    Returns (height_grid, (min_x, min_y), (rows, cols)).
    height_grid has NaN for empty/sparse cells.
    """
    xy = points[:, :2].astype(np.float32)
    min_xy = xy.min(axis=0) - padding_m
    max_xy = xy.max(axis=0) + padding_m
    width = max(1, int(math.ceil((max_xy[0] - min_xy[0]) / cell_size)) + 1)
    height = max(1, int(math.ceil((max_xy[1] - min_xy[1]) / cell_size)) + 1)

    gx = np.floor((xy[:, 0] - min_xy[0]) / cell_size).astype(np.int32)
    gy = np.floor((xy[:, 1] - min_xy[1]) / cell_size).astype(np.int32)
    gx = np.clip(gx, 0, width - 1)
    gy = np.clip(gy, 0, height - 1)

    # Compute ground height as low percentile per cell
    ground = np.full((height, width), np.nan, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)

    linear = gy * width + gx
    order = np.argsort(linear, kind="mergesort")
    linear_s = linear[order]
    z_s = points[order, 2].astype(np.float32)

    uniq, starts, cnts = np.unique(linear_s, return_index=True, return_counts=True)
    for cell_id, start, count in zip(uniq.tolist(), starts.tolist(), cnts.tolist()):
        row = cell_id // width
        col = cell_id % width
        counts[row, col] = count
        if count >= min_points_per_cell:
            zs = z_s[start:start + count]
            ground[row, col] = np.percentile(zs, ground_percentile)

    return ground, (float(min_xy[0]), float(min_xy[1])), (height, width)


# ── Smoothing ─────────────────────────────────────────────────────────────────

def _median_3x3(z: np.ndarray) -> np.ndarray:
    """NaN-aware 3x3 median filter."""
    h, w = z.shape
    padded = np.pad(z.astype(np.float32), 1, mode="constant", constant_values=np.nan)
    stack = np.stack([padded[r:r+h, c:c+w] for r in range(3) for c in range(3)], axis=0)
    with np.errstate(all="ignore"):
        out = np.nanmedian(stack, axis=0).astype(np.float32)
    out[np.isnan(z)] = np.nan
    return out


def _gaussian_smooth(z: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """NaN-aware Gaussian smoothing."""
    radius = int(math.ceil(2.0 * sigma))
    size = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel = np.outer(kernel, kernel).astype(np.float32)

    zz = np.asarray(z, dtype=np.float32)
    valid = (~np.isnan(zz)).astype(np.float32)
    filled = np.where(np.isnan(zz), 0.0, zz).astype(np.float32)

    zp = np.pad(filled, radius, mode="constant", constant_values=0.0)
    vp = np.pad(valid, radius, mode="constant", constant_values=0.0)
    h, w = zz.shape

    wsum = np.zeros((h, w), dtype=np.float64)
    vsum = np.zeros((h, w), dtype=np.float64)
    for dy in range(size):
        for dx in range(size):
            k = float(kernel[dy, dx])
            wsum += k * zp[dy:dy+h, dx:dx+w]
            vsum += k * vp[dy:dy+h, dx:dx+w]

    out = np.full((h, w), np.nan, dtype=np.float32)
    good = vsum > 1e-12
    out[good] = (wsum[good] / vsum[good]).astype(np.float32)
    out[~valid.astype(bool)] = np.nan
    return out


# ── Slope Computation ─────────────────────────────────────────────────────────

def _compute_slope_and_gradient(
    ground: np.ndarray, cell_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-cell slope and gradient from a ground heightmap.

    Returns (slope_deg, dzdx, dzdy).
    """
    # Fill small NaN gaps for gradient computation
    zz = ground.astype(np.float32).copy()
    for _ in range(4):
        nan_mask = np.isnan(zz)
        if not nan_mask.any():
            break
        neighbors = np.stack(
            [np.roll(zz, 1, 0), np.roll(zz, -1, 0),
             np.roll(zz, 1, 1), np.roll(zz, -1, 1)], axis=0)
        valid = ~np.isnan(neighbors)
        counts = valid.sum(axis=0)
        sums = np.nansum(neighbors, axis=0)
        means = np.full_like(zz, np.nan)
        ok = counts > 0
        means[ok] = (sums[ok] / counts[ok]).astype(np.float32)
        fillable = nan_mask & ~np.isnan(means)
        if not fillable.any():
            break
        zz[fillable] = means[fillable]

    dzdy, dzdx = np.gradient(zz, cell_size, cell_size)
    dzdx = dzdx.astype(np.float32)
    dzdy = dzdy.astype(np.float32)
    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)
    # Restore NaN where original was NaN
    nan_orig = np.isnan(ground)
    slope_deg[nan_orig] = np.nan
    dzdx[nan_orig] = np.nan
    dzdy[nan_orig] = np.nan
    return slope_deg, dzdx, dzdy


def _gradient_direction_consistency(
    dzdx: np.ndarray, dzdy: np.ndarray, rows: np.ndarray, cols: np.ndarray,
) -> float:
    """Measure how consistent the gradient direction is within a set of cells.

    Returns a score between 0 (random directions) and 1 (perfectly uniform direction).
    A real ramp has consistent direction (score > 0.5).
    """
    gx = dzdx[rows, cols]
    gy = dzdy[rows, cols]
    valid = np.isfinite(gx) & np.isfinite(gy)
    gx, gy = gx[valid], gy[valid]
    if len(gx) < 3:
        return 0.0

    # Normalize each gradient to unit vector
    mag = np.sqrt(gx**2 + gy**2)
    nonzero = mag > 1e-8
    if nonzero.sum() < 3:
        return 0.0
    gx_n = gx[nonzero] / mag[nonzero]
    gy_n = gy[nonzero] / mag[nonzero]

    # Mean direction vector — its magnitude indicates consistency
    # (1.0 = all same direction, 0.0 = uniformly random)
    mean_x = float(np.mean(gx_n))
    mean_y = float(np.mean(gy_n))
    consistency = math.sqrt(mean_x**2 + mean_y**2)
    return consistency


# ── Ramp Detection ────────────────────────────────────────────────────────────

def detect_ramps_by_slope(
    points: np.ndarray,
    cell_size: float = 0.20,
    min_ramp_slope_deg: float = 5.0,
    min_area_m2: float = 1.0,
    min_length_m: float = 1.0,
    min_width_m: float = 0.5,
    ground_percentile: float = 10.0,
    z_band: float = 1.0,
    log: callable = None,
) -> GroundAnalysisResult:
    """Detect ramp regions by finding connected areas with significant slope.

    Parameters
    ----------
    points : (N, 3) point cloud
    cell_size : grid cell size in meters
    min_ramp_slope_deg : minimum slope to be considered a ramp candidate
    min_area_m2 : minimum area for a valid ramp region
    min_length_m : minimum ramp length
    min_width_m : minimum ramp width
    ground_percentile : percentile for ground height estimation
    z_band : height band above floor anchor for ground points
    log : optional logging callback(msg, level)
    """
    if log is None:
        log = lambda m, l="info": None

    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 100:
        return GroundAnalysisResult(
            levels=[], transitions=[], cell_size=cell_size,
            grid_origin=(0.0, 0.0), grid_shape=(0, 0),
        )

    # Filter to floor band
    floor_anchor = float(np.percentile(pts[:, 2], 5.0))
    z_mask = (pts[:, 2] >= floor_anchor) & (pts[:, 2] <= floor_anchor + z_band)
    floor_pts = pts[z_mask]
    log(f"[Ramp] Floor anchor: {floor_anchor:.3f}m, ground points: {floor_pts.shape[0]:,}", "info")

    if floor_pts.shape[0] < 100:
        return GroundAnalysisResult(
            levels=[], transitions=[], cell_size=cell_size,
            grid_origin=(0.0, 0.0), grid_shape=(0, 0),
        )

    # Build ground heightmap
    log("[Ramp] Building ground heightmap...", "info")
    ground, origin, shape = _build_ground_heightmap(
        floor_pts, cell_size=cell_size, ground_percentile=ground_percentile,
    )
    h, w = shape
    min_x, min_y = origin

    # Smooth
    log("[Ramp] Smoothing heightmap...", "info")
    ground_smooth = _gaussian_smooth(_median_3x3(ground), sigma=2.0)

    # Compute slope and gradient direction
    log("[Ramp] Computing slope...", "info")
    slope_deg, dzdx, dzdy = _compute_slope_and_gradient(ground_smooth, cell_size)

    # Threshold → ramp candidate mask
    valid = ~np.isnan(slope_deg)
    candidate = valid & (slope_deg >= min_ramp_slope_deg)

    n_candidates = int(candidate.sum())
    log(f"[Ramp] Candidate cells (slope >= {min_ramp_slope_deg}°): {n_candidates}", "info")

    if n_candidates == 0:
        return GroundAnalysisResult(
            levels=[], transitions=[], cell_size=cell_size,
            grid_origin=origin, grid_shape=shape,
        )

    # Connected component analysis
    min_cells = max(3, int(min_area_m2 / (cell_size * cell_size)))
    seen = np.zeros((h, w), dtype=bool)
    components: List[List[Tuple[int, int]]] = []

    rows_all, cols_all = np.where(candidate)
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
                if 0 <= nr < h and 0 <= nc < w and candidate[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
                    comp.append((nr, nc))
        if len(comp) >= min_cells:
            components.append(comp)

    log(f"[Ramp] Connected components (>= {min_cells} cells): {len(components)}", "info")

    # Analyze each component → TransitionInfo
    transitions: List[TransitionInfo] = []
    tid = 0

    for comp in components:
        cells = np.array(comp, dtype=np.int32)
        rows = cells[:, 0]
        cols = cells[:, 1]

        # World coordinates
        wx = cols.astype(np.float32) * cell_size + min_x + cell_size * 0.5
        wy = rows.astype(np.float32) * cell_size + min_y + cell_size * 0.5

        # Heights at ramp cells
        cell_h = ground_smooth[rows, cols]
        valid_h = ~np.isnan(cell_h)
        if valid_h.sum() < 3:
            continue
        ch = cell_h[valid_h]
        z_range = float(ch.max() - ch.min())

        # PCA for ramp direction
        xy = np.column_stack((wx[valid_h], wy[valid_h]))
        center = xy.mean(axis=0)
        centered = xy - center

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
        length_m = max(float(np.ptp(proj)), cell_size)
        perp = np.array([-principal[1], principal[0]])
        width_m = max(float(np.ptp(centered @ perp)), cell_size)

        # Filter by size
        area = len(comp) * cell_size * cell_size
        if length_m < min_length_m or width_m < min_width_m:
            continue
        if area < min_area_m2:
            continue

        # Filter by gradient direction consistency
        # Real ramps have consistent slope direction.
        # Large ramps (>5m²) naturally have more gradient variation, so use a lower threshold.
        consistency = _gradient_direction_consistency(dzdx, dzdy, rows, cols)
        consistency_threshold = 0.4 if area > 5.0 else 0.6
        if consistency < consistency_threshold:
            continue

        # Filter by minimum height difference — real ramps have significant rise
        if z_range < 0.15:
            continue

        # Ramp angle: use median per-cell slope (more robust than height_range/length
        # which gets diluted for large components that include flat areas)
        cell_slopes = slope_deg[rows, cols]
        valid_slopes = cell_slopes[~np.isnan(cell_slopes)]
        angle_deg = float(np.median(valid_slopes)) if valid_slopes.size > 0 else 0.0

        # Start/end points (low → high)
        min_idx = int(np.argmin(proj))
        max_idx = int(np.argmax(proj))
        start_xy = (float(xy[min_idx, 0]), float(xy[min_idx, 1]))
        end_xy = (float(xy[max_idx, 0]), float(xy[max_idx, 1]))

        # Ensure start is the low end
        if ch[min_idx] > ch[max_idx]:
            start_xy, end_xy = end_xy, start_xy

        transitions.append(TransitionInfo(
            transition_id=tid,
            type="ramp",
            level_from=0,
            level_to=0,
            start_xy=start_xy,
            end_xy=end_xy,
            angle_deg=angle_deg,
            width_m=width_m,
            length_m=length_m,
            step_height_m=z_range,
            height_from=float(ch.min()),
            height_to=float(ch.max()),
            cells=cells,
        ))
        tid += 1

    log(f"[Ramp] Ramps after size filter: {len(transitions)}", "info")

    return GroundAnalysisResult(
        levels=[],
        transitions=transitions,
        cell_size=cell_size,
        grid_origin=origin,
        grid_shape=shape,
    )


# ── Cross-Section Angle Refinement ─────────────────────────────────────────────

def _refine_angle_cross_section(
    points: np.ndarray,
    transition: TransitionInfo,
    cell_size: float,
    origin: Tuple[float, float],
    strip_width: float = 0.5,
    bin_size: float = 0.15,
    margin_m: float = 2.0,
) -> float:
    """Refine a ramp's angle using cross-section profiles through the ramp region.

    Takes 1D cross-sections along the ramp's principal direction (from PCA),
    fits a linear segment, and returns the measured slope angle.
    This is more accurate than per-cell median slope because it measures
    the actual height change over distance in 1D, averaging out noise.

    Returns the refined angle in degrees, or the original if refinement fails.
    """
    min_x, min_y = origin
    cells = transition.cells
    rows, cols = cells[:, 0], cells[:, 1]

    # World coordinates of ramp cells
    wx = cols.astype(np.float32) * cell_size + min_x + cell_size * 0.5
    wy = rows.astype(np.float32) * cell_size + min_y + cell_size * 0.5

    # Ramp bounding box with margin
    ramp_x_min = float(wx.min()) - margin_m
    ramp_x_max = float(wx.max()) + margin_m
    ramp_y_min = float(wy.min()) - margin_m
    ramp_y_max = float(wy.max()) + margin_m

    # Extract points within the ramp bounding box
    mask = (
        (points[:, 0] >= ramp_x_min) & (points[:, 0] <= ramp_x_max) &
        (points[:, 1] >= ramp_y_min) & (points[:, 1] <= ramp_y_max)
    )
    local_pts = points[mask]
    if local_pts.shape[0] < 20:
        return transition.angle_deg

    # Compute ramp principal direction from start→end
    dx = transition.end_xy[0] - transition.start_xy[0]
    dy = transition.end_xy[1] - transition.start_xy[1]
    ramp_length = math.sqrt(dx * dx + dy * dy)
    if ramp_length < 0.1:
        return transition.angle_deg
    dir_x = dx / ramp_length
    dir_y = dy / ramp_length

    # Ramp center
    cx = (transition.start_xy[0] + transition.end_xy[0]) * 0.5
    cy = (transition.start_xy[1] + transition.end_xy[1]) * 0.5

    # Project local points onto ramp direction (along) and perpendicular (across)
    rel_x = local_pts[:, 0] - cx
    rel_y = local_pts[:, 1] - cy
    along = rel_x * dir_x + rel_y * dir_y      # distance along ramp
    across = -rel_x * dir_y + rel_y * dir_x    # distance perpendicular to ramp
    z = local_pts[:, 2]

    # Take multiple strips across the ramp width
    across_min = float(across.min())
    across_max = float(across.max())
    n_strips = max(1, int((across_max - across_min) / strip_width))

    fitted_angles = []

    for si in range(n_strips):
        strip_lo = across_min + si * strip_width
        strip_hi = strip_lo + strip_width
        strip_mask = (across >= strip_lo) & (across < strip_hi)
        if strip_mask.sum() < 10:
            continue

        s_along = along[strip_mask]
        s_z = z[strip_mask]

        # Bin along the ramp direction
        a_min = float(s_along.min())
        a_max = float(s_along.max())
        n_bins = max(1, int((a_max - a_min) / bin_size))

        bin_x = []
        bin_z = []
        for bi in range(n_bins):
            blo = a_min + bi * bin_size
            bhi = blo + bin_size
            bm = (s_along >= blo) & (s_along < bhi)
            if bm.sum() >= 3:
                bin_x.append(blo + bin_size * 0.5)
                bin_z.append(float(np.percentile(s_z[bm], 10)))

        if len(bin_x) < 4:
            continue

        bx = np.array(bin_x, dtype=np.float32)
        bz = np.array(bin_z, dtype=np.float32)

        # Linear fit: z = a * x + b
        n = len(bx)
        mx = float(bx.mean())
        mz = float(bz.mean())
        ss_xx = float(((bx - mx) ** 2).sum())
        ss_xz = float(((bx - mx) * (bz - mz)).sum())
        ss_zz = float(((bz - mz) ** 2).sum())

        if ss_xx < 1e-12:
            continue

        a = ss_xz / ss_xx
        ss_res = ss_zz - a * ss_xz
        r2 = max(0.0, 1.0 - ss_res / ss_zz) if ss_zz > 1e-12 else 0.0

        # Only accept fits with decent R² (the profile should be roughly linear)
        if r2 >= 0.5:
            angle = math.degrees(math.atan(abs(a)))
            fitted_angles.append(angle)

    if not fitted_angles:
        return transition.angle_deg

    # Return median of fitted angles across strips
    return float(np.median(fitted_angles))


def refine_with_cross_sections(
    points: np.ndarray,
    result: GroundAnalysisResult,
    log: callable = None,
) -> GroundAnalysisResult:
    """Refine all ramp angles using cross-section profiles.

    For each detected ramp region, takes 1D cross-sections along the ramp
    direction and fits linear segments to get a precise angle measurement.
    """
    if log is None:
        log = lambda m, l="info": None

    # Filter points to floor band for cross-section analysis
    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    floor_anchor = float(np.percentile(pts[:, 2], 5.0))
    z_mask = (pts[:, 2] >= floor_anchor) & (pts[:, 2] <= floor_anchor + 1.0)
    ground_pts = pts[z_mask]

    for t in result.transitions:
        old_angle = t.angle_deg
        new_angle = _refine_angle_cross_section(
            ground_pts, t,
            cell_size=result.cell_size,
            origin=result.grid_origin,
        )
        t.angle_deg = new_angle
        if abs(new_angle - old_angle) > 0.5:
            log(f"[Hybrid] Ramp #{t.transition_id}: {old_angle:.1f}° → {new_angle:.1f}° (cross-section refined)", "info")

    return result


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
    """Block non-traversable transitions on the obstacle map."""
    grid = obstacle_grid.copy()
    h, w = grid.shape
    ox, oy = map_origin

    for t in result.transitions:
        if t.traversable:
            continue

        analysis_origin = result.grid_origin
        analysis_cell = result.cell_size

        for ci in range(t.cells.shape[0]):
            row_a, col_a = int(t.cells[ci, 0]), int(t.cells[ci, 1])
            wx = analysis_origin[0] + (col_a + 0.5) * analysis_cell
            wy = analysis_origin[1] + (row_a + 0.5) * analysis_cell
            px = int((wx - ox) / map_resolution)
            py = int((wy - oy) / map_resolution)
            if 0 <= px < w and 0 <= py < h:
                grid[py, px] = 0

    return grid


# ── Report ────────────────────────────────────────────────────────────────────

def generate_accessibility_report(result: GroundAnalysisResult) -> str:
    """Generate a human-readable accessibility report."""
    lines = []
    lines.append(f"Ramp Regions Detected: {len(result.transitions)}")
    for t in result.transitions:
        status = "PASS" if t.traversable else "FAIL"
        lines.append(
            f"  [{status}] Ramp #{t.transition_id}: "
            f"angle={t.angle_deg:.1f}°, length={t.length_m:.1f}m, "
            f"width={t.width_m:.1f}m, height_diff={t.step_height_m:.2f}m "
            f"({t.start_xy[0]:.1f},{t.start_xy[1]:.1f}) → ({t.end_xy[0]:.1f},{t.end_xy[1]:.1f})"
        )
    pass_count = sum(1 for t in result.transitions if t.traversable)
    total = len(result.transitions)
    lines.append(f"\nAccessibility: {pass_count}/{total} ramps passable")
    return "\n".join(lines)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_ground_analysis(
    points: np.ndarray,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
    cell_size: float = 0.20,
    min_ramp_slope_deg: float = 5.0,
    log: callable = None,
    **kwargs,
) -> GroundAnalysisResult:
    """Run slope-based ramp detection + accessibility assessment.

    Parameters
    ----------
    points : (N, 3) point cloud
    max_slope_deg : robot maximum traversable slope
    max_step_m : robot maximum climbable step height
    cell_size : analysis grid cell size
    min_ramp_slope_deg : minimum slope to detect as ramp (default 4°)
    log : optional logging callback(message, level)
    """
    if log is None:
        log = lambda msg, lvl="info": None

    log("[Hybrid] Step 1: Slope grid — locating ramp regions...", "info")

    result = detect_ramps_by_slope(
        points,
        cell_size=cell_size,
        min_ramp_slope_deg=min_ramp_slope_deg,
        log=log,
    )

    log(f"[Hybrid] Step 2: Cross-section — refining angles for {len(result.transitions)} ramps...", "info")
    result = refine_with_cross_sections(points, result, log=log)

    log(f"[Hybrid] Step 3: Assessing accessibility (max_slope={max_slope_deg}°)...", "info")
    result = assess_transitions(result, max_slope_deg=max_slope_deg, max_step_m=max_step_m)

    for t in result.transitions:
        status = "PASS" if t.traversable else "FAIL"
        log(f"[Hybrid]   [{status}] {t.angle_deg:.1f}° ramp, {t.length_m:.1f}m x {t.width_m:.1f}m", "info")

    return result
