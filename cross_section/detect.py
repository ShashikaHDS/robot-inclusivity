"""Cross-section profile ramp detection.

Algorithm:
  1. Sweep cross-sections through the point cloud at multiple directions
  2. For each strip: build a 1D height profile
  3. Fit piecewise-linear segments to each profile
  4. A ramp = a sloped segment flanked by flat segments
  5. Merge detections from all directions → final ramp list
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class Segment:
    """A piecewise-linear segment of a height profile."""
    x_start: float
    x_end: float
    z_start: float
    z_end: float
    slope_deg: float
    is_flat: bool
    r_squared: float = 1.0


@dataclass
class RampCandidate:
    """A ramp detected in a single profile strip."""
    x_start: float          # along sweep axis (rotated frame)
    x_end: float
    strip_y: float          # center of the strip (rotated frame)
    angle_deg: float
    height_diff: float
    length_m: float
    r_squared: float
    direction_deg: float    # sweep direction in world frame
    # World coordinates (filled after inverse rotation)
    world_start: Optional[Tuple[float, float]] = None
    world_end: Optional[Tuple[float, float]] = None


@dataclass
class Ramp:
    """A confirmed ramp after merging detections from multiple directions."""
    ramp_id: int
    start_xy: Tuple[float, float]
    end_xy: Tuple[float, float]
    angle_deg: float
    length_m: float
    width_m: float
    height_diff_m: float
    traversable: bool = False
    confidence: float = 0.0     # best R² across detections
    n_detections: int = 1       # how many strips/directions detected this


@dataclass
class DetectionResult:
    """Full result of cross-section ramp detection."""
    ramps: List[Ramp]
    n_profiles: int = 0
    n_candidates: int = 0


# ── Profile Extraction ────────────────────────────────────────────────────────

def _rotate_points_2d(
    xy: np.ndarray, angle_deg: float,
) -> np.ndarray:
    """Rotate XY coordinates by angle_deg around origin."""
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (xy @ rot.T).astype(np.float32)


def _inverse_rotate_2d(
    xy: np.ndarray, angle_deg: float,
) -> np.ndarray:
    """Inverse rotate XY coordinates."""
    return _rotate_points_2d(xy, -angle_deg)


def extract_profiles(
    points: np.ndarray,
    direction_deg: float,
    strip_width: float = 0.5,
    bin_size: float = 0.20,
    min_points_per_bin: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Extract 1D height profiles along a given direction.

    Parameters
    ----------
    points : (N, 3) point cloud (already Z-filtered)
    direction_deg : sweep direction in degrees (0=X axis, 90=Y axis)
    strip_width : width of each cross-section strip (meters)
    bin_size : bin size along sweep axis for height estimation
    min_points_per_bin : minimum points per bin

    Returns
    -------
    List of (x_positions, z_heights, strip_center_y) tuples.
    x_positions and z_heights are 1D arrays of the profile.
    """
    xy = points[:, :2].copy()
    z = points[:, 2].copy()

    # Rotate so sweep direction aligns with X-axis
    xy_rot = _rotate_points_2d(xy, -direction_deg)
    x_rot = xy_rot[:, 0]
    y_rot = xy_rot[:, 1]

    # Bin into strips along Y
    y_min = float(y_rot.min())
    y_max = float(y_rot.max())
    n_strips = max(1, int(math.ceil((y_max - y_min) / strip_width)))

    profiles = []
    for si in range(n_strips):
        y_lo = y_min + si * strip_width
        y_hi = y_lo + strip_width
        strip_mask = (y_rot >= y_lo) & (y_rot < y_hi)
        if strip_mask.sum() < 10:
            continue

        sx = x_rot[strip_mask]
        sz = z[strip_mask]

        # Bin along X axis
        x_min_s = float(sx.min())
        x_max_s = float(sx.max())
        n_bins = max(1, int(math.ceil((x_max_s - x_min_s) / bin_size)))

        x_pos = []
        z_height = []
        for bi in range(n_bins):
            bx_lo = x_min_s + bi * bin_size
            bx_hi = bx_lo + bin_size
            bin_mask = (sx >= bx_lo) & (sx < bx_hi)
            count = int(bin_mask.sum())
            if count >= min_points_per_bin:
                x_pos.append(bx_lo + bin_size * 0.5)
                # Use 10th percentile as ground height (robust to outliers)
                z_height.append(float(np.percentile(sz[bin_mask], 10)))

        if len(x_pos) >= 5:
            profiles.append((
                np.array(x_pos, dtype=np.float32),
                np.array(z_height, dtype=np.float32),
                float((y_lo + y_hi) * 0.5),
            ))

    return profiles


# ── Segment Fitting ───────────────────────────────────────────────────────────

def _linear_fit(x: np.ndarray, z: np.ndarray) -> Tuple[float, float, float]:
    """Fit z = a*x + b. Returns (slope_a, intercept_b, r_squared)."""
    n = len(x)
    if n < 2:
        return 0.0, float(z[0]) if n == 1 else 0.0, 0.0
    mx = float(x.mean())
    mz = float(z.mean())
    dx = x - mx
    dz = z - mz
    ss_xx = float((dx * dx).sum())
    ss_xz = float((dx * dz).sum())
    ss_zz = float((dz * dz).sum())
    if ss_xx < 1e-12:
        return 0.0, mz, 0.0
    a = ss_xz / ss_xx
    b = mz - a * mx
    ss_res = ss_zz - a * ss_xz
    r2 = max(0.0, 1.0 - ss_res / ss_zz) if ss_zz > 1e-12 else 1.0
    return float(a), float(b), float(r2)


def fit_segments(
    x: np.ndarray,
    z: np.ndarray,
    flat_slope_threshold: float = 0.05,  # ~2.9 degrees
    min_segment_points: int = 3,
    r2_threshold: float = 0.85,
) -> List[Segment]:
    """Fit piecewise-linear segments to a 1D height profile.

    Uses a greedy approach: grow a segment while R² stays high,
    then start a new segment when the fit degrades.
    """
    n = len(x)
    if n < min_segment_points:
        return []

    segments = []
    start = 0

    while start < n - 1:
        best_end = start + min_segment_points - 1
        if best_end >= n:
            break

        # Grow the segment as long as the fit is good
        for end in range(start + min_segment_points - 1, n):
            seg_x = x[start:end + 1]
            seg_z = z[start:end + 1]
            a, b, r2 = _linear_fit(seg_x, seg_z)

            if r2 >= r2_threshold or end == start + min_segment_points - 1:
                best_end = end
                best_a = a
                best_b = b
                best_r2 = r2
            else:
                break

        # Create segment
        seg_x = x[start:best_end + 1]
        seg_z = z[start:best_end + 1]
        a, b, r2 = _linear_fit(seg_x, seg_z)
        slope_deg = float(math.degrees(math.atan(abs(a))))
        is_flat = abs(a) < flat_slope_threshold

        segments.append(Segment(
            x_start=float(seg_x[0]),
            x_end=float(seg_x[-1]),
            z_start=float(a * seg_x[0] + b),
            z_end=float(a * seg_x[-1] + b),
            slope_deg=slope_deg,
            is_flat=is_flat,
            r_squared=r2,
        ))

        # Next segment starts where this one ends (overlap by 1 for continuity)
        start = best_end

    return segments


# ── Ramp Finding in Profile ───────────────────────────────────────────────────

def find_ramps_in_profile(
    segments: List[Segment],
    min_ramp_slope_deg: float = 4.0,
    min_ramp_length_m: float = 1.0,
    min_height_diff_m: float = 0.05,
) -> List[dict]:
    """Find ramp patterns in a list of fitted segments.

    A ramp = a sloped segment (slope >= threshold) that:
    - Has length >= min_ramp_length_m
    - Has height difference >= min_height_diff_m
    - Ideally flanked by flat segments (but not required)
    """
    ramps = []
    for i, seg in enumerate(segments):
        if seg.is_flat:
            continue
        if seg.slope_deg < min_ramp_slope_deg:
            continue

        length = seg.x_end - seg.x_start
        if length < min_ramp_length_m:
            continue

        height_diff = abs(seg.z_end - seg.z_start)
        if height_diff < min_height_diff_m:
            continue

        # A real ramp MUST be flanked by flat ground on at least one side.
        # This eliminates walls, step edges, and terrain boundaries.
        has_flat_before = (i > 0 and segments[i - 1].is_flat)
        has_flat_after = (i < len(segments) - 1 and segments[i + 1].is_flat)
        if not (has_flat_before or has_flat_after):
            continue

        ramps.append({
            "x_start": seg.x_start,
            "x_end": seg.x_end,
            "z_start": seg.z_start,
            "z_end": seg.z_end,
            "angle_deg": seg.slope_deg,
            "height_diff": height_diff,
            "length": length,
            "r_squared": seg.r_squared,
            "flanked": has_flat_before and has_flat_after,
        })

    return ramps


# ── Coordinate Conversion ─────────────────────────────────────────────────────

def _rotated_to_world(
    x_rot: float, y_rot: float, direction_deg: float,
) -> Tuple[float, float]:
    """Convert rotated (x, y) back to world (x, y)."""
    xy = np.array([[x_rot, y_rot]], dtype=np.float32)
    world = _inverse_rotate_2d(xy, -direction_deg)
    return (float(world[0, 0]), float(world[0, 1]))


# ── Clustering / Merging ──────────────────────────────────────────────────────

def _cluster_candidates(
    candidates: List[RampCandidate],
    merge_radius_m: float = 3.0,
) -> List[List[RampCandidate]]:
    """Cluster ramp candidates by spatial proximity of their midpoints."""
    if not candidates:
        return []

    # Compute world midpoints
    mids = []
    for c in candidates:
        if c.world_start and c.world_end:
            mx = (c.world_start[0] + c.world_end[0]) * 0.5
            my = (c.world_start[1] + c.world_end[1]) * 0.5
        else:
            mx = my = 0.0
        mids.append((mx, my))

    mids_arr = np.array(mids, dtype=np.float32)
    n = len(candidates)
    assigned = [False] * n
    clusters: List[List[RampCandidate]] = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [candidates[i]]
        assigned[i] = True

        for j in range(i + 1, n):
            if assigned[j]:
                continue
            dist = math.sqrt(
                (mids_arr[i, 0] - mids_arr[j, 0]) ** 2 +
                (mids_arr[i, 1] - mids_arr[j, 1]) ** 2
            )
            if dist < merge_radius_m:
                cluster.append(candidates[j])
                assigned[j] = True

        clusters.append(cluster)

    return clusters


# ── Main Detection ────────────────────────────────────────────────────────────

def detect_ramps(
    points: np.ndarray,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
    directions: List[float] = None,
    strip_width: float = 0.5,
    bin_size: float = 0.20,
    min_ramp_slope_deg: float = 4.0,
    min_ramp_length_m: float = 1.0,
    z_band: float = 1.5,
    merge_radius_m: float = 5.0,
    log: callable = None,
) -> DetectionResult:
    """Detect ramps using cross-section profile analysis.

    Parameters
    ----------
    points : (N, 3) point cloud
    max_slope_deg : robot max traversable slope
    max_step_m : robot max step height (unused for now)
    directions : sweep directions in degrees (default: 0, 45, 90, 135)
    strip_width : width of each cross-section strip
    bin_size : bin size along sweep axis
    min_ramp_slope_deg : minimum slope to detect as ramp
    min_ramp_length_m : minimum ramp length
    z_band : height band above floor for ground points
    merge_radius_m : cluster radius for merging detections
    log : logging callback
    """
    if log is None:
        log = lambda m, l="info": None

    if directions is None:
        directions = [0.0, 45.0, 90.0, 135.0]

    pts = np.asarray(points, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]

    # Filter to floor band
    floor_anchor = float(np.percentile(pts[:, 2], 5.0))
    z_mask = (pts[:, 2] >= floor_anchor) & (pts[:, 2] <= floor_anchor + z_band)
    ground_pts = pts[z_mask]
    log(f"[XSec] Floor anchor: {floor_anchor:.3f}m, ground points: {ground_pts.shape[0]:,}", "info")

    if ground_pts.shape[0] < 100:
        return DetectionResult(ramps=[], n_profiles=0, n_candidates=0)

    # Collect ramp candidates from all directions
    all_candidates: List[RampCandidate] = []
    total_profiles = 0

    for direction in directions:
        log(f"[XSec] Scanning direction {direction:.0f}°...", "info")
        profiles = extract_profiles(
            ground_pts, direction,
            strip_width=strip_width,
            bin_size=bin_size,
        )
        total_profiles += len(profiles)

        for x_arr, z_arr, strip_y in profiles:
            segments = fit_segments(x_arr, z_arr)
            ramp_dicts = find_ramps_in_profile(
                segments,
                min_ramp_slope_deg=min_ramp_slope_deg,
                min_ramp_length_m=min_ramp_length_m,
            )

            for rd in ramp_dicts:
                # Convert start/end to world coordinates
                ws = _rotated_to_world(rd["x_start"], strip_y, direction)
                we = _rotated_to_world(rd["x_end"], strip_y, direction)

                all_candidates.append(RampCandidate(
                    x_start=rd["x_start"],
                    x_end=rd["x_end"],
                    strip_y=strip_y,
                    angle_deg=rd["angle_deg"],
                    height_diff=rd["height_diff"],
                    length_m=rd["length"],
                    r_squared=rd["r_squared"],
                    direction_deg=direction,
                    world_start=ws,
                    world_end=we,
                ))

    log(f"[XSec] Total profiles: {total_profiles}, candidates: {len(all_candidates)}", "info")

    if not all_candidates:
        return DetectionResult(ramps=[], n_profiles=total_profiles, n_candidates=0)

    # Cluster nearby candidates
    clusters = _cluster_candidates(all_candidates, merge_radius_m=merge_radius_m)
    log(f"[XSec] Clusters after merging: {len(clusters)}", "info")

    # Build final ramp list from clusters — apply strict filters
    ramps: List[Ramp] = []
    for ci, cluster in enumerate(clusters):
        # ── Filter: require at least 3 strip detections ──
        # A real ramp spans multiple parallel strips. Noise appears in 1-2 strips only.
        if len(cluster) < 3:
            continue

        # Pick the detection with the best R² as representative
        best = max(cluster, key=lambda c: c.r_squared)

        # ── Filter: ramp length must be >= 1m (user-confirmed) ──
        if best.length_m < min_ramp_length_m:
            continue

        # ── Filter: minimum height difference 0.10m ──
        if best.height_diff < 0.10:
            continue

        # Estimate width from the spread of strip_y values across detections
        strip_ys = [c.strip_y for c in cluster]
        width = max(float(max(strip_ys) - min(strip_ys)), strip_width)

        # ── Filter: width must be >= 0.8m (a real ramp, not a pipe or edge) ──
        if width < 0.8:
            continue

        # Average angle across detections
        angles = [c.angle_deg for c in cluster]
        avg_angle = float(np.mean(angles))

        ramp = Ramp(
            ramp_id=ci,
            start_xy=best.world_start,
            end_xy=best.world_end,
            angle_deg=avg_angle,
            length_m=best.length_m,
            width_m=width,
            height_diff_m=best.height_diff,
            traversable=avg_angle <= max_slope_deg,
            confidence=best.r_squared,
            n_detections=len(cluster),
        )
        ramps.append(ramp)

    # Sort by angle descending (steepest first)
    ramps.sort(key=lambda r: r.angle_deg, reverse=True)
    for i, r in enumerate(ramps):
        r.ramp_id = i

    log(f"[XSec] Final ramps: {len(ramps)}", "info")
    for r in ramps:
        status = "PASS" if r.traversable else "FAIL"
        log(f"[XSec]   [{status}] {r.angle_deg:.1f}° ramp, {r.length_m:.1f}m x {r.width_m:.1f}m, "
            f"dh={r.height_diff_m:.2f}m, R²={r.confidence:.2f}, detections={r.n_detections}",
            "info")

    return DetectionResult(
        ramps=ramps,
        n_profiles=total_profiles,
        n_candidates=len(all_candidates),
    )
