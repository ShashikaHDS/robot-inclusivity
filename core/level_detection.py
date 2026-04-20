"""Detect distinct floor levels in a point cloud via Z-histogram analysis.

A multi-floor building produces tall peaks in the Z histogram at each floor's
height (the floor surface itself has the highest XY density at its Z). We
find those peaks, pair neighbours to form slabs, and return per-level
z-bounds that the map pipeline can iterate over.

Pure numpy — no scipy, no new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class FloorLevel:
    index: int            # 0 = lowest level
    anchor_z: float       # detected floor height (center of the peak bin)
    z_low: float          # lower bound of this level's slab (absolute z)
    z_high: float         # upper bound of this level's slab (absolute z)
    point_count: int      # points that fall inside [z_low, z_high]


def detect_floor_levels(
    points: np.ndarray,
    bin_size_m: float = 0.10,
    peak_prominence_ratio: float = 0.15,
    min_floor_separation_m: float = 1.8,
    default_ceiling_height_m: float = 3.0,
    smooth_window: int = 3,
) -> List[FloorLevel]:
    """Detect floor levels from a point cloud.

    Parameters
    ----------
    points : (N, 3) float array
        X, Y, Z point cloud.
    bin_size_m : float
        Z-histogram bin width. Smaller = more resolution, more noise.
    peak_prominence_ratio : float
        A peak must be ≥ this fraction of the tallest peak to count as a floor.
        0.15 rejects tiny "furniture-top" peaks.
    min_floor_separation_m : float
        Two peaks closer than this are merged (keeping the taller).
        Typical indoor floor-to-floor is 2.5–3.5 m, so 1.8 is a safe minimum.
    default_ceiling_height_m : float
        Slab height assigned to the topmost level (no next-peak to bound it).
    smooth_window : int
        Simple moving-average width applied to the histogram before peak-find.

    Returns
    -------
    List[FloorLevel], ordered from lowest to highest. Always returns at least
    one level; falls back to a single-level result if no clear peaks are found.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3 or pts.shape[0] == 0:
        raise ValueError("points must be a non-empty (N, 3) array")

    z = pts[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    total_range = z_max - z_min

    # Too short to have multiple floors — single level
    if total_range < min_floor_separation_m:
        return [FloorLevel(0, z_min, z_min, z_max, int(pts.shape[0]))]

    # Build histogram
    n_bins = max(8, int(np.ceil(total_range / bin_size_m)))
    counts, edges = np.histogram(z, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Simple moving-average smooth
    if smooth_window > 1 and len(counts) > smooth_window:
        k = smooth_window
        kernel = np.ones(k, dtype=np.float32) / k
        counts_s = np.convolve(counts.astype(np.float32), kernel, mode="same")
    else:
        counts_s = counts.astype(np.float32)

    peak_threshold = float(counts_s.max()) * peak_prominence_ratio

    # Find local maxima above threshold
    peaks: List[int] = []
    for i in range(len(counts_s)):
        if counts_s[i] < peak_threshold:
            continue
        left_ok = (i == 0) or counts_s[i] >= counts_s[i - 1]
        right_ok = (i == len(counts_s) - 1) or counts_s[i] >= counts_s[i + 1]
        strictly_greater = (
            (i > 0 and counts_s[i] > counts_s[i - 1])
            or (i < len(counts_s) - 1 and counts_s[i] > counts_s[i + 1])
        )
        if left_ok and right_ok and strictly_greater:
            peaks.append(i)

    if not peaks:
        # Fallback to single level
        return [FloorLevel(0, z_min, z_min, z_max, int(pts.shape[0]))]

    # Merge peaks closer than min_floor_separation_m (keep the taller)
    merged: List[int] = [peaks[0]]
    for p in peaks[1:]:
        if centers[p] - centers[merged[-1]] < min_floor_separation_m:
            if counts_s[p] > counts_s[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)

    # Build slabs
    levels: List[FloorLevel] = []
    for idx, p in enumerate(merged):
        anchor = float(centers[p])
        # Lower bound: midpoint to previous peak (or z_min)
        if idx == 0:
            z_low = z_min
        else:
            z_low = 0.5 * (anchor + float(centers[merged[idx - 1]]))
        # Upper bound: midpoint to next peak (or anchor + default ceiling)
        if idx == len(merged) - 1:
            z_high = min(z_max, anchor + default_ceiling_height_m)
        else:
            z_high = 0.5 * (anchor + float(centers[merged[idx + 1]]))
        mask = (z >= z_low) & (z <= z_high)
        levels.append(FloorLevel(
            index=idx,
            anchor_z=anchor,
            z_low=float(z_low),
            z_high=float(z_high),
            point_count=int(mask.sum()),
        ))

    return levels


def summarize_levels(levels: List[FloorLevel]) -> str:
    """Pretty one-line summary for the log panel."""
    if len(levels) == 1:
        lv = levels[0]
        return (f"Single level detected: z=[{lv.z_low:.2f}, {lv.z_high:.2f}] m, "
                f"{lv.point_count:,} points")
    parts = [f"{len(levels)} floor levels detected:"]
    for lv in levels:
        parts.append(
            f"  L{lv.index}: anchor={lv.anchor_z:+.2f} m, "
            f"slab=[{lv.z_low:+.2f}, {lv.z_high:+.2f}] m, "
            f"{lv.point_count:,} pts"
        )
    return "\n".join(parts)
