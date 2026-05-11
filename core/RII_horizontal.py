"""RII Horizontal — inflation, BFS reachability, path planners, coverage computation."""

from __future__ import annotations

import os
import sys
import time
import math
import numpy as np
from collections import deque

try:
    from scipy.ndimage import binary_dilation as _scipy_dilate
    from scipy.ndimage import label as _scipy_label
    from scipy.ndimage import convolve as _scipy_convolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from core.map_io import parse_pgm, parse_yaml

from config import PCD_PACKAGE_DIR
if PCD_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PCD_PACKAGE_DIR)
from pcd_package.pcd_tools import estimate_ground_preserving_preset


STC_ANALYSIS_CELL_M = 0.20
TERRAIN_SIDECAR_MIN_BAND_M = 2.0
BLOCKED_MAP_VIEW = "Obstacle Map"
PRIMARY_SELECTION_VIEW = "Obstacle Map"


def derive_terrain_sidecar_bounds(
    points: np.ndarray,
    obstacle_min_z: float,
    obstacle_max_z: float,
    min_band_m: float = TERRAIN_SIDECAR_MIN_BAND_M,
):
    """Choose a floor-preserving z range for RII terrain sidecars."""
    preset = estimate_ground_preserving_preset(points)
    obstacle_min_z = float(obstacle_min_z)
    obstacle_max_z = float(obstacle_max_z)
    band_ok = obstacle_max_z > obstacle_min_z and (obstacle_max_z - obstacle_min_z) >= float(min_band_m)
    if band_ok:
        return obstacle_min_z, obstacle_max_z, {
            "source": "requested",
            "floor_anchor_z": float(preset["floor_anchor_z"]),
            "cleanup_min_z": float(preset["cleanup_min_z"]),
            "cleanup_max_z": float(preset["cleanup_max_z"]),
        }
    return float(preset["cleanup_min_z"]), float(preset["cleanup_max_z"]), {
        "source": "preset_cleanup",
        "floor_anchor_z": float(preset["floor_anchor_z"]),
        "cleanup_min_z": float(preset["cleanup_min_z"]),
        "cleanup_max_z": float(preset["cleanup_max_z"]),
    }


def _largest_component_on_coarse_mask(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Approximate STC-reachable area as the largest connected free region on a coarse grid."""
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw = math.ceil(w / step)
    ch = math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels = np.full((ch, cw), -1, dtype=np.int32)
    component_sizes = []
    for row in range(ch):
        for col in range(cw):
            if coarse[row, col] == 0 or labels[row, col] != -1:
                continue
            comp_id = len(component_sizes)
            q = deque([(row, col)])
            labels[row, col] = comp_id
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < ch and 0 <= nc < cw and coarse[nr, nc] == 1 and labels[nr, nc] == -1:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
            component_sizes.append(size)

    if not component_sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []

    largest_id = int(np.argmax(component_sizes))
    largest_tiles = (labels == largest_id)
    up = np.repeat(np.repeat(largest_tiles, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & (accessible2d.astype(bool))).astype(np.uint8)
    return mask, len(component_sizes), int(component_sizes[largest_id]), step, _stc_stroke_on_mask(largest_tiles)


def _stc_stroke_on_mask(mask: np.ndarray):
    """Return an STC-style Euler tour over a 4-connected boolean grid."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return []
    start = (int(idx[0][0]), int(idx[0][1]))

    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parent = np.full((height, width, 2), -1, dtype=np.int32)
    nbrs = ((0, 1), (0, -1), (1, 0), (-1, 0))

    stack = [start]
    seen[start] = True
    while stack:
        row, col = stack.pop()
        for dr, dc in nbrs:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                parent[nr, nc] = (row, col)
                stack.append((row, col))
                stack.append((nr, nc))
                break

    children = [[[] for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for col in range(width):
            pr, pc = parent[row, col]
            if pr >= 0:
                children[int(pr)][int(pc)].append((row, col))
    for row in range(height):
        for col in range(width):
            children[row][col].sort()

    stroke = [start]
    stack2 = [(start, 0)]
    while stack2:
        node, child_idx = stack2[-1]
        row, col = node
        ch_list = children[row][col]
        if child_idx < len(ch_list):
            child = ch_list[child_idx]
            stack2[-1] = (node, child_idx + 1)
            stroke.append(child)
            stack2.append((child, 0))
        else:
            stack2.pop()
            if stack2:
                stroke.append(stack2[-1][0])
    return stroke


# ── Path planner registry ──────────────────────────────────────────────
PLANNER_NAMES = [
    "Spanning Tree Coverage (STC)",
    "Boustrophedon Cellular Decomposition (BCD)",
    "Wavefront Coverage",
    "Morse-based Cellular Decomposition (Morse)",
    "Frontier-based Exploration (Frontier)",
]
# Map display name → internal dispatch key
_PLANNER_KEY = {
    "Spanning Tree Coverage (STC)": "STC",
    "Boustrophedon Cellular Decomposition (BCD)": "BCD",
    "Wavefront Coverage": "Wavefront",
    "Morse-based Cellular Decomposition (Morse)": "Morse",
    "Frontier-based Exploration (Frontier)": "Frontier",
}


def _connect_path(waypoints, free_mask):
    """Insert BFS shortest-path segments between non-adjacent waypoints.

    Given a list of (row, col) waypoints on a boolean *free_mask*, return a
    fully 4-connected path that never crosses blocked cells.  Adjacent
    waypoints (Manhattan distance 1) are kept as-is; distant ones are bridged
    with BFS through free space.
    """
    if len(waypoints) <= 1:
        return list(waypoints)
    h, w = free_mask.shape
    connected = [waypoints[0]]
    for i in range(1, len(waypoints)):
        prev = waypoints[i - 1]
        cur = waypoints[i]
        dr = abs(cur[0] - prev[0])
        dc = abs(cur[1] - prev[1])
        if dr + dc <= 1:
            connected.append(cur)
            continue
        # BFS from prev to cur on free_mask
        seen = np.zeros((h, w), dtype=bool)
        parent = {}
        q = deque([prev])
        seen[prev] = True
        found = False
        while q:
            rr, cc = q.popleft()
            if (rr, cc) == cur:
                found = True
                break
            for d_r, d_c in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + d_r, cc + d_c
                if 0 <= nr < h and 0 <= nc < w and free_mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    parent[(nr, nc)] = (rr, cc)
                    q.append((nr, nc))
        if found:
            seg = []
            node = cur
            while node != prev:
                seg.append(node)
                node = parent[node]
            seg.reverse()
            connected.extend(seg)
        else:
            # Unreachable — just append the waypoint (gap in path)
            connected.append(cur)
    return connected


def _bfs_largest_component(mask2d):
    """Return (labels, component_sizes) for 4-connected components on a boolean 2D mask."""
    if _HAS_SCIPY:
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)  # 4-connected
        labeled, n = _scipy_label(mask2d.astype(bool), structure=struct)
        labels = labeled.astype(np.int32) - 1  # scipy uses 1-based, we use 0-based (-1 = unlabeled)
        labels[~mask2d.astype(bool)] = -1
        # Use bincount for fast size computation
        counts = np.bincount(labeled.ravel())
        sizes = [int(counts[i]) for i in range(1, n + 1)]
        return labels, sizes
    # Fallback: pure Python BFS
    h, w = mask2d.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    sizes = []
    for r in range(h):
        for c in range(w):
            if not mask2d[r, c] or labels[r, c] != -1:
                continue
            cid = len(sizes)
            q = deque([(r, c)])
            labels[r, c] = cid
            sz = 0
            while q:
                rr, cc = q.popleft()
                sz += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask2d[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            sizes.append(sz)
    return labels, sizes


def _keep_largest(accessible2d, labels, sizes):
    """Zero out everything except the largest connected component; return (mask, n_comps, largest_size)."""
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0
    lid = int(np.argmax(sizes))
    keep = (labels == lid)
    mask = (keep & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid])


# ── BCD (Boustrophedon Cellular Decomposition) ────────────────────────
def _run_bcd(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Boustrophedon path over the largest connected accessible region.

    Chobanyan & Choset, "Coverage of Known Spaces: The Boustrophedon
    Cellular Decomposition", 1998.

    The decomposition is approximated on a coarse grid: each column is swept
    top-to-bottom then bottom-to-top in alternating directions (ox-plough
    pattern). Only the largest connected component is kept.
    """
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw, ch = math.ceil(w / step), math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels, sizes = _bfs_largest_component(coarse.astype(bool))
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []
    lid = int(np.argmax(sizes))
    comp = (labels == lid)

    # Boustrophedon sweep: alternate column direction
    waypoints = []
    for col in range(cw):
        rows = range(ch) if col % 2 == 0 else range(ch - 1, -1, -1)
        for row in rows:
            if comp[row, col]:
                waypoints.append((row, col))

    path = _connect_path(waypoints, comp)
    up = np.repeat(np.repeat(comp, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid]), step, path


# ── Wavefront Coverage ─────────────────────────────────────────────────
def _run_wavefront(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Wavefront (distance-transform) coverage path.

    Zelinsky et al., "Planning Paths of Complete Coverage of an
    Unstructured Environment by a Mobile Robot", 1993.

    A BFS wavefront expands from the centre of the largest component,
    assigning distance values. The path then follows cells in descending
    distance order (farthest-first), producing a spiral-inward trajectory.
    """
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw, ch = math.ceil(w / step), math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels, sizes = _bfs_largest_component(coarse.astype(bool))
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []
    lid = int(np.argmax(sizes))
    comp = (labels == lid)

    # Find centroid of largest component as wavefront seed
    idx = np.argwhere(comp)
    cr, cc = int(idx[:, 0].mean()), int(idx[:, 1].mean())
    if not comp[cr, cc]:
        cr, cc = int(idx[0, 0]), int(idx[0, 1])

    # BFS wavefront from centroid
    dist = np.full((ch, cw), -1, dtype=np.int32)
    dist[cr, cc] = 0
    q = deque([(cr, cc)])
    while q:
        rr, rc_ = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, rc_ + dc
            if 0 <= nr < ch and 0 <= nc < cw and comp[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[rr, rc_] + 1
                q.append((nr, nc))

    # Collect cells with valid distance, sort descending (farthest first → spiral inward)
    cells = [(int(dist[r, c]), r, c) for r in range(ch) for c in range(cw) if dist[r, c] >= 0]
    cells.sort(key=lambda x: -x[0])

    # Greedy nearest-neighbour ordering to reduce path jumps
    waypoints = _greedy_nearest_order([(r, c) for _, r, c in cells])
    path = _connect_path(waypoints, comp)

    up = np.repeat(np.repeat(comp, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid]), step, path


def _greedy_nearest_order(cells):
    """Reorder cells via greedy nearest-neighbour to produce a smooth path."""
    if len(cells) <= 1:
        return list(cells)
    remaining = set(range(len(cells)))
    order = [0]
    remaining.discard(0)
    while remaining:
        cr, cc = cells[order[-1]]
        best = None
        best_d = float('inf')
        for i in remaining:
            d = abs(cells[i][0] - cr) + abs(cells[i][1] - cc)
            if d < best_d:
                best_d = d
                best = i
        order.append(best)
        remaining.discard(best)
    return [cells[i] for i in order]


# ── Morse-based Cellular Decomposition ─────────────────────────────────
def _run_morse(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Morse-based cellular decomposition coverage path.

    Acar & Choset, "Sensor-Based Coverage of Unknown Environments: Incremental
    Construction of Morse Decompositions", 2002.

    The free space is sliced into vertical strips at each column. Connected
    vertical segments within a column form Morse cells. Cells are linked to
    neighbours in adjacent columns and the path traverses cells in a
    depth-first order, sweeping each cell top-to-bottom or bottom-to-top.
    """
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw, ch = math.ceil(w / step), math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels, sizes = _bfs_largest_component(coarse.astype(bool))
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []
    lid = int(np.argmax(sizes))
    comp = (labels == lid)

    # Decompose into vertical segments (Morse cells) per column
    cells_by_col = {}  # col -> list of (start_row, end_row) tuples
    for col in range(cw):
        segs = []
        in_seg = False
        start = 0
        for row in range(ch):
            if comp[row, col]:
                if not in_seg:
                    start = row
                    in_seg = True
            else:
                if in_seg:
                    segs.append((start, row - 1))
                    in_seg = False
        if in_seg:
            segs.append((start, ch - 1))
        cells_by_col[col] = segs

    # Build adjacency graph between segments in neighbouring columns
    seg_id = {}
    seg_list = []
    for col in sorted(cells_by_col.keys()):
        for seg in cells_by_col[col]:
            seg_id[(col, seg)] = len(seg_list)
            seg_list.append((col, seg))

    adj = [[] for _ in seg_list]
    for col in range(cw - 1):
        for s1 in cells_by_col.get(col, []):
            for s2 in cells_by_col.get(col + 1, []):
                if s1[1] >= s2[0] and s2[1] >= s1[0]:  # overlapping row ranges
                    i, j = seg_id[(col, s1)], seg_id[(col + 1, s2)]
                    adj[i].append(j)
                    adj[j].append(i)

    # DFS traversal of segment graph
    visited = [False] * len(seg_list)
    waypoints = []
    if seg_list:
        stack = [0]
        sweep_down = True
        while stack:
            sid = stack.pop()
            if visited[sid]:
                continue
            visited[sid] = True
            col, (r0, r1) = seg_list[sid]
            if sweep_down:
                for r in range(r0, r1 + 1):
                    waypoints.append((r, col))
            else:
                for r in range(r1, r0 - 1, -1):
                    waypoints.append((r, col))
            sweep_down = not sweep_down
            for nb in adj[sid]:
                if not visited[nb]:
                    stack.append(nb)

    path = _connect_path(waypoints, comp)
    up = np.repeat(np.repeat(comp, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid]), step, path


# ── Frontier-based Exploration ─────────────────────────────────────────
def _run_frontier(accessible2d: np.ndarray, resolution: float, target_cell_m: float = STC_ANALYSIS_CELL_M):
    """Frontier-based exploration coverage path.

    Yamauchi, "A Frontier-Based Approach for Autonomous Exploration", 1997.

    In a coverage context the 'frontier' is the boundary between visited
    and unvisited free cells. Starting from the centroid, the planner
    repeatedly moves to the nearest frontier cell, marking cells as visited
    until all reachable cells are covered. This greedy strategy naturally
    prioritises nearby uncovered regions.
    """
    h, w = accessible2d.shape
    step = max(1, round(max(float(resolution), float(target_cell_m)) / float(resolution)))
    cw, ch = math.ceil(w / step), math.ceil(h / step)

    free = accessible2d.astype(np.uint8)
    ph, pw = ch * step, cw * step
    fp = np.zeros((ph, pw), dtype=np.uint8)
    fp[:h, :w] = free
    coarse = fp.reshape(ch, step, cw, step).min(axis=(1, 3)).astype(np.uint8)

    labels, sizes = _bfs_largest_component(coarse.astype(bool))
    if not sizes:
        return np.zeros_like(accessible2d, dtype=np.uint8), 0, 0, step, []
    lid = int(np.argmax(sizes))
    comp = (labels == lid)

    # Start at centroid of largest component
    idx = np.argwhere(comp)
    cr, cc = int(idx[:, 0].mean()), int(idx[:, 1].mean())
    if not comp[cr, cc]:
        cr, cc = int(idx[0, 0]), int(idx[0, 1])

    visited = np.zeros((ch, cw), dtype=bool)
    path = []
    cur = (cr, cc)
    visited[cur] = True
    path.append(cur)
    total_free = int(comp.sum())

    # Precompute BFS distance field for nearest-frontier lookups
    def _bfs_to_nearest_frontier(start):
        """BFS from start; return first unvisited free cell found."""
        q2 = deque([start])
        seen = np.zeros((ch, cw), dtype=bool)
        seen[start] = True
        parent = {}
        while q2:
            rr, rc_ = q2.popleft()
            if comp[rr, rc_] and not visited[rr, rc_]:
                # Trace back to start to build path segment
                seg = []
                node = (rr, rc_)
                while node != start:
                    seg.append(node)
                    node = parent[node]
                seg.reverse()
                return seg
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, rc_ + dc
                if 0 <= nr < ch and 0 <= nc < cw and comp[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    parent[(nr, nc)] = (rr, rc_)
                    q2.append((nr, nc))
        return None

    while len(path) < total_free:
        seg = _bfs_to_nearest_frontier(cur)
        if seg is None:
            break
        for cell in seg:
            visited[cell] = True
            path.append(cell)
        cur = path[-1]

    up = np.repeat(np.repeat(comp, step, axis=0), step, axis=1)[:h, :w]
    mask = (up & accessible2d.astype(bool)).astype(np.uint8)
    return mask, len(sizes), int(sizes[lid]), step, path


# ── Planner dispatcher ─────────────────────────────────────────────────
_PLANNER_DISPATCH = {
    "STC": lambda acc, res: _largest_component_on_coarse_mask(acc, res),
    "BCD": _run_bcd,
    "Wavefront": _run_wavefront,
    "Morse": _run_morse,
    "Frontier": _run_frontier,
}


def run_planner(name: str, accessible2d: np.ndarray, resolution: float):
    """Run a named path planner.  Returns (mask, n_components, largest_size, step, path)."""
    key = _PLANNER_KEY.get(name, name)  # accept display name or short key
    fn = _PLANNER_DISPATCH.get(key)
    if fn is None:
        raise ValueError(f"Unknown planner: {name!r}. Choose from {list(_PLANNER_DISPATCH)}")
    return fn(accessible2d, resolution)


def _footprint_inflation_pixels(params, resolution: float) -> tuple:
    shape = params.get('shape', 'circular')
    halfW = params.get('halfW', params.get('radius', 0.35))
    halfL = params.get('halfL', params.get('radius', 0.35))
    inflX = max(0, math.ceil(halfW / resolution))
    inflY = max(0, math.ceil(halfL / resolution))
    isRect = (shape == 'rectangular')
    return inflX, inflY, isRect


def _dilate_binary_mask(mask2d: np.ndarray, inflX: int, inflY: int, isRect: bool) -> np.ndarray:
    base = np.asarray(mask2d, dtype=np.uint8)
    if base.ndim != 2:
        raise ValueError("mask2d must be 2D")
    if inflX <= 0 and inflY <= 0:
        return base.copy()

    if _HAS_SCIPY:
        # Build structuring element (circular or rectangular)
        ky, kx = 2 * inflY + 1, 2 * inflX + 1
        if isRect:
            struct = np.ones((ky, kx), dtype=bool)
        else:
            yy, xx = np.ogrid[-inflY:inflY + 1, -inflX:inflX + 1]
            struct = (xx * xx + yy * yy) <= (inflX * inflX)
        return _scipy_dilate(base.astype(bool), structure=struct).astype(np.uint8)

    # Fallback: pure Python slice-based dilation
    h, w = base.shape
    out = np.zeros((h, w), dtype=np.uint8)
    inflSq = inflX * inflX
    for dy in range(-inflY, inflY + 1):
        dy2 = dy * dy
        for dx in range(-inflX, inflX + 1):
            if not isRect and dx * dx + dy2 > inflSq:
                continue
            sy0 = max(0, -dy)
            sy1 = min(h, h - dy)
            sx0 = max(0, -dx)
            sx1 = min(w, w - dx)
            dy0 = max(0, dy)
            dy1 = min(h, h + dy)
            dx0 = max(0, dx)
            dx1 = min(w, w + dx)
            if sy1 > sy0 and sx1 > sx0:
                out[dy0:dy1, dx0:dx1] |= base[sy0:sy1, sx0:sx1]
    return out


def _score_accessibility_from_masks(
    blocked2d: np.ndarray,
    floor_mask2d,
    resolution: float,
    params: dict,
    label: str,
    logf=None,
    use_stc: bool = False,
    trav_mask2d=None,
    planner: str | None = None,
    start_cell=None,
) -> dict:
    L = logf if logf else lambda m, c="": None
    t0 = time.time()
    h, w = blocked2d.shape
    inflX, inflY, isRect = _footprint_inflation_pixels(params, resolution)
    halfW = params.get('halfW', params.get('radius', 0.35))
    halfL = params.get('halfL', params.get('radius', 0.35))
    min_gap = (2 * inflX + 1) * resolution
    L(f"[{label}] Inflate: {'rect' if isRect else 'circle'} {inflX}x{inflY}px "
      f"(halfW={halfW:.3f}m, halfL={halfL:.3f}m, min passable gap={min_gap:.2f}m)", "info")

    blocked_src = np.asarray(blocked2d, dtype=np.uint8)
    inflated2d = _dilate_binary_mask(blocked_src, inflX, inflY, isRect)
    if inflX > 0 or inflY > 0:
        L(f"[{label}] Inflation done: {time.time()-t0:.2f}s", "info")

    accessible2d = (inflated2d == 0).astype(np.uint8)
    if trav_mask2d is not None and floor_mask2d is not None:
        trav = np.asarray(trav_mask2d, dtype=np.uint8)
        floor = np.asarray(floor_mask2d, dtype=np.uint8)
        known_non_trav = (floor > 0) & (trav == 0)
        before = int(accessible2d.sum())
        accessible2d[known_non_trav] = 0
        excluded = before - int(accessible2d.sum())
        if excluded > 0:
            L(f"[{label}] Terrain constraint excluded {excluded} known-non-traversable cells", "info")
    if floor_mask2d is None:
        floor_mask2d = (inflated2d == 0).astype(np.uint8)
        L(f"[{label}] Floor denominator fallback: using post-mask free cells.", "warn")
    else:
        floor_mask2d = np.asarray(floor_mask2d, dtype=np.uint8)

    accessible2d &= floor_mask2d

    # Filter to the connected component containing the start point (or largest if no start).
    # This ensures only cells physically reachable from the robot's position are counted.
    start_r, start_c = None, None
    if start_cell is not None:
        start_r, start_c = int(start_cell[0]), int(start_cell[1])
        if not (0 <= start_r < h and 0 <= start_c < w and accessible2d[start_r, start_c]):
            # Start point is blocked — find nearest accessible cell
            if accessible2d.any():
                free_r, free_c = np.where(accessible2d > 0)
                dists = (free_r - start_r) ** 2 + (free_c - start_c) ** 2
                nearest = int(dists.argmin())
                orig_r, orig_c = start_r, start_c
                start_r, start_c = int(free_r[nearest]), int(free_c[nearest])
                L(f"[{label}] Start ({orig_r},{orig_c}) is blocked, snapped to nearest free cell ({start_r},{start_c})", "info")

    if _HAS_SCIPY and accessible2d.any():
        struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        labeled, n_comp = _scipy_label(accessible2d.astype(bool), structure=struct4)
        if n_comp > 1:
            # Pick the component containing the start point, or largest if no start
            if start_r is not None and labeled[start_r, start_c] > 0:
                keep_id = int(labeled[start_r, start_c])
                L(f"[{label}] Using start-point component (id={keep_id})", "info")
            else:
                comp_sizes = np.bincount(labeled.ravel())
                comp_sizes[0] = 0
                keep_id = int(comp_sizes.argmax())
            pre_filter = int(accessible2d.sum())
            accessible2d = (labeled == keep_id).astype(np.uint8)
            excluded = pre_filter - int(accessible2d.sum())
            if excluded > 0:
                L(f"[{label}] Connected-component filter: kept region of {n_comp}, excluded {excluded} unreachable cells", "info")
    elif accessible2d.any():
        labels, sizes = _bfs_largest_component(accessible2d)
        if len(sizes) > 1:
            if start_r is not None and labels[start_r, start_c] >= 0:
                keep_id = int(labels[start_r, start_c])
            else:
                keep_id = int(np.argmax(sizes))
            pre_filter = int(accessible2d.sum())
            accessible2d = (labels == keep_id).astype(np.uint8)
            excluded = pre_filter - int(accessible2d.sum())
            if excluded > 0:
                L(f"[{label}] BFS component filter: excluded {excluded} unreachable cells", "info")

    # Backwards compat: use_stc=True without planner → STC
    if planner is None and use_stc:
        planner = "STC"

    stc_components = 0
    stc_largest_tiles = 0
    stc_step = 1
    stc_path = []
    planner_name = _PLANNER_KEY.get(planner, planner) if planner else ""
    if planner:
        accessible2d, stc_components, stc_largest_tiles, stc_step, stc_path = run_planner(
            planner, accessible2d, resolution
        )
        L(
            f"[{label}] {planner} planner: largest connected region kept "
            f"({stc_components} components, largest={stc_largest_tiles} coarse cells, cell={stc_step * resolution:.3f} m)",
            "info",
        )

        # Compute TRUE covered area by sweeping the robot footprint along the planner path.
        # Only cells physically touched by the robot body as it follows the path are "covered".
        if stc_path:
            swept = np.zeros((h, w), dtype=np.uint8)
            step = max(1, stc_step)
            for pr, pc in stc_path:
                # Convert coarse cell to fine grid center
                cr = int((pr + 0.5) * step)
                cc = int((pc + 0.5) * step)
                # Stamp robot footprint at this position
                r0 = max(0, cr - inflY)
                r1 = min(h, cr + inflY + 1)
                c0 = max(0, cc - inflX)
                c1 = min(w, cc + inflX + 1)
                if isRect:
                    swept[r0:r1, c0:c1] = 1
                else:
                    for dy in range(r0, r1):
                        for dx in range(c0, c1):
                            if (dy - cr) ** 2 + (dx - cc) ** 2 <= inflX * inflX:
                                swept[dy, dx] = 1
            # Only count swept cells that are on known floor
            swept &= floor_mask2d
            swept_cells = int(swept.sum())
            swept_area = float(swept_cells) * resolution * resolution
            L(f"[{label}] Swept footprint: {swept_area:.2f}m² ({swept_cells} cells) along {len(stc_path)} waypoints", "info")
            # Use swept area as the coverage result instead of inflated-free-space
            accessible2d = swept

    accessible_cells = int(accessible2d.sum())
    accessible_area = float(accessible_cells) * resolution * resolution
    total_floor_cells = int(floor_mask2d.sum())
    total_floor_area = float(total_floor_cells) * resolution * resolution
    rii_horizontal = (accessible_area / total_floor_area * 100.0) if total_floor_area > 0 else 0.0

    L(f"[{label}] Total floor: {total_floor_area:.2f}m² ({total_floor_cells} px)", "info")
    L(f"[{label}] Inflated accessible: {accessible_area:.2f}m² ({accessible_cells} px)", "info")
    L(f"[{label}] Area paint: {time.time()-t0:.2f}s", "info")
    L(f"[{label}] Done: RII Horizontal={rii_horizontal:.1f}% ({accessible_area:.2f}/{total_floor_area:.2f}m²)", "success")

    return dict(
        coveredArea=accessible_area,
        reachableArea=accessible_area,
        accessibleArea=accessible_area,
        accessibleCells=accessible_cells,
        totalFloorArea=total_floor_area,
        totalFloorCells=total_floor_cells,
        riiHorizontal=rii_horizontal,
        useSTC=bool(planner),
        planner=planner_name or "",
        stcComponents=int(stc_components),
        stcLargestTiles=int(stc_largest_tiles),
        stcStep=int(stc_step),
        stcPath=stc_path,
        reachableCells=accessible_cells,
        waypoints=accessible_cells,
        blocked=inflated2d.ravel().copy(),
        sourceBlocked=blocked_src.ravel().copy(),
        floorPx=floor_mask2d.ravel().copy(),
        covPx=accessible2d.ravel().copy(),
        params=params,
        resolution=resolution,
        coarsePath=[],
        step=1,
        cw=w,
        ch=h,
        w=w,
        h=h,
    )


def run_coverage(
    pgm_path,
    yaml_path,
    params,
    start_x,
    start_y,
    sel_mask,
    label,
    logf=None,
    traversable_pgm_path=None,
    floor_pgm_path=None,
    use_stc=False,
    planner=None,
    ground_analysis_result=None,
    accurate_footprint=False,
    footprint_motion="differential",
    coverage_mode="inflation",  # "inflation" | "footprint_fit" | "coverage_path"
    wall_safety_cells=0,
):
    """Horizontal RII area computation."""
    L = logf if logf else lambda m, c="": None
    t_total = time.time()
    P = lambda m: print(f"  [{label}] {m}")

    w, h, pixels = parse_pgm(pgm_path)
    yd = parse_yaml(yaml_path)
    res = yd['resolution']
    ox, oy = yd['origin'][0], yd['origin'][1]
    ft = yd['free_thresh']
    neg = yd['negate']
    fpt = math.ceil(255 * (1 - ft))

    P(f"Map: {w}x{h}, res={res}, fpt={fpt}")
    L(f"[{label}] Map: {w}x{h}, res={res}", "info")
    _shape_str = params.get('shape', 'circular')
    if _shape_str == 'rectangular':
        _fp_str = f"rect W={params.get('halfW',0)*2:.2f} × L={params.get('halfL',0)*2:.2f} m"
    else:
        _fp_str = f"circle r={params.get('radius', params.get('halfW',0)):.2f} m"
    L(f"[{label}] Coverage mode: {coverage_mode}  |  Footprint: {_fp_str}", "info")

    t0 = time.time()
    pix2d = pixels.reshape(h, w)
    flipped = pix2d[::-1, :].ravel().copy()
    if neg: flipped = 255 - flipped
    blocked = (flipped < fpt).astype(np.uint8)
    P(f"Blocked map: {time.time()-t0:.3f}s, blocked={int(np.sum(blocked))}, free={int(np.sum(blocked==0))}")
    L(f"[{label}] Blocked map: {time.time()-t0:.3f}s", "info")

    floor_mask = None
    if floor_pgm_path and os.path.isfile(floor_pgm_path):
        t0 = time.time()
        fw, fh, floor_pixels = parse_pgm(floor_pgm_path)
        if fw == w and fh == h:
            floor_flip = floor_pixels.reshape(h, w)[::-1, :].ravel().copy()
            if neg:
                floor_flip = 255 - floor_flip
            floor_mask = (floor_flip >= fpt).astype(np.uint8)
            L(f"[{label}] Floor mask: {time.time()-t0:.3f}s", "info")
        else:
            L(
                f"[{label}] Floor mask size mismatch: "
                f"{fw}x{fh} vs map {w}x{h}. Ignoring floor sidecar.",
                "warn",
            )

    trav_mask = None
    if traversable_pgm_path and os.path.isfile(traversable_pgm_path):
        t0 = time.time()
        tw, th, trav_pixels = parse_pgm(traversable_pgm_path)
        if tw == w and th == h:
            trav_flip = trav_pixels.reshape(h, w)[::-1, :].ravel().copy()
            if neg:
                trav_flip = 255 - trav_flip
            trav_mask = (trav_flip >= fpt).astype(np.uint8)
            if floor_mask is None:
                floor_mask = trav_mask.copy()
                L(f"[{label}] Floor denominator fallback: using traversability sidecar.", "warn")
            L(f"[{label}] Traversability mask: {time.time()-t0:.3f}s", "info")
        else:
            L(
                f"[{label}] Traversability mask size mismatch: "
                f"{tw}x{th} vs map {w}x{h}. Ignoring sidecar.",
                "warn",
            )

    # NOTE: Previously this code cleared obstacle cells wherever the traversability
    # mask said "traversable" (blocked[trav_mask == 1] = 0). This was WRONG because
    # the traversability map uses a wider Z-range that can mark wall bases as
    # "traversable floor", punching holes in walls and letting the robot through gaps
    # it physically cannot fit. The obstacle map is authoritative for walls/obstacles.
    # The traversability mask is now only used to RESTRICT access (non-traversable
    # floor stays blocked), never to UNBLOCK obstacle cells.
    if trav_mask is not None:
        if floor_mask is not None:
            floor_mask[trav_mask == 1] = 1

    # ── Ground analysis: block non-traversable ramps/steps on the obstacle map ──
    if ground_analysis_result is not None:
        t0 = time.time()
        n_blocked_cells = 0
        analysis_origin = ground_analysis_result.grid_origin
        analysis_cell = ground_analysis_result.cell_size
        for t in ground_analysis_result.transitions:
            if t.traversable:
                continue
            is_manual = getattr(t, '_is_manual', False)
            for ci in range(t.cells.shape[0]):
                row_a, col_a = int(t.cells[ci, 0]), int(t.cells[ci, 1])
                if is_manual:
                    # Manual ramps: cells are PGM pixel coords (row=py_pgm, col=px)
                    # The blocked array is flipped: row 0 = bottom of world
                    # PGM pixel (row_a, col_a) maps to flipped row = (h-1-row_a)
                    fpx = col_a
                    fpy = h - 1 - row_a
                    if 0 <= fpx < w and 0 <= fpy < h:
                        idx = fpy * w + fpx
                        if blocked[idx] == 0:
                            blocked[idx] = 1
                            n_blocked_cells += 1
                else:
                    # Auto-detected: convert from analysis grid to map pixels
                    wx = analysis_origin[0] + (col_a + 0.5) * analysis_cell
                    wy = analysis_origin[1] + (row_a + 0.5) * analysis_cell
                    px = int((wx - ox) / res)
                    py = int((wy - oy) / res)
                    if 0 <= px < w and 0 <= py < h:
                        idx = py * w + px
                        if blocked[idx] == 0:
                            blocked[idx] = 1
                            n_blocked_cells += 1
        n_blocked_transitions = sum(
            1 for t in ground_analysis_result.transitions if not t.traversable
        )
        L(f"[{label}] Ground analysis: {n_blocked_transitions} non-traversable "
          f"transition(s), {n_blocked_cells} cells blocked ({time.time()-t0:.3f}s)", "info")

    if sel_mask is not None:
        t0 = time.time()
        blocked[sel_mask == 0] = 1
        if floor_mask is not None:
            floor_mask[sel_mask == 0] = 0
        if trav_mask is not None:
            trav_mask[sel_mask == 0] = 0
        L(f"[{label}] Selection mask: {time.time()-t0:.3f}s", "info")

    source_blocked = blocked.copy()

    # ═══════════════════════════════════════════════════════════════════════
    # Accurate footprint-fit branch — rotation-aware + differential drive.
    # Skips the inflation + coarse-grid planner entirely.
    # coverage_mode="footprint_fit" → BFS reachable swept-body mask
    # coverage_mode="coverage_path" → BFS + boustrophedon simulation
    # (The older accurate_footprint=True flag still maps to "footprint_fit".)
    # ═══════════════════════════════════════════════════════════════════════
    if accurate_footprint or coverage_mode in ("footprint_fit", "coverage_path"):
        from core.footprint_coverage import compute_footprint_reachable, simulate_coverage_path
        shape = params.get('shape', 'circular')
        halfW = params.get('halfW', params.get('radius', 0.35))
        halfL = params.get('halfL', params.get('radius', 0.35))
        blocked2d = blocked.reshape(h, w)

        # Seed from a SINGLE start cell (the user's start point). The
        # selection rectangle is already applied to `blocked` earlier
        # as a BOUNDARY; if we also used it as seeds, cells inside sealed
        # rooms enclosed by the selection would independently flood-fill
        # and look reachable even though the robot can't reach them.
        sx_fine = max(0, min(int((start_x - ox) / res), w - 1))
        sy_fine = max(0, min(int((start_y - oy) / res), h - 1))
        seed_mask = np.zeros((h, w), dtype=np.uint8)
        seed_mask[sy_fine, sx_fine] = 1

        t0 = time.time()
        use_coverage_path = (coverage_mode == "coverage_path")
        if use_coverage_path:
            reachable2d, fp_meta = simulate_coverage_path(
                blocked2d, halfW, halfL, shape, res, seed_mask,
                wall_safety_cells=wall_safety_cells,
                motion_model=footprint_motion,
                logf=L,
            )
            L(f"[{label}] Coverage path simulation done: {time.time()-t0:.2f}s  "
              f"swept={fp_meta['reachable_cells']}  path={fp_meta.get('path_length', 0)}", "success")
        else:
            reachable2d, fp_meta = compute_footprint_reachable(
                blocked2d, halfW, halfL, shape, res, seed_mask,
                motion_model=footprint_motion,
                wall_safety_cells=wall_safety_cells, logf=L,
            )
            fp_meta.pop("visited_cube", None)
            L(f"[{label}] Accurate footprint fit done: {time.time()-t0:.2f}s  "
              f"reachable={fp_meta['reachable_cells']}  states={fp_meta['states_expanded']}", "success")

        # Floor denominator (same rule as the inflation path)
        if trav_mask is not None:
            floor_mask2d = trav_mask.reshape(h, w)
        elif floor_mask is not None:
            floor_mask2d = floor_mask.reshape(h, w)
        else:
            floor_mask2d = (blocked2d == 0).astype(np.uint8)

        total_floor_cells = int(floor_mask2d.sum())
        total_floor_area = float(total_floor_cells) * res * res
        accessible_cells = int(reachable2d.sum())
        accessible_area = float(accessible_cells) * res * res
        rii_horizontal = (accessible_area / total_floor_area * 100.0) if total_floor_area > 0 else 0.0
        L(f"[{label}] Total floor: {total_floor_area:.2f}m², Covered: {accessible_area:.2f}m²", "info")
        L(f"[{label}] RII Horizontal = {rii_horizontal:.1f}% "
          f"(footprint-fit, {fp_meta['orientations']} orientations, {footprint_motion})", "success")

        sx_rep = max(0, min(int((start_x - ox) / res), w - 1))
        sy_rep = max(0, min(int((start_y - oy) / res), h - 1))
        result = dict(
            coveredArea=accessible_area,
            reachableArea=accessible_area,
            accessibleArea=accessible_area,
            accessibleCells=accessible_cells,
            totalFloorArea=total_floor_area,
            totalFloorCells=total_floor_cells,
            riiHorizontal=rii_horizontal,
            useSTC=False,
            planner=planner or "footprint-fit",
            stcComponents=1,
            stcLargestTiles=accessible_cells,
            stcStep=1,
            stcPath=[],
            reachableCells=accessible_cells,
            waypoints=0,
            blocked=blocked.copy(),
            sourceBlocked=source_blocked.copy(),
            floorPx=floor_mask2d.ravel().copy(),
            covPx=reachable2d.copy(),
            trafficHeatmap=np.zeros((h, w), dtype=np.int32).ravel(),
            params=dict(params),
            resolution=float(res),
            coarsePath=[],
            step=1,
            cw=w,
            ch=h,
            w=w,
            h=h,
            footprintMeta=fp_meta,
            accurateFootprint=True,
            footprintMotion=footprint_motion,
        )
        result.update(
            origin=(float(ox), float(oy)),
            requestedStartWorld=(float(start_x), float(start_y)),
            effectiveStartWorld=(float(start_x), float(start_y)),
            requestedStartCell=(sy_rep, sx_rep),
            effectiveStartCell=(sy_rep, sx_rep),
            startAdjusted=True,
            startAdjustmentReason=None,
            startComponentSize=accessible_cells,
            largestComponentSize=accessible_cells,
        )
        L(f"[{label}] Total time: {time.time() - t_total:.2f}s", "info")
        return result

    # ── Build the driveable map from traversability (primary) + obstacles (fallback) ──
    # The traversability map already encodes where the robot can physically drive
    # (slope, step height checked). We use it as the base, then inflate
    # obstacle walls on top to prevent the robot center from getting too close.
    shape = params.get('shape', 'circular')
    halfW = params.get('halfW', params.get('radius', 0.35))
    halfL = params.get('halfL', params.get('radius', 0.35))
    inflX = max(0, math.ceil(halfW / res))
    inflY = max(0, math.ceil(halfL / res))
    isRect = (shape == 'rectangular')
    min_gap = (2 * inflX + 1) * res
    L(f"[{label}] Inflate: {'rect' if isRect else 'circle'} {inflX}x{inflY}px "
      f"(halfW={halfW:.3f}m, halfL={halfL:.3f}m, min passable gap={min_gap:.2f}m)", "info")

    t0 = time.time()
    b2d = blocked.reshape(h, w)

    if trav_mask is not None:
        # Use traversability map as the primary driveable surface.
        # The traversability map already encodes walls, slopes, and steps.
        # We only inflate the traversability boundary (edges of non-traversable regions)
        # by the robot half-footprint so the robot center stays safely away from edges.
        trav2d = trav_mask.reshape(h, w)
        non_trav = (trav2d == 0).astype(np.uint8)
        inflated_boundary = _dilate_binary_mask(non_trav, inflX, inflY, isRect)
        blocked = inflated_boundary.ravel()
        free_count = int((blocked == 0).sum())
        L(f"[{label}] Traversability-based: inflated non-traversable boundary → "
          f"free={free_count} cells (trav={int(trav2d.sum())}, inflated edge={int(inflated_boundary.sum()) - int(non_trav.sum())})", "info")
    else:
        # No traversability map — fall back to inflating the obstacle map
        inflated2d = _dilate_binary_mask(b2d, inflX, inflY, isRect)
        blocked = inflated2d.ravel()
        L(f"[{label}] No traversability map — using inflated obstacle map, "
          f"free={int((blocked==0).sum())}", "info")
    L(f"[{label}] Inflation done: {time.time()-t0:.2f}s", "info")

    # ── Coarse grid for coverage planner (cell size ≈ robot body HALF) ──
    # Cell size used to be 2*bodyHalf (full body width). That made the
    # planner miss gaps that weren't aligned to a 1-body-width grid: a
    # 1.5 m gap with a 1 m robot landed as one isolated free coarse cell
    # straddled by blocked neighbours, so BFS couldn't enter.
    # Halving the cell to a half-overlap grid removes the alignment
    # lottery; gaps wider than ~1.5 × robot body always show a chain of
    # free coarse cells regardless of where the gap falls.
    bodyHalf = min(halfW, halfL)
    stepM = bodyHalf           # was 2.0 * bodyHalf
    MIN_STEP = 4  # minimum coarse cell size in pixels
    step = max(MIN_STEP, round(stepM / res))
    cw = math.ceil(w / step)
    ch = math.ceil(h / step)

    # Build coarse grid: a coarse cell is free if the majority of its fine cells are free.
    # Using a 50% threshold prevents sparse traversability maps from making everything blocked.
    ph, pw = ch * step, cw * step
    bp = np.ones((ph, pw), dtype=np.uint8)
    bp[:h, :w] = blocked.reshape(h, w)
    tile_sums = bp.reshape(ch, step, cw, step).sum(axis=(1, 3))
    tile_total = step * step
    # A coarse cell is blocked if ANY of its fine cells are blocked.
    # This ensures the full robot body fits — no squeezing through gaps.
    coarse = (tile_sums > 0).astype(np.uint8).ravel()
    coarseFree = int((coarse == 0).sum())
    L(f"[{label}] Coarse grid: {cw}x{ch}, step={step} ({step*res:.3f}m), free={coarseFree}", "info")

    # ── Find the largest connected free region and place start there ──
    sx = int((start_x - ox) / res / step) if start_x != 0 or start_y != 0 else cw // 2
    sy = int((start_y - oy) / res / step) if start_x != 0 or start_y != 0 else ch // 2
    sx = max(0, min(sx, cw - 1))
    sy = max(0, min(sy, ch - 1))

    if coarseFree > 0:
        coarse2d = coarse.reshape(ch, cw)
        # Find connected components and keep the largest one
        labels_c, sizes_c = _bfs_largest_component(coarse2d == 0)
        if sizes_c:
            # If user specified a start and it's in a component, use that component
            user_comp = -1
            if coarse2d[sy, sx] == 0 and labels_c[sy, sx] >= 0:
                user_comp = int(labels_c[sy, sx])
            # Otherwise use the largest component
            if user_comp >= 0 and sizes_c[user_comp] > 10:
                keep_comp = user_comp
            else:
                keep_comp = int(np.argmax(sizes_c))
            # Zero out all cells not in the chosen component
            coarse2d[labels_c != keep_comp] = 1
            coarse = coarse2d.ravel()
            coarseFree = int(sizes_c[keep_comp])
            L(f"[{label}] Kept largest connected region: {coarseFree} cells (of {len(sizes_c)} components)", "info")

            # Place start in the chosen component, nearest to requested position
            comp_mask = (labels_c == keep_comp)
            comp_indices = np.argwhere(comp_mask)
            dists = (comp_indices[:, 0] - sy) ** 2 + (comp_indices[:, 1] - sx) ** 2
            nearest = int(dists.argmin())
            sy, sx = int(comp_indices[nearest, 0]), int(comp_indices[nearest, 1])
            L(f"[{label}] Start: coarse ({sx}, {sy}) in component of {coarseFree} cells", "info")
        else:
            L(f"[{label}] No connected components found in coarse grid", "warn")
    else:
        L(f"[{label}] No free coarse cells", "warn")

    # ── Coverage planner (spiral STC with BFS+A* fallback) ──
    DX = [1, 0, -1, 0]
    DY = [0, 1, 0, -1]
    covPx = np.zeros(w * h, dtype=np.uint8)
    covArea = 0.0
    path = []

    if coarseFree == 0:
        L(f"[{label}] No free coarse cells — zero coverage.", "warn")
    else:
        def _bfs_nearest(co, vis, sx, sy):
            seen = np.zeros(cw * ch, dtype=np.uint8)
            q = deque([(sx, sy)])
            seen[sy * cw + sx] = 1
            while q:
                bx, by = q.popleft()
                k = by * cw + bx
                if not vis[k] and not co[k]:
                    return (bx, by)
                for d in range(4):
                    nx, ny = bx + DX[d], by + DY[d]
                    if 0 <= nx < cw and 0 <= ny < ch:
                        nk = ny * cw + nx
                        if not seen[nk] and not co[nk]:
                            seen[nk] = 1
                            q.append((nx, ny))
            return None

        def _astar(co, sx, sy, gx, gy):
            if sx == gx and sy == gy:
                return [(sx, sy)]
            t = cw * ch
            g = np.full(t, 1e9, dtype=np.float32)
            p = np.full(t, -1, dtype=np.int32)
            sk = sy * cw + sx
            gk = gy * cw + gx
            g[sk] = 0
            import heapq
            heap = [(abs(gx - sx) + abs(gy - sy), sk)]
            found = False
            while heap:
                f, ck = heapq.heappop(heap)
                if ck == gk:
                    found = True
                    break
                bx, by = ck % cw, ck // cw
                cg = g[ck]
                if cg > f:
                    continue
                for d in range(4):
                    nx, ny = bx + DX[d], by + DY[d]
                    if 0 <= nx < cw and 0 <= ny < ch:
                        nk = ny * cw + nx
                        if not co[nk]:
                            ng = cg + 1
                            if ng < g[nk]:
                                g[nk] = ng
                                p[nk] = ck
                                heapq.heappush(heap, (ng + abs(gx - nx) + abs(gy - ny), nk))
            if not found:
                return None
            rpath = []
            k = gk
            while k != -1:
                rpath.append((k % cw, k // cw))
                k = p[k]
            rpath.reverse()
            return rpath

        # Spiral coverage
        t0 = time.time()
        visited = np.zeros(cw * ch, dtype=np.uint8)
        cx, cy, d = sx, sy, 0
        visited[cy * cw + cx] = 1
        vc = 1
        path.append((cx, cy))

        for _ in range(coarseFree * 4):
            if vc >= coarseFree:
                break
            moved = False
            for nd in [(d + 1) % 4, d, (d + 3) % 4, (d + 2) % 4]:
                nx, ny = cx + DX[nd], cy + DY[nd]
                if 0 <= nx < cw and 0 <= ny < ch and not coarse[ny * cw + nx] and not visited[ny * cw + nx]:
                    cx, cy, d = nx, ny, nd
                    visited[cy * cw + cx] = 1
                    vc += 1
                    path.append((cx, cy))
                    moved = True
                    break
            if not moved:
                tgt = _bfs_nearest(coarse, visited, cx, cy)
                if not tgt:
                    break
                trans = _astar(coarse, cx, cy, tgt[0], tgt[1])
                if not trans:
                    visited[tgt[1] * cw + tgt[0]] = 1
                    vc += 1
                    continue
                for ti in range(1, len(trans)):
                    tx, ty = trans[ti]
                    if not visited[ty * cw + tx]:
                        visited[ty * cw + tx] = 1
                        vc += 1
                    path.append(trans[ti])
                cx, cy = trans[-1]
                if len(trans) >= 2:
                    ddx = cx - trans[-2][0]
                    ddy = cy - trans[-2][1]
                    for dd in range(4):
                        if DX[dd] == ddx and DY[dd] == ddy:
                            d = dd
                            break

        L(f"[{label}] Planner: {time.time()-t0:.2f}s, visited={vc}/{coarseFree}, path={len(path)} waypoints", "info")

        # ── Paint coverage: each visited coarse cell → robot covers that tile ──
        cov2d = covPx.reshape(h, w)
        blk2d = blocked.reshape(h, w)
        visited_cells = set()
        for (ccx, ccy) in path:
            if (ccx, ccy) in visited_cells:
                continue
            visited_cells.add((ccx, ccy))
            py0 = ccy * step
            py1 = min(py0 + step, h)
            px0 = ccx * step
            px1 = min(px0 + step, w)
            tile = cov2d[py0:py1, px0:px1]
            blk_tile = blk2d[py0:py1, px0:px1]
            new_covered = (~tile.astype(bool)) & (~blk_tile.astype(bool))
            covArea += float(np.sum(new_covered)) * res * res
            tile[new_covered] = 1

        L(f"[{label}] Coverage: {covArea:.2f}m², {len(visited_cells)} tiles, {len(path)} waypoints", "info")

    # ── Build traffic heatmap: how many times each cell is visited by the planner ──
    traffic = np.zeros((h, w), dtype=np.int32)
    for (ccx, ccy) in path:
        py0 = ccy * step; py1 = min(py0 + step, h)
        px0 = ccx * step; px1 = min(px0 + step, w)
        traffic[py0:py1, px0:px1] += 1

    # ── Build result ──
    if trav_mask is not None:
        # Use traversability as the total driveable floor denominator
        floor_mask2d = trav_mask.reshape(h, w)
    elif floor_mask is not None:
        floor_mask2d = floor_mask.reshape(h, w)
    else:
        floor_mask2d = (blocked.reshape(h, w) == 0).astype(np.uint8)
    total_floor_cells = int(floor_mask2d.sum())
    total_floor_area = float(total_floor_cells) * res * res
    accessible_cells = int(covPx.sum())
    accessible_area = float(accessible_cells) * res * res
    rii_horizontal = (accessible_area / total_floor_area * 100.0) if total_floor_area > 0 else 0.0

    L(f"[{label}] Total floor: {total_floor_area:.2f}m², Covered: {accessible_area:.2f}m²", "info")
    L(f"[{label}] RII Horizontal = {rii_horizontal:.1f}%", "success")

    result = dict(
        coveredArea=accessible_area,
        reachableArea=accessible_area,
        accessibleArea=accessible_area,
        accessibleCells=accessible_cells,
        totalFloorArea=total_floor_area,
        totalFloorCells=total_floor_cells,
        riiHorizontal=rii_horizontal,
        useSTC=True,
        planner=planner or "STC",
        stcComponents=1,
        stcLargestTiles=coarseFree,
        stcStep=step,
        stcPath=path,
        reachableCells=accessible_cells,
        waypoints=len(path),
        blocked=blocked.copy(),
        sourceBlocked=source_blocked.copy(),
        floorPx=floor_mask2d.ravel().copy(),
        covPx=covPx.copy(),
        trafficHeatmap=traffic.ravel().copy(),
        params=dict(params),
        resolution=float(res),
        coarsePath=path,
        step=step,
        cw=cw,
        ch=ch,
        w=w,
        h=h,
    )
    eff_start_world = (ox + sx * step * res, oy + sy * step * res)
    result.update(
        origin=(float(ox), float(oy)),
        requestedStartWorld=(float(start_x), float(start_y)),
        effectiveStartWorld=eff_start_world,
        requestedStartCell=(sy * step, sx * step),
        effectiveStartCell=(sy * step, sx * step),
        startAdjusted=True,
        startAdjustmentReason=None,
        startComponentSize=0,
        largestComponentSize=coarseFree,
    )
    return result
