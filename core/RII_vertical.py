import time
import math
import numpy as np

_VOXLABEL_EMPTY = 0
_VOXLABEL_WALL  = 1
_VOXLABEL_OBS   = 2

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

try:
    from scipy.ndimage import label as _ndimage_label
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _voxel_index(p, origin, vs):
    q = np.floor((p - origin) / vs).astype(np.int32)
    return (int(q[0]), int(q[1]), int(q[2]))


def _voxel_center(k, origin, vs):
    return origin + (np.array(k, dtype=np.float32) + 0.5) * vs


# ── Numba-accelerated raycasting kernel ──────────────────────────────
if _HAS_NUMBA:
    @njit(cache=True)
    def _raycast_single(ox, oy, oz, dx, dy, dz, vs, t_max,
                        occ_grid, nx, ny, nz, max_steps):
        """Single-ray Amanatides & Woo traversal on a dense 3D grid.
        Returns (hit_ix, hit_iy, hit_iz, label).  label==0 means miss."""
        INF = 1e30
        ovx = ox / vs
        ovy = oy / vs
        ovz = oz / vs
        vx = int(math.floor(ovx))
        vy = int(math.floor(ovy))
        vz = int(math.floor(ovz))

        sx = 1 if dx > 1e-12 else (-1 if dx < -1e-12 else 0)
        sy = 1 if dy > 1e-12 else (-1 if dy < -1e-12 else 0)
        sz = 1 if dz > 1e-12 else (-1 if dz < -1e-12 else 0)

        tdx = abs(1.0 / dx) if abs(dx) > 1e-12 else INF
        tdy = abs(1.0 / dy) if abs(dy) > 1e-12 else INF
        tdz = abs(1.0 / dz) if abs(dz) > 1e-12 else INF

        if abs(dx) > 1e-12:
            nb = vx + (1 if sx > 0 else 0)
            tmx = (nb - ovx) / dx
        else:
            tmx = INF
        if abs(dy) > 1e-12:
            nb = vy + (1 if sy > 0 else 0)
            tmy = (nb - ovy) / dy
        else:
            tmy = INF
        if abs(dz) > 1e-12:
            nb = vz + (1 if sz > 0 else 0)
            tmz = (nb - ovz) / dz
        else:
            tmz = INF

        t_limit = t_max / vs  # work in voxel units
        t = 0.0
        for _ in range(max_steps):
            if 0 <= vx < nx and 0 <= vy < ny and 0 <= vz < nz:
                lab = occ_grid[vx, vy, vz]
                if lab != 0:
                    if t * vs <= t_max + 1e-6:
                        return vx, vy, vz, lab
                    break
            if tmx <= tmy and tmx <= tmz:
                t = tmx
                tmx += tdx
                vx += sx
            elif tmy <= tmz:
                t = tmy
                tmy += tdy
                vy += sy
            else:
                t = tmz
                tmz += tdz
                vz += sz
            if t > t_limit:
                break
        return -1, -1, -1, 0

    @njit(parallel=True, cache=True)
    def _batch_raycast(origins, dirs, vs, t_max,
                       occ_grid, wall_grid, painted_grid,
                       nx, ny, nz,
                       half_w, half_s, max_steps):
        """Batch raycast all origins × dirs.  Writes directly into painted_grid.
        Returns (hit_wall_count, hit_obs_count, miss_count, useful_ground array)."""
        n_origins = origins.shape[0]
        n_dirs = dirs.shape[0]
        # Per-thread stats stored in arrays for parallel reduction
        hit_w = np.zeros(n_origins, dtype=np.int64)
        hit_o = np.zeros(n_origins, dtype=np.int64)
        miss_c = np.zeros(n_origins, dtype=np.int64)
        useful = np.zeros(n_origins, dtype=np.uint8)

        for gi in prange(n_origins):
            ox_g = origins[gi, 0]
            oy_g = origins[gi, 1]
            oz_g = origins[gi, 2]
            for di in range(n_dirs):
                dx = dirs[di, 0]
                dy = dirs[di, 1]
                dz = dirs[di, 2]
                hx, hy, hz, lab = _raycast_single(
                    ox_g, oy_g, oz_g, dx, dy, dz,
                    vs, t_max, occ_grid, nx, ny, nz, max_steps)
                if lab == 0:
                    miss_c[gi] += 1
                elif lab == _VOXLABEL_WALL and wall_grid[hx, hy, hz]:
                    hit_w[gi] += 1
                    useful[gi] = 1
                    # Paint patch around hit
                    for ddx in range(-half_w, half_w + 1):
                        for ddy in range(-half_w, half_w + 1):
                            for ddz in range(-half_s, half_s + 1):
                                px = hx + ddx
                                py = hy + ddy
                                pz = hz + ddz
                                if 0 <= px < nx and 0 <= py < ny and 0 <= pz < nz:
                                    if wall_grid[px, py, pz]:
                                        painted_grid[px, py, pz] = 1
                else:
                    hit_o[gi] += 1
        return hit_w.sum(), hit_o.sum(), miss_c.sum(), useful


# ── Fallback pure-Python raycasting (no Numba) ──────────────────────
def _raycast_first_hit(origin_p, dir_u, voxel_origin, vs, occ, t_max, max_steps=4000):
    """Amanatides & Woo voxel traversal (pure-Python fallback). Returns (key|None, dist_m, label)."""
    d = dir_u.astype(np.float64)
    o = origin_p.astype(np.float64)
    o_v = (o - voxel_origin) / vs
    vx, vy, vz = np.floor(o_v).astype(np.int64)
    inf = 1e30
    step = np.where(np.abs(d) > 1e-12, np.sign(d), 0.0).astype(int)
    tDelta = np.full(3, inf, dtype=np.float64)
    np.divide(1.0, np.abs(d), out=tDelta, where=(np.abs(d) > 1e-12))

    def _tmax(o_ax, v_ax, s_ax, d_ax):
        if abs(d_ax) <= 1e-12:
            return inf
        nb = v_ax + (1 if s_ax > 0 else 0)
        return (nb - o_ax) / d_ax

    tMax = np.array([_tmax(o_v[0], vx, step[0], d[0]),
                     _tmax(o_v[1], vy, step[1], d[1]),
                     _tmax(o_v[2], vz, step[2], d[2])], dtype=np.float64)
    t = 0.0
    for _ in range(max_steps):
        key = (int(vx), int(vy), int(vz))
        lab = occ.get(key, _VOXLABEL_EMPTY)
        if lab != _VOXLABEL_EMPTY:
            dist_m = t * vs
            if dist_m <= t_max + 1e-6:
                return key, float(dist_m), lab
        axis = int(np.argmin(tMax))
        t = tMax[axis]
        tMax[axis] += tDelta[axis]
        if axis == 0: vx += step[0]
        elif axis == 1: vy += step[1]
        else: vz += step[2]
        if t * vs > t_max:
            break
    return None, float("inf"), _VOXLABEL_EMPTY


# ── Vectorized voxel building ────────────────────────────────────────
def _build_wall_voxels(pts, labels, wall_label_ids, voxel_size, wall_majority_thr=0.60,
                       ground_z_ref=None, wall_min_h=0.40, wall_max_h=2.00):
    """Voxelise point cloud, classify wall vs obstacle, extract wall band.
    Returns (occ_grid, voxel_origin, ground_z_ref, wall_grid, vs, grid_shape)
    where occ_grid and wall_grid are dense 3D uint8 arrays."""
    vs = voxel_size
    pts_min = pts.min(axis=0)
    pts_max = pts.max(axis=0)
    voxel_origin = pts_min - np.array([vs, vs, vs], dtype=np.float32)
    if ground_z_ref is None:
        ground_z_ref = float(np.percentile(pts[:, 2], 5.0))

    # Vectorized voxel index computation for ALL points at once
    indices = np.floor((pts - voxel_origin) / vs).astype(np.int32)  # (N, 3)
    grid_max = indices.max(axis=0)
    nx, ny, nz = int(grid_max[0]) + 2, int(grid_max[1]) + 2, int(grid_max[2]) + 2

    # Flatten 3D index to 1D for numpy bincount
    flat = indices[:, 0].astype(np.int64) * (ny * nz) + indices[:, 1].astype(np.int64) * nz + indices[:, 2].astype(np.int64)
    total_counts = np.bincount(flat, minlength=nx * ny * nz)

    wall_like = np.isin(labels, np.array(sorted(wall_label_ids), dtype=np.int32))
    wall_counts = np.bincount(flat[wall_like], minlength=nx * ny * nz)

    # Build dense occupancy grid
    occ_grid = np.zeros(nx * ny * nz, dtype=np.uint8)
    occupied = total_counts > 0
    frac = np.zeros(nx * ny * nz, dtype=np.float32)
    np.divide(wall_counts.astype(np.float32), total_counts.astype(np.float32),
              out=frac, where=occupied)
    occ_grid[occupied & (frac >= wall_majority_thr)] = _VOXLABEL_WALL
    occ_grid[occupied & (frac < wall_majority_thr)] = _VOXLABEL_OBS
    occ_grid = occ_grid.reshape(nx, ny, nz)

    # Extract wall band (wall voxels in height range)
    wall_grid = np.zeros((nx, ny, nz), dtype=np.uint8)
    zmin = ground_z_ref + wall_min_h
    zmax = ground_z_ref + wall_max_h
    # Compute z-center for each voxel layer
    z_centers = voxel_origin[2] + (np.arange(nz, dtype=np.float32) + 0.5) * vs
    z_mask = (z_centers >= zmin) & (z_centers <= zmax)
    wall_grid[:, :, z_mask] = (occ_grid[:, :, z_mask] == _VOXLABEL_WALL).astype(np.uint8)

    return occ_grid, voxel_origin, ground_z_ref, wall_grid, vs, (nx, ny, nz)


# ── Vectorized ground sampling ───────────────────────────────────────
def _sample_ground_from_rii(act_result, ground_z_ref, stride=3, max_samples=60000):
    """Extract reachable ground XY positions from horizontal RII accessible mask."""
    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result["resolution"])
    ox, oy = act_result["origin"]
    cov = np.asarray(act_result["covPx"], dtype=np.uint8).reshape(h, w)
    # Vectorized: subsample then find nonzero
    sub = cov[::stride, ::stride]
    rows, cols = np.nonzero(sub)
    # Convert back to original pixel coords
    rows = rows * stride
    cols = cols * stride
    if len(rows) > max_samples:
        sel = np.linspace(0, len(rows) - 1, max_samples, dtype=np.int64)
        rows = rows[sel]
        cols = cols[sel]
    if len(rows) == 0:
        return np.zeros((0, 2), dtype=np.float32), ground_z_ref
    xs = ox + (cols + 0.5) * res
    ys = oy + (rows + 0.5) * res
    out = np.stack([xs, ys], axis=1).astype(np.float32)
    return out, ground_z_ref


def compute_rii_vertical(
    pts_3d, labels, act_result,
    wall_label_ids=None,
    voxel_size=0.05, max_reach=1.0, angle_step_deg=10.0,
    wall_min_h=0.40, wall_max_h=2.00,
    ground_stride=3, max_ground_samples=60000,
    paint_width=0.25, paint_vertical_span=0.30, sweep_step=0.20,
    logf=None, progress_cb=None,
):
    """
    Compute RII Vertical: fraction of wall surface reachable from accessible floor.
    Uses STVL raycasting from reachable ground positions to wall voxels.
    Accelerated with Numba JIT when available, pure-Python fallback otherwise.

    Returns dict with:
      - tcr: Task Coverage Rate (painted/total wall) = RII_V
      - painted_area_m2, total_wall_area_m2
      - painted_voxels, wall_band (sets of voxel keys for visualization)
      - voxel_origin, ground_z_ref, voxel_size
      - ray stats
    """
    L = logf if logf else lambda m, c="": None
    P = progress_cb if progress_cb else lambda v: None

    if wall_label_ids is None:
        wall_label_ids = {1}

    t0 = time.time()
    P(5)
    L("[RII_V] Building wall voxel grid...", "info")

    occ_grid, voxel_origin, gref, wall_grid, vs, (nx, ny, nz) = _build_wall_voxels(
        pts_3d, labels, wall_label_ids, voxel_size,
        wall_min_h=wall_min_h, wall_max_h=wall_max_h,
    )
    n_occ = int((occ_grid > 0).sum())
    n_wall = int(wall_grid.sum())
    L(f"[RII_V] Voxels built: {n_occ} total, {n_wall} wall-band voxels", "info")

    if n_wall == 0:
        L("[RII_V] No wall voxels found in height band. Check wall_label_ids and height range.", "warn")
        return dict(tcr=0.0, painted_area_m2=0.0, total_wall_area_m2=0.0,
                    painted_voxels=set(), wall_band=set(), voxel_origin=voxel_origin,
                    ground_z_ref=gref, voxel_size=vs, rays_wall=0, rays_obstacle=0, rays_miss=0,
                    ground_samples=0, oe=0.0, sc=0.0, combined=0.0)

    P(15)
    L("[RII_V] Sampling reachable ground from RII_H...", "info")
    ground_xy, _ = _sample_ground_from_rii(act_result, gref, stride=ground_stride, max_samples=max_ground_samples)
    L(f"[RII_V] Ground samples: {len(ground_xy)}", "info")

    if len(ground_xy) == 0:
        L("[RII_V] No reachable ground cells. Run RII Horizontal first.", "warn")
        wall_band_set = set(zip(*np.where(wall_grid)))
        return dict(tcr=0.0, painted_area_m2=0.0, total_wall_area_m2=0.0,
                    painted_voxels=set(), wall_band=wall_band_set, voxel_origin=voxel_origin,
                    ground_z_ref=gref, voxel_size=vs, rays_wall=0, rays_obstacle=0, rays_miss=0,
                    ground_samples=0, oe=0.0, sc=0.0, combined=0.0)

    # Tool heights for vertical sweep
    z0 = gref + wall_min_h
    z1 = gref + wall_max_h
    tool_heights = np.arange(z0, z1 + 1e-6, max(1e-6, sweep_step), dtype=np.float32)

    # Ray directions (horizontal fan)
    ang = np.deg2rad(np.arange(0.0, 360.0, angle_step_deg, dtype=np.float32))
    dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1).astype(np.float32)

    half_span_vox = int(math.ceil((0.5 * paint_vertical_span) / vs))
    half_width_vox = int(math.ceil((0.5 * paint_width) / vs))

    # Build flat list of all ray origins: ground × heights
    # Each origin = (gx, gy, z) relative to voxel_origin (for Numba grid coords)
    n_ground = len(ground_xy)
    n_heights = len(tool_heights)
    all_origins = np.empty((n_ground * n_heights, 3), dtype=np.float32)
    for hi, z in enumerate(tool_heights):
        sl = slice(hi * n_ground, (hi + 1) * n_ground)
        all_origins[sl, 0] = ground_xy[:, 0] - voxel_origin[0]
        all_origins[sl, 1] = ground_xy[:, 1] - voxel_origin[1]
        all_origins[sl, 2] = z - voxel_origin[2]

    P(20)
    n_rays = len(all_origins) * len(dirs)
    L(f"[RII_V] Raycasting: {len(ground_xy)} ground x {n_heights} heights x {len(dirs)} dirs = {n_rays:,} rays...", "info")
    L(f"[RII_V] Using {'Numba JIT (parallel)' if _HAS_NUMBA else 'pure Python (slow)'}", "info")

    painted_grid = np.zeros((nx, ny, nz), dtype=np.uint8)

    if _HAS_NUMBA:
        # ── Fast path: Numba parallel raycasting ──
        hit_w, hit_o, miss_total, useful = _batch_raycast(
            all_origins, dirs, vs, max_reach,
            occ_grid, wall_grid, painted_grid,
            nx, ny, nz,
            half_width_vox, half_span_vox, 4000)
        hit_w = int(hit_w)
        hit_o = int(hit_o)
        miss_total = int(miss_total)
        n_useful_ground = int(useful.sum())
    else:
        # ── Slow path: pure Python fallback ──
        occ_dict = {}
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    if occ_grid[ix, iy, iz] != 0:
                        occ_dict[(ix, iy, iz)] = int(occ_grid[ix, iy, iz])
        wall_band_set_fb = set(zip(*[c.tolist() for c in np.where(wall_grid)]))

        painted_set = set()
        hit_w = hit_o = miss_total = 0
        n_useful_ground = 0
        total_iters = len(all_origins)
        for gi in range(total_iters):
            if gi % max(1, total_iters // 20) == 0:
                pct = 20 + int(70 * gi / max(1, total_iters))
                P(pct)
            ox_l, oy_l, oz_l = all_origins[gi]
            ground_useful = False
            for d in dirs:
                k, dist, lab = _raycast_first_hit(
                    np.array([ox_l, oy_l, oz_l], dtype=np.float32),
                    d, np.zeros(3, dtype=np.float32), vs, occ_dict, max_reach)
                if k is None:
                    miss_total += 1
                    continue
                if lab == _VOXLABEL_WALL and k in wall_band_set_fb:
                    hit_w += 1
                    ground_useful = True
                    hx, hy, hz = k
                    for ddx in range(-half_width_vox, half_width_vox + 1):
                        for ddy in range(-half_width_vox, half_width_vox + 1):
                            for ddz in range(-half_span_vox, half_span_vox + 1):
                                pk = (hx + ddx, hy + ddy, hz + ddz)
                                if pk in wall_band_set_fb:
                                    painted_set.add(pk)
                else:
                    hit_o += 1
            if ground_useful:
                n_useful_ground += 1
        # Write fallback results into painted_grid
        for (px, py, pz) in painted_set:
            if 0 <= px < nx and 0 <= py < ny and 0 <= pz < nz:
                painted_grid[px, py, pz] = 1

    P(92)
    n_painted = int(painted_grid.sum())
    painted_area = n_painted * (vs ** 2)
    total_wall_area = n_wall * (vs ** 2)
    tcr = (painted_area / total_wall_area) if total_wall_area > 0 else 0.0

    L(f"[RII_V] Rays: wall_hits={hit_w}, obstacle_hits={hit_o}, miss={miss_total}", "info")
    L(f"[RII_V] Painted: {n_painted}/{n_wall} voxels ({painted_area:.2f}/{total_wall_area:.2f} m²)", "info")
    L(f"[RII_V] TCR (RII_V) = {tcr*100:.1f}%", "success")
    L(f"[RII_V] Done in {time.time()-t0:.1f}s", "info")

    # ── Surface Continuity (SC) via scipy or fallback BFS ──
    sc = 0.0
    if n_painted > 0:
        if _HAS_SCIPY:
            labeled, n_components = _ndimage_label(painted_grid)
            if n_components > 0:
                comp_sizes = np.bincount(labeled.ravel())[1:]  # skip background (0)
                largest_component = int(comp_sizes.max())
                sc = largest_component / n_painted
        else:
            # Fallback: Python BFS on painted voxels
            painted_coords = set(zip(*[c.tolist() for c in np.where(painted_grid)]))
            remaining = set(painted_coords)
            largest_component = 0
            while remaining:
                seed = next(iter(remaining))
                queue = [seed]
                remaining.discard(seed)
                comp_size = 0
                while queue:
                    cur = queue.pop()
                    comp_size += 1
                    cx, cy, cz = cur
                    for ddx, ddy, ddz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                        nb = (cx+ddx, cy+ddy, cz+ddz)
                        if nb in remaining:
                            remaining.discard(nb)
                            queue.append(nb)
                if comp_size > largest_component:
                    largest_component = comp_size
            sc = largest_component / n_painted
        L(f"[RII_V] Surface Continuity (SC) = {sc*100:.1f}%", "info")

    # ── Operational Efficiency (OE) ──
    # Fraction of ground samples that could reach at least one wall voxel.
    oe = n_useful_ground / max(1, len(all_origins)) if n_useful_ground > 0 else 0.0
    oe = min(1.0, oe)

    P(95)

    # Convert dense grids back to sets of (ix, iy, iz) tuples for visualization compatibility
    wb_coords = np.argwhere(wall_grid)
    wall_band_set = set(map(tuple, wb_coords.tolist()))
    pp_coords = np.argwhere(painted_grid)
    painted_set = set(map(tuple, pp_coords.tolist()))

    result = dict(
        tcr=tcr,
        riiVertical=tcr * 100.0,
        painted_area_m2=painted_area,
        total_wall_area_m2=total_wall_area,
        painted_voxels=painted_set,
        wall_band=wall_band_set,
        voxel_origin=voxel_origin,
        ground_z_ref=gref,
        voxel_size=vs,
        rays_wall=hit_w,
        rays_obstacle=hit_o,
        rays_miss=miss_total,
        ground_samples=len(ground_xy),
        sc=sc,
        oe=oe,
    )
    P(100)
    return result


def compute_combined_rii(rii_h_pct, rii_v_result, gamma=0.5):
    """
    Combined RII for painting robot accessibility.

    RII_Combined = TCR × (γ · OE + (1-γ) · SC)

    Where:
      TCR = Task Coverage Rate = RII_V (painted wall / total wall)
      OE  = Operational Efficiency (useful floor / traversable floor)
      SC  = Surface Continuity (largest contiguous paintable / total paintable)
      γ   = balance factor (0-1), default 0.5

    Additionally reports a simple weighted average:
      RII_Weighted = α · RII_H + (1-α) · RII_V
    """
    tcr = rii_v_result.get("tcr", 0.0)
    oe = rii_v_result.get("oe", 0.0)
    sc = rii_v_result.get("sc", 0.0)

    # Task-completion oriented combined score
    combined_paint = tcr * (gamma * oe + (1 - gamma) * sc)

    # Simple weighted average (equal weighting)
    rii_v_pct = tcr * 100.0
    weighted_avg = 0.5 * rii_h_pct + 0.5 * rii_v_pct

    return dict(
        combined_paint=combined_paint * 100.0,
        weighted_avg=weighted_avg,
        tcr=tcr * 100.0,
        oe=oe * 100.0,
        sc=sc * 100.0,
        rii_h=rii_h_pct,
        rii_v=rii_v_pct,
        gamma=gamma,
    )


def identify_wall_segments(pts_3d, labels, wall_label_ids=None, voxel_size=0.20, min_area_m2=0.5):
    """
    Find distinct wall segments from labelled 3D point cloud.
    Groups wall points into connected components using 2D XY voxel grid.
    Returns list of dicts with: id, point_indices, area_m2, centroid, bbox_3d.
    """
    if wall_label_ids is None:
        wall_label_ids = {1}
    wall_mask = np.isin(labels, np.array(sorted(wall_label_ids), dtype=np.int32))
    wall_idx = np.where(wall_mask)[0]
    if len(wall_idx) == 0:
        return []

    wall_pts = pts_3d[wall_idx]
    vs = voxel_size
    # Project to 2D XY grid for connected component analysis
    xy = wall_pts[:, :2]
    mn = xy.min(axis=0)
    grid_coords = np.floor((xy - mn) / vs).astype(np.int32)
    gw = int(grid_coords[:, 0].max()) + 1
    gh = int(grid_coords[:, 1].max()) + 1

    # Map grid cell → list of point indices (into wall_idx)
    cell_to_pts = {}
    for i in range(len(wall_idx)):
        cell = (int(grid_coords[i, 0]), int(grid_coords[i, 1]))
        if cell not in cell_to_pts:
            cell_to_pts[cell] = []
        cell_to_pts[cell].append(i)

    # BFS connected components on grid cells (4-connected)
    visited = set()
    segments = []
    seg_id = 0
    for start_cell in cell_to_pts:
        if start_cell in visited:
            continue
        visited.add(start_cell)
        queue = [start_cell]
        comp_pts = []
        while queue:
            c = queue.pop()
            comp_pts.extend(cell_to_pts[c])
            cx, cy = c
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                nb = (cx+dx, cy+dy)
                if nb in cell_to_pts and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

        area = len(set((grid_coords[i, 0], grid_coords[i, 1]) for i in comp_pts)) * (vs ** 2)
        if area < min_area_m2:
            continue

        global_indices = wall_idx[np.array(comp_pts)]
        seg_pts = pts_3d[global_indices]
        centroid = seg_pts.mean(axis=0)
        bbox_min = seg_pts.min(axis=0)
        bbox_max = seg_pts.max(axis=0)
        seg_id += 1
        segments.append(dict(
            id=seg_id,
            point_indices=global_indices,
            num_points=len(global_indices),
            area_m2=area,
            centroid=centroid.tolist(),
            bbox_min=bbox_min.tolist(),
            bbox_max=bbox_max.tolist(),
            height_span=float(bbox_max[2] - bbox_min[2]),
            width_span=float(np.linalg.norm(bbox_max[:2] - bbox_min[:2])),
        ))

    segments.sort(key=lambda s: s["area_m2"], reverse=True)
    # Re-number
    for i, seg in enumerate(segments):
        seg["id"] = i + 1
    return segments


def colorize_cloud_with_walls(pts_3d, labels, wall_segments, selected_wall_ids=None, focused_wall_id=None, wall_label_ids=None):
    """
    Create per-point RGB colors: gray for non-wall, orange for wall, cyan for focused, pink for selected.
    Returns Nx3 uint8 array.
    """
    if wall_label_ids is None:
        wall_label_ids = {1}
    n = pts_3d.shape[0]
    colors = np.full((n, 3), 140, dtype=np.uint8)  # gray default

    # Wall points → orange
    wall_mask = np.isin(labels, np.array(sorted(wall_label_ids), dtype=np.int32))
    colors[wall_mask] = [255, 160, 0]

    selected = set(selected_wall_ids or [])
    focused = focused_wall_id

    for seg in wall_segments:
        sid = seg["id"]
        idx = seg["point_indices"]
        if sid == focused:
            colors[idx] = [80, 220, 255]  # cyan
        elif sid in selected:
            colors[idx] = [255, 90, 180]  # pink
        # else: keep orange from wall_mask above

    return colors
