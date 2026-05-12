import sys
import math
import time
import numpy as np
from collections import deque

from PyQt5.QtGui import QImage

from config import PCD_PACKAGE_DIR
if PCD_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PCD_PACKAGE_DIR)
from pcd_package.pcd_tools import load_xyz_and_labels

from core.RII_horizontal import _score_accessibility_from_masks, _footprint_inflation_pixels, _dilate_binary_mask
from core.rendering import _build_bg


SEMANTIC_LABEL_NAMES = {
    0: "Unlabelled / Background",
    1: "Wall",
    3: "Staircase",
    4: "Fixed Obstacles",
    5: "Temporary Ramps",
    6: "Safety Barriers and Signs",
    7: "Temporary Utilities",
    8: "Scaffold Structure",
    9: "Semi-Fixed Obstacles",
    10: "Large Materials",
    11: "Stored Equipment",
    12: "Mobile Machines and Vehicles",
    13: "Movable Objects",
    14: "Containers and Pallets",
    15: "Small Tools",
    17: "Portable Objects",
}

# All valid raw label IDs (used for iteration instead of range(16))
SEMANTIC_RAW_LABEL_IDS = sorted(SEMANTIC_LABEL_NAMES.keys())

SEMANTIC_FIXATION_GROUPS = {
    0: "Fixed",        # Unlabelled / Background
    1: "Fixed",        # Wall
    3: "Fixed",        # Staircase
    4: "Fixed",        # Fixed Obstacles
    5: "Semi-Fixed",   # Temporary Ramps
    6: "Semi-Fixed",   # Safety Barriers and Signs
    7: "Semi-Fixed",   # Temporary Utilities
    8: "Semi-Fixed",   # Scaffold Structure
    9: "Semi-Fixed",   # Semi-Fixed Obstacles
    10: "Movable",     # Large Materials
    11: "Movable",     # Stored Equipment
    12: "Movable",     # Mobile Machines and Vehicles
    13: "Movable",     # Movable Objects
    14: "Portable",    # Containers and Pallets
    15: "Portable",    # Small Tools
    17: "Portable",    # Portable Objects
}

SEMANTIC_RECOMMENDATIONS = {
    0: "Large unlabeled regions are reducing interpretability. Revisit the CloudCompare annotations so the gap can be tied to the correct construction elements.",
    1: "Wall-related gap is a hard structural constraint. Improve RII Horizontal by changing the route, robot footprint, or deployment zone rather than trying to move the wall.",
    3: "Staircases are fixed vertical barriers. Improve horizontal accessibility with alternative ramped routes, lift access, or a different robot platform.",
    4: "Fixed obstacles are permanent constraints. Consider site-layout changes, alternate access corridors, or a smaller robot footprint.",
    5: "Temporary ramps are semi-fixed and usually offer the best short-term intervention: adjust placement, width, or gradient so the robot can traverse them safely.",
    6: "Barrier/sign placement is a recoverable gap. Reposition compliant safety barriers and signs to reopen the corridor while maintaining site safety rules.",
    7: "Temporary utilities are often recoverable. Reroute hoses, cables, or power runs overhead or along protected edges to free the travel corridor.",
    8: "Scaffold placement is a semi-fixed constraint. Leave a protected robot corridor or reschedule traversal around scaffold-heavy phases.",
    9: "Semi-fixed obstacles should be reviewed during phase planning. Reposition them or create a temporary bypass to recover accessible area.",
    10: "Large materials are movable. Improve RII Horizontal through material staging zones, just-in-time delivery, or clearing buffer space around routes.",
    11: "Stored equipment is movable. Define dedicated parking/storage zones so equipment does not reduce corridor clearance.",
    12: "Mobile machines and vehicles are dynamic movable constraints. Use time windows, exclusion zones, or coordinated traffic rules to recover access.",
    13: "Movable objects indicate housekeeping issues. Regular clearing or better local storage can recover this portion of the RII gap quickly.",
    14: "Containers and pallets are portable and highly actionable. Relocating pallet stacks or container staging areas should recover this gap first.",
    15: "Small tools are portable clutter. Apply 5S-style housekeeping or tool-drop zones to restore traversable space.",
    17: "Portable objects are the most easily recoverable source of lost accessibility. Focus cleanup here before changing semi-fixed or fixed site elements.",
}

SEMANTIC_3D_COLORS = {
    0: (80, 80, 80),      # Unlabelled — dark gray (black in CC, brightened for visibility)
    1: (255, 170, 255),   # Wall — pink
    3: (249, 241, 0),     # Staircase — yellow
    4: (85, 170, 255),    # Fixed Obstacles — blue
    5: (255, 0, 0),       # Temporary Ramps — red
    6: (142, 106, 36),    # Safety Barriers — brown
    7: (255, 0, 127),     # Temporary Utilities — magenta-red
    8: (170, 255, 255),   # Scaffold Structure — cyan
    9: (170, 170, 255),   # Semi-Fixed Obstacles — periwinkle
    10: (85, 85, 0),      # Large Materials — olive
    11: (132, 132, 140),  # Stored Equipment — gray
    12: (170, 255, 0),    # Mobile Machines — lime
    13: (179, 20, 176),   # Movable Objects — purple
    14: (0, 170, 255),    # Containers — sky blue
    15: (255, 85, 127),   # Small Tools — coral
    17: (250, 255, 147),  # Portable Objects — pale yellow
}

SEMANTIC_REMOVABLE_FIXATIONS = ("Portable", "Movable", "Semi-Fixed")
SEMANTIC_LAYER_SEQUENCE = [
    ("Actual site state", ()),
    ("Portable removed", ("Portable",)),
    ("Portable + Movable removed", ("Portable", "Movable")),
    ("Structural maximum (only Fixed remains)", ("Portable", "Movable", "Semi-Fixed")),
]


def load_semantic_pcd(pcd_path):
    """Load a CloudCompare-labeled PCD or PLY using the shared parser."""
    try:
        pts, labels, label_field = load_xyz_and_labels(pcd_path)
    except Exception:
        return None, None, None

    if labels is not None:
        labels = np.clip(labels, 0, 17)
    return pts, labels, label_field


def project_labels_to_2d_grid(pts, labels, yaml_data, map_w, map_h):
    """Project 3D semantic labels onto the 2D occupancy grid.

    For each grid cell, assigns the most common label among all 3D points
    that fall within that cell's horizontal footprint.

    Returns:
        label_grid: (map_h * map_w) int array, -1 = no label data
    """
    res = yaml_data['resolution']
    ox, oy = yaml_data['origin'][0], yaml_data['origin'][1]

    label_grid = np.full(map_h * map_w, -1, dtype=np.int32)

    # Convert 3D points to grid indices
    gx = ((pts[:, 0] - ox) / res).astype(np.int32)
    gy = ((pts[:, 1] - oy) / res).astype(np.int32)

    # Filter points within grid bounds
    valid = (gx >= 0) & (gx < map_w) & (gy >= 0) & (gy < map_h)
    gx, gy, lab = gx[valid], gy[valid], labels[valid]

    # For each cell, use the most frequent label (mode)
    # Use accumulation: for each cell, track label counts
    cell_indices = gy * map_w + gx

    # Group by cell and find mode
    for label_val in SEMANTIC_RAW_LABEL_IDS:
        mask = lab == label_val
        if not np.any(mask):
            continue
        cells = cell_indices[mask]
        unique_cells, cell_counts = np.unique(cells, return_counts=True)
        for ci, count in zip(unique_cells, cell_counts):
            if label_grid[ci] == -1:
                label_grid[ci] = label_val
            else:
                # Check if this label has more points in this cell
                # Simple approach: last-wins for ties, most-common otherwise
                # We'll do a simpler approach: highest count wins
                pass  # first-assigned approach for speed

    # More accurate approach: direct assignment by most common
    # Build count arrays per cell
    if len(cell_indices) > 0:
        label_counts = np.zeros((map_h * map_w, max(SEMANTIC_RAW_LABEL_IDS) + 1), dtype=np.int32)
        for i in range(len(cell_indices)):
            ci = cell_indices[i]
            li = lab[i]
            label_counts[ci, li] += 1

        has_data = label_counts.sum(axis=1) > 0
        label_grid[has_data] = label_counts[has_data].argmax(axis=1)

    return label_grid


def analyze_semantic_rii(ref_result, act_result, label_grid, yaml_data, label_names=None):
    """Analyze which semantic categories contribute to the RII gap.

    Args:
        ref_result: reference coverage result dict
        act_result: actual coverage result dict
        label_grid: (h*w) array of semantic labels (-1 = no data)
        yaml_data: map YAML config
        label_names: dict mapping label_id -> name

    Returns:
        dict with:
          - total_missed_area: float (m²)
          - label_breakdown: list of {label, name, area, pct, recommendation}
          - top_recommendations: list of strings
    """
    if label_names is None:
        label_names = SEMANTIC_LABEL_NAMES

    w, h = ref_result['w'], ref_result['h']
    res = yaml_data['resolution']
    cell_area = res * res

    # Normalise to 1-D (h*w,) so the boolean math broadcasts against
    # label_grid regardless of whether Step 3 produced a 1-D mask
    # (Default inflation path) or a 2-D one (Accurate footprint fit /
    # Coverage path).
    ref_cov = np.asarray(ref_result['covPx']).reshape(-1)
    act_cov = np.asarray(act_result['covPx']).reshape(-1)

    # Missed = covered by reference but NOT by actual
    missed = (ref_cov == 1) & (act_cov == 0)
    total_missed = float(np.sum(missed)) * cell_area

    # Break down by semantic label
    breakdown = []
    fixation_totals = {}
    for label_id in SEMANTIC_RAW_LABEL_IDS:
        in_label = (label_grid == label_id)
        label_missed = missed & in_label
        area = float(np.sum(label_missed)) * cell_area
        if area > 0:
            pct = (area / total_missed * 100) if total_missed > 0 else 0
            fixation = SEMANTIC_FIXATION_GROUPS.get(label_id, "Unknown")
            fixation_totals[fixation] = fixation_totals.get(fixation, 0.0) + area
            breakdown.append({
                'label': label_id,
                'name': label_names.get(label_id, f"Label {label_id}"),
                'fixation': fixation,
                'area': area,
                'pct': pct,
                'recommendation': SEMANTIC_RECOMMENDATIONS.get(label_id, "Review this area."),
            })

    # Also count missed area with no label data
    no_label_missed = missed & (label_grid == -1)
    no_label_area = float(np.sum(no_label_missed)) * cell_area
    if no_label_area > 0:
        pct = (no_label_area / total_missed * 100) if total_missed > 0 else 0
        breakdown.append({
            'label': -1,
            'name': "No Label Data",
            'fixation': "Unknown",
            'area': no_label_area,
            'pct': pct,
            'recommendation': "No semantic data for this area — ensure the labeled point cloud covers the full map extent.",
        })

    # Sort by area descending
    breakdown.sort(key=lambda x: x['area'], reverse=True)

    # Prioritize interventions that are easiest to act on first.
    priority = {"Portable": 0, "Movable": 1, "Semi-Fixed": 2, "Fixed": 3, "Unknown": 4}
    actionable = [b for b in breakdown if b['label'] not in (-1, 0, 1) and b['area'] > 0]
    actionable.sort(key=lambda b: (priority.get(b['fixation'], 99), -b['area']))
    top_recs = []
    for b in actionable[:3]:
        top_recs.append(
            f"• {b['name']} [{b['fixation']}] ({b['pct']:.1f}% of gap, {b['area']:.2f} m²): {b['recommendation']}"
        )

    fixation_breakdown = [
        {
            "fixation": fixation,
            "area": area,
            "pct": (area / total_missed * 100) if total_missed > 0 else 0.0,
        }
        for fixation, area in sorted(
            fixation_totals.items(),
            key=lambda item: (priority.get(item[0], 99), -item[1]),
        )
    ]

    return {
        'total_missed_area': total_missed,
        'label_breakdown': breakdown,
        'fixation_breakdown': fixation_breakdown,
        'top_recommendations': top_recs,
    }


def compute_semantic_layered_rii(act_result, label_grid, logf=None, progress_cb=None):
    """Recompute horizontal RII while progressively removing fixation groups."""
    L = logf if logf else lambda m, c="": None
    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result["resolution"])
    params = dict(act_result.get("params", {}))
    use_stc = bool(act_result.get("useSTC"))
    source_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]),
        dtype=np.uint8,
    ).reshape(h, w)
    floor_mask = act_result.get("floorPx")
    floor2d = np.asarray(floor_mask, dtype=np.uint8).reshape(h, w) if isinstance(floor_mask, np.ndarray) else None
    label2d = np.asarray(label_grid, dtype=np.int32).reshape(h, w)
    cell_area = res * res

    layers = []
    for idx, (name, excluded_fixations) in enumerate(SEMANTIC_LAYER_SEQUENCE):
        removed_mask = np.zeros((h, w), dtype=bool)
        if idx == 0:
            layer_result = dict(act_result)
        else:
            excluded_labels = [
                label_id
                for label_id, fixation in SEMANTIC_FIXATION_GROUPS.items()
                if fixation in excluded_fixations
            ]
            if excluded_labels:
                removed_mask = (source_blocked == 1) & np.isin(label2d, excluded_labels)
            test_blocked = np.asarray(source_blocked, dtype=np.uint8).copy()
            test_blocked[removed_mask] = 0
            layer_result = _score_accessibility_from_masks(
                test_blocked,
                floor2d,
                res,
                params,
                f"LAYER {idx}",
                L,
                use_stc=use_stc,
            )
            layer_result.update(
                sourceBlocked=test_blocked.ravel().copy(),
                params=dict(params),
                resolution=float(res),
                origin=tuple(act_result.get("origin", (0.0, 0.0))),
                removedCells=int(removed_mask.sum()),
                removedArea=float(removed_mask.sum()) * cell_area,
            )

        rii = float(layer_result.get("riiHorizontal", 0.0))
        accessible_area = float(layer_result.get("accessibleArea", layer_result.get("reachableArea", 0.0)))
        total_floor_area = float(layer_result.get("totalFloorArea", 0.0))
        removed_area = float(layer_result.get("removedArea", 0.0))
        if idx == 0:
            removed_area = 0.0
        layers.append({
            "layer": idx,
            "name": name,
            "excludedFixations": list(excluded_fixations),
            "riiHorizontal": rii,
            "accessibleArea": accessible_area,
            "totalFloorArea": total_floor_area,
            "removedArea": removed_area,
        })
        if progress_cb:
            progress_cb(idx + 1, len(SEMANTIC_LAYER_SEQUENCE), name)

    for idx, layer in enumerate(layers):
        prev = layers[idx - 1] if idx > 0 else None
        layer["deltaPts"] = 0.0 if prev is None else float(layer["riiHorizontal"] - prev["riiHorizontal"])
        layer["deltaArea"] = 0.0 if prev is None else float(layer["accessibleArea"] - prev["accessibleArea"])

    portable_delta = float(layers[1]["riiHorizontal"] - layers[0]["riiHorizontal"])
    movable_delta = float(layers[2]["riiHorizontal"] - layers[1]["riiHorizontal"])
    semi_fixed_delta = float(layers[3]["riiHorizontal"] - layers[2]["riiHorizontal"])
    return {
        "layers": layers,
        "rii_actual": float(layers[0]["riiHorizontal"]),
        "rii_structural_max": float(layers[-1]["riiHorizontal"]),
        "delta_portable": portable_delta,
        "delta_movable": movable_delta,
        "delta_semi_fixed": semi_fixed_delta,
        "improvement_potential": float(layers[-1]["riiHorizontal"] - layers[0]["riiHorizontal"]),
        "delta_portable_area": float(layers[1]["accessibleArea"] - layers[0]["accessibleArea"]),
        "delta_movable_area": float(layers[2]["accessibleArea"] - layers[1]["accessibleArea"]),
        "delta_semi_fixed_area": float(layers[3]["accessibleArea"] - layers[2]["accessibleArea"]),
    }


def simulate_removed_fixations(act_result, label_grid, excluded_fixations, label="FIXATION", logf=None):
    """Recompute horizontal RII after removing every blocked cell from the selected fixation groups."""
    excluded = tuple(dict.fromkeys(excluded_fixations))
    if not excluded:
        raise ValueError("Select one or more fixation groups first")

    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result["resolution"])
    params = dict(act_result.get("params", {}))
    use_stc = bool(act_result.get("useSTC"))
    source_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]),
        dtype=np.uint8,
    ).reshape(h, w)
    floor_mask = act_result.get("floorPx")
    floor2d = np.asarray(floor_mask, dtype=np.uint8).reshape(h, w) if isinstance(floor_mask, np.ndarray) else None
    label2d = np.asarray(label_grid, dtype=np.int32).reshape(h, w)
    excluded_labels = [
        label_id
        for label_id, fixation in SEMANTIC_FIXATION_GROUPS.items()
        if fixation in excluded
    ]
    remove_mask = (source_blocked == 1) & np.isin(label2d, excluded_labels)
    modified = np.asarray(source_blocked, dtype=np.uint8).copy()
    modified[remove_mask] = 0
    improved = _score_accessibility_from_masks(
        modified,
        floor2d,
        res,
        params,
        label,
        logf,
        use_stc=use_stc,
    )
    improved.update(
        sourceBlocked=modified.ravel().copy(),
        params=dict(params),
        resolution=float(res),
        origin=tuple(act_result.get("origin", (0.0, 0.0))),
        removedCells=int(remove_mask.sum()),
        removedArea=float(remove_mask.sum()) * res * res,
        removedMode="fixation",
        excludedFixations=list(excluded),
    )
    return improved


def render_semantic_missed(ref_result, act_result, label_grid, bg_pgm=None):
    """Render an image showing missed areas colored by semantic label, overlaid on map.pgm."""
    w, h = ref_result['w'], ref_result['h']
    ref_cov = ref_result['covPx'].reshape(h, w)[::-1, :]
    act_cov = act_result['covPx'].reshape(h, w)[::-1, :]
    lg = label_grid.reshape(h, w)[::-1, :]
    blk = ref_result['blocked'].reshape(h, w)[::-1, :]

    LABEL_COLORS = SEMANTIC_3D_COLORS

    buf = _build_bg(h, w, ref_result, bg_pgm)
    _blend = bg_pgm is not None

    # Actual covered = green (blended with background)
    cov_mask = act_cov == 1
    if _blend:
        buf[cov_mask] = (0.35 * np.array([0, 200, 130], dtype=np.float32) + 0.65 * buf[cov_mask].astype(np.float32)).astype(np.uint8)
    else:
        buf[cov_mask] = [0, 200, 130]

    # Missed areas colored by semantic label
    missed = (ref_cov == 1) & (act_cov == 0)
    for label_id in SEMANTIC_RAW_LABEL_IDS:
        mask = missed & (lg == label_id)
        if np.any(mask):
            if _blend:
                buf[mask] = (0.45 * np.array(LABEL_COLORS[label_id], dtype=np.float32) + 0.55 * buf[mask].astype(np.float32)).astype(np.uint8)
            else:
                buf[mask] = LABEL_COLORS[label_id]

    # Missed with no label data = dim red
    mask_no_label = missed & (lg == -1)
    buf[mask_no_label] = [100, 30, 30]

    # Blocked areas (only darken if no bg_pgm, since bg already shows them)
    if not _blend:
        buf[(act_cov == 0) & (ref_cov == 0) & (blk == 1)] = [20, 24, 28]

    return QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


def _binary_components(mask2d: np.ndarray) -> list[np.ndarray]:
    height, width = mask2d.shape
    try:
        from scipy.ndimage import label as _label
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        labeled, n = _label(mask2d.astype(bool), structure=struct)
        comps = []
        for i in range(1, n + 1):
            flat = np.where(labeled.ravel() == i)[0].astype(np.int32)
            comps.append(flat)
        return comps
    except ImportError:
        pass
    # Fallback: pure Python BFS
    seen = np.zeros_like(mask2d, dtype=np.uint8)
    comps = []
    for row in range(height):
        for col in range(width):
            if mask2d[row, col] == 0 or seen[row, col]:
                continue
            q = deque([(row, col)])
            seen[row, col] = 1
            flat = []
            while q:
                rr, cc = q.popleft()
                flat.append(rr * width + cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask2d[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = 1
                        q.append((nr, nc))
            comps.append(np.asarray(flat, dtype=np.int32))
    return comps


def _integral_rect_sum(integral: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> int:
    if r0 >= r1 or c0 >= c1:
        return 0
    total = int(integral[r1 - 1, c1 - 1])
    if r0 > 0:
        total -= int(integral[r0 - 1, c1 - 1])
    if c0 > 0:
        total -= int(integral[r1 - 1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(integral[r0 - 1, c0 - 1])
    return total


def identify_semantic_removal_candidates(
    act_result,
    label_grid,
    yaml_data,
    max_candidates: int = 120,
    min_component_area_m2: float = 0.04,
    progress_cb=None,
):
    """Find connected removable-object components that plausibly reduce horizontal accessibility."""
    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(yaml_data["resolution"])
    ox, oy = float(yaml_data["origin"][0]), float(yaml_data["origin"][1])
    cell_area = res * res

    label2d = np.asarray(label_grid, dtype=np.int32).reshape(h, w)
    source_blocked = np.asarray(act_result.get("sourceBlocked", act_result["blocked"]), dtype=np.uint8).reshape(h, w)
    floor2d = np.asarray(act_result.get("floorPx"), dtype=np.uint8).reshape(h, w)
    cov2d = np.asarray(act_result["covPx"], dtype=np.uint8).reshape(h, w)
    inaccessible_floor = (floor2d == 1) & (cov2d == 0)
    inaccessible_integral = np.cumsum(
        np.cumsum(inaccessible_floor.astype(np.int32), axis=0),
        axis=1,
    )

    inflX, inflY, isRect = _footprint_inflation_pixels(act_result.get("params", {}), res)
    min_cells = max(1, int(math.ceil(min_component_area_m2 / max(cell_area, 1e-9))))

    priority = {"Portable": 0, "Movable": 1, "Semi-Fixed": 2}
    candidates = []
    candidate_id = 1
    removable_label_ids = [
        label_id for label_id, fixation in SEMANTIC_FIXATION_GROUPS.items()
        if fixation in SEMANTIC_REMOVABLE_FIXATIONS
    ]
    total_labels = max(1, len(removable_label_ids))
    for label_index, label_id in enumerate(removable_label_ids, start=1):
        fixation = SEMANTIC_FIXATION_GROUPS.get(label_id, "Unknown")
        base = ((source_blocked == 1) & (label2d == label_id)).astype(np.uint8)
        if not np.any(base):
            if progress_cb:
                progress_cb(label_index, total_labels, fixation, label_id)
            continue
        for flat_idx in _binary_components(base):
            if flat_idx.size < min_cells:
                continue
            rows = flat_idx // w
            cols = flat_idx % w
            r0 = max(0, int(rows.min()) - inflY)
            r1 = min(h, int(rows.max()) + inflY + 1)
            c0 = max(0, int(cols.min()) - inflX)
            c1 = min(w, int(cols.max()) + inflX + 1)
            # Approximate unlockable area from the inflation-expanded component bbox.
            # Exact recomputation still uses the real component mask; this fast pass is
            # only for ranking and listing semantic candidates interactively.
            potential_unlock_cells = _integral_rect_sum(
                inaccessible_integral,
                r0,
                r1,
                c0,
                c1,
            )
            if potential_unlock_cells <= 0:
                continue
            x0 = ox + float(cols.min()) * res
            x1 = ox + float(cols.max() + 1) * res
            y0 = oy + float(rows.min()) * res
            y1 = oy + float(rows.max() + 1) * res
            candidates.append({
                "id": candidate_id,
                "label": int(label_id),
                "name": SEMANTIC_LABEL_NAMES.get(label_id, f"Label {label_id}"),
                "fixation": fixation,
                "area": float(flat_idx.size) * cell_area,
                "cells": int(flat_idx.size),
                "potentialUnlockArea": float(potential_unlock_cells) * cell_area,
                "potentialUnlockCells": potential_unlock_cells,
                "indices": flat_idx,
                "bboxWorld": (x0, x1, y0, y1),
                "recommendation": SEMANTIC_RECOMMENDATIONS.get(label_id, "Review this object."),
            })
            candidate_id += 1
        if progress_cb:
            progress_cb(label_index, total_labels, fixation, label_id)

    candidates.sort(
        key=lambda c: (
            priority.get(c["fixation"], 99),
            -c["potentialUnlockArea"],
            -c["area"],
        )
    )
    return candidates[:max_candidates]


def simulate_removed_candidates(act_result, candidate_list, selected_ids,
                                  label="IMPROVED", logf=None,
                                  label_grid=None):
    """Recompute horizontal RII after intelligently RELOCATING the
    selected semantic components.

    Behaviour change: this used to literally remove the objects (zero
    their cells) and score the resulting map. That treated every
    suggestion as "delete the chair", which doesn't match how a real
    layout improvement works. The function now RELOCATES each selected
    object using the same rule-based + simulated-net-gain pipeline that
    Optimize Layout uses (find_relocation_zones with progressive
    relaxation), and scores the resulting map with the same
    _score_accessibility_from_masks routine used by the Reference and
    Actual coverage runs — so the headline RII is directly comparable
    with the Step 3 numbers.

    Result dict carries `relocations` (list of placement records) and
    still includes `removedCells`/`removedArea` (both zero) for callers
    that read the legacy fields.
    """
    L = logf if logf else (lambda m, c="": None)
    selected = {int(v) for v in selected_ids}
    if not selected:
        raise ValueError("No removable objects were selected")

    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result["resolution"])
    cell_area = res * res
    params = dict(act_result.get("params", {}))
    floor_mask = np.asarray(act_result.get("floorPx"), dtype=np.uint8).reshape(h, w)

    working_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]),
        dtype=np.uint8,
    ).reshape(h, w).copy()

    picked = [c for c in candidate_list if int(c["id"]) in selected]
    if not picked:
        raise ValueError("Selected removable objects are no longer available")

    # Process biggest unlock-potential first so dependencies between moves
    # are captured greedily (same idea as optimize_multi_object_relocation).
    picked.sort(key=lambda c: -float(c.get("potentialUnlockArea", c.get("area", 0))))

    placements: list[dict] = []
    skipped: list[dict] = []

    for candidate in picked:
        flat = candidate["indices"]
        rows = flat // w
        cols = flat % w
        if working_blocked[rows, cols].sum() == 0:
            # Already cleared by a previous move (overlap edge case).
            continue

        # Build a temp result reflecting the current working state so
        # find_relocation_zones sees the right baseline.
        temp_result = dict(act_result)
        temp_result["sourceBlocked"] = working_blocked.ravel().copy()
        temp_result["blocked"] = working_blocked.ravel().copy()
        _, acc = _quick_reachable_area(working_blocked, floor_mask, params, res)
        temp_result["covPx"] = acc.ravel().copy()

        zones = []
        used_relax = 0
        for relax in (0, 1, 2, 3):
            zones = find_relocation_zones(
                temp_result, candidate, max_zones=3,
                label_grid=label_grid,
                peer_label_id=candidate.get("label"),
                relaxation_level=relax,
            )
            if zones:
                used_relax = relax
                break

        if not zones:
            skipped.append({
                "candidate_id": int(candidate["id"]),
                "name": candidate.get("name", f"Object #{candidate['id']}"),
                "reason": "no rule-safe, accessibility-positive zone",
            })
            L(f"[{label}] Skip {candidate.get('name', candidate['id'])} — no safe relocation slot.", "warn")
            continue

        # Apply: remove origin cells, stamp destination
        working_blocked[rows, cols] = 0
        zr, zc = zones[0]["top_left_rc"]
        fh, fw = zones[0]["footprint_hw"]
        working_blocked[zr:zr + fh, zc:zc + fw] = 1

        placements.append({
            "candidate_id": int(candidate["id"]),
            "name": candidate.get("name", f"Object #{candidate['id']}"),
            "to_rc": (int(zr), int(zc)),
            "footprint_hw": (int(fh), int(fw)),
            "relaxation_level": int(used_relax),
            "net_area_gain": float(zones[0].get("net_area_gain", 0.0)),
            "origin_distance_m": float(zones[0].get("origin_distance_m", 0.0)),
            "peer_score": float(zones[0].get("peer_score", 0.0)),
        })
        L(
            f"[{label}] Relocate {candidate.get('name', candidate['id'])} → ({zr}, {zc})  "
            f"(relax={used_relax}, +{zones[0].get('net_area_gain', 0):.2f} m²)",
            "info",
        )

    # Final RII via the SAME pipeline used by Reference / Actual coverage
    # runs — so this matches Step 3's RII numbers directly.
    improved = _score_accessibility_from_masks(
        working_blocked,
        floor_mask,
        res,
        params,
        label,
        logf,
        use_stc=bool(act_result.get("useSTC")),
    )
    improved.update(
        sourceBlocked=working_blocked.ravel().copy(),
        params=params,
        resolution=res,
        origin=tuple(act_result.get("origin", (0.0, 0.0))),
        selectedCandidateIds=sorted(selected),
        relocations=placements,
        skippedCandidates=skipped,
        # Legacy fields — kept zero so old callers don't break.
        removedCells=0,
        removedArea=0.0,
    )
    return improved


def render_semantic_candidates(ref_result, act_result, label_grid, candidates, selected_ids=None, focused_id=None, bg_pgm=None):
    """Render the semantic gap plus removable-object candidates, overlaid on map.pgm."""
    selected = {int(v) for v in (selected_ids or [])}
    focused = None if focused_id is None else int(focused_id)
    w, h = ref_result['w'], ref_result['h']
    ref_cov = ref_result['covPx'].reshape(h, w)[::-1, :]
    act_cov = act_result['covPx'].reshape(h, w)[::-1, :]
    lg = label_grid.reshape(h, w)[::-1, :]
    blk = np.asarray(act_result.get('sourceBlocked', act_result['blocked']), dtype=np.uint8).reshape(h, w)[::-1, :]

    buf = _build_bg(h, w, act_result, bg_pgm)
    _blend = bg_pgm is not None

    missed = (ref_cov == 1) & (act_cov == 0)
    # Color missed areas by semantic label (matching the 3D viewer palette)
    for label_id in SEMANTIC_RAW_LABEL_IDS:
        mask = missed & (lg == label_id)
        if np.any(mask):
            if _blend:
                buf[mask] = (0.45 * np.array(SEMANTIC_3D_COLORS[label_id], dtype=np.float32) + 0.55 * buf[mask].astype(np.float32)).astype(np.uint8)
            else:
                buf[mask] = SEMANTIC_3D_COLORS[label_id]
    # Missed with no label data = dim red
    mask_no_label = missed & (lg == -1)
    if np.any(mask_no_label):
        buf[mask_no_label] = [100, 30, 30]

    cov_mask = act_cov == 1
    if _blend:
        buf[cov_mask] = (0.35 * np.array([0, 200, 130], dtype=np.float32) + 0.65 * buf[cov_mask].astype(np.float32)).astype(np.uint8)
    else:
        buf[cov_mask] = [0, 200, 130]

    if not _blend:
        buf[(act_cov == 0) & (blk == 1)] = [20, 24, 28]

    for candidate in candidates:
        rows = candidate["indices"] // w
        cols = candidate["indices"] % w
        disp_rows = h - 1 - rows
        cid = int(candidate["id"])
        lid = candidate.get("label", -1)
        if cid == focused:
            color = np.array([80, 220, 255], dtype=np.uint8)
        elif cid in selected:
            color = np.array([255, 90, 180], dtype=np.uint8)
        else:
            color = np.array(SEMANTIC_3D_COLORS.get(lid, (255, 190, 0)), dtype=np.uint8)
        buf[disp_rows, cols] = color
        if cid == focused:
            r0 = max(0, int(disp_rows.min()) - 2)
            r1 = min(h - 1, int(disp_rows.max()) + 2)
            c0 = max(0, int(cols.min()) - 2)
            c1 = min(w - 1, int(cols.max()) + 2)
            buf[r0:r1 + 1, c0] = [80, 220, 255]
            buf[r0:r1 + 1, c1] = [80, 220, 255]
            buf[r0, c0:c1 + 1] = [80, 220, 255]
            buf[r1, c0:c1 + 1] = [80, 220, 255]

    return QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


# ── Bottleneck Analysis & Placement Optimization ─────────────────────────────


def _quick_reachable_area(blocked2d, floor2d, params, resolution, _struct_cache={}):
    """Lightweight accessibility: dilate + count largest connected accessible region."""
    inflX, inflY, isRect = _footprint_inflation_pixels(params, resolution)
    inflated = _dilate_binary_mask(blocked2d, inflX, inflY, isRect)
    accessible = ((inflated == 0) & (floor2d > 0)).astype(np.uint8)
    # Count only the largest connected component (true reachability)
    if accessible.any():
        try:
            from scipy.ndimage import label as _label
            struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            labeled, n = _label(accessible, structure=struct4)
            if n > 0:
                sizes = np.bincount(labeled.ravel())[1:]  # skip background (0)
                largest = int(sizes.argmax()) + 1
                accessible = (labeled == largest).astype(np.uint8)
        except ImportError:
            pass  # Without scipy, count all accessible cells (less accurate)
    return int(accessible.sum()), accessible


def score_bottleneck_candidates(act_result, candidates, top_n=20, progress_cb=None):
    """Compute true reachable-area gain for the top candidates using full dilation + count.

    Enriches each candidate dict with:
        trueUnlockArea (float, m^2), bottleneckRatio (float), isBottleneck (bool)
    """
    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result.get("resolution", 0.05))
    cell_area = res * res
    params = act_result.get("params", {})
    source_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]), dtype=np.uint8
    ).reshape(h, w)
    floor2d = np.asarray(act_result.get("floorPx"), dtype=np.uint8).reshape(h, w)

    # Precompute baseline inflation structuring element
    inflX, inflY, isRect = _footprint_inflation_pixels(params, res)
    baseline_cells, _ = _quick_reachable_area(source_blocked, floor2d, params, res)

    scored = sorted(candidates, key=lambda c: -c.get("potentialUnlockArea", 0))[:top_n]
    scored_ids = {c["id"] for c in scored}

    for i, cand in enumerate(scored):
        test_blocked = source_blocked.copy()
        flat = cand["indices"]
        rows, cols = flat // w, flat % w
        test_blocked[rows, cols] = 0

        new_cells, _ = _quick_reachable_area(test_blocked, floor2d, params, res)
        delta_cells = max(0, new_cells - baseline_cells)
        true_unlock = float(delta_cells) * cell_area
        obj_area = max(cand["area"], cell_area)
        ratio = true_unlock / obj_area

        cand["trueUnlockArea"] = true_unlock
        cand["bottleneckRatio"] = ratio
        cand["isBottleneck"] = ratio > 5.0

        if progress_cb:
            progress_cb(i + 1, len(scored), cand.get("name", ""))

    for cand in candidates:
        if cand["id"] not in scored_ids:
            cand["trueUnlockArea"] = cand.get("potentialUnlockArea", 0.0)
            cand["bottleneckRatio"] = 0.0
            cand["isBottleneck"] = False

    return candidates


def find_relocation_zones(act_result, candidate, max_zones=5,
                           label_grid=None,
                           peer_label_id=None,
                           relaxation_level=0):
    """Find valid placement positions for a candidate object.

    Two-layer pipeline (replaces the old wall+traffic heuristic):

    **Layer A — rule-based hard filter** (deterministic):
      1. Footprint fits in the accessible obstacle-free area (existing
         erosion logic).
      2. Destination doesn't overlap the current coverage path (robot's
         own routes stay clear).
      3. Destination doesn't touch a doorway / pinch cell — cells whose
         free 4-neighbour count in the *raw* obstacle map is ≤ 2.

    **Layer B — score the survivors** with simulated net RII gain plus
    soft preferences:
      score = net_gain_m2  +  α · peer_proximity  −  β · origin_distance
      Tie-breakers in order: peer_proximity, origin_distance, wall adjacency.
      Zones with net_gain ≤ 0 are dropped (safety from the old code).

    The simulation runs over **every** zone that survives Layer A (after
    20-cell deduplication), not just the top-K from a local heuristic.

    label_grid is optional; if supplied alongside peer_label_id the peer-
    proximity score is non-zero (objects of the same fixation type
    attract).

    Returns list of zone dicts with:
        zone_id, top_left_rc, wall_score, net_area_gain,
        peer_score, origin_distance_m, combined_score,
        footprint_hw, rules_passed
    """
    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result.get("resolution", 0.05))
    cell_area = res * res
    params = act_result.get("params", {})
    source_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]), dtype=np.uint8
    ).reshape(h, w)
    floor2d = np.asarray(act_result.get("floorPx"), dtype=np.uint8).reshape(h, w)

    flat = candidate["indices"]
    rows_obj, cols_obj = flat // w, flat % w
    obj_h = int(rows_obj.max() - rows_obj.min()) + 1
    obj_w = int(cols_obj.max() - cols_obj.min()) + 1

    # Baseline: remove candidate and measure accessible area
    test_blocked = source_blocked.copy()
    test_blocked[rows_obj, cols_obj] = 0
    baseline_cells, accessible = _quick_reachable_area(test_blocked, floor2d, params, res)

    # Erode accessible mask by object bbox — valid top-left corners for placement
    if obj_h >= h or obj_w >= w:
        return []
    try:
        from scipy.ndimage import binary_erosion, convolve
        # Erode accessible area by object footprint: remaining cells are valid top-left corners
        footprint = np.ones((obj_h, obj_w), dtype=bool)
        valid = binary_erosion(accessible.astype(bool), structure=footprint, border_value=False).astype(np.uint8)
        # Also exclude positions that overlap with obstacles
        obstacle_free = binary_erosion((test_blocked == 0).astype(bool), structure=footprint, border_value=False).astype(np.uint8)
        valid &= obstacle_free
    except ImportError:
        valid = accessible.copy()
        if obj_h > 1:
            valid[-(obj_h - 1):, :] = 0
        if obj_w > 1:
            valid[:, -(obj_w - 1):] = 0
        for dr in range(obj_h):
            for dc in range(obj_w):
                shifted = np.zeros_like(accessible)
                shifted[:h - dr, :w - dc] = accessible[dr:, dc:]
                valid &= shifted
        for dr in range(obj_h):
            for dc in range(obj_w):
                shifted_blocked = np.zeros_like(test_blocked)
                shifted_blocked[:h - dr, :w - dc] = test_blocked[dr:, dc:]
                valid[shifted_blocked > 0] = 0

    if not np.any(valid):
        return []

    # ── Layer A rule-based pre-filter masks ────────────────────────────
    # Wall adjacency (kept as a soft tie-breaker, not the primary score)
    try:
        from scipy.ndimage import convolve as _convolve
        kernel = np.ones((3, 3), dtype=np.int32); kernel[1, 1] = 0
        wall_score = _convolve(source_blocked.astype(np.int32), kernel,
                                mode='constant', cval=0).astype(np.float32)
    except ImportError:
        wall_score = np.zeros((h, w), dtype=np.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                shifted = np.zeros_like(wall_score)
                sr0, sr1 = max(0, -dr), min(h, h - dr)
                sc0, sc1 = max(0, -dc), min(w, w - dc)
                dr0, dr1 = max(0, dr), min(h, h + dr)
                dc0, dc1 = max(0, dc), min(w, w + dc)
                shifted[dr0:dr1, dc0:dc1] = source_blocked[sr0:sr1, sc0:sc1].astype(np.float32)
                wall_score += shifted

    # RULE 2 — coverage-path clearance: forbid zones whose footprint
    # overlaps the buffer around the robot's current routes.
    inflX, inflY, _ = _footprint_inflation_pixels(params, res)
    robot_short_px = max(1, min(int(inflX), int(inflY)))
    cov2d_orig = np.asarray(act_result['covPx'], dtype=np.uint8).reshape(h, w)
    try:
        from scipy.ndimage import binary_dilation as _bd
        path_struct = np.ones((2 * robot_short_px + 1, 2 * robot_short_px + 1), dtype=bool)
        path_buffer = _bd(cov2d_orig.astype(bool), structure=path_struct).astype(np.uint8)
    except ImportError:
        path_buffer = cov2d_orig.copy()

    # RULE 3 — doorway / pinch-point preservation. A doorway cell is a
    # free cell with ≤ 2 free 4-neighbours in the RAW obstacle map.
    free_raw = (source_blocked == 0).astype(np.int32)
    nbr_free = np.zeros_like(free_raw)
    nbr_free[1:, :] += free_raw[:-1, :]
    nbr_free[:-1, :] += free_raw[1:, :]
    nbr_free[:, 1:] += free_raw[:, :-1]
    nbr_free[:, :-1] += free_raw[:, 1:]
    doorway_cells = ((free_raw == 1) & (nbr_free <= 2)).astype(np.uint8)

    # Peer-attractor map: distance to nearest cell whose label matches the
    # candidate's fixation. Used as a tie-breaker so movable items cluster
    # with peers instead of sitting in random corners.
    peer_dist = None
    if label_grid is not None and peer_label_id is not None:
        try:
            label2d = np.asarray(label_grid, dtype=np.int32).reshape(h, w)
            peer_mask = (label2d == int(peer_label_id))
            # Exclude the candidate's own cells from the peer mask
            peer_mask[rows_obj, cols_obj] = False
            if peer_mask.any():
                from scipy.ndimage import distance_transform_edt as _dte
                peer_dist = _dte(~peer_mask).astype(np.float32) * res  # in meters
        except Exception:
            peer_dist = None

    # Object origin (centroid in pixels) for the origin-distance penalty
    origin_r = float(rows_obj.mean())
    origin_c = float(cols_obj.mean())

    # ── Layer A: enumerate valid zones, dedupe by 20-cell Manhattan ─────
    flat_valid = np.where(valid.ravel() > 0)[0]
    if flat_valid.size == 0:
        return []

    # Sort valid cells by (low path-buffer overlap, then wall_score) so
    # the deduplication keeps zones that already look promising.
    pb_score = -path_buffer.ravel()[flat_valid].astype(np.float32)
    ws_score = wall_score.ravel()[flat_valid]
    pre_score = pb_score * 1000.0 + ws_score  # lex sort
    order = np.argsort(-pre_score)
    valid_ordered = flat_valid[order]

    zone_centroids: list[tuple[int, int]] = []
    DEDUP_CELLS = 20
    MAX_LAYER_A_ZONES = 80  # hard cap so BFS stays under ~5 s per object
    # Progressive rule relaxation. Higher relaxation_level skips rules so we
    # can ALWAYS find a relocation slot — never fall back to "Remove".
    #   0 = all rules (default, strict)
    #   1 = drop doorway rule (allow placement near narrow passages)
    #   2 = drop path-clearance rule (allow placement on robot's current path)
    #   3 = footprint fit only (anywhere it physically fits)
    enforce_path = (relaxation_level < 2)
    enforce_doorway = (relaxation_level < 1)
    for fi in valid_ordered:
        if len(zone_centroids) >= MAX_LAYER_A_ZONES:
            break
        r, c = int(fi // w), int(fi % w)

        # Rule 2 — coverage-path clearance
        if enforce_path and path_buffer[r:r + obj_h, c:c + obj_w].any():
            continue
        # Rule 3 — no doorway seal
        if enforce_doorway and doorway_cells[r:r + obj_h, c:c + obj_w].any():
            continue

        # 20-cell Manhattan deduplication
        too_close = False
        for pr, pc in zone_centroids:
            if abs(r - pr) + abs(c - pc) < DEDUP_CELLS:
                too_close = True
                break
        if too_close:
            continue
        zone_centroids.append((r, c))

    if not zone_centroids:
        return []

    # ── Layer B: score each surviving zone with simulated net RII ──────
    original_accessible = int(cov2d_orig.sum())
    ALPHA_PEER = 0.05     # m² per peer score — gentle
    BETA_ORIGIN = 0.25    # m² per metre of origin distance — STRONG penalty
                          # so a 10 m corner is heavily disfavoured vs a 1 m nudge
    MAX_DISPLACEMENT_M = 8.0  # hard cap — never propose moves farther than this

    zones = []
    # Always require strictly positive net_gain. If no relaxation level
    # produces a valid zone, the optimiser will skip the candidate
    # rather than emit a harmful move. "Always relocate" never means
    # "relocate to somewhere worse".
    min_acceptable_gain = 0.0
    for zone_id, (zr, zc) in enumerate(zone_centroids):
        # Simulate placement
        place_blocked = test_blocked.copy()
        place_blocked[zr:zr + obj_h, zc:zc + obj_w] = 1
        placed_cells, _ = _quick_reachable_area(place_blocked, floor2d, params, res)
        net_gain = float(placed_cells - original_accessible) * cell_area
        if net_gain <= min_acceptable_gain:
            continue  # would hurt or no-op

        # Peer proximity (smaller distance is better → invert)
        if peer_dist is not None:
            zone_center_r = zr + obj_h // 2
            zone_center_c = zc + obj_w // 2
            d_peer = float(peer_dist[zone_center_r, zone_center_c])
            peer_score_m = 1.0 / (1.0 + d_peer)  # 1.0 at peer, → 0 far away
        else:
            d_peer = float('inf'); peer_score_m = 0.0

        # Origin-distance penalty (Euclidean, in metres)
        d_origin = (((zr + obj_h / 2.0) - origin_r) ** 2 +
                    ((zc + obj_w / 2.0) - origin_c) ** 2) ** 0.5 * res

        # Hard cap on physical move distance — no "shove to far corner" moves.
        if d_origin > MAX_DISPLACEMENT_M:
            continue

        combined = (net_gain
                    + ALPHA_PEER * peer_score_m
                    - BETA_ORIGIN * d_origin)

        zones.append({
            "zone_id": zone_id,
            "top_left_rc": (int(zr), int(zc)),
            "wall_score": float(wall_score[zr, zc]),
            "net_area_gain": net_gain,
            "peer_score": peer_score_m,
            "peer_distance_m": d_peer if peer_score_m else None,
            "origin_distance_m": float(d_origin),
            "combined_score": float(combined),
            "footprint_hw": (obj_h, obj_w),
            "rules_passed": True,
            "relaxation_level": int(relaxation_level),
        })

    # Rank: combined score, then peer proximity, then small origin move,
    # then wall adjacency as final tie-breaker.
    zones.sort(key=lambda z: (
        -z["combined_score"],
        -z["peer_score"],
        z["origin_distance_m"],
        -z["wall_score"],
    ))
    return zones[:max_zones]


def classify_candidate_actions(candidates, relocation_results):
    """Assign actionType to each candidate: Relocate / Remove / Cannot optimize.

    Args:
        candidates: list of candidate dicts (enriched by score_bottleneck_candidates)
        relocation_results: dict mapping candidate id -> list of zone dicts
    """
    for cand in candidates:
        fixation = cand.get("fixation", "Fixed")
        if fixation == "Fixed":
            cand["actionType"] = "Cannot optimize"
            cand["relocationZones"] = []
            cand["bestZone"] = None
            continue

        zones = relocation_results.get(cand["id"], [])
        if zones:
            cand["actionType"] = "Relocate"
            cand["relocationZones"] = zones
            cand["bestZone"] = zones[0]
        else:
            cand["actionType"] = "Remove"
            cand["relocationZones"] = []
            cand["bestZone"] = None

    return candidates


def optimize_multi_object_relocation(act_result, candidates, max_moves=10,
                                       progress_cb=None, label_grid=None,
                                       yaml_data=None):
    """Greedy iterative multi-object relocation.

    Iteratively picks the best bottleneck, relocates it, then re-evaluates
    remaining candidates on the updated map. This captures dependencies
    where moving object A unlocks new relocation options for object B.

    Returns dict with:
        moves: list of move dicts
        original_accessible_area, optimized_accessible_area, total_gain (float, m^2)
        optimized_blocked: numpy array (h, w) — final blocked state
    """
    w, h = int(act_result["w"]), int(act_result["h"])
    res = float(act_result.get("resolution", 0.05))
    cell_area = res * res
    params = act_result.get("params", {})
    floor2d = np.asarray(act_result.get("floorPx"), dtype=np.uint8).reshape(h, w)

    current_blocked = np.asarray(
        act_result.get("sourceBlocked", act_result["blocked"]), dtype=np.uint8
    ).reshape(h, w).copy()

    baseline_cells, _ = _quick_reachable_area(current_blocked, floor2d, params, res)
    original_cells = baseline_cells

    moved_ids = set()
    moves = []
    cumulative_gain = 0.0

    # Pre-compute narrow-gap (widen-corridor) candidates once. They compete
    # head-to-head with object-relocation candidates inside the greedy loop.
    try:
        from core.corridor_diagnosis import find_narrow_gaps
        narrow_gaps = find_narrow_gaps(act_result, yaml_data=yaml_data)
    except Exception:
        narrow_gaps = []
    used_gap_ids: set[int] = set()

    for step_i in range(max_moves):
        # ── 1. Best OBJECT candidate (gain if vacated) ──────────────────
        best_cand = None
        best_gain_cells = 0
        for cand in candidates:
            if cand.get("fixation") == "Fixed" or cand["id"] in moved_ids:
                continue
            flat = cand["indices"]
            rows, cols = flat // w, flat % w
            if current_blocked[rows, cols].sum() == 0:
                continue
            test = current_blocked.copy()
            test[rows, cols] = 0
            new_cells, _ = _quick_reachable_area(test, floor2d, params, res)
            gain = new_cells - baseline_cells
            if gain > best_gain_cells:
                best_cand = cand
                best_gain_cells = gain

        # ── 2. Best CORRIDOR widening candidate (re-scored against current map)
        best_gap = None
        best_gap_gain_cells = 0
        for gap in narrow_gaps:
            if gap.id in used_gap_ids:
                continue
            # Skip if any trim cell isn't blocked any more (a previous move
            # already cleared the obstacle).
            tr = gap.trim_cells[:, 0]; tc = gap.trim_cells[:, 1]
            if current_blocked[tr, tc].sum() == 0:
                continue
            test = current_blocked.copy()
            test[tr, tc] = 0
            new_cells, _ = _quick_reachable_area(test, floor2d, params, res)
            gain = new_cells - baseline_cells
            if gain > best_gap_gain_cells:
                best_gap = gap
                best_gap_gain_cells = gain

        # ── 3. Pick the better of the two ───────────────────────────────
        if best_gap is not None and best_gap_gain_cells > best_gain_cells:
            # Apply corridor widening
            tr = best_gap.trim_cells[:, 0]; tc = best_gap.trim_cells[:, 1]
            current_blocked[tr, tc] = 0
            new_cells, _ = _quick_reachable_area(current_blocked, floor2d, params, res)
            step_gain = float(new_cells - baseline_cells) * cell_area
            baseline_cells = new_cells
            cumulative_gain += step_gain
            used_gap_ids.add(best_gap.id)
            moves.append({
                "candidate_id": -1 - best_gap.id,  # negative ids = non-object
                "name": f"Corridor at {best_gap.throat_world}",
                "fixation": "Structural",
                "from_indices": np.array([], dtype=np.int64),
                "to_rc": (int(best_gap.throat_rc[0]), int(best_gap.throat_rc[1])),
                "footprint_hw": (1, 1),
                "step_gain": step_gain,
                "cumulative_gain": cumulative_gain,
                "new_accessible_area": float(new_cells) * cell_area,
                "action": "Widen corridor",
                "kind": "widen_corridor",
                "description": best_gap.description,
                "zone_meta": None,
                "trim_cells": best_gap.trim_cells.copy(),
                "current_width_m": best_gap.current_width_m,
                "required_width_m": best_gap.required_width_m,
                "throat_world": best_gap.throat_world,
            })
            if progress_cb:
                progress_cb(step_i + 1, max_moves, f"widen corridor #{best_gap.id}")
            continue

        if best_cand is None or best_gain_cells <= 0:
            break

        flat = best_cand["indices"]
        rows, cols = flat // w, flat % w
        obj_h = int(rows.max() - rows.min()) + 1
        obj_w = int(cols.max() - cols.min()) + 1

        temp_result = dict(act_result)
        temp_result["sourceBlocked"] = current_blocked.ravel().copy()
        temp_result["blocked"] = current_blocked.ravel().copy()
        _, acc = _quick_reachable_area(current_blocked, floor2d, params, res)
        temp_result["covPx"] = acc.ravel().copy()

        # Try strict rules first, relax progressively. Every level
        # requires net_gain > 0 (always-positive guarantee), so if no
        # level finds a valid zone within MAX_DISPLACEMENT, we SKIP this
        # candidate — we don't emit a harmful or degenerate move.
        zones = []
        used_relax = 0
        for relax in (0, 1, 2, 3):
            zones = find_relocation_zones(
                temp_result, best_cand, max_zones=3,
                label_grid=label_grid,
                peer_label_id=best_cand.get("label"),
                relaxation_level=relax,
            )
            if zones:
                used_relax = relax
                break

        if not zones:
            # No accessibility-positive, within-cap relocation exists.
            # Mark this candidate as tried and move on — DO NOT emit a
            # negative-gain move.
            moved_ids.add(best_cand["id"])
            continue

        # Apply the chosen move on the working obstacle map.
        current_blocked[rows, cols] = 0
        zr, zc = zones[0]["top_left_rc"]
        fh, fw = zones[0]["footprint_hw"]
        current_blocked[zr:zr + fh, zc:zc + fw] = 1
        to_rc = (int(zr), int(zc))

        new_cells, _ = _quick_reachable_area(current_blocked, floor2d, params, res)
        step_gain = float(new_cells - baseline_cells) * cell_area
        baseline_cells = new_cells
        cumulative_gain += step_gain
        moved_ids.add(best_cand["id"])

        kind = "relocate_object"
        name = best_cand.get('name', best_cand['id'])
        # we always have a valid to_rc here (the no-zone case `continue`s above)
        wx = (to_rc[1] + obj_w / 2.0) * res
        wy = (to_rc[0] + obj_h / 2.0) * res
        d_origin_m = zones[0].get("origin_distance_m", 0.0)
        description = (
            f"Relocate {name} → ({wx:.1f}, {wy:.1f}) m  "
            f"[move {d_origin_m:.1f} m]"
        )
        if used_relax == 0 and zones[0].get("peer_score", 0) > 0:
            description += "  (near peers)"
        elif used_relax == 1:
            description += "  (near a doorway — pick an alternative if possible)"
        elif used_relax == 2:
            description += "  (close to a robot route — keep clear when moving)"
        elif used_relax == 3:
            description += "  (best-fit only — verify accessibility on site)"
        moves.append({
            "candidate_id": best_cand["id"],
            "name": name,
            "fixation": best_cand.get("fixation", ""),
            "from_indices": flat.copy(),
            "to_rc": to_rc,
            "footprint_hw": (obj_h, obj_w),
            "step_gain": step_gain,
            "cumulative_gain": cumulative_gain,
            "new_accessible_area": float(new_cells) * cell_area,
            "action": "Relocate",
            "kind": kind,
            "description": description,
            "zone_meta": (zones[0] if zones else None),
            "relaxation_level": int(used_relax) if to_rc else None,
        })

        if progress_cb:
            progress_cb(step_i + 1, max_moves, best_cand.get("name", ""))

    # Final numbers are computed by _quick_reachable_area on the modified
    # blocked mask — same dilation + largest-component logic the
    # reference / actual runs use under the hood, so before/after are
    # directly comparable.
    return {
        "moves": moves,
        "original_accessible_area": float(original_cells) * cell_area,
        "optimized_accessible_area": float(baseline_cells) * cell_area,
        "total_gain": float(baseline_cells - original_cells) * cell_area,
        "optimized_blocked": current_blocked.copy(),
    }
