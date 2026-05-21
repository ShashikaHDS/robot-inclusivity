"""Merge two Nav2-format 2D maps (.pgm + .yaml) into one.

Two maps are assumed to be in the same world frame (same Z plane, same
coordinate axes). Each has its own resolution and origin (x, y) in
metres. The merge resamples both onto a common grid that spans their
combined world bounds and combines pixel values with an
"obstacle wins" rule:

    obstacle (any input)  → obstacle
    free     (any input)  → free   (unless any input was obstacle)
    unknown  (both)       → unknown

Resolution defaults to the *finer* of the two inputs so we never lose
detail; the user can override with a coarser value to shrink the
output.

Map B can also be repositioned: `b_translate_world=(dx, dy)` shifts B's
world position, `b_rotation_deg` rotates B about its (translated) centre
(positive = counter-clockwise in world frame), and `b_mask` is an
optional (hB, wB) uint8 array where 0-cells are skipped during the
merge so the user can erase parts of B before combining.

The output is a Nav2-conformant PGM + YAML pair ready to drop into
Step 2 / Step 3 of the pipeline.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

from core.map_io import parse_pgm, parse_yaml


# Conventional Nav2 PGM bytes:
#   0     = occupied (max-confidence wall)
#   254   = free
#   205   = unknown (mid grey)
_NAV2_OCC = np.uint8(0)
_NAV2_FREE = np.uint8(254)
_NAV2_UNKNOWN = np.uint8(205)


def _classify(pixels: np.ndarray, yaml_data: dict) -> np.ndarray:
    """Return a uint8 class map: 0=occupied, 1=free, 2=unknown."""
    if yaml_data.get("negate"):
        pixels = 255 - pixels
    p_occ = 1.0 - (pixels.astype(np.float32) / 255.0)
    occupied_thresh = float(yaml_data.get("occupied_thresh", 0.65))
    free_thresh = float(yaml_data.get("free_thresh", 0.196))
    cls = np.full_like(pixels, 2, dtype=np.uint8)  # default unknown
    cls[p_occ >= occupied_thresh] = 0              # occupied
    cls[p_occ <= free_thresh] = 1                  # free
    return cls


def _world_bounds(yaml_data: dict, w: int, h: int) -> Tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) of the (untransformed) map in metres."""
    res = float(yaml_data["resolution"])
    ox = float(yaml_data["origin"][0])
    oy = float(yaml_data["origin"][1])
    return ox, oy, ox + w * res, oy + h * res


def _transformed_bounds(
    yaml_data: dict, w: int, h: int,
    translate: Tuple[float, float], rotation_deg: float,
) -> Tuple[float, float, float, float]:
    """World bounds of a map after translate + rotation about its centre."""
    res = float(yaml_data["resolution"])
    ox = float(yaml_data["origin"][0])
    oy = float(yaml_data["origin"][1])
    cx0 = ox + w * res / 2.0
    cy0 = oy + h * res / 2.0
    tx, ty = float(translate[0]), float(translate[1])
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    corners = [
        (ox, oy), (ox + w * res, oy),
        (ox + w * res, oy + h * res), (ox, oy + h * res),
    ]
    xs, ys = [], []
    for px, py in corners:
        dx = px - cx0
        dy = py - cy0
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        xs.append(cx0 + tx + rx)
        ys.append(cy0 + ty + ry)
    return min(xs), min(ys), max(xs), max(ys)


def merge_two_maps(
    pgm_a: str, yaml_a: str,
    pgm_b: str, yaml_b: str,
    out_pgm: str, out_yaml: str,
    *,
    target_resolution: Optional[float] = None,
    padding_m: float = 0.0,
    b_translate_world: Tuple[float, float] = (0.0, 0.0),
    b_rotation_deg: float = 0.0,
    b_mask: Optional[np.ndarray] = None,
) -> dict:
    """Merge two maps into out_pgm + out_yaml. Returns a stats dict.

    target_resolution: defaults to min of the two input resolutions.
    padding_m: optional extra world-space margin around the merged bounds.
    b_translate_world: (dx, dy) extra world offset applied to map B.
    b_rotation_deg: rotation of map B about its (translated) centre,
        counter-clockwise positive in the world frame.
    b_mask: optional (hB, wB) uint8 mask where 0 = erase, 1 = keep.
    """
    if not (os.path.isfile(pgm_a) and os.path.isfile(yaml_a)):
        raise FileNotFoundError(f"Map A is missing files:\n{pgm_a}\n{yaml_a}")
    if not (os.path.isfile(pgm_b) and os.path.isfile(yaml_b)):
        raise FileNotFoundError(f"Map B is missing files:\n{pgm_b}\n{yaml_b}")

    wA, hA, pixA = parse_pgm(pgm_a)
    wB, hB, pixB = parse_pgm(pgm_b)
    yA = parse_yaml(yaml_a)
    yB = parse_yaml(yaml_b)

    clsA = _classify(pixA, yA).reshape(hA, wA)
    clsB = _classify(pixB, yB).reshape(hB, wB)

    if b_mask is not None:
        b_mask = np.asarray(b_mask, dtype=np.uint8).reshape(hB, wB)

    # World bounds — A is identity, B has the user transform applied.
    axmin, aymin, axmax, aymax = _world_bounds(yA, wA, hA)
    bxmin, bymin, bxmax, bymax = _transformed_bounds(
        yB, wB, hB, b_translate_world, b_rotation_deg,
    )
    xmin = min(axmin, bxmin) - padding_m
    ymin = min(aymin, bymin) - padding_m
    xmax = max(axmax, bxmax) + padding_m
    ymax = max(aymax, bymax) + padding_m

    res = float(target_resolution) if target_resolution else min(yA["resolution"], yB["resolution"])
    out_w = max(1, int(math.ceil((xmax - xmin) / res)))
    out_h = max(1, int(math.ceil((ymax - ymin) / res)))

    # Start everything as UNKNOWN
    merged_cls = np.full((out_h, out_w), 2, dtype=np.uint8)

    def _stamp(src_cls: np.ndarray, yaml_in: dict, *,
               translate=(0.0, 0.0), rotation_deg=0.0,
               mask: Optional[np.ndarray] = None):
        """Resample src_cls into the merged frame and combine.

        For each output cell we compute its world centre, undo the input's
        translate/rotation to get the corresponding (pre-transform) world
        point, then floor-divide by the input's resolution to gather a
        pixel. Cells outside the input rectangle (or erased via `mask`)
        stay at their current merged class.
        """
        in_h, in_w = src_cls.shape
        res_in = float(yaml_in["resolution"])
        ox_in = float(yaml_in["origin"][0])
        oy_in = float(yaml_in["origin"][1])
        cx0 = ox_in + in_w * res_in / 2.0
        cy0 = oy_in + in_h * res_in / 2.0
        tx, ty = float(translate[0]), float(translate[1])
        cx_s = cx0 + tx
        cy_s = cy0 + ty

        theta = math.radians(rotation_deg)
        c, s = math.cos(theta), math.sin(theta)  # noqa: E741
        # Inverse rotation: R(-theta) → cos(t)·Δ + sin(t)·Δ_y for x, etc.

        cc_out = np.arange(out_w, dtype=np.float32)
        rr_out = np.arange(out_h, dtype=np.float32)
        world_x = xmin + (cc_out + 0.5) * res
        world_y = ymax - (rr_out + 0.5) * res

        # 2D broadcast
        Wx = np.broadcast_to(world_x[np.newaxis, :], (out_h, out_w))
        Wy = np.broadcast_to(world_y[:, np.newaxis], (out_h, out_w))

        dx = Wx - cx_s
        dy = Wy - cy_s
        # P_orig = C0 + R(-theta) · (P_screen - (C0 + t))
        P_orig_x = cx0 + c * dx + s * dy
        P_orig_y = cy0 - s * dx + c * dy

        col_idx = np.floor((P_orig_x - ox_in) / res_in).astype(np.int32)
        ymax_in = oy_in + in_h * res_in
        row_idx = np.floor((ymax_in - P_orig_y) / res_in).astype(np.int32)

        in_bounds = (
            (col_idx >= 0) & (col_idx < in_w)
            & (row_idx >= 0) & (row_idx < in_h)
        )
        if not in_bounds.any():
            return

        safe_col = np.clip(col_idx, 0, in_w - 1)
        safe_row = np.clip(row_idx, 0, in_h - 1)
        gathered = src_cls[safe_row, safe_col]

        if mask is not None:
            in_bounds = in_bounds & (mask[safe_row, safe_col] == 1)
            if not in_bounds.any():
                return

        # Combine rule: obstacle wins; then free wins over unknown.
        occ = in_bounds & (gathered == 0)
        free = in_bounds & (gathered == 1) & (merged_cls != 0)
        merged_cls[occ] = 0
        merged_cls[free] = 1

    _stamp(clsA, yA)
    _stamp(clsB, yB,
           translate=b_translate_world,
           rotation_deg=b_rotation_deg,
           mask=b_mask)

    # Encode back to a Nav2-style uint8 PGM
    out_pix = np.full_like(merged_cls, _NAV2_UNKNOWN, dtype=np.uint8)
    out_pix[merged_cls == 0] = _NAV2_OCC
    out_pix[merged_cls == 1] = _NAV2_FREE

    # Write PGM (P5 binary)
    os.makedirs(os.path.dirname(out_pgm) or ".", exist_ok=True)
    with open(out_pgm, "wb") as f:
        f.write(f"P5\n{out_w} {out_h}\n255\n".encode("ascii"))
        f.write(out_pix.tobytes())

    # Write YAML — origin uses the bottom-left corner of the merged map
    out_origin = [float(xmin), float(ymin), 0.0]
    with open(out_yaml, "w") as f:
        f.write(
            f"image: {os.path.basename(out_pgm)}\n"
            f"resolution: {res:.4f}\n"
            f"origin: [{out_origin[0]:.4f}, {out_origin[1]:.4f}, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
        )

    n_occ = int((merged_cls == 0).sum())
    n_free = int((merged_cls == 1).sum())
    n_unk = int((merged_cls == 2).sum())
    return {
        "width": out_w,
        "height": out_h,
        "resolution": res,
        "origin": out_origin,
        "occupied_cells": n_occ,
        "free_cells": n_free,
        "unknown_cells": n_unk,
        "occupied_area_m2": n_occ * res * res,
        "free_area_m2": n_free * res * res,
    }
