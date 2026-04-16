"""Pure-Python helpers for PCD/PLY I/O, cleanup, map export, and semantic labels."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
import os
from collections import deque
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class PCDHeader:
    """Parsed PCD header metadata."""

    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    types: tuple[str, ...]
    counts: tuple[int, ...]
    width: int
    height: int
    points: int
    data: str
    data_offset: int


@dataclass(frozen=True)
class PLYHeader:
    """Parsed PLY header metadata for the vertex element."""

    fields: tuple[str, ...]
    types: tuple[str, ...]
    points: int
    data: str
    data_offset: int


_PCD_TYPE_MAP = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
}

_PLY_TYPE_MAP = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "float32": np.float32,
    "float64": np.float64,
}

_LABEL_FIELD_CANDIDATES = (
    "classification",
    "label",
    "class",
    "scalar_classification",
    "scalar_label",
    "scalar_class",
    "scalar field",
)


def _structured_dtype(header: PCDHeader) -> np.dtype:
    dtype_fields = []
    for name, typ, size, count in zip(header.fields, header.types, header.sizes, header.counts):
        np_type = _PCD_TYPE_MAP.get((typ, size))
        if np_type is None:
            raise ValueError(f"Unsupported PCD field type/size: {name} {typ}{size}")
        if count == 1:
            dtype_fields.append((name, np_type))
        else:
            dtype_fields.append((name, np_type, (count,)))
    return np.dtype(dtype_fields)


def parse_pcd_header(path: str) -> PCDHeader:
    """Parse a PCD header and return structured metadata."""
    fields: tuple[str, ...] = ()
    sizes: tuple[int, ...] = ()
    types: tuple[str, ...] = ()
    counts: tuple[int, ...] = ()
    width = 0
    height = 1
    points = 0
    data = ""

    with open(path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PCD header: {path}")
            decoded = line.decode("ascii", errors="ignore").strip()
            if not decoded or decoded.startswith("#"):
                continue
            parts = decoded.split()
            key = parts[0].upper()
            values = parts[1:]
            if key == "FIELDS":
                fields = tuple(values)
            elif key == "SIZE":
                sizes = tuple(int(v) for v in values)
            elif key == "TYPE":
                types = tuple(values)
            elif key == "COUNT":
                counts = tuple(int(v) for v in values)
            elif key == "WIDTH":
                width = int(values[0])
            elif key == "HEIGHT":
                height = int(values[0])
            elif key == "POINTS":
                points = int(values[0])
            elif key == "DATA":
                data = values[0].lower()
                break

        if not counts:
            counts = tuple(1 for _ in fields)
        if not points and width:
            points = width * height

        if not fields or not sizes or not types or len(fields) != len(sizes) or len(fields) != len(types):
            raise ValueError(f"Unsupported or incomplete PCD header: {path}")

        return PCDHeader(
            fields=fields,
            sizes=sizes,
            types=types,
            counts=counts,
            width=width,
            height=height,
            points=points,
            data=data,
            data_offset=handle.tell(),
        )


def load_pcd_array(path: str) -> tuple[PCDHeader, np.ndarray]:
    """Load a PCD file into a structured NumPy array."""
    header = parse_pcd_header(path)
    dtype = _structured_dtype(header)

    if header.data == "binary":
        with open(path, "rb") as handle:
            handle.seek(header.data_offset)
            array = np.fromfile(handle, dtype=dtype, count=header.points)
    elif header.data == "ascii":
        with open(path, "rb") as handle:
            handle.seek(header.data_offset)
            text = handle.read().decode("ascii", errors="ignore")
        array = np.loadtxt(io.StringIO(text), dtype=dtype)
        if array.shape == ():
            array = np.array([array], dtype=dtype)
    elif header.data == "binary_compressed":
        raise ValueError("binary_compressed PCD is not supported by this pipeline")
    else:
        raise ValueError(f"Unsupported PCD DATA mode: {header.data}")

    if array.shape[0] != header.points:
        raise ValueError(f"Expected {header.points} points but loaded {array.shape[0]} from {path}")

    return header, array


def _ply_dtype(header: PLYHeader) -> np.dtype:
    dtype_fields = []
    for name, typ in zip(header.fields, header.types):
        np_type = _PLY_TYPE_MAP.get(typ.lower())
        if np_type is None:
            raise ValueError(f"Unsupported PLY property type: {name} {typ}")
        dtype_fields.append((name, np_type))
    dtype = np.dtype(dtype_fields)
    if header.data == "binary_big_endian":
        dtype = dtype.newbyteorder(">")
    return dtype


def parse_ply_header(path: str) -> PLYHeader:
    """Parse a PLY header and return vertex metadata."""
    fields: list[str] = []
    types: list[str] = []
    vertex_count = 0
    data = ""
    current_element = None

    with open(path, "rb") as handle:
        first = handle.readline().decode("ascii", errors="ignore").strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"Incomplete PLY header: {path}")
            decoded = line.decode("ascii", errors="ignore").strip()
            if not decoded or decoded.startswith("comment"):
                continue
            parts = decoded.split()
            key = parts[0].lower()
            if key == "format":
                data = parts[1].lower()
            elif key == "element":
                current_element = parts[1].lower()
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif key == "property" and current_element == "vertex":
                if parts[1].lower() == "list":
                    raise ValueError("PLY vertex list properties are not supported")
                types.append(parts[1])
                fields.append(parts[2])
            elif key == "end_header":
                break

        if data not in {"ascii", "binary_little_endian", "binary_big_endian"}:
            raise ValueError(f"Unsupported PLY format: {data}")
        if vertex_count <= 0 or not fields:
            raise ValueError(f"PLY file has no vertex data: {path}")

        return PLYHeader(
            fields=tuple(fields),
            types=tuple(types),
            points=vertex_count,
            data=data,
            data_offset=handle.tell(),
        )


def load_ply_array(path: str) -> tuple[PLYHeader, np.ndarray]:
    """Load a PLY file into a structured NumPy array."""
    header = parse_ply_header(path)
    dtype = _ply_dtype(header)

    if header.data in {"binary_little_endian", "binary_big_endian"}:
        with open(path, "rb") as handle:
            handle.seek(header.data_offset)
            array = np.fromfile(handle, dtype=dtype, count=header.points)
    else:
        rows = []
        with open(path, "rb") as handle:
            handle.seek(header.data_offset)
            for _ in range(header.points):
                line = handle.readline()
                if not line:
                    break
                rows.append(line.decode("ascii", errors="ignore"))
        array = np.loadtxt(io.StringIO("".join(rows)), dtype=dtype)
        if array.shape == ():
            array = np.array([array], dtype=dtype)

    if array.shape[0] != header.points:
        raise ValueError(f"Expected {header.points} vertices but loaded {array.shape[0]} from {path}")
    return header, array


def load_point_cloud_array(path: str) -> tuple[tuple[str, ...], np.ndarray]:
    """Load a supported point cloud file into a structured NumPy array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pcd":
        header, array = load_pcd_array(path)
        return header.fields, array
    if ext == ".ply":
        header, array = load_ply_array(path)
        return header.fields, array
    raise ValueError(f"Unsupported point cloud format: {path}")


def load_xyz_points(path: str) -> np.ndarray:
    """Load XYZ coordinates from a PCD or PLY file."""
    _, array = load_point_cloud_array(path)
    lowered = {name.lower(): name for name in array.dtype.names or ()}
    try:
        x_name = lowered["x"]
        y_name = lowered["y"]
        z_name = lowered["z"]
    except KeyError as exc:
        raise ValueError(f"Missing XYZ fields in {path}") from exc

    return np.column_stack((array[x_name], array[y_name], array[z_name])).astype(np.float32, copy=False)


def detect_label_field(field_names: list[str] | tuple[str, ...]) -> str | None:
    """Detect a semantic label field from PCD or PLY field names."""
    lowered = {name.lower(): name for name in field_names}
    for candidate in _LABEL_FIELD_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    for name in field_names:
        low = name.lower()
        if low.startswith("scalar_") or "class" in low or "label" in low:
            return name
    extras = [
        name for name in field_names
        if name.lower() not in {
            "x", "y", "z", "nx", "ny", "nz",
            "normal_x", "normal_y", "normal_z",
            "intensity", "rgb", "rgba",
            "red", "green", "blue", "alpha",
            "confidence",
        }
    ]
    return extras[-1] if extras else None


def load_xyz_and_labels(path: str, label_field: str | None = None) -> tuple[np.ndarray, np.ndarray | None, str | None]:
    """Load XYZ coordinates and semantic labels from a PCD or PLY file."""
    fields, array = load_point_cloud_array(path)
    lowered = {name.lower(): name for name in array.dtype.names or ()}
    try:
        x_name = lowered["x"]
        y_name = lowered["y"]
        z_name = lowered["z"]
    except KeyError as exc:
        raise ValueError(f"Missing XYZ fields in {path}") from exc

    points = np.column_stack((array[x_name], array[y_name], array[z_name])).astype(np.float32, copy=False)
    field_name = label_field or detect_label_field(fields)
    if field_name is None:
        return points, None, None

    labels = np.asarray(array[field_name]).reshape(-1)
    if np.issubdtype(labels.dtype, np.floating):
        labels = np.rint(labels)
    labels = labels.astype(np.int32, copy=False)
    return points, labels, field_name


def write_xyz_pcd(path: str, points: np.ndarray) -> None:
    """Write a binary XYZ-only PCD file."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("write_xyz_pcd expects an Nx3 array")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {pts.shape[0]}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {pts.shape[0]}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.ascontiguousarray(pts).tobytes())


def filter_non_finite(points: np.ndarray) -> np.ndarray:
    """Drop points with NaN/Inf coordinates."""
    mask = np.all(np.isfinite(points), axis=1)
    return points[mask]


def estimate_ground_preserving_preset(
    points: np.ndarray,
    floor_low_percentile: float = 2.0,
    floor_anchor_percentile: float = 5.0,
    cleanup_floor_margin_m: float = 0.25,
    cleanup_height_m: float = 4.0,
    map_band_min_above_floor_m: float = 0.05,
    map_band_max_above_floor_m: float = 1.00,
) -> dict[str, float]:
    """Suggest absolute z-bounds for cleanup and map projection from a cloud's floor anchor.

    The GUI preset used to assume the floor lived near z=0. For merged PLY files whose
    vertical datum is shifted, that removes the floor instead of the ceiling. This helper
    derives absolute values from the cloud itself so the preset remains ground-preserving.
    """
    pts = filter_non_finite(np.asarray(points, dtype=np.float32))
    if pts.size == 0:
        raise ValueError("Cannot estimate a preset from an empty point cloud")

    z = pts[:, 2]
    floor_low = float(np.percentile(z, floor_low_percentile))
    floor_anchor = float(np.percentile(z, floor_anchor_percentile))
    cleanup_min_z = float(floor_low - cleanup_floor_margin_m)
    cleanup_max_z = float(floor_anchor + cleanup_height_m)
    map_min_z = float(floor_anchor + map_band_min_above_floor_m)
    map_max_z = float(floor_anchor + map_band_max_above_floor_m)
    return {
        "floor_low_z": floor_low,
        "floor_anchor_z": floor_anchor,
        "cleanup_min_z": cleanup_min_z,
        "cleanup_max_z": cleanup_max_z,
        "map_min_z": map_min_z,
        "map_max_z": map_max_z,
    }


def slice_points_by_z(points: np.ndarray, min_z: float, max_z: float) -> np.ndarray:
    """Keep only points inside the requested z band."""
    if points.size == 0:
        return points
    mask = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    return points[mask]


def resolve_projection_z_bounds(
    points: np.ndarray,
    min_z: float | None = None,
    max_z: float | None = None,
    *,
    floor_anchor_percentile: float = 5.0,
    z_mode: str = "auto",
) -> tuple[float, float, dict[str, float | str | None]]:
    """Resolve z bounds for map projection.

    z_mode controls how min_z/max_z are interpreted:
      - "auto"     : positive-only bounds are floor-relative offsets; any negative
                     bound switches to absolute world coordinates (legacy heuristic).
      - "absolute" : bounds are always absolute world z values.
      - "floor_relative" : bounds are always offsets above the detected floor anchor.

    Positive-only bounds are treated as offsets above the detected floor anchor so
    clouds whose floor is not near z=0 still use the intended slice. Any negative
    bound keeps the request in absolute world coordinates for explicit overrides.
    """
    pts = filter_non_finite(np.asarray(points, dtype=np.float32))
    if pts.size == 0:
        raise ValueError("Cannot resolve z bounds from an empty point cloud")

    if z_mode == "absolute":
        use_floor_relative = False
    elif z_mode == "floor_relative":
        use_floor_relative = (min_z is not None or max_z is not None)
    else:
        use_floor_relative = (
            (min_z is not None or max_z is not None)
            and (min_z is None or float(min_z) >= 0.0)
            and (max_z is None or float(max_z) >= 0.0)
        )
    floor_anchor = None
    if use_floor_relative:
        floor_anchor = float(np.percentile(pts[:, 2], floor_anchor_percentile))
        low = floor_anchor if min_z is None else floor_anchor + float(min_z)
        high = np.inf if max_z is None else floor_anchor + float(max_z)
        mode = "floor_relative"
    else:
        low = -np.inf if min_z is None else float(min_z)
        high = np.inf if max_z is None else float(max_z)
        mode = "absolute"

    return low, high, {
        "projection_z_mode": mode,
        "projection_floor_anchor_z": floor_anchor,
        "projection_min_z": float(low),
        "projection_max_z": float(high),
    }


def _voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    grid = np.floor(points / voxel_size).astype(np.int64)
    return np.ascontiguousarray(grid).view([("x", np.int64), ("y", np.int64), ("z", np.int64)]).reshape(-1)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Average points that fall into the same voxel."""
    if voxel_size <= 0 or points.size == 0:
        return points

    keys = _voxel_keys(points, voxel_size)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    sums = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis], minlength=counts.shape[0]) for axis in range(3)]
    )
    return (sums / counts[:, None]).astype(np.float32, copy=False)


def approximate_density_filter(
    points: np.ndarray,
    radius: float,
    min_neighbors: int,
    sor_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate local-density filtering using hashed voxel neighborhoods."""
    if points.size == 0 or radius <= 0 or (min_neighbors <= 0 and sor_std <= 0):
        return points, np.zeros(points.shape[0], dtype=np.int32)

    cell = max(radius, 1e-6)
    grid = np.floor(points / cell).astype(np.int32)
    unique_voxels, inverse, counts = np.unique(grid, axis=0, return_inverse=True, return_counts=True)
    lookup = {tuple(v): idx for idx, v in enumerate(unique_voxels.tolist())}

    density = np.zeros(unique_voxels.shape[0], dtype=np.int32)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for idx, voxel in enumerate(unique_voxels):
        total = 0
        vx, vy, vz = voxel.tolist()
        for dx, dy, dz in offsets:
            neighbor_idx = lookup.get((vx + dx, vy + dy, vz + dz))
            if neighbor_idx is not None:
                total += counts[neighbor_idx]
        density[idx] = total

    thresholds = []
    if min_neighbors > 0:
        thresholds.append(float(min_neighbors))
    if sor_std > 0:
        mean = float(density.mean()) if density.size else 0.0
        std = float(density.std()) if density.size else 0.0
        thresholds.append(max(1.0, mean - sor_std * std))
    threshold = max(thresholds) if thresholds else 0.0
    keep_mask = density[inverse] >= threshold
    return points[keep_mask], density[inverse]


def preclean_point_cloud(
    points: np.ndarray,
    min_z: float,
    max_z: float,
    voxel_size: float,
    sor_k: int,
    sor_std: float,
    ror_radius: float,
    ror_min: int,
    log_fn: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Apply the lightweight cleanup steps used by the GUI."""

    def log(message: str) -> None:
        if log_fn:
            log_fn(message)

    current = filter_non_finite(points)
    log(f"[preclean] finite points: {current.shape[0]:,}")

    current = slice_points_by_z(current, min_z, max_z)
    log(f"[preclean] after z-slice [{min_z:.2f}, {max_z:.2f}] m: {current.shape[0]:,}")

    if voxel_size > 0:
        current = voxel_downsample(current, voxel_size)
        log(f"[preclean] after voxel {voxel_size:.3f} m: {current.shape[0]:,}")

    if ror_min > 0 and ror_radius > 0:
        current, _ = approximate_density_filter(current, ror_radius, ror_min)
        log(f"[preclean] after density radius {ror_radius:.3f} m / min {ror_min}: {current.shape[0]:,}")

    if sor_k > 0 and sor_std > 0:
        current, density = approximate_density_filter(current, max(ror_radius, voxel_size, 0.05), 0, sor_std=sor_std)
        mean_density = float(density.mean()) if density.size else 0.0
        log(f"[preclean] after statistical density std {sor_std:.2f}: {current.shape[0]:,} (mean density {mean_density:.1f})")

    return current.astype(np.float32, copy=False)


def _internal_traversability_cell_size(resolution: float, target: float = 0.20) -> float:
    """Choose a coarser internal grid so terrain metrics stay stable and tractable."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    scale = max(1, int(math.ceil(target / resolution)))
    return float(resolution * scale)


def _build_xy_grid(
    points: np.ndarray,
    cell_size: float,
    padding_m: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int]]:
    xy = np.asarray(points[:, :2], dtype=np.float32)
    min_xy = xy.min(axis=0) - float(padding_m)
    max_xy = xy.max(axis=0) + float(padding_m)
    width = max(1, int(math.ceil((max_xy[0] - min_xy[0]) / cell_size)) + 1)
    height = max(1, int(math.ceil((max_xy[1] - min_xy[1]) / cell_size)) + 1)
    gx = np.floor((xy[:, 0] - min_xy[0]) / cell_size).astype(np.int32)
    gy = np.floor((xy[:, 1] - min_xy[1]) / cell_size).astype(np.int32)
    gx = np.clip(gx, 0, width - 1)
    gy = np.clip(gy, 0, height - 1)
    return np.stack([gy, gx], axis=1), (float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1])), (height, width)


def _ground_heightmap(
    points: np.ndarray,
    grid_xy: np.ndarray,
    shape: tuple[int, int],
    ground_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate ground height as a low-percentile z value in each observed cell."""
    height, width = shape
    ground = np.full((height, width), np.nan, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.int32)

    linear = grid_xy[:, 0] * width + grid_xy[:, 1]
    order = np.argsort(linear, kind="mergesort")
    linear_s = linear[order]
    z_s = np.asarray(points[order, 2], dtype=np.float32)

    uniq, starts, cnts = np.unique(linear_s, return_index=True, return_counts=True)
    for cell_id, start, count in zip(uniq.tolist(), starts.tolist(), cnts.tolist()):
        row = cell_id // width
        col = cell_id % width
        counts[row, col] = int(count)
        zs = z_s[start:start + count]
        ground[row, col] = np.percentile(zs, ground_percentile)

    return ground, counts


def _resolve_terrain_min_points_threshold(
    counts: np.ndarray,
    requested_min_points_per_cell: int,
    target_keep_fraction: float = 0.75,
) -> tuple[int, dict[str, float | int | str]]:
    """Choose a floor-cell density threshold that does not collapse sparse clouds."""
    requested = max(1, int(requested_min_points_per_cell))
    observed = np.asarray(counts, dtype=np.int32)
    observed = observed[observed > 0]
    if observed.size == 0:
        return requested, {
            "requested_min_points_per_cell": requested,
            "applied_min_points_per_cell": requested,
            "threshold_mode": "requested",
            "observed_cells": 0,
            "requested_keep_fraction": 0.0,
            "applied_keep_fraction": 0.0,
        }

    keep_requested = float(np.mean(observed >= requested))
    applied = requested
    mode = "requested"
    floor_min = max(2, requested // 2)  # never drop below 2 points per cell
    if requested > 1 and keep_requested < float(target_keep_fraction):
        mode = "adaptive"
        applied = floor_min
        for candidate in range(requested - 1, floor_min - 1, -1):
            keep_candidate = float(np.mean(observed >= candidate))
            applied = candidate
            if keep_candidate >= float(target_keep_fraction):
                break

    keep_applied = float(np.mean(observed >= applied))
    return applied, {
        "requested_min_points_per_cell": requested,
        "applied_min_points_per_cell": int(applied),
        "threshold_mode": mode,
        "observed_cells": int(observed.size),
        "requested_keep_fraction": keep_requested,
        "applied_keep_fraction": keep_applied,
    }


def _nan_safe_smooth(z: np.ndarray, sigma_cells: float = 1.5) -> np.ndarray:
    """NaN-aware Gaussian smoothing of a 2D heightmap (pure numpy, no scipy).

    Replaces each cell with a weighted average of its neighbors using a Gaussian
    kernel, ignoring NaN cells.  This removes per-cell noise from the ground
    heightmap so that gradient-based slope estimates are stable and match the true
    physical slope of ramps / terrain.
    """
    radius = int(math.ceil(2.0 * sigma_cells))
    size = 2 * radius + 1
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-0.5 * (ax / sigma_cells) ** 2)
    kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)

    zz = np.asarray(z, dtype=np.float32).copy()
    valid = (~np.isnan(zz)).astype(np.float32)
    zz_filled = np.where(np.isnan(zz), 0.0, zz).astype(np.float32)

    pad = radius
    zp = np.pad(zz_filled, pad, mode="constant", constant_values=0.0)
    vp = np.pad(valid, pad, mode="constant", constant_values=0.0)
    h, w = zz.shape
    weighted_sum = np.zeros((h, w), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    for dy in range(size):
        for dx in range(size):
            k = float(kernel[dy, dx])
            weighted_sum += k * zp[dy:dy + h, dx:dx + w]
            weight_sum += k * vp[dy:dy + h, dx:dx + w]

    out = np.full((h, w), np.nan, dtype=np.float32)
    good = weight_sum > 1e-12
    out[good] = (weighted_sum[good] / weight_sum[good]).astype(np.float32)
    # Keep original NaN cells as NaN
    out[~valid.astype(bool)] = np.nan
    return out


def _nan_safe_median_3x3(z: np.ndarray) -> np.ndarray:
    """NaN-aware 3x3 median filter — removes outlier height spikes without blurring edges."""
    h, w = z.shape
    padded = np.pad(z.astype(np.float32), 1, mode="constant", constant_values=np.nan)
    stack = np.stack([padded[r:r + h, c:c + w] for r in range(3) for c in range(3)], axis=0)
    with np.errstate(all="ignore"):
        out = np.nanmedian(stack, axis=0).astype(np.float32)
    out[np.isnan(z)] = np.nan
    return out


def _nan_safe_gradients(z: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Fill small NaN holes from 4-neighbors before computing terrain gradients."""
    zz = np.asarray(z, dtype=np.float32).copy()
    # Fill small NaN holes from 4-neighbors
    for _ in range(6):
        nan_mask = np.isnan(zz)
        if not nan_mask.any():
            break
        neighbors = np.stack(
            [np.roll(zz, 1, 0), np.roll(zz, -1, 0), np.roll(zz, 1, 1), np.roll(zz, -1, 1)],
            axis=0,
        )
        valid = ~np.isnan(neighbors)
        counts = valid.sum(axis=0)
        sums = np.nansum(neighbors, axis=0)
        means = np.full_like(zz, np.nan)
        fillable_counts = counts > 0
        means[fillable_counts] = (sums[fillable_counts] / counts[fillable_counts]).astype(np.float32, copy=False)
        fillable = nan_mask & ~np.isnan(means)
        if not np.any(fillable):
            break
        zz[fillable] = means[fillable]

    dzdy, dzdx = np.gradient(zz, cell_size, cell_size)
    return dzdx.astype(np.float32, copy=False), dzdy.astype(np.float32, copy=False)



def _step_height(z: np.ndarray) -> np.ndarray:
    """Maximum absolute height change to 4-neighbors without edge wrap-around."""
    height, width = z.shape
    out = np.zeros((height, width), dtype=np.float32)

    dz = np.full((height, width), np.nan, dtype=np.float32)
    dz[1:, :] = np.abs(z[1:, :] - z[:-1, :])
    valid = ~np.isnan(dz)
    out[valid] = np.maximum(out[valid], dz[valid])

    dz = np.full((height, width), np.nan, dtype=np.float32)
    dz[:-1, :] = np.abs(z[:-1, :] - z[1:, :])
    valid = ~np.isnan(dz)
    out[valid] = np.maximum(out[valid], dz[valid])

    dz = np.full((height, width), np.nan, dtype=np.float32)
    dz[:, 1:] = np.abs(z[:, 1:] - z[:, :-1])
    valid = ~np.isnan(dz)
    out[valid] = np.maximum(out[valid], dz[valid])

    dz = np.full((height, width), np.nan, dtype=np.float32)
    dz[:, :-1] = np.abs(z[:, :-1] - z[:, 1:])
    valid = ~np.isnan(dz)
    out[valid] = np.maximum(out[valid], dz[valid])
    return out


def _morph_close(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """Morphological closing (dilate then erode) to bridge small gaps in a bool mask."""
    h, w = mask.shape
    arr = mask.astype(np.uint8)
    # Dilate: set cell to 1 if any neighbor within radius is 1
    dilated = arr.copy()
    for _ in range(radius):
        padded = np.pad(dilated, 1, mode="constant", constant_values=0)
        dilated = (
            padded[1:h+1, 1:w+1] | padded[0:h, 1:w+1] | padded[2:h+2, 1:w+1] |
            padded[1:h+1, 0:w] | padded[1:h+1, 2:w+2]
        ).astype(np.uint8)
    # Erode: pad with 1s (edge replication) so boundaries are not eaten away
    eroded = dilated.copy()
    for _ in range(radius):
        padded = np.pad(eroded, 1, mode="edge")
        eroded = (
            padded[1:h+1, 1:w+1] & padded[0:h, 1:w+1] & padded[2:h+2, 1:w+1] &
            padded[1:h+1, 0:w] & padded[1:h+1, 2:w+2]
        ).astype(np.uint8)
    return eroded.astype(bool)


def _remove_small_components(mask: np.ndarray, min_cells: int = 50) -> np.ndarray:
    """Remove connected components smaller than min_cells from a boolean mask."""
    height, width = mask.shape
    labels = np.zeros_like(mask, dtype=np.int32)
    comp_id = 0
    comp_sizes: list[int] = [0]  # 0 = background

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col]:
                continue
            comp_id += 1
            queue = deque([(row, col)])
            labels[row, col] = comp_id
            size = 0
            while queue:
                rr, cc = queue.popleft()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not labels[nr, nc]:
                        labels[nr, nc] = comp_id
                        queue.append((nr, nc))
            comp_sizes.append(size)

    comp_sizes_arr = np.array(comp_sizes, dtype=np.int32)
    keep = comp_sizes_arr >= min_cells
    keep[0] = False  # background always excluded
    return keep[labels]


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected True component."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=np.uint8)
    best: list[tuple[int, int]] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            queue = deque([(row, col)])
            seen[row, col] = 1
            comp = [(row, col)]
            while queue:
                rr, cc = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = 1
                        queue.append((nr, nc))
                        comp.append((nr, nc))
            if len(comp) > len(best):
                best = comp

    out = np.zeros_like(mask, dtype=bool)
    for row, col in best:
        out[row, col] = True
    return out


def _terrain_masks_from_points(
    points: np.ndarray,
    resolution: float = 0.05,
    padding_m: float = 0.5,
    min_points_per_cell: int = 3,
    min_z: float | None = None,
    max_z: float | None = None,
    ground_percentile: float = 10.0,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
    origin_xy: tuple[float, float] | None = None,
    shape: tuple[int, int] | None = None,
    z_mode: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | list[float]], dict[str, float | int]]:
    """Compute aligned known/traversable/reachable terrain masks from observed floor-like terrain."""
    if points.size == 0:
        raise ValueError("Cannot build a map from an empty point cloud")

    pts = filter_non_finite(np.asarray(points, dtype=np.float32))
    floor_anchor = float(np.percentile(pts[:, 2], ground_percentile))
    floor_low = float(floor_anchor)
    floor_high = float(floor_anchor + 1.0)
    resolved_z_mode = "floor_relative"
    if min_z is not None or max_z is not None:
        low, high, z_info = resolve_projection_z_bounds(
            pts,
            min_z=min_z,
            max_z=max_z,
            floor_anchor_percentile=ground_percentile,
            z_mode=z_mode,
        )
        floor_low = float(low)
        floor_high = float(high)
        resolved_z_mode = str(z_info["projection_z_mode"])
        pts = pts[(pts[:, 2] >= floor_low) & (pts[:, 2] <= floor_high)]
    else:
        pts = pts[(pts[:, 2] >= floor_low) & (pts[:, 2] <= floor_high)]
    if pts.size == 0:
        raise ValueError("All points were removed by the traversable floor-height band")

    internal_cell = _internal_traversability_cell_size(resolution)
    upsample = max(1, int(round(internal_cell / resolution)))

    if origin_xy is not None and shape is not None:
        min_x = float(origin_xy[0])
        min_y = float(origin_xy[1])
        out_h, out_w = int(shape[0]), int(shape[1])
        coarse_h = max(1, int(math.ceil(out_h / upsample)))
        coarse_w = max(1, int(math.ceil(out_w / upsample)))
        gx = np.floor((pts[:, 0] - min_x) / internal_cell).astype(np.int32)
        gy = np.floor((pts[:, 1] - min_y) / internal_cell).astype(np.int32)
        valid = (gx >= 0) & (gx < coarse_w) & (gy >= 0) & (gy < coarse_h)
        pts = pts[valid]
        if pts.size == 0:
            raise ValueError("No traversable-floor points overlap the reference map bounds")
        grid_xy = np.stack([gy[valid], gx[valid]], axis=1)
        bounds = (
            min_x,
            min_y,
            min_x + max(0, out_w - 1) * resolution,
            min_y + max(0, out_h - 1) * resolution,
        )
        coarse_shape = (coarse_h, coarse_w)
        out_shape = (out_h, out_w)
    else:
        grid_xy, bounds, coarse_shape = _build_xy_grid(pts, internal_cell, padding_m)
        min_x, min_y, max_x, max_y = bounds
        out_shape = (
            max(1, int(math.ceil((max_y - min_y) / resolution)) + 1),
            max(1, int(math.ceil((max_x - min_x) / resolution)) + 1),
        )

    ground_all, counts = _ground_heightmap(
        pts,
        grid_xy,
        coarse_shape,
        ground_percentile=ground_percentile,
    )
    applied_min_points_per_cell, threshold_stats = _resolve_terrain_min_points_threshold(
        counts,
        min_points_per_cell,
    )
    ground = np.asarray(ground_all, dtype=np.float32).copy()
    ground[counts < int(applied_min_points_per_cell)] = np.nan
    known = ~np.isnan(ground)

    # ── Denoise the ground heightmap ──
    # 1. Median filter: removes outlier spikes without blurring real edges (ramps, steps)
    ground_clean = _nan_safe_median_3x3(ground)
    # 2. Gaussian smooth: suppresses remaining noise so gradients are stable
    ground_smooth = _nan_safe_smooth(ground_clean, sigma_cells=2.0)

    dzdx, dzdy = _nan_safe_gradients(ground_smooth, internal_cell)
    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
    step = _step_height(ground_smooth)

    traversable = (
        known
        & (slope_deg <= float(max_slope_deg))
        & (step <= float(max_step_m))
    )

    # ── Clean the traversable mask ──
    # 1. Remove small isolated known-cell clusters (scattered LiDAR noise)
    known = _remove_small_components(known, min_cells=20)
    traversable = traversable & known
    # 2. Remove small isolated traversable specks
    traversable = _remove_small_components(traversable, min_cells=20)
    # 3. Morphological closing: bridge small gaps so areas stay connected
    traversable = _morph_close(traversable, radius=3) & known
    # 4. Fill small non-traversable holes inside traversable regions
    holes_mask = _remove_small_components(~traversable & known, min_cells=20)
    traversable = traversable | (~holes_mask & known)

    reachable = _largest_component(traversable)

    min_x, min_y, _, _ = bounds
    height, width = out_shape

    def upsample_mask(mask: np.ndarray) -> np.ndarray:
        if upsample > 1:
            out = np.repeat(np.repeat(mask, upsample, axis=0), upsample, axis=1)
        else:
            out = mask
        return out[:height, :width]

    known_mask = upsample_mask(known)
    traversable_mask = upsample_mask(traversable)
    reachable_mask = upsample_mask(reachable)


    yaml_data = {
        "resolution": float(resolution),
        "origin": [float(min_x), float(min_y), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
        "mode": "trinary",
    }
    stats = {
        "input_points": int(pts.shape[0]),
        "known_cells": int(known.sum()),
        "traversable_cells": int(traversable.sum()),
        "reachable_cells": int(reachable.sum()),
        "output_known_cells": int(known_mask.sum()),
        "output_traversable_cells": int(traversable_mask.sum()),
        "output_reachable_cells": int(reachable_mask.sum()),
        "internal_cell_size": float(internal_cell),
        "floor_anchor_z": float(floor_anchor),
        "floor_band_min_z": float(floor_low),
        "floor_band_max_z": float(floor_high),
        "floor_band_height": float(floor_high - floor_low),
        "projection_z_mode": resolved_z_mode,
        **threshold_stats,
    }
    return known_mask, traversable_mask, reachable_mask, yaml_data, stats


def _mask_to_occ_grid(mask: np.ndarray) -> np.ndarray:
    grid_occ = np.zeros(mask.shape, dtype=np.uint8)
    grid_occ[mask] = 254
    grid_occ[0, :] = 0
    grid_occ[-1, :] = 0
    grid_occ[:, 0] = 0
    grid_occ[:, -1] = 0
    return grid_occ


def build_traversability_image(
    points: np.ndarray,
    resolution: float = 0.05,
    padding_m: float = 0.5,
    min_points_per_cell: int = 3,
    min_z: float | None = None,
    max_z: float | None = None,
    ground_percentile: float = 10.0,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
    reachable_only: bool = False,
    origin_xy: tuple[float, float] | None = None,
    shape: tuple[int, int] | None = None,
    z_mode: str = "auto",

) -> tuple[np.ndarray, dict[str, float | int | list[float]], dict[str, float | int]]:
    """Build a binary traversability map from observed floor-like terrain."""
    known_mask, traversable_mask, reachable_mask, yaml_data, stats = _terrain_masks_from_points(
        points,
        resolution=resolution,
        padding_m=padding_m,
        min_points_per_cell=min_points_per_cell,
        min_z=min_z,
        max_z=max_z,
        ground_percentile=ground_percentile,
        max_slope_deg=max_slope_deg,
        max_step_m=max_step_m,
        origin_xy=origin_xy,
        shape=shape,
        z_mode=z_mode,

    )
    free_mask = reachable_mask if reachable_only else traversable_mask
    stats["output_free_cells"] = int(free_mask.sum())
    return _mask_to_occ_grid(free_mask), yaml_data, stats


def build_known_floor_image(
    points: np.ndarray,
    resolution: float = 0.05,
    padding_m: float = 0.5,
    min_points_per_cell: int = 3,
    min_z: float | None = None,
    max_z: float | None = None,
    ground_percentile: float = 10.0,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
    origin_xy: tuple[float, float] | None = None,
    shape: tuple[int, int] | None = None,
    z_mode: str = "auto",
) -> tuple[np.ndarray, dict[str, float | int | list[float]], dict[str, float | int]]:
    """Build a binary observed-floor map from cells with a valid local ground estimate."""
    known_mask, _trav_mask, _reachable_mask, yaml_data, stats = _terrain_masks_from_points(
        points,
        resolution=resolution,
        padding_m=padding_m,
        min_points_per_cell=min_points_per_cell,
        min_z=min_z,
        max_z=max_z,
        ground_percentile=ground_percentile,
        max_slope_deg=max_slope_deg,
        max_step_m=max_step_m,
        origin_xy=origin_xy,
        shape=shape,
        z_mode=z_mode,

    )
    floor_mask = known_mask
    stats["output_free_cells"] = int(floor_mask.sum())
    return _mask_to_occ_grid(floor_mask), yaml_data, stats


def build_occupancy_image(
    points: np.ndarray,
    resolution: float = 0.05,
    padding_m: float = 0.5,
    min_points_per_cell: int = 1,
    min_z: float | None = None,
    max_z: float | None = None,
) -> tuple[np.ndarray, dict[str, float | int | list[float]]]:
    """Project a cleaned PCD into a simple 2D occupancy map."""
    if points.size == 0:
        raise ValueError("Cannot build a map from an empty point cloud")

    pts = filter_non_finite(np.asarray(points, dtype=np.float32))
    if min_z is not None or max_z is not None:
        low, high, z_info = resolve_projection_z_bounds(pts, min_z=min_z, max_z=max_z)
        pts = slice_points_by_z(pts, low, high)
    else:
        z_info = {
            "projection_z_mode": "absolute",
            "projection_floor_anchor_z": None,
            "projection_min_z": float("-inf"),
            "projection_max_z": float("inf"),
        }
    if pts.size == 0:
        raise ValueError("All points were removed by the requested z limits")

    xy = pts[:, :2]
    min_xy = xy.min(axis=0) - float(padding_m)
    max_xy = xy.max(axis=0) + float(padding_m)
    width = max(1, int(math.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1)
    height = max(1, int(math.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1)

    gx = np.floor((pts[:, 0] - min_xy[0]) / resolution).astype(np.int32)
    gy = np.floor((pts[:, 1] - min_xy[1]) / resolution).astype(np.int32)
    valid = (gx >= 0) & (gx < width) & (gy >= 0) & (gy < height)
    cell_ids = gy[valid] * width + gx[valid]
    counts = np.bincount(cell_ids, minlength=width * height)

    grid_occ = np.full((height, width), 254, dtype=np.uint8)
    grid_occ.reshape(-1)[counts >= int(max(1, min_points_per_cell))] = 0
    grid_occ[0, :] = 0
    grid_occ[-1, :] = 0
    grid_occ[:, 0] = 0
    grid_occ[:, -1] = 0

    yaml_data = {
        "resolution": float(resolution),
        "origin": [float(min_xy[0]), float(min_xy[1]), 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
        "mode": "trinary",
    }
    yaml_data.update(z_info)
    return grid_occ, yaml_data


def write_pgm(path: str, image: np.ndarray) -> None:
    """Write a grayscale NumPy image as a binary PGM."""
    img = np.asarray(image, dtype=np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode("ascii"))
        handle.write(img.tobytes())


def write_nav2_yaml(path: str, image_path: str, yaml_data: dict[str, float | int | list[float]]) -> None:
    """Write a Nav2-compatible YAML map config."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"image: {os.path.basename(image_path)}\n")
        handle.write(f"mode: {yaml_data.get('mode', 'trinary')}\n")
        handle.write(f"resolution: {yaml_data['resolution']}\n")
        origin = yaml_data["origin"]
        handle.write(f"origin: [{origin[0]}, {origin[1]}, {origin[2]}]\n")
        handle.write(f"negate: {yaml_data.get('negate', 0)}\n")
        handle.write(f"occupied_thresh: {yaml_data.get('occupied_thresh', 0.65)}\n")
        handle.write(f"free_thresh: {yaml_data.get('free_thresh', 0.25)}\n")


def export_occupancy_map(
    points: np.ndarray,
    out_prefix: str,
    resolution: float = 0.05,
    padding_m: float = 0.5,
    min_points_per_cell: int = 1,
    min_z: float | None = None,
    max_z: float | None = None,
) -> tuple[str, str, np.ndarray, dict[str, float | int | list[float]]]:
    """Create and save a PGM/YAML occupancy map from a point cloud."""
    occ_grid, yaml_data = build_occupancy_image(
        points,
        resolution=resolution,
        padding_m=padding_m,
        min_points_per_cell=min_points_per_cell,
        min_z=min_z,
        max_z=max_z,
    )
    pgm_path = out_prefix + ".pgm"
    yaml_path = out_prefix + ".yaml"
    write_pgm(pgm_path, occ_grid[::-1, :])
    write_nav2_yaml(yaml_path, pgm_path, yaml_data)
    return pgm_path, yaml_path, occ_grid, yaml_data


def export_traversability_map(
    points: np.ndarray,
    out_prefix: str,
    resolution: float = 0.05,
    padding_m: float = 0.5,
    min_points_per_cell: int = 3,
    min_z: float | None = None,
    max_z: float | None = None,
    ground_percentile: float = 10.0,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
    reachable_only: bool = False,
    origin_xy: tuple[float, float] | None = None,
    shape: tuple[int, int] | None = None,
    z_mode: str = "auto",

) -> tuple[str, str, np.ndarray, dict[str, float | int | list[float]], dict[str, float | int]]:
    """Create and save a 2D traversability map from a point cloud."""
    occ_grid, yaml_data, stats = build_traversability_image(
        points,
        resolution=resolution,
        padding_m=padding_m,
        min_points_per_cell=min_points_per_cell,
        min_z=min_z,
        max_z=max_z,
        ground_percentile=ground_percentile,
        max_slope_deg=max_slope_deg,
        max_step_m=max_step_m,
        reachable_only=reachable_only,
        origin_xy=origin_xy,
        shape=shape,
        z_mode=z_mode,

    )
    pgm_path = out_prefix + ".pgm"
    yaml_path = out_prefix + ".yaml"
    write_pgm(pgm_path, occ_grid[::-1, :])
    write_nav2_yaml(yaml_path, pgm_path, yaml_data)
    return pgm_path, yaml_path, occ_grid, yaml_data, stats


def export_known_floor_map(
    points: np.ndarray,
    out_prefix: str,
    resolution: float = 0.05,
    padding_m: float = 0.5,
    min_points_per_cell: int = 3,
    min_z: float | None = None,
    max_z: float | None = None,
    ground_percentile: float = 10.0,
    max_slope_deg: float = 35.0,
    max_step_m: float = 0.25,
    origin_xy: tuple[float, float] | None = None,
    shape: tuple[int, int] | None = None,
    z_mode: str = "auto",
) -> tuple[str, str, np.ndarray, dict[str, float | int | list[float]], dict[str, float | int]]:
    """Create and save a 2D observed-floor map from a point cloud."""
    occ_grid, yaml_data, stats = build_known_floor_image(
        points,
        resolution=resolution,
        padding_m=padding_m,
        min_points_per_cell=min_points_per_cell,
        min_z=min_z,
        max_z=max_z,
        ground_percentile=ground_percentile,
        max_slope_deg=max_slope_deg,
        max_step_m=max_step_m,
        origin_xy=origin_xy,
        shape=shape,
        z_mode=z_mode,
    )
    pgm_path = out_prefix + ".pgm"
    yaml_path = out_prefix + ".yaml"
    write_pgm(pgm_path, occ_grid[::-1, :])
    write_nav2_yaml(yaml_path, pgm_path, yaml_data)
    return pgm_path, yaml_path, occ_grid, yaml_data, stats
