"""Minimal PCD/PLY loader — self-contained, no external dependencies beyond numpy."""

from __future__ import annotations

import io
import os
import struct
from typing import Tuple

import numpy as np


def load_xyz_points(path: str) -> np.ndarray:
    """Load XYZ coordinates from a PCD or PLY file. Returns (N, 3) float32."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        return _load_ply_xyz(path)
    return _load_pcd_xyz(path)


def _load_pcd_xyz(path: str) -> np.ndarray:
    """Load XYZ from a PCD file (ascii or binary)."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line.startswith("DATA"):
                break

        fields = []
        sizes = []
        types = []
        counts = []
        n_points = 0
        data_mode = "ascii"

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            key = parts[0].upper()
            if key == "FIELDS":
                fields = [p.lower() for p in parts[1:]]
            elif key == "SIZE":
                sizes = [int(p) for p in parts[1:]]
            elif key == "TYPE":
                types = parts[1:]
            elif key == "COUNT":
                counts = [int(p) for p in parts[1:]]
            elif key == "POINTS" or key == "WIDTH":
                n_points = max(n_points, int(parts[1]))
            elif key == "DATA":
                data_mode = parts[1].lower() if len(parts) > 1 else "ascii"

        if not fields or n_points == 0:
            raise ValueError(f"Invalid PCD header in {path}")

        # Find x, y, z field indices
        try:
            xi = fields.index("x")
            yi = fields.index("y")
            zi = fields.index("z")
        except ValueError as exc:
            raise ValueError(f"Missing XYZ fields in {path}: {fields}") from exc

        if data_mode == "ascii":
            data = f.read().decode("ascii", errors="replace")
            rows = [line.split() for line in data.strip().split("\n") if line.strip()]
            pts = np.array(
                [[float(row[xi]), float(row[yi]), float(row[zi])] for row in rows[:n_points]],
                dtype=np.float32,
            )
        else:
            # Binary
            point_size = sum(s * c for s, c in zip(sizes, counts))
            offsets = []
            off = 0
            for s, c in zip(sizes, counts):
                offsets.append(off)
                off += s * c

            raw = f.read(n_points * point_size)
            pts = np.zeros((n_points, 3), dtype=np.float32)
            for i in range(n_points):
                base = i * point_size
                for dim, fi in enumerate([xi, yi, zi]):
                    o = base + offsets[fi]
                    s = sizes[fi]
                    t = types[fi]
                    if t == "F" and s == 4:
                        pts[i, dim] = struct.unpack_from("<f", raw, o)[0]
                    elif t == "F" and s == 8:
                        pts[i, dim] = struct.unpack_from("<d", raw, o)[0]
                    elif t == "I" and s == 4:
                        pts[i, dim] = struct.unpack_from("<i", raw, o)[0]
                    elif t == "U" and s == 4:
                        pts[i, dim] = struct.unpack_from("<I", raw, o)[0]

    return _filter_finite(pts)


def _load_ply_xyz(path: str) -> np.ndarray:
    """Load XYZ from a PLY file (ascii or binary_little_endian)."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_vertices = 0
        props = []
        fmt = "ascii"

        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "element" and parts[1] == "vertex":
                n_vertices = int(parts[2])
            elif parts[0] == "property":
                props.append(parts[-1].lower())
            elif parts[0] == "format":
                fmt = parts[1]

        try:
            xi = props.index("x")
            yi = props.index("y")
            zi = props.index("z")
        except ValueError as exc:
            raise ValueError(f"Missing XYZ in PLY: {props}") from exc

        if fmt == "ascii":
            data = f.read().decode("ascii", errors="replace")
            rows = data.strip().split("\n")
            pts = np.zeros((min(len(rows), n_vertices), 3), dtype=np.float32)
            for i, row in enumerate(rows[:n_vertices]):
                vals = row.split()
                pts[i] = [float(vals[xi]), float(vals[yi]), float(vals[zi])]
        else:
            # binary_little_endian — assume all float32 properties
            row_size = len(props) * 4
            raw = f.read(n_vertices * row_size)
            all_data = np.frombuffer(raw, dtype=np.float32).reshape(n_vertices, len(props))
            pts = np.column_stack((all_data[:, xi], all_data[:, yi], all_data[:, zi]))

    return _filter_finite(pts)


def _filter_finite(pts: np.ndarray) -> np.ndarray:
    """Drop NaN/Inf points."""
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask].astype(np.float32, copy=False)
