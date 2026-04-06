#!/usr/bin/env python3
"""Generate a 2D traversability or obstacle map from a cleaned PCD or PLY file."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from pcd_package.pcd_tools import (
    export_known_floor_map,
    export_occupancy_map,
    export_traversability_map,
    load_xyz_points,
)


def read_pgm_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as handle:
        data = handle.read(256)
    i = 0

    def skip_ws_comments() -> None:
        nonlocal i
        while i < len(data):
            if data[i] == 35:
                while i < len(data) and data[i] != 10:
                    i += 1
                i += 1
            elif data[i] <= 32:
                i += 1
            else:
                break

    def read_token() -> str:
        nonlocal i
        skip_ws_comments()
        tok = bytearray()
        while i < len(data) and data[i] > 32:
            tok.append(data[i])
            i += 1
        return tok.decode()

    magic = read_token()
    if magic not in {"P5", "P2"}:
        raise ValueError(f"Unsupported PGM header in {path}")
    width = int(read_token())
    height = int(read_token())
    return width, height


def read_nav2_yaml(path: str) -> dict[str, float | int | list[float]]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    def get(key: str, default: str) -> str:
        match = re.search(rf"^{key}\s*:\s*(.+)", text, re.MULTILINE)
        return match.group(1).strip() if match else default

    origin_s = get("origin", "[0,0,0]").replace("[", "").replace("]", "")
    return {
        "resolution": float(get("resolution", "0.05")),
        "origin": [float(part) for part in origin_s.split(",")],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project a cleaned PCD or PLY into a 2D map.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input point cloud path")
    parser.add_argument("--out-prefix", required=True, help="Output path prefix, e.g. /tmp/map")
    parser.add_argument(
        "--mode",
        choices=["traversability", "obstacle", "floor"],
        default="obstacle",
        help="Map semantics: obstacle marks any point hit as occupied; traversability keeps terrain-passable floor; floor keeps all observed floor cells.",
    )
    parser.add_argument("--resolution", type=float, default=0.05, help="Map resolution in metres")
    parser.add_argument("--padding", type=float, default=0.5, help="Map padding around XY bounds in metres")
    parser.add_argument("--min-points-per-cell", type=int, default=3, help="Minimum points to treat a cell as observed floor")
    parser.add_argument(
        "--min_z",
        type=float,
        default=None,
        help="Optional lower z bound before projection. Positive-only bounds are treated as offsets above the detected floor.",
    )
    parser.add_argument(
        "--max_z",
        type=float,
        default=None,
        help="Optional upper z bound before projection. Positive-only bounds are treated as offsets above the detected floor.",
    )
    parser.add_argument("--ground-percentile", type=float, default=10.0, help="Low-percentile height used as local ground estimate")
    parser.add_argument("--max-slope-deg", type=float, default=35.0, help="Maximum terrain slope for traversability")
    parser.add_argument("--max-step-m", type=float, default=0.25, help="Maximum step height between neighboring cells")
    parser.add_argument("--max-roughness-m", type=float, default=0.15, help="Maximum 3x3 local height roughness")
    parser.add_argument("--reachable-only", action="store_true", help="Keep only the largest connected traversable region")
    parser.add_argument("--absolute-z", action="store_true", help="Treat --min_z/--max_z as absolute world z values (skip floor-relative heuristic)")
    parser.add_argument("--align-pgm", default=None, help="Optional reference PGM to align traversability output dimensions")
    parser.add_argument("--align-yaml", default=None, help="Optional reference map YAML to align traversability origin/resolution")
    args = parser.parse_args()

    points = load_xyz_points(args.in_path)
    if args.mode == "obstacle":
        pgm_path, yaml_path, occ_grid, yaml_data = export_occupancy_map(
            points,
            out_prefix=args.out_prefix,
            resolution=args.resolution,
            padding_m=args.padding,
            min_points_per_cell=args.min_points_per_cell,
            min_z=args.min_z,
            max_z=args.max_z,
        )
        stats = None
    else:
        resolution = args.resolution
        origin_xy = None
        shape = None
        if args.align_pgm or args.align_yaml:
            if not args.align_pgm or not args.align_yaml:
                raise ValueError("--align-pgm and --align-yaml must be provided together")
            ref_width, ref_height = read_pgm_size(args.align_pgm)
            ref_yaml = read_nav2_yaml(args.align_yaml)
            resolution = float(ref_yaml["resolution"])
            origin = ref_yaml["origin"]
            origin_xy = (float(origin[0]), float(origin[1]))
            shape = (int(ref_height), int(ref_width))
        export_fn = export_traversability_map if args.mode == "traversability" else export_known_floor_map
        extra = {"reachable_only": args.reachable_only} if args.mode == "traversability" else {}
        z_mode = "absolute" if args.absolute_z else "auto"
        pgm_path, yaml_path, occ_grid, yaml_data, stats = export_fn(
            points,
            out_prefix=args.out_prefix,
            resolution=resolution,
            padding_m=args.padding,
            min_points_per_cell=args.min_points_per_cell,
            min_z=args.min_z,
            max_z=args.max_z,
            ground_percentile=args.ground_percentile,
            max_slope_deg=args.max_slope_deg,
            max_step_m=args.max_step_m,
            max_roughness_m=args.max_roughness_m,
            origin_xy=origin_xy,
            shape=shape,
            z_mode=z_mode,
            **extra,
        )

    occupied = int((occ_grid == 0).sum())
    free = int((occ_grid == 254).sum())
    print(f"[map] wrote {pgm_path}")
    print(f"[map] wrote {yaml_path}")
    print(
        "[map] size: "
        f"{occ_grid.shape[1]}x{occ_grid.shape[0]} @ {yaml_data['resolution']:.3f} m | "
        f"occupied={occupied:,} free={free:,}"
    )
    z_mode = yaml_data.get("projection_z_mode")
    if z_mode is not None:
        low = float(yaml_data["projection_min_z"])
        high = float(yaml_data["projection_max_z"])
        floor_anchor = yaml_data.get("projection_floor_anchor_z")
        floor_text = "n/a" if floor_anchor is None else f"{float(floor_anchor):.3f}"
        print(
            "[map] z-band: "
            f"mode={z_mode} "
            f"applied=[{low:.3f}, {high:.3f}] "
            f"floor_anchor={floor_text}"
        )
    if stats is not None:
        print(
            f"[map] {args.mode}: "
            f"known={stats['known_cells']:,} "
            f"trav={stats['traversable_cells']:,} "
            f"reachable={stats['reachable_cells']:,} "
            f"free_out={stats['output_free_cells']:,} "
            f"internal_cell={stats['internal_cell_size']:.3f} m "
            f"floor_z=[{stats['floor_band_min_z']:.3f}, {stats['floor_band_max_z']:.3f}] "
            f"mode={stats['projection_z_mode']} "
            f"min_pts={stats['applied_min_points_per_cell']}/{stats['requested_min_points_per_cell']} "
            f"threshold={stats['threshold_mode']} "
            f"keep={100.0 * stats['applied_keep_fraction']:.1f}%"
        )


if __name__ == "__main__":
    main()
