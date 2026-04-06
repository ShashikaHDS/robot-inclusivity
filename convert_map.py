#!/usr/bin/env python3
"""
Convert a 3D point cloud (.pcd/.ply) to a 2D Nav2 occupancy map (.pgm + .yaml).

Auto-detects the ground plane so you never need to guess absolute Z values.
Produces three map layers: obstacle, traversability, and floor.

Usage:
    python3 convert_map.py --in GlobalMap.pcd --out-dir ./map_out
    python3 convert_map.py --in GlobalMap.pcd --out-dir ./map_out --min-z 0.05 --max-z 1.0
    python3 convert_map.py --in GlobalMap.pcd --out-dir ./map_out --analyze-z
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PCD_PKG = SCRIPT_DIR / "src" / "pcd_package"
if str(PCD_PKG) not in sys.path:
    sys.path.insert(0, str(PCD_PKG))

import numpy as np
from pcd_package.pcd_tools import (
    estimate_ground_preserving_preset,
    export_known_floor_map,
    export_occupancy_map,
    export_traversability_map,
    load_xyz_points,
)


def analyze_z(points: np.ndarray) -> None:
    """Print Z distribution analysis to help the user pick bounds."""
    z = points[:, 2]
    preset = estimate_ground_preserving_preset(points)

    print("\n=== Z Distribution ===")
    print(f"  Min:    {z.min():.3f} m")
    print(f"  Max:    {z.max():.3f} m")
    print(f"  Mean:   {z.mean():.3f} m")
    print(f"  Median: {np.median(z):.3f} m")

    print(f"\n=== Auto-Detected Ground ===")
    print(f"  Floor anchor (5th pctile): {preset['floor_anchor_z']:.3f} m")
    print(f"  Floor low    (2nd pctile): {preset['floor_low_z']:.3f} m")

    print(f"\n=== Recommended Ranges ===")
    print(f"  Obstacle map Z:  [{preset['map_min_z']:.3f}, {preset['map_max_z']:.3f}] m  (absolute)")
    print(f"  Cleanup Z:       [{preset['cleanup_min_z']:.3f}, {preset['cleanup_max_z']:.3f}] m  (absolute)")
    print(f"  Floor-relative:  min_z=0.05  max_z=1.00  (offsets above floor anchor)")

    print(f"\n=== Z Histogram (top bins) ===")
    bins = np.arange(z.min(), z.max() + 0.25, 0.25)
    counts, edges = np.histogram(z, bins=bins)
    top_indices = np.argsort(counts)[::-1][:15]
    for idx in sorted(top_indices):
        if counts[idx] > 0:
            bar = "#" * min(int(counts[idx] / 20000), 50)
            print(f"  Z [{edges[idx]:+7.2f}, {edges[idx+1]:+7.2f}): {counts[idx]:>10,} {bar}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert 3D PCD/PLY to 2D Nav2 map with auto floor detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Auto floor detection (recommended):
  %(prog)s --in map.pcd --out-dir ./maps

  # Custom floor-relative offsets (positive = above detected floor):
  %(prog)s --in map.pcd --out-dir ./maps --min-z 0.1 --max-z 1.5

  # Absolute Z values (negative min_z triggers absolute mode):
  %(prog)s --in map.pcd --out-dir ./maps --min-z -0.5 --max-z 0.8

  # Just analyze the Z distribution:
  %(prog)s --in map.pcd --analyze-z
""",
    )
    parser.add_argument("--in", dest="in_path", required=True, help="Input .pcd or .ply file")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: next to input file)")
    parser.add_argument("--out-name", default="map", help="Output base name (default: map)")
    parser.add_argument("--min-z", type=float, default=0.05,
                        help="Min Z for obstacle map. Positive = floor-relative offset (default: 0.05)")
    parser.add_argument("--max-z", type=float, default=1.00,
                        help="Max Z for obstacle map. Positive = floor-relative offset (default: 1.00)")
    parser.add_argument("--resolution", type=float, default=0.05, help="Grid cell size in metres (default: 0.05)")
    parser.add_argument("--padding", type=float, default=0.50, help="Padding around map bounds in metres (default: 0.50)")
    parser.add_argument("--max-slope-deg", type=float, default=35.0, help="Max traversable slope in degrees (default: 35)")
    parser.add_argument("--max-step-m", type=float, default=0.25, help="Max step height in metres (default: 0.25)")
    parser.add_argument("--analyze-z", action="store_true", help="Print Z distribution analysis and exit")
    parser.add_argument("--obstacle-only", action="store_true", help="Only generate obstacle map (skip traversability/floor)")
    parser.add_argument("--copy-to", default=None,
                        help="Also copy output maps to this directory (e.g. Nav2 maps folder)")
    args = parser.parse_args()

    # ── Load ────────────────────────────────────────────────────────────
    print(f"[load] Reading {args.in_path}")
    points = load_xyz_points(args.in_path)
    print(f"[load] {points.shape[0]:,} points loaded")

    if args.analyze_z:
        analyze_z(points)
        return

    # ── Output directory ────────────────────────────────────────────────
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.in_path), "map_out")
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, args.out_name)

    # ── Show detected floor ─────────────────────────────────────────────
    preset = estimate_ground_preserving_preset(points)
    print(f"[floor] Auto-detected floor anchor: {preset['floor_anchor_z']:.3f} m")
    if args.min_z >= 0 and args.max_z >= 0:
        actual_min = preset['floor_anchor_z'] + args.min_z
        actual_max = preset['floor_anchor_z'] + args.max_z
        print(f"[floor] Floor-relative mode: [{args.min_z}, {args.max_z}] → absolute [{actual_min:.3f}, {actual_max:.3f}] m")
    else:
        print(f"[floor] Absolute mode: [{args.min_z:.3f}, {args.max_z:.3f}] m")

    # ── 1. Obstacle map ────────────────────────────────────────────────
    print(f"\n[step 1/3] Generating obstacle map...")
    pgm_path, yaml_path, occ_grid, yaml_data = export_occupancy_map(
        points,
        out_prefix=out_prefix,
        resolution=args.resolution,
        padding_m=args.padding,
        min_z=args.min_z,
        max_z=args.max_z,
    )
    occupied = int((occ_grid == 0).sum())
    free = int((occ_grid == 254).sum())
    print(f"[step 1/3] {pgm_path}  ({occ_grid.shape[1]}x{occ_grid.shape[0]}, occupied={occupied:,}, free={free:,})")
    print(f"[step 1/3] {yaml_path}")

    if args.obstacle_only:
        print(f"\n[done] Obstacle map saved to {out_dir}/")
        _maybe_copy(args.copy_to, out_prefix, [".pgm", ".yaml"])
        return

    # ── 2. Traversability sidecar ──────────────────────────────────────
    trav_prefix = out_prefix + "_traversable"
    print(f"\n[step 2/3] Generating traversability map...")
    # Derive terrain z bounds (may widen range to preserve floor)
    from core.RII_horizontal import derive_terrain_sidecar_bounds
    terrain_mz, terrain_xz, terrain_meta = derive_terrain_sidecar_bounds(points, args.min_z, args.max_z)
    z_mode = "absolute" if terrain_meta["source"] == "preset_cleanup" else "auto"
    if terrain_meta["source"] == "preset_cleanup":
        print(f"[step 2/3] Using wider terrain Z range [{terrain_mz:.2f}, {terrain_xz:.2f}] m (absolute, floor-preserving)")
    ref_yaml = yaml_data
    origin_xy = (float(ref_yaml["origin"][0]), float(ref_yaml["origin"][1]))
    shape = (occ_grid.shape[0], occ_grid.shape[1])
    trav_pgm, trav_yaml, _, _, trav_stats = export_traversability_map(
        points,
        out_prefix=trav_prefix,
        resolution=args.resolution,
        padding_m=args.padding,
        min_z=terrain_mz,
        max_z=terrain_xz,
        ground_percentile=10.0,
        max_slope_deg=args.max_slope_deg,
        max_step_m=args.max_step_m,
        max_roughness_m=9999.0,
        origin_xy=origin_xy,
        shape=shape,
        z_mode=z_mode,
    )
    print(f"[step 2/3] {trav_pgm}  (traversable={trav_stats['traversable_cells']:,})")

    # ── 3. Floor sidecar ──────────────────────────────────────────────
    floor_prefix = out_prefix + "_floor"
    print(f"\n[step 3/3] Generating floor map...")
    floor_pgm, floor_yaml, _, _, floor_stats = export_known_floor_map(
        points,
        out_prefix=floor_prefix,
        resolution=args.resolution,
        padding_m=args.padding,
        min_z=terrain_mz,
        max_z=terrain_xz,
        ground_percentile=10.0,
        max_slope_deg=args.max_slope_deg,
        max_step_m=args.max_step_m,
        max_roughness_m=9999.0,
        origin_xy=origin_xy,
        shape=shape,
        z_mode=z_mode,
    )
    print(f"[step 3/3] {floor_pgm}  (known_floor={floor_stats['known_cells']:,})")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Output directory: {out_dir}/")
    print(f"  Obstacle map:     {args.out_name}.pgm + .yaml")
    print(f"  Traversability:   {args.out_name}_traversable.pgm")
    print(f"  Floor:            {args.out_name}_floor.pgm")
    print(f"  Resolution:       {args.resolution} m/cell")
    print(f"  Grid size:        {occ_grid.shape[1]} x {occ_grid.shape[0]}")
    print(f"  Floor anchor:     {preset['floor_anchor_z']:.3f} m")
    print(f"{'='*60}")

    _maybe_copy(args.copy_to, out_prefix, [".pgm", ".yaml"])
    _maybe_copy(args.copy_to, trav_prefix, [".pgm", ".yaml"])
    _maybe_copy(args.copy_to, floor_prefix, [".pgm", ".yaml"])


def _maybe_copy(dest_dir: str | None, prefix: str, suffixes: list[str]) -> None:
    if not dest_dir:
        return
    import shutil
    os.makedirs(dest_dir, exist_ok=True)
    for suffix in suffixes:
        src = prefix + suffix
        if os.path.isfile(src):
            dst = os.path.join(dest_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            print(f"[copy] {dst}")


if __name__ == "__main__":
    main()
