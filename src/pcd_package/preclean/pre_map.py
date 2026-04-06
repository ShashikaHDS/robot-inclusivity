#!/usr/bin/env python3
"""Safe preclean for large PCD/PLY point clouds without Open3D."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcd_package.pcd_tools import load_xyz_points, preclean_point_cloud, write_xyz_pcd


def main() -> None:
    parser = argparse.ArgumentParser(description="Preclean a PCD or PLY file for 2D map generation.")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--min_z", type=float, default=0.01, help="Slice bottom in metres")
    parser.add_argument("--max_z", type=float, default=0.80, help="Slice top in metres")
    parser.add_argument("--voxel", type=float, default=0.05, help="Voxel leaf size in metres")
    parser.add_argument("--sor_k", type=int, default=50, help="Retained for CLI compatibility")
    parser.add_argument("--sor_std", type=float, default=1.2, help="Approximate density std-ratio filter")
    parser.add_argument("--ror_radius", type=float, default=0.15, help="Density neighbourhood radius in metres")
    parser.add_argument("--ror_min", type=int, default=5, help="Minimum approximate neighbours")
    args = parser.parse_args()

    print("[step] load_xyz_points")
    points = load_xyz_points(args.in_path)
    print(f"[preclean] input points: {points.shape[0]:,}")

    cleaned = preclean_point_cloud(
        points,
        min_z=args.min_z,
        max_z=args.max_z,
        voxel_size=args.voxel,
        sor_k=args.sor_k,
        sor_std=args.sor_std,
        ror_radius=args.ror_radius,
        ror_min=args.ror_min,
        log_fn=print,
    )

    kept_pct = 100.0 * cleaned.shape[0] / max(points.shape[0], 1)
    print(f"[preclean] output points: {cleaned.shape[0]:,}  (kept {kept_pct:.2f}%)")
    print("[step] write_xyz_pcd")
    write_xyz_pcd(args.out_path, cleaned)
    print(f"[preclean] wrote {args.out_path}")
    print("[done]")


if __name__ == "__main__":
    main()
