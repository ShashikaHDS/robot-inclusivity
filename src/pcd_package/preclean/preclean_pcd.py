#!/usr/bin/env python3
"""Legacy-compatible wrapper for the lightweight PCD/PLY preclean pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcd_package.pcd_tools import load_xyz_points, preclean_point_cloud, write_xyz_pcd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--min_z", type=float, default=0.01, help="Slice bottom (m)")
    parser.add_argument("--max_z", type=float, default=0.80, help="Slice top (m)")
    parser.add_argument("--voxel", type=float, default=0.05, help="Voxel leaf (m)")
    parser.add_argument("--sor_k", type=int, default=50, help="Retained for CLI compatibility")
    parser.add_argument("--sor_std", type=float, default=1.2, help="Approximate density std-ratio")
    parser.add_argument("--ror_radius", type=float, default=0.15, help="Density radius (m)")
    parser.add_argument("--ror_min", type=int, default=5, help="Approximate minimum neighbours")
    args = parser.parse_args()

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
    print(
        f"[preclean] output points: {cleaned.shape[0]:,}  "
        f"(kept {100.0 * cleaned.shape[0] / max(points.shape[0], 1):.1f}%)"
    )
    if cleaned.shape[0] == 0:
        print("[WARN] all points removed; relax parameters.")
    write_xyz_pcd(args.out_path, cleaned)
    print(f"[preclean] wrote {args.out_path}")


if __name__ == "__main__":
    main()
