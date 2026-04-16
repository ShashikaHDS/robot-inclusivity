#!/usr/bin/env python3
"""Cross-section ramp detection — CLI entry point.

Usage:
    python -m cross_section.run --pcd /path/to/map.pcd --max-slope 8
    python -m cross_section.run --pcd /path/to/map.pcd --max-slope 6 --max-step 0.15
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="Cross-section profile ramp detection",
    )
    parser.add_argument("--pcd", type=str, required=True, help="Path to PCD/PLY file")
    parser.add_argument("--max-slope", type=float, default=35.0, help="Robot max traversable slope (degrees)")
    parser.add_argument("--max-step", type=float, default=0.25, help="Robot max step height (meters)")
    parser.add_argument("--min-ramp-slope", type=float, default=4.0, help="Min slope to detect as ramp (degrees)")
    parser.add_argument("--min-ramp-length", type=float, default=1.0, help="Min ramp length (meters)")
    parser.add_argument("--strip-width", type=float, default=0.5, help="Cross-section strip width (meters)")
    parser.add_argument("--bin-size", type=float, default=0.20, help="Height profile bin size (meters)")
    parser.add_argument("--z-band", type=float, default=1.5, help="Height band above floor (meters)")
    parser.add_argument("--merge-radius", type=float, default=3.0, help="Cluster merge radius (meters)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if not os.path.isfile(args.pcd):
        print(f"Error: File not found: {args.pcd}")
        sys.exit(1)

    from .pcd_io import load_xyz_points
    from .detect import detect_ramps
    from .report import generate_report

    def log(msg, level="info"):
        if args.verbose or level in ("warn", "error"):
            print(msg)

    print(f"Loading: {args.pcd}")
    t0 = time.time()
    points = load_xyz_points(args.pcd)
    print(f"Loaded {points.shape[0]:,} points ({time.time() - t0:.1f}s)")
    print(f"Robot: max_slope={args.max_slope}°, max_step={args.max_step}m")
    print()

    t0 = time.time()
    result = detect_ramps(
        points,
        max_slope_deg=args.max_slope,
        max_step_m=args.max_step,
        strip_width=args.strip_width,
        bin_size=args.bin_size,
        min_ramp_slope_deg=args.min_ramp_slope,
        min_ramp_length_m=args.min_ramp_length,
        z_band=args.z_band,
        merge_radius_m=args.merge_radius,
        log=log,
    )
    elapsed = time.time() - t0
    print(f"\nDetection completed in {elapsed:.1f}s")
    print()

    report = generate_report(result, args.max_slope, args.max_step)
    print(report)


if __name__ == "__main__":
    main()
