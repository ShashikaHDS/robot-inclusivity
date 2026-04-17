#!/usr/bin/env python3
"""
RII Pipeline v3.0 — Single Map Pipeline.

V3 eliminates the separate traversability map. The obstacle map IS the
traversability map. Ramps and step-climbable features are projected
directly onto it.

Key differences from v1/v2:
  - No traversability sidecar (map_traversable.pgm) generated
  - min_z auto-linked to robot's max_step (steps robot can climb are ignored)
  - Ramp detection (hybrid: slope grid + cross-section) overlaid on obstacle map
  - Manual slope marking supported
  - Coverage runs directly on obstacle map + ramp blocking

Usage:
    python rii_pipeline_v3.py          # Launch GUI
    python rii_pipeline_v3.py --test   # Run ramp detection test
"""
import sys
import os
import signal
import argparse

# Set v3 mode flag BEFORE importing GUI modules
os.environ["RII_PIPELINE_VERSION"] = "3"


def _run_gui():
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    from gui.main_window import MainWin

    sigint_state = {"pending": False}

    def _handle_sigint(signum, frame):
        sigint_state["pending"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWin()
    w.setWindowTitle("Robot Inclusivity Index (RII) — v3.0")
    w.show()
    sig_timer = QTimer()
    sig_timer.timeout.connect(lambda: w.close() if sigint_state.pop("pending", False) else None)
    sig_timer.start(100)
    app._sigint_timer = sig_timer
    sys.exit(app.exec_())


def _run_test(pcd_path: str, max_slope: float, max_step: float):
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "pcd_package", "pcd_package"))
    from pcd_tools import load_xyz_points
    from core.ground_analysis import run_ground_analysis, generate_accessibility_report

    print(f"Loading: {pcd_path}")
    points = load_xyz_points(pcd_path)
    print(f"Loaded {points.shape[0]:,} points")
    print(f"Robot: max_slope={max_slope}°, max_step={max_step}m")
    print(f"min_z will be set to {max_step}m (steps ≤ {max_step}m ignored)")
    print()

    def log(msg, lvl="info"):
        print(msg)

    result = run_ground_analysis(points, max_slope_deg=max_slope, max_step_m=max_step, log=log)
    print()
    print(generate_accessibility_report(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RII Pipeline v3.0 — Single Map Pipeline")
    parser.add_argument("--test", action="store_true", help="Run ramp detection test (no GUI)")
    parser.add_argument("--pcd", type=str, default=None, help="PCD file for --test mode")
    parser.add_argument("--max-slope", type=float, default=35.0, help="Max slope (degrees)")
    parser.add_argument("--max-step", type=float, default=0.25, help="Max step height (meters)")
    args = parser.parse_args()

    if args.test:
        pcd = args.pcd
        if pcd is None:
            candidates = [
                "/media/teal/ssd1tb/lio_sam_maps/130426/ver7/GlobalMap.pcd",
            ]
            for c in candidates:
                if os.path.isfile(c):
                    pcd = c
                    break
            if pcd is None:
                print("Error: No PCD file found. Use --pcd <path>")
                sys.exit(1)
        _run_test(pcd, args.max_slope, args.max_step)
    else:
        _run_gui()
