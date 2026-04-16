#!/usr/bin/env python3
"""
RII Pipeline v2.0 — entry point.

v2.0 uses RANSAC-based ground segmentation with ramp/step detection
instead of the noisy per-cell traversability map.

Requires: pip install PyQt5 numpy Pillow pyqtgraph PyOpenGL --break-system-packages

Usage:
    python rii_pipeline_v2.py          # Launch GUI (same as v1)
    python rii_pipeline_v2.py --test   # Run ground analysis on default PCD and print report
"""
import sys
import signal
import argparse


def _run_gui():
    """Launch the GUI application."""
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
    w.setWindowTitle("Robot Inclusivity Index (RII) — v2.0")
    w.show()
    sig_timer = QTimer()
    sig_timer.timeout.connect(lambda: w.close() if sigint_state.pop("pending", False) else None)
    sig_timer.start(100)
    app._sigint_timer = sig_timer
    sys.exit(app.exec_())


def _run_test(pcd_path: str, max_slope: float, max_step: float):
    """Run ground analysis on a PCD file and print the report."""
    import numpy as np
    from src.pcd_package.pcd_package.pcd_tools import load_xyz_points
    from core.ground_analysis import run_ground_analysis, generate_accessibility_report

    print(f"Loading point cloud: {pcd_path}")
    points = load_xyz_points(pcd_path)
    print(f"Loaded {points.shape[0]:,} points")
    print(f"Robot capabilities: max_slope={max_slope} deg, max_step={max_step}m")
    print()

    def log(msg, lvl="info"):
        print(msg)

    result = run_ground_analysis(
        points,
        max_slope_deg=max_slope,
        max_step_m=max_step,
        log=log,
    )

    print()
    print("=" * 60)
    print("ACCESSIBILITY REPORT")
    print("=" * 60)
    print(generate_accessibility_report(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RII Pipeline v2.0 — RANSAC ground segmentation with ramp/step detection",
    )
    parser.add_argument("--test", action="store_true", help="Run ground analysis test (no GUI)")
    parser.add_argument("--pcd", type=str, default=None, help="Path to PCD file for --test mode")
    parser.add_argument("--max-slope", type=float, default=35.0, help="Max traversable slope (degrees)")
    parser.add_argument("--max-step", type=float, default=0.25, help="Max climbable step height (meters)")
    args = parser.parse_args()

    if args.test:
        pcd = args.pcd
        if pcd is None:
            # Try default path
            import os
            candidates = [
                "/media/teal/ssd1tb/lio_sam_maps/130426/ver7/GlobalMap.pcd",
                os.path.expanduser("~/GlobalMap.pcd"),
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
