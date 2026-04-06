#!/usr/bin/env python3
"""
Build a standalone Windows executable for the RII Pipeline.

Prerequisites (run on Windows):
    pip install pyinstaller PyQt5 numpy Pillow pyqtgraph PyOpenGL open3d scipy numba

Usage:
    python build_windows.py

Output:
    dist/RII_Pipeline/RII_Pipeline.exe   (folder mode — self-contained)

The output folder can be zipped and distributed to any Windows PC.
No Python installation required on the target machine.
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    # Ensure PyInstaller is available
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Build paths
    entry = os.path.join(SCRIPT_DIR, "rii_pipeline.py")
    core_dir = os.path.join(SCRIPT_DIR, "core")
    gui_dir = os.path.join(SCRIPT_DIR, "gui")
    src_dir = os.path.join(SCRIPT_DIR, "src")
    config_file = os.path.join(SCRIPT_DIR, "config.py")
    convert_file = os.path.join(SCRIPT_DIR, "convert_map.py")

    # Collect data directories
    datas = [
        (core_dir, "core"),
        (gui_dir, "gui"),
        (src_dir, "src"),
        (config_file, "."),
        (convert_file, "."),
    ]

    # Color scale directory (if exists)
    color_scale = os.path.join(SCRIPT_DIR, "color_scale")
    if os.path.isdir(color_scale):
        datas.append((color_scale, "color_scale"))

    # Map directory (if exists)
    map_dir = os.path.join(SCRIPT_DIR, "map")
    if os.path.isdir(map_dir):
        datas.append((map_dir, "map"))

    # Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "RII_Pipeline",
        "--noconfirm",
        "--windowed",              # No console window
        "--collect-all", "open3d",
        "--collect-all", "pyqtgraph",
    ]

    # Add hidden imports
    hidden = [
        "numpy",
        "PIL",
        "PyQt5",
        "PyQt5.QtWidgets",
        "PyQt5.QtCore",
        "PyQt5.QtGui",
        "pyqtgraph",
        "pyqtgraph.opengl",
        "OpenGL",
        "open3d",
        "scipy",
        "scipy.ndimage",
    ]
    # Numba is optional
    try:
        import numba
        hidden.append("numba")
        cmd.extend(["--collect-all", "numba"])
    except ImportError:
        pass

    for h in hidden:
        cmd.extend(["--hidden-import", h])

    # Add data files
    sep = ";" if sys.platform == "win32" else ":"
    for src, dst in datas:
        cmd.extend(["--add-data", f"{src}{sep}{dst}"])

    # Add paths so imports resolve
    cmd.extend(["--paths", SCRIPT_DIR])
    cmd.extend(["--paths", os.path.join(SCRIPT_DIR, "src", "pcd_package")])

    # Entry point
    cmd.append(entry)

    print("Building RII Pipeline standalone executable...")
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.check_call(cmd, cwd=SCRIPT_DIR)

    dist_path = os.path.join(SCRIPT_DIR, "dist", "RII_Pipeline")
    print()
    print("=" * 60)
    print(f"  Build complete!")
    print(f"  Output: {dist_path}")
    print(f"  Run:    {os.path.join(dist_path, 'RII_Pipeline.exe')}")
    print()
    print("  To distribute: zip the dist/RII_Pipeline folder.")
    print("  No Python needed on the target Windows PC.")
    print("=" * 60)


if __name__ == "__main__":
    main()
