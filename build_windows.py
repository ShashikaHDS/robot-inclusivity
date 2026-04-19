#!/usr/bin/env python3
"""
Build a standalone Windows executable for the RII Pipeline.

Prerequisites (run on Windows 10/11 with Python 3.10 or 3.11):
    pip install -r requirements.txt pyinstaller

Usage:
    python build_windows.py

Output:
    dist/RII_Pipeline/RII_Pipeline.exe   (folder-mode, self-contained)

The output folder can be zipped and distributed, OR wrapped in the
Inno Setup installer (installer.iss) to produce a proper setup.exe.
No Python installation is required on the target machine.
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_icon():
    """Ensure icon.ico exists — generate from icon.png via Pillow if needed."""
    ico = os.path.join(SCRIPT_DIR, "icon.ico")
    png = os.path.join(SCRIPT_DIR, "icon.png")
    if os.path.isfile(ico):
        return ico
    if not os.path.isfile(png):
        print("[icon] No icon.png found — building without a custom icon.")
        return None
    try:
        from PIL import Image
    except ImportError:
        print("[icon] Pillow not installed — can't generate icon.ico. Install with `pip install Pillow`.")
        return None
    print(f"[icon] Generating {ico} from {png}…")
    img = Image.open(png)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # Multi-resolution ICO — Windows picks the best size for each context
    img.save(ico, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return ico


def main():
    # Ensure PyInstaller is available
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("PyInstaller not found. Installing…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    icon_path = ensure_icon()

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
    # Bundle the icon inside the app so code that reads icon.png still works
    png = os.path.join(SCRIPT_DIR, "icon.png")
    if os.path.isfile(png):
        datas.append((png, "."))

    # Optional asset directories
    color_scale = os.path.join(SCRIPT_DIR, "color_scale")
    if os.path.isdir(color_scale):
        datas.append((color_scale, "color_scale"))
    map_dir = os.path.join(SCRIPT_DIR, "map")
    if os.path.isdir(map_dir):
        datas.append((map_dir, "map"))

    # Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "RII_Pipeline",
        "--noconfirm",
        "--windowed",              # No console window
        "--collect-all", "pyqtgraph",
    ]
    if icon_path:
        cmd.extend(["--icon", icon_path])

    # Hidden imports — things PyInstaller's static scan might miss
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
        "scipy",
        "scipy.ndimage",
    ]
    # Numba is optional
    try:
        import numba  # noqa: F401
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

    print("Building RII Pipeline standalone executable…")
    print(f"Command: {' '.join(cmd)}")
    print()
    subprocess.check_call(cmd, cwd=SCRIPT_DIR)

    dist_path = os.path.join(SCRIPT_DIR, "dist", "RII_Pipeline")
    print()
    print("=" * 60)
    print("  Build complete!")
    print(f"  Output folder: {dist_path}")
    print(f"  Launch:        {os.path.join(dist_path, 'RII_Pipeline.exe')}")
    print()
    print("  Distribute as:")
    print("    - a zip of the dist/RII_Pipeline folder, OR")
    print("    - a setup.exe built with Inno Setup from installer.iss")
    print("=" * 60)


if __name__ == "__main__":
    main()
