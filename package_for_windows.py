#!/usr/bin/env python3
"""Assemble a self-contained `windows_build_kit/` folder.

Run on Linux (or Windows). Produces a folder you can copy to any Windows 11 PC
and run the included BUILD.bat to get an installer .exe.

    python3 package_for_windows.py

Output folder: windows_build_kit/
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KIT = ROOT / "windows_build_kit"
ZIP_PATH = ROOT / "windows_build_kit.zip"

# Files to copy verbatim into the kit (relative to ROOT)
FILES_TO_COPY = [
    "rii_pipeline.py",
    "rii_pipeline_v2.py",
    "rii_pipeline_v3.py",
    "config.py",
    "convert_map.py",
    "build_windows.py",
    "installer.iss",
    "requirements.txt",
    "icon.png",
    "README_WINDOWS.md",
    "README.md",
    # Windows-specific helpers (live at repo root so `git clone` on
    # Windows gives you BUILD.bat directly; they also go into the kit)
    "BUILD.bat",
    "INSTALL_DEPS.bat",
    "RUN_DEV.bat",
    "README_FIRST.txt",
]

# Directories to copy (relative to ROOT). Python cache and editor junk are
# skipped by the ignore filter below.
DIRS_TO_COPY = [
    "core",
    "gui",
    "cross_section",
    "src",
    "color_scale",
    "map",
]


_IGNORE_NAMES = {
    "__pycache__", ".DS_Store", ".pytest_cache", ".mypy_cache",
    ".vscode", ".idea", ".git", "node_modules", "build", "dist",
    "Output", "windows_build_kit",
}
_IGNORE_SUFFIXES = (".pyc", ".pyo", ".db", ".log", ".tmp")


def _ignore(_path, names):
    """shutil.copytree ignore filter — skip caches, editor junk, and large
    generated artifacts that shouldn't travel with the source kit."""
    return [
        n for n in names
        if n in _IGNORE_NAMES
        or n.endswith(_IGNORE_SUFFIXES)
    ]


def copy_tree(name: str) -> bool:
    src = ROOT / name
    if not src.exists():
        print(f"  - skip   {name} (not found)")
        return False
    dst = KIT / name
    if dst.exists():
        shutil.rmtree(dst)
    if src.is_dir():
        shutil.copytree(src, dst, ignore=_ignore)
    else:
        shutil.copy2(src, dst)
    print(f"  + copied {name}")
    return True




def make_zip() -> Path:
    """Create windows_build_kit.zip from the kit folder."""
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    print(f"Creating zip at: {ZIP_PATH}")
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, dirs, files in os.walk(KIT):
            dirs[:] = [d for d in dirs if d not in _IGNORE_NAMES]
            for f in files:
                if f.endswith(_IGNORE_SUFFIXES):
                    continue
                full = Path(root) / f
                arc = full.relative_to(ROOT)
                zf.write(full, arc)
    return ZIP_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-zip", action="store_true",
                        help="skip the zip step, only create the folder")
    parser.add_argument("--zip-only", action="store_true",
                        help="skip folder creation, only rebuild the zip from an existing kit")
    args = parser.parse_args()

    if args.zip_only:
        if not KIT.exists():
            print(f"[ERROR] {KIT} does not exist. Run without --zip-only first.")
            return 1
        z = make_zip()
        print(f"\nZip ready: {z}  ({z.stat().st_size / 1024:.0f} KB)")
        return 0

    if KIT.exists():
        print(f"Removing existing kit at {KIT}")
        shutil.rmtree(KIT)
    KIT.mkdir(parents=True)
    print(f"Creating Windows build kit at: {KIT}\n")

    print("Copying files…")
    for name in FILES_TO_COPY:
        copy_tree(name)
    print()

    print("Copying directories…")
    for name in DIRS_TO_COPY:
        copy_tree(name)
    print()

    # Compute total size
    total = 0
    files = 0
    for root, _dirs, filenames in os.walk(KIT):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(root, f))
                files += 1
            except OSError:
                pass
    zip_info = ""
    if not args.no_zip:
        print()
        z = make_zip()
        zip_kb = z.stat().st_size / 1024
        zip_info = f"  Kit zip:   {z}  ({zip_kb:.0f} KB)\n"

    print()
    print("=" * 58)
    print(f"  Kit ready: {KIT}  ({files} files, {total / (1024 * 1024):.1f} MB)")
    if zip_info:
        print(zip_info, end="")
    print()
    print("  Next steps:")
    print(f"    1. Copy {'the zip' if zip_info else f'the {KIT.name} folder'} to a Windows 11 PC.")
    print("    2. Unzip (if needed) and read README_FIRST.txt.")
    print("    3. Double-click BUILD.bat.")
    print("=" * 58)
    return 0


if __name__ == "__main__":
    sys.exit(main())
