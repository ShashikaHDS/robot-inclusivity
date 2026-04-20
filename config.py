"""Configuration constants for the RII Pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_VERSION = "2.0"
GITHUB_REPO = "ShashikaHDS/Teal-Robot"  # owner/repo — used by the updater
UPDATE_ASSET_PREFIX = "RII_Pipeline_Setup"  # matches the Inno Setup output name

IS_WINDOWS = sys.platform == "win32"

WORKSPACE = str(Path(__file__).resolve().parent)
MAP_IN_DIR = os.path.join(WORKSPACE, "src", "pcd_package", "preclean", "map_in")
DEFAULT_PCD_OUT = os.path.join(WORKSPACE, "src", "pcd_package", "preclean", "map_out")
DEFAULT_MAP_SAVE = os.path.join(WORKSPACE, "src", "pcd_package", "final_2d_represntation")
PRECLEAN_DIR = os.path.join(WORKSPACE, "src", "pcd_package", "preclean")
PCD_PACKAGE_DIR = os.path.join(WORKSPACE, "src", "pcd_package")


def detect_default_point_cloud() -> str:
    """Return a likely default raw point-cloud path inside this workspace."""
    candidates = (
        Path(MAP_IN_DIR) / "GlobalMap.pcd",
        Path(MAP_IN_DIR) / "GlobalMap.ply",
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    return MAP_IN_DIR


DEFAULT_PCD_IN = detect_default_point_cloud()


# ── ROS detection (optional — not required on Windows) ───────────────────────

ROS_AVAILABLE = False
ROS_DISTRO = ""
ROS_SETUP = ""
SOURCE_CMD = ""

def _detect_ros() -> None:
    global ROS_AVAILABLE, ROS_DISTRO, ROS_SETUP, SOURCE_CMD
    if IS_WINDOWS:
        return
    candidates = []
    env_distro = os.environ.get("ROS_DISTRO")
    if env_distro:
        candidates.append(env_distro)
    candidates.extend(["jazzy", "humble", "iron", "rolling"])
    for distro in candidates:
        if distro and os.path.isfile(f"/opt/ros/{distro}/setup.bash"):
            ROS_AVAILABLE = True
            ROS_DISTRO = distro
            ROS_SETUP = f"/opt/ros/{distro}/setup.bash"
            install_setup = os.path.join(WORKSPACE, "install", "setup.bash")
            SOURCE_CMD = f"source {ROS_SETUP}" + (f" && source {install_setup}" if os.path.isfile(install_setup) else "")
            return

_detect_ros()
