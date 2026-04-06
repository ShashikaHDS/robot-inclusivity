#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Robot Inclusivity Index (RII) — Launch Script
#
# Usage:
#   ./launch.sh                          # Launch the GUI
#   ./launch.sh convert --in map.pcd     # Convert 3D PCD → 2D Nav2 map (auto floor detection)
#   ./launch.sh convert --help           # Show all conversion options
#   ./launch.sh analyze --in map.pcd     # Analyze Z distribution only
# ──────────────────────────────────────────────────────────────────────
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source ROS if available (needed for shell workers)
for distro in "${ROS_DISTRO:-}" jazzy humble iron rolling; do
    if [ -n "${distro:-}" ] && [ -f "/opt/ros/${distro}/setup.bash" ]; then
        source "/opt/ros/${distro}/setup.bash"
        break
    fi
done

cd "${SCRIPT_DIR}"

case "${1:-}" in
    convert)
        shift
        exec python3 convert_map.py "$@"
        ;;
    analyze)
        shift
        exec python3 convert_map.py --analyze-z "$@"
        ;;
    *)
        exec python3 rii_pipeline.py "$@"
        ;;
esac
