#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# RII Pipeline — Bootstrap Script
# Detects platform (x86_64 / aarch64) and installs all dependencies.
# Usage:  ./bootstrap.sh
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$(uname -m)"
PIP_FLAGS="--break-system-packages"

# ── Colors ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Detect platform ─────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  RII Pipeline — Bootstrap"
echo "============================================"
echo ""

info "Architecture: ${ARCH}"
info "Python:       $(python3 --version 2>&1)"

case "${ARCH}" in
    x86_64)
        info "Platform:     Desktop (x86_64)"
        ;;
    aarch64)
        info "Platform:     Jetson / ARM (aarch64)"
        # Check for JetPack
        if [ -f /etc/nv_tegra_release ]; then
            JETPACK_INFO=$(cat /etc/nv_tegra_release | head -1)
            info "Tegra:        ${JETPACK_INFO}"
        fi
        ;;
    *)
        warn "Unknown architecture: ${ARCH} — will attempt standard pip install"
        ;;
esac

echo ""

# ── Check pip ────────────────────────────────────────────────────────
if ! python3 -m pip --version &>/dev/null; then
    info "pip not found, installing..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip
fi

# ── Helper: install a pip package ────────────────────────────────────
pip_install() {
    local pkg="$1"
    local desc="${2:-$pkg}"
    if python3 -c "import ${pkg%%[>=<]*}" &>/dev/null 2>&1; then
        ok "${desc} — already installed"
    else
        info "Installing ${desc}..."
        if python3 -m pip install "${pkg}" ${PIP_FLAGS} 2>&1 | tail -1; then
            ok "${desc} — installed"
        else
            warn "${desc} — install failed (non-critical if optional)"
            return 1
        fi
    fi
    return 0
}

# ── Core dependencies ────────────────────────────────────────────────
info "Installing core dependencies..."
echo ""

pip_install "numpy"       "numpy (array operations)"
pip_install "PyQt5"       "PyQt5 (GUI framework)"
pip_install "Pillow"      "Pillow (image processing)"
pip_install "pyqtgraph"   "pyqtgraph (plotting / 3D viewer)"
pip_install "PyOpenGL"    "PyOpenGL (3D rendering)"

echo ""

# ── Performance dependencies ─────────────────────────────────────────
info "Installing performance optimizations..."
echo ""

# scipy
pip_install "scipy" "scipy (fast connected-component labeling)" || true

# numba — the main speedup for raycasting
# On aarch64, llvmlite may need system LLVM
if [ "${ARCH}" = "aarch64" ]; then
    if ! python3 -c "import numba" &>/dev/null 2>&1; then
        info "Installing llvmlite build dependencies for ARM..."
        sudo apt-get install -y -qq llvm-dev libllvm14 2>/dev/null || \
        sudo apt-get install -y -qq llvm-dev 2>/dev/null || true
    fi
fi

pip_install "numba" "numba (JIT-compiled parallel raycasting)" || {
    warn "numba install failed — RII Vertical will use pure-Python fallback (slower)"
    warn "On Jetson, try: sudo apt-get install -y llvm-dev && pip install numba ${PIP_FLAGS}"
}

echo ""

# ── Verify ───────────────────────────────────────────────────────────
info "Verifying installation..."
echo ""

python3 -c "
import sys
ok = True

def check(name, import_name=None):
    global ok
    import_name = import_name or name
    try:
        __import__(import_name)
        print(f'  \033[0;32m[OK]\033[0m    {name}')
    except Exception as e:
        print(f'  \033[1;33m[MISS]\033[0m  {name} — {e}')
        ok = False

print('Core:')
check('numpy')
check('PyQt5', 'PyQt5.QtWidgets')
check('Pillow', 'PIL')
check('pyqtgraph')
check('PyOpenGL', 'OpenGL')

print()
print('Performance:')
check('numba')
check('scipy')

print()
if ok:
    print('\033[0;32mAll dependencies installed successfully.\033[0m')
else:
    print('\033[1;33mSome optional dependencies missing — pipeline will still work.\033[0m')
"

echo ""
ok "Bootstrap complete!"
echo ""
info "Run the pipeline with:"
echo "    cd ${SCRIPT_DIR}"
echo "    python3 rii_pipeline.py"
echo ""
