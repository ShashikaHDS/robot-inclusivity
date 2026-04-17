"""QThread workers — ShellW, ViewW, MapBuildW."""

import os
import sys
import signal
import subprocess
import shlex
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from config import SOURCE_CMD, PCD_PACKAGE_DIR, IS_WINDOWS
if PCD_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PCD_PACKAGE_DIR)
from pcd_package.pcd_tools import load_xyz_points

from core.RII_horizontal import derive_terrain_sidecar_bounds

try:
    import pyqtgraph.opengl as pgl
    PYQTGRAPH_GL_AVAILABLE = True
except Exception:
    PYQTGRAPH_GL_AVAILABLE = False


# ── Cross-platform subprocess helpers ────────────────────────────────────────

def _popen_kwargs():
    """Return platform-specific Popen kwargs for process group management."""
    if IS_WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"preexec_fn": os.setsid}


def _kill_proc(p):
    """Terminate a subprocess and its process group (cross-platform)."""
    if p.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            p.terminate()
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass


def _python_exe():
    """Return the Python executable name for subprocesses."""
    return sys.executable


class ShellW(QThread):
    log = pyqtSignal(str, str); done = pyqtSignal(bool, str)
    def __init__(s, cmd, label="CMD", source=True):
        super().__init__(); s.cmd = cmd; s.label = label; s.source = source; s._p = None
    def run(s):
        if IS_WINDOWS:
            full = s.cmd
        else:
            full = f"bash -c '{SOURCE_CMD} && {s.cmd}'" if s.source else f"bash -c '{s.cmd}'"
        s.log.emit(f"[{s.label}] $ {s.cmd[:200]}", "info")
        try:
            s._p = subprocess.Popen(full, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, **_popen_kwargs())
            for ln in iter(s._p.stdout.readline, ''):
                if ln.strip(): s.log.emit(f"  {ln.rstrip()}", "")
            s._p.wait()
            ok = s._p.returncode == 0
            s.log.emit(f"[{s.label}] {'Done' if ok else 'Failed'}", "success" if ok else "warn")
            s.done.emit(ok, "OK")
        except Exception as e:
            s.log.emit(f"[{s.label}] {e}", "warn"); s.done.emit(False, str(e))
    def cancel(s):
        if s._p:
            _kill_proc(s._p)

class ViewW(QThread):
    log = pyqtSignal(str, str); done = pyqtSignal(bool, str); loaded = pyqtSignal(object)
    def __init__(s, path, label="V"):
        super().__init__(); s.path = path; s.label = label; s._c = False
    def run(s):
        try:
            s.log.emit(f"[{s.label}] Loading point cloud into the UI...", "info")
            points = load_xyz_points(s.path)
            total = int(points.shape[0])
            max_points = 2_000_000 if PYQTGRAPH_GL_AVAILABLE else 250_000
            sampled = total > max_points
            if sampled:
                rng = np.random.default_rng(42)
                keep = rng.choice(total, size=max_points, replace=False)
                display_points = np.ascontiguousarray(points[keep], dtype=np.float32)
            else:
                display_points = np.ascontiguousarray(points, dtype=np.float32)
            if s._c:
                s.done.emit(False, "Cancelled")
                return
            s.loaded.emit({
                "points": display_points,
                "path": s.path,
                "label": s.label,
                "total_points": total,
                "display_points": int(display_points.shape[0]),
                "sampled": sampled,
            })
            suffix = f"{display_points.shape[0]:,}" + (" sampled " if sampled else " ") + "points"
            s.log.emit(f"[{s.label}] Loaded {total:,} points, showing {suffix}", "success")
            s.done.emit(True, "OK")
        except Exception as e:
            s.done.emit(False, str(e))
    def cancel(s):
        s._c = True

class MapBuildW(QThread):
    log = pyqtSignal(str, str); done = pyqtSignal(bool, str); prog = pyqtSignal(int)
    def __init__(s, pcd, sd, mz, xz, max_slope_deg=35.0, max_step_m=0.25, wait_seconds=12, v3_mode=False):
        super().__init__()
        s.pcd = pcd
        s.sd = sd
        s.mz = mz
        s.xz = xz
        s.max_slope_deg = max_slope_deg
        s.max_step_m = max_step_m
        s.wait_seconds = wait_seconds
        s.v3_mode = v3_mode
        s._ps = []
        s._c = False
    def _killall(s):
        for p, label in s._ps:
            _kill_proc(p)
            s.log.emit(f"[{label}] Stopped", "")
        s._ps.clear()

    def _run_subprocess(s, cmd, label):
        """Run a subprocess, stream output, handle cancellation. Returns success bool."""
        s.log.emit(f"[{label}] $ {' '.join(shlex.quote(str(part)) for part in cmd)}", "info")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            **_popen_kwargs(),
        )
        s._ps.append((proc, label))
        for ln in iter(proc.stdout.readline, ''):
            if s._c:
                _kill_proc(proc)
                s._killall()
                return False
            if ln.strip():
                s.log.emit(f"  {ln.rstrip()}", "")
        proc.wait(timeout=60)
        return proc.returncode == 0

    def run(s):
        try:
            os.makedirs(s.sd, exist_ok=True)
            out_prefix = os.path.join(s.sd, "map")
            trav_prefix = out_prefix + "_traversable"
            floor_prefix = out_prefix + "_floor"
            for prefix in (out_prefix, trav_prefix, floor_prefix):
                for suffix in (".pgm", ".yaml"):
                    path = prefix + suffix
                    if os.path.exists(path):
                        os.remove(path)

            terrain_mz = float(s.mz)
            terrain_xz = float(s.xz)
            terrain_z_absolute = False
            try:
                points = load_xyz_points(s.pcd)
                terrain_mz, terrain_xz, terrain_meta = derive_terrain_sidecar_bounds(points, s.mz, s.xz)
                if terrain_meta["source"] == "preset_cleanup":
                    terrain_z_absolute = True
                    s.log.emit(
                        "[Mask] Floor/traversability sidecars use the broader floor-preserving z range "
                        f"[{terrain_mz:.2f}, {terrain_xz:.2f}] m (absolute) "
                        f"(floor anchor≈{terrain_meta['floor_anchor_z']:.2f} m). "
                        f"The obstacle map stays on [{float(s.mz):.2f}, {float(s.xz):.2f}] m.",
                        "info",
                    )
                else:
                    s.log.emit(
                        "[Mask] Reusing the requested z range for floor/traversability sidecars: "
                        f"[{terrain_mz:.2f}, {terrain_xz:.2f}] m.",
                        "info",
                    )
            except Exception as exc:
                s.log.emit(
                    "[Mask] Terrain z-range auto-estimate failed; reusing the obstacle-map z range "
                    f"[{float(s.mz):.2f}, {float(s.xz):.2f}] m. {exc}",
                    "warn",
                )

            s.prog.emit(5)
            s.log.emit("[Map] Generating obstacle map plus traversability and floor sidecars from the selected point cloud...", "info")

            py = _python_exe()
            script = os.path.join(PCD_PACKAGE_DIR, "pcd_package", "pcd_to_occupancy_map.py")

            obstacle_cmd = [
                py, script,
                "--in", s.pcd,
                "--out-prefix", out_prefix,
                "--mode", "obstacle",
                "--resolution", "0.05",
                "--padding", "0.50",
                "--min_z", str(s.mz),
                "--max_z", str(s.xz),
            ]
            trav_cmd = [
                py, script,
                "--in", s.pcd,
                "--out-prefix", trav_prefix,
                "--mode", "traversability",
                "--align-pgm", out_prefix + ".pgm",
                "--align-yaml", out_prefix + ".yaml",
                "--min_z", str(terrain_mz),
                "--max_z", str(terrain_xz),
                "--max-slope-deg", str(s.max_slope_deg),
                "--max-step-m", str(s.max_step_m),
            ]
            if terrain_z_absolute:
                trav_cmd.append("--absolute-z")
            floor_cmd = [
                py, script,
                "--in", s.pcd,
                "--out-prefix", floor_prefix,
                "--mode", "floor",
                "--align-pgm", out_prefix + ".pgm",
                "--align-yaml", out_prefix + ".yaml",
                "--min_z", str(terrain_mz),
                "--max_z", str(terrain_xz),
                "--max-slope-deg", str(s.max_slope_deg),
                "--max-step-m", str(s.max_step_m),
            ]
            if terrain_z_absolute:
                floor_cmd.append("--absolute-z")

            # Step 1: Obstacle map
            s.prog.emit(20)
            if not s._run_subprocess(obstacle_cmd, "Map"):
                if s._c:
                    return
                s._killall()
                s.log.emit("Obstacle map files were not produced by the direct projection step.", "warn")
                s.done.emit(False, "2D map generation failed")
                return
            if not os.path.isfile(out_prefix + ".pgm") or not os.path.isfile(out_prefix + ".yaml"):
                s._killall()
                s.log.emit("Obstacle map files were not produced.", "warn")
                s.done.emit(False, "2D map generation failed")
                return

            if s.v3_mode:
                # V3: obstacle map only — no traversability or floor sidecars
                s.prog.emit(95)
                if os.path.isfile(out_prefix + ".pgm") and os.path.isfile(out_prefix + ".yaml"):
                    s.log.emit(f"Saved: {out_prefix}.pgm", "success")
                    s.log.emit(f"Saved: {out_prefix}.yaml", "success")
                    s.log.emit("[V3] Traversability/floor sidecars skipped — obstacle map is the base.", "info")
                    s._killall()
                    s.prog.emit(100)
                    s.done.emit(True, out_prefix)
                else:
                    s._killall()
                    s.log.emit("Obstacle map files were not produced.", "warn")
                    s.done.emit(False, "2D map generation failed")
            else:
                # V1/V2: generate traversability + floor sidecars
                # Step 2: Traversability sidecar
                s.prog.emit(70)
                if not s._run_subprocess(trav_cmd, "Mask"):
                    if s._c:
                        return
                    s._killall()
                    s.log.emit("Traversability sidecar files were not produced.", "warn")
                    s.done.emit(False, "2D map generation failed")
                    return
                if not os.path.isfile(trav_prefix + ".pgm"):
                    s._killall()
                    s.log.emit("Traversability sidecar files were not produced.", "warn")
                    s.done.emit(False, "2D map generation failed")
                    return

                # Step 3: Floor sidecar
                s.prog.emit(90)
                s._run_subprocess(floor_cmd, "Floor")
                s.prog.emit(95)

                if (
                    os.path.isfile(out_prefix + ".pgm") and
                    os.path.isfile(out_prefix + ".yaml") and
                    os.path.isfile(trav_prefix + ".pgm") and
                    os.path.isfile(floor_prefix + ".pgm")
                ):
                    s.log.emit(f"Saved: {out_prefix}.pgm", "success")
                    s.log.emit(f"Saved: {out_prefix}.yaml", "success")
                    s.log.emit(f"Saved: {trav_prefix}.pgm", "success")
                    s.log.emit(f"Saved: {floor_prefix}.pgm", "success")
                    s._killall()
                    s.prog.emit(100)
                    s.done.emit(True, out_prefix)
                else:
                    s._killall()
                    s.log.emit("Map, traversability sidecar, or floor sidecar files were not produced.", "warn")
                    s.done.emit(False, "2D map generation failed")
        except Exception as e:
            s._killall()
            s.log.emit(f"[Map] {e}", "warn")
            s.done.emit(False, str(e))
    def cancel(s):
        s._c = True
        s._killall()


class GroundAnalysisW(QThread):
    """Worker thread for RANSAC ground segmentation + transition detection."""
    log = pyqtSignal(str, str)
    done = pyqtSignal(bool, str)
    result_ready = pyqtSignal(object)  # emits GroundAnalysisResult
    prog = pyqtSignal(int)

    def __init__(s, pcd_path, max_slope_deg=35.0, max_step_m=0.25):
        super().__init__()
        s.pcd_path = pcd_path
        s.max_slope_deg = max_slope_deg
        s.max_step_m = max_step_m
        s._c = False

    def run(s):
        try:
            from core.ground_analysis import run_ground_analysis, generate_accessibility_report

            s.log.emit("[Ground] Loading point cloud...", "info")
            s.prog.emit(10)
            points = load_xyz_points(s.pcd_path)
            if s._c:
                s.done.emit(False, "Cancelled"); return
            s.log.emit(f"[Ground] Loaded {points.shape[0]:,} points", "info")
            s.prog.emit(30)

            def log_fn(msg, lvl="info"):
                s.log.emit(msg, lvl)

            result = run_ground_analysis(
                points,
                max_slope_deg=s.max_slope_deg,
                max_step_m=s.max_step_m,
                log=log_fn,
            )
            if s._c:
                s.done.emit(False, "Cancelled"); return
            s.prog.emit(90)

            report = generate_accessibility_report(result)
            s.log.emit(report, "info")
            s.prog.emit(100)

            s.result_ready.emit(result)
            s.done.emit(True, "OK")
        except Exception as e:
            s.log.emit(f"[Ground] Error: {e}", "warn")
            s.done.emit(False, str(e))

    def cancel(s):
        s._c = True
