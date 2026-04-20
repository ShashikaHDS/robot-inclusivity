"""Main application window — MainWin."""

from __future__ import annotations

import os
import sys
import math
import time
import shlex
import signal
import shutil
import tempfile
import threading

import numpy as np

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QApplication,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QSplitter, QStackedWidget, QTabBar,
    QGroupBox, QFrame, QLabel,
    QPushButton, QLineEdit, QComboBox,
    QDoubleSpinBox, QSpinBox,
    QProgressBar, QTextEdit,
    QCheckBox, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QSettings
from PyQt5.QtGui import QImage, QColor, QIcon, QKeySequence
from PyQt5.QtWidgets import QAction

from config import (
    DEFAULT_PCD_IN as DEF_IN,
    PCD_PACKAGE_DIR,
    PRECLEAN_DIR as PRECLEAN,
    WORKSPACE,
)

if PCD_PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PCD_PACKAGE_DIR)

from pcd_package.pcd_tools import (
    estimate_ground_preserving_preset,
    load_xyz_points,
)

from core.map_io import (
    parse_pgm, parse_yaml,
    resolve_point_cloud_path,
    filtered_point_cloud_filename,
    filtered_point_cloud_stem_candidates,
    rewrite_nav2_yaml_image,
    traversability_sidecar_path,
    floor_sidecar_path,
)
from core.semantic_selection import (
    selection_kind, selection_bounds_px,
    selection_center_px, selection_mask_from_display,
    selection_to_world_bounds,
)
from core.RII_horizontal import (
    BLOCKED_MAP_VIEW, PRIMARY_SELECTION_VIEW,
    PLANNER_NAMES,
    run_coverage,
)
from core.rendering import (
    render_coverage_fast, render_compare_fast,
    render_stc_path_fast, make_info_image,
    render_bottleneck_overlay,
    render_optimization_overlay,
)
from core.semantic_analysis import (
    SEMANTIC_LABEL_NAMES, SEMANTIC_FIXATION_GROUPS,
    SEMANTIC_3D_COLORS, SEMANTIC_REMOVABLE_FIXATIONS,
    load_semantic_pcd,
    project_labels_to_2d_grid, analyze_semantic_rii,
    compute_semantic_layered_rii,
    simulate_removed_fixations,
    identify_semantic_removal_candidates,
    simulate_removed_candidates,
    render_semantic_candidates,
    score_bottleneck_candidates,
    find_relocation_zones,
    classify_candidate_actions,
    optimize_multi_object_relocation,
)
from core.RII_vertical import (
    compute_rii_vertical, compute_combined_rii,
    identify_wall_segments, colorize_cloud_with_walls,
)
from gui.workers import ShellW, ViewW, MapBuildW, GroundAnalysisW
from gui.widgets import (
    MapW, DragScrollArea,
    PointCloudPreviewW, PointCloudW,
    PYQTGRAPH_GL_AVAILABLE,
)
from gui.step_indicator import StepIndicator
from gui.collapsible import CollapsibleSection


class MainWin(QMainWindow):
    ui_log_sig = pyqtSignal(str, str)
    ref_result_sig = pyqtSignal(object, str)
    act_result_sig = pyqtSignal(object, str)
    ref_error_sig = pyqtSignal(str)
    act_error_sig = pyqtSignal(str)
    sem_loaded_sig = pyqtSignal(int, object, object, object)
    sem_load_error_sig = pyqtSignal(int, str)
    sem_result_sig = pyqtSignal(int, object, object, object)
    sem_error_sig = pyqtSignal(int, str)
    sem_improved_sig = pyqtSignal(int, object)
    sem_improved_error_sig = pyqtSignal(int, str)
    bottleneck_done_sig = pyqtSignal()
    bottleneck_error_sig = pyqtSignal(str)
    optimization_done_sig = pyqtSignal(object)
    optimization_error_sig = pyqtSignal(str)
    sem_progress_sig = pyqtSignal(int, int, str)
    rv_result_sig = pyqtSignal(object)
    rv_error_sig = pyqtSignal(str)
    rv_progress_sig = pyqtSignal(int, str)

    def __init__(s):
        super().__init__()
        s._v3_mode = os.environ.get("RII_PIPELINE_VERSION") == "3"
        s.setWindowTitle("Robot Inclusivity Index (RII)")
        _icon_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "icon.png")
        if os.path.isfile(_icon_path):
            s.setWindowIcon(QIcon(_icon_path))
        s.setMinimumSize(1300, 800)
        default_in = resolve_point_cloud_path(DEF_IN, ["GlobalMap"])
        s._wk = []
        s._cache_root = None
        s._init_session_cache()
        s.pcd_in = default_in if os.path.isfile(default_in) else ""
        s.ref_r = s.act_r = None; s._imgs = {}; s._clouds = {}; s._map_w = s._map_h = 0
        s._loaded_map_path = None
        s._pgm_pixels = None  # raw PGM pixels for selection mask
        s._sem_pts = None; s._sem_labels = None; s._label_grid = None
        s._sem_analysis = None
        s._sem_candidates = []
        s._sem_improved = None
        s._sem_layered_result = None
        s._sem_focused_candidate_id = None
        s._sem_session_token = 0
        s._sem_load_active = False
        s._sem_analysis_active = False
        s._build(); s._theme()
        s.ui_log_sig.connect(s._log)
        s.ref_result_sig.connect(s._ref_done)
        s.act_result_sig.connect(s._act_done)
        s.ref_error_sig.connect(s._ref_failed)
        s.act_error_sig.connect(s._act_failed)
        s.sem_loaded_sig.connect(s._sem_loaded)
        s.sem_load_error_sig.connect(s._sem_load_failed)
        s.sem_result_sig.connect(s._sem_done)
        s.sem_error_sig.connect(s._sem_failed)
        s.sem_improved_sig.connect(s._sem_improved_done)
        s.sem_improved_error_sig.connect(s._sem_improved_failed)
        s.bottleneck_done_sig.connect(s._bottleneck_done)
        s.bottleneck_error_sig.connect(s._bottleneck_failed)
        s.optimization_done_sig.connect(s._optimization_done)
        s.optimization_error_sig.connect(s._optimization_failed)
        s.sem_progress_sig.connect(s._sem_progress)
        s.rv_result_sig.connect(s._rv_done)
        s.rv_error_sig.connect(s._rv_failed)
        s.rv_progress_sig.connect(s._rv_progress)
        s._rv_active = False
        s._rv_result = None
        s._rv_wall_segments = []
        s._rv_focused_wall_id = None
        s._project_path = None
        s._log(s._viewer_backend_startup_message(), "info" if PYQTGRAPH_GL_AVAILABLE else "warn")
        s._log(f"Session cache: {s._cache_root}", "info")
        s._log("Pipeline ready. Steps 1→6.", "info")
        # Background update check — fires 5s after launch so it doesn't block
        # the window from appearing. Silent when no update / no internet.
        QTimer.singleShot(5000, s._startup_update_check)

    def _viewer_backend_startup_message(s):
        if PYQTGRAPH_GL_AVAILABLE:
            return "Embedded 3D point-cloud viewer ready via pyqtgraph.opengl."
        return "pyqtgraph.opengl is unavailable; using lightweight point-cloud preview."

    # ── Theme constants ───────────────────────────────────────────────
    _ACCENT = "#2563eb"           # primary blue
    _ACCENT_HOVER = "#1d4ed8"     # darker blue on hover
    _ACCENT_SECONDARY = "#3b82f6" # lighter blue for secondary actions
    _DANGER = "#dc2626"           # red for destructive actions
    _BG = "#ffffff"
    _BG_PANEL = "#f8f9fa"
    _BG_INPUT = "#ffffff"
    _BORDER = "#d1d5db"
    _BORDER_FOCUS = "#2563eb"
    _TEXT = "#1f2937"
    _TEXT_SECONDARY = "#6b7280"
    _TEXT_MUTED = "#9ca3af"

    def _theme(s):
        s.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {s._BG}; color: {s._TEXT};
                font-family: -apple-system, "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
            }}
            QLabel {{ color: {s._TEXT_SECONDARY}; font-size: 12px; }}
            QScrollArea {{ border: none; background: {s._BG}; }}

            /* ── Input fields ───────────────────────────────── */
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
                background: {s._BG_INPUT}; border: 1px solid {s._BORDER}; border-radius: 6px;
                padding: 6px 10px; color: {s._TEXT};
                font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
                font-size: 12px; min-height: 26px;
                selection-background-color: #dbeafe; selection-color: #1e3a8a;
            }}
            QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:hover {{
                border-color: #9ca3af;
            }}
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
                border: 2px solid {s._BORDER_FOCUS}; padding: 5px 9px;
            }}
            QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QComboBox:disabled {{
                background: #f9fafb; color: {s._TEXT_MUTED}; border-color: #e5e7eb;
            }}
            QDoubleSpinBox::up-button, QSpinBox::up-button {{
                subcontrol-origin: border; subcontrol-position: top right;
                width: 16px; border-left: 1px solid {s._BORDER}; background: transparent;
                border-top-right-radius: 5px;
            }}
            QDoubleSpinBox::down-button, QSpinBox::down-button {{
                subcontrol-origin: border; subcontrol-position: bottom right;
                width: 16px; border-left: 1px solid {s._BORDER}; background: transparent;
                border-bottom-right-radius: 5px;
            }}
            QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
                background: #f3f4f6;
            }}
            QComboBox::drop-down {{ border: none; width: 24px; }}
            QComboBox QAbstractItemView {{
                background: {s._BG_INPUT}; border: 1px solid {s._BORDER};
                border-radius: 6px; padding: 4px; selection-background-color: #dbeafe;
                selection-color: {s._ACCENT_HOVER}; outline: none;
            }}

            /* ── Nested QGroupBox (sub-panels inside a section) ── */
            QGroupBox {{
                background: #fafbfc; border: 1px solid #e5e7eb; border-radius: 8px;
                margin-top: 14px; padding: 14px; padding-top: 22px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 12px; padding: 2px 8px;
                color: {s._TEXT}; font-size: 11px; font-weight: 700;
                letter-spacing: 0.3px; text-transform: uppercase;
            }}

            /* ── Progress bar ───────────────────────────────── */
            QProgressBar {{
                background: #f3f4f6; border: none; border-radius: 4px;
                height: 8px; max-height: 8px; text-align: center;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 #3b82f6, stop:1 {s._ACCENT});
                border-radius: 4px;
            }}

            /* ── Log panel ──────────────────────────────────── */
            QTextEdit {{
                background: #fafbfc; border: 1px solid #e5e7eb; border-radius: 8px;
                color: {s._TEXT_SECONDARY};
                font-family: "JetBrains Mono", "SF Mono", "Consolas", monospace;
                font-size: 11px; padding: 8px 10px;
                selection-background-color: #dbeafe;
            }}

            /* ── Splitter ───────────────────────────────────── */
            QSplitter::handle {{
                background: transparent; width: 8px;
                border-left: 1px solid {s._BORDER};
            }}
            QSplitter::handle:hover {{ border-left: 2px solid {s._ACCENT}; }}

            /* ── Checkbox ───────────────────────────────────── */
            QCheckBox {{ color: {s._TEXT_SECONDARY}; font-size: 11px; spacing: 8px; }}
            QCheckBox::indicator {{
                width: 15px; height: 15px; border: 1.5px solid {s._BORDER};
                border-radius: 4px; background: {s._BG_INPUT};
            }}
            QCheckBox::indicator:hover {{ border-color: {s._ACCENT}; }}
            QCheckBox::indicator:checked {{
                background: {s._ACCENT}; border-color: {s._ACCENT_HOVER};
                image: url();  /* drawn as solid fill; Qt renders a check glyph */
            }}

            /* ── List widget (used in Help dialog etc.) ─────── */
            QListWidget {{
                background: {s._BG_INPUT}; border: 1px solid #e5e7eb; border-radius: 8px;
                color: {s._TEXT}; font-size: 12px; padding: 4px; outline: none;
            }}
            QListWidget::item {{ padding: 8px 12px; border-radius: 6px; margin: 1px 0; }}
            QListWidget::item:hover {{ background: #f9fafb; }}
            QListWidget::item:selected {{ background: #dbeafe; color: {s._ACCENT_HOVER}; }}

            /* ── Scrollbars (subtle, macOS-style) ───────────── */
            QScrollBar:vertical {{
                background: transparent; width: 10px; margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: #d1d5db; border-radius: 5px; min-height: 24px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{ background: #9ca3af; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
            QScrollBar:horizontal {{
                background: transparent; height: 10px; margin: 0;
            }}
            QScrollBar::handle:horizontal {{
                background: #d1d5db; border-radius: 5px; min-width: 24px; margin: 2px;
            }}
            QScrollBar::handle:horizontal:hover {{ background: #9ca3af; }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}

            /* ── ToolTip ────────────────────────────────────── */
            QToolTip {{
                background: #1f2937; color: #f9fafb; border: 1px solid #374151;
                border-radius: 6px; padding: 6px 10px; font-size: 11px;
            }}

            /* ── Tab bar (right panel) ──────────────────────── */
            QTabBar {{ qproperty-drawBase: 0; background: transparent; }}
            QTabBar::tab {{
                background: {s._BG_PANEL};
                color: {s._TEXT_SECONDARY};
                border: 1px solid #e5e7eb;
                border-radius: 6px;
                padding: 6px 14px; margin: 3px 2px;
                font-size: 11px; font-weight: 600;
            }}
            QTabBar::tab:hover {{ background: #eff6ff; color: {s._ACCENT}; border-color: #bfdbfe; }}
            QTabBar::tab:selected {{
                background: #dbeafe; color: {s._ACCENT_HOVER};
                border: 1px solid {s._ACCENT};
            }}
        """)

    # Flat button styling — solid colors, simple hover/press states.
    _BTN_BASE = (
        "QPushButton {"
        "  color: #ffffff;"
        "  border: none;"
        "  border-radius: 4px;"
        "  padding: 10px;"
        "  font-weight: bold;"
        "  font-size: 13px;"
        "}"
        "QPushButton:disabled { background: #e5e7eb; color: #9ca3af; }"
    )

    def _B(s, _c=None):
        return (
            s._BTN_BASE +
            f"QPushButton {{ background: {s._ACCENT}; }}"
            f"QPushButton:hover {{ background: {s._ACCENT_HOVER}; }}"
            f"QPushButton:pressed {{ background: #1e40af; }}"
        )

    def _B_secondary(s):
        return (
            "QPushButton {"
            f"  background: #ffffff; color: {s._ACCENT};"
            f"  border: 1px solid {s._BORDER};"
            "  border-radius: 4px; padding: 10px;"
            "  font-weight: bold; font-size: 13px;"
            "}"
            "QPushButton:hover { background: #f0f4ff; border-color: #2563eb; }"
            "QPushButton:pressed { background: #dbeafe; }"
            "QPushButton:disabled { background: #f3f4f6; color: #9ca3af; border-color: #e5e7eb; }"
        )

    def _B_danger(s):
        return (
            s._BTN_BASE +
            f"QPushButton {{ background: {s._DANGER}; }}"
            "QPushButton:hover { background: #b91c1c; }"
            "QPushButton:pressed { background: #991b1b; }"
        )

    def _B_success(s):
        return (
            s._BTN_BASE +
            "QPushButton { background: #10b981; }"
            "QPushButton:hover { background: #059669; }"
            "QPushButton:pressed { background: #047857; }"
        )

    def _log(s, m, c=""):
        cl = {"info": "#2563eb", "success": "#16a34a", "warn": "#dc2626", "gold": "#d97706"}.get(c, "#6b7280")
        s.log_box.append(f'<span style="color:{cl}">[{time.strftime("%H:%M:%S")}] {m}</span>')

    def _init_session_cache(s):
        cache_base = os.path.join(tempfile.gettempdir(), "rii_pipeline_cache")
        os.makedirs(cache_base, exist_ok=True)
        s._cache_root = tempfile.mkdtemp(prefix="session_", dir=cache_base)
        s.pcd_out = os.path.join(s._cache_root, "pcd")
        s.map_dir = os.path.join(s._cache_root, "map")
        os.makedirs(s.pcd_out, exist_ok=True)
        os.makedirs(s.map_dir, exist_ok=True)

    def _mk_browse_btn(s, tooltip, on_click):
        from PyQt5.QtWidgets import QStyle
        b = QPushButton(s.style().standardIcon(QStyle.SP_DirOpenIcon), "")
        b.setFixedSize(36, 34); b.setToolTip(tooltip)
        b.setStyleSheet(
            "QPushButton { background:#ffffff; border:1px solid #d1d5db;"
            "              border-radius:6px; padding:4px; }"
            "QPushButton:hover { background:#eff6ff; border-color:#93c5fd; }"
            "QPushButton:pressed { background:#dbeafe; }"
        )
        b.clicked.connect(on_click)
        return b

    def _browse_dir(s, le, attr):
        d = QFileDialog.getExistingDirectory(s, "Select", getattr(s, attr))
        if d: le.setText(d); setattr(s, attr, d)

    def _browse_point_cloud(s, le, attr):
        start = getattr(s, attr)
        base = start if os.path.isdir(start) else os.path.dirname(start) if start else ""
        f, _ = QFileDialog.getOpenFileName(s, "Select Point Cloud", base, "Point Cloud (*.pcd *.ply)")
        if f:
            le.setText(f)
            setattr(s, attr, f)

    def _set_clean_param(s, name, value):
        widget = s.cp[name]
        if isinstance(widget, QSpinBox):
            widget.setValue(int(value))
        else:
            widget.setValue(float(value))

    def _sync_map_z_from_cleanup(s):
        follow = hasattr(s, "map_z_follow_cleanup") and s.map_z_follow_cleanup.isChecked()
        if hasattr(s, "oz1") and hasattr(s, "oz2"):
            s.oz1.setEnabled(not follow)
            s.oz2.setEnabled(not follow)
            if follow and hasattr(s, "cp"):
                s.oz1.setValue(float(s.cp["min_z"].value()))
                s.oz2.setValue(float(s.cp["max_z"].value()))

    def _flash_widgets(s, widgets, duration_ms=1600):
        active = [w for w in widgets if w is not None]
        if not active:
            return
        for widget in active:
            widget.setStyleSheet("background:#dbeafe;border:1px solid #2563eb;color:#1e40af;")

        def clear():
            for widget in active:
                widget.setStyleSheet("")

        QTimer.singleShot(duration_ms, clear)

    def _apply_ground_preset(s, log=True):
        preset = {
            "voxel": 0.05,
            "sor_k": 50,
            "sor_std": 0.0,
            "ror_radius": 0.20,
            "ror_min": 2,
        }
        z_preset = {
            "cleanup_min_z": -3.0,
            "cleanup_max_z": 3.0,
            "map_min_z": 0.05,
            "map_max_z": 1.00,
            "floor_anchor_z": 0.0,
        }
        source_desc = "fallback absolute z preset"
        pi = s.e_in.text().strip() if hasattr(s, "e_in") else ""
        if os.path.isfile(pi):
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                points = load_xyz_points(pi)
                z_preset = estimate_ground_preserving_preset(points)
                source_desc = f"floor-anchored from {os.path.basename(pi)}"
            except Exception as exc:
                if log:
                    s._log(f"Preset auto-anchor failed; using fallback z values. {exc}", "warn")
            finally:
                QApplication.restoreOverrideCursor()
        preset["min_z"] = z_preset["cleanup_min_z"]
        preset["max_z"] = z_preset["cleanup_max_z"]
        for key, value in preset.items():
            s._set_clean_param(key, value)
        if hasattr(s, "map_z_follow_cleanup"):
            s.map_z_follow_cleanup.setChecked(False)
        if hasattr(s, "oz1") and hasattr(s, "oz2"):
            s.oz1.setValue(float(z_preset["map_min_z"]))
            s.oz2.setValue(float(z_preset["map_max_z"]))
        s._sync_map_z_from_cleanup()
        changed_widgets = [s.cp.get(key) for key in preset]
        if hasattr(s, "oz1") and hasattr(s, "oz2"):
            changed_widgets.extend([s.oz1, s.oz2])
        s._flash_widgets(changed_widgets)
        if hasattr(s, "preset_status"):
            s.preset_status.setText(
                "Preset applied: "
                f"floor≈{z_preset['floor_anchor_z']:.2f} m, "
                f"cleanup z={z_preset['cleanup_min_z']:.2f}..{z_preset['cleanup_max_z']:.2f} m, "
                f"map z={z_preset['map_min_z']:.2f}..{z_preset['map_max_z']:.2f} m, "
                "voxel=0.05, sor_k=50, sor_std=0.0, ror_radius=0.20, ror_min=2 "
                f"({source_desc})"
            )
        if log:
            s._log(
                "Applied ground-preserving cleanup preset. "
                f"Floor anchor≈{z_preset['floor_anchor_z']:.2f} m, "
                f"cleanup z={z_preset['cleanup_min_z']:.2f}..{z_preset['cleanup_max_z']:.2f} m, "
                f"map z={z_preset['map_min_z']:.2f}..{z_preset['map_max_z']:.2f} m.",
                "info",
            )

    def _apply_map_ground_preset(s, log=True):
        pi = s.e_in.text().strip() if hasattr(s, "e_in") else ""
        if not os.path.isfile(pi):
            QMessageBox.warning(s, "Error", f"Raw point cloud not found:\n{pi}")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        source_desc = "fallback absolute z preset"
        try:
            points = load_xyz_points(pi)
            z_preset = estimate_ground_preserving_preset(points)
            source_desc = f"floor-anchored from {os.path.basename(pi)}"
        except Exception as exc:
            QMessageBox.warning(s, "Error", f"Failed to estimate map z preset:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        s.oz1.setValue(float(z_preset["map_min_z"]))
        s.oz2.setValue(float(z_preset["map_max_z"]))
        s._flash_widgets([s.oz1, s.oz2])
        if hasattr(s, "floor_status"):
            s.floor_status.setText(
                f"Floor detected at {z_preset['floor_anchor_z']:.3f} m  |  "
                f"Recommended obstacle slice: [{z_preset['map_min_z']:.2f}, {z_preset['map_max_z']:.2f}] m (floor-relative)"
            )
        if hasattr(s, "map_preset_status"):
            s.map_preset_status.setText(
                "Preset applied: "
                f"floor≈{z_preset['floor_anchor_z']:.2f} m, "
                f"map z={z_preset['map_min_z']:.2f}..{z_preset['map_max_z']:.2f} m "
                f"({source_desc})"
            )
        if log:
            s._log(
                "Applied raw-cloud map z preset. "
                f"Floor anchor≈{z_preset['floor_anchor_z']:.2f} m, "
                f"map z={z_preset['map_min_z']:.2f}..{z_preset['map_max_z']:.2f} m.",
                "info",
            )

    def _browse_pgm(s):
        f, _ = QFileDialog.getOpenFileName(s, "PGM", "", "PGM (*.pgm)")
        if f:
            s.e_pgm.setText(f); s._load_map(f)
            # Auto-fill yaml if matching file exists
            y = f.replace('.pgm', '.yaml')
            if os.path.isfile(y) and not s.e_yaml.text():
                s.e_yaml.setText(y)

    def _browse_yaml(s):
        f, _ = QFileDialog.getOpenFileName(s, "YAML", "", "YAML (*.yaml *.yml)")
        if f:
            s.e_yaml.setText(f)
            s._update_actual_start_bounds(f)

    def _save_filtered_pcd_as(s):
        src = resolve_point_cloud_path(
            s.e_out.text(),
            filtered_point_cloud_stem_candidates(s.e_in.text().strip()),
        )
        if not os.path.isfile(src):
            QMessageBox.warning(s, "Error", f"Run Step 2 first\n{src}")
            return
        dst, _ = QFileDialog.getSaveFileName(
            s,
            "Save Filtered Point Cloud",
            os.path.join(WORKSPACE, filtered_point_cloud_filename(s.e_in.text().strip())),
            "Point Cloud (*.pcd)",
        )
        if not dst:
            return
        if not dst.lower().endswith(".pcd"):
            dst += ".pcd"
        shutil.copy2(src, dst)
        s._log(f"Saved filtered PCD: {dst}", "success")

    def _save_map_bundle(s):
        src_dir = s.e_save.text().strip()
        pgm = os.path.join(src_dir, "map.pgm")
        yml = os.path.join(src_dir, "map.yaml")
        if not (os.path.isfile(pgm) and os.path.isfile(yml)):
            QMessageBox.warning(s, "Error", f"Run Step 2 first\n{pgm}\n{yml}")
            return
        dst_yaml, _ = QFileDialog.getSaveFileName(
            s,
            "Save Map Bundle As",
            os.path.join(WORKSPACE, "map.yaml"),
            "Nav2 Map YAML (*.yaml)",
        )
        if not dst_yaml:
            return
        if not dst_yaml.lower().endswith(".yaml"):
            dst_yaml += ".yaml"
        dst_pgm = os.path.splitext(dst_yaml)[0] + ".pgm"
        shutil.copy2(pgm, dst_pgm)
        with open(yml, "r", encoding="utf-8") as handle:
            yaml_text = handle.read()
        yaml_text = rewrite_nav2_yaml_image(yaml_text, os.path.basename(dst_pgm))
        with open(dst_yaml, "w", encoding="utf-8") as handle:
            handle.write(yaml_text)
        extras = [
            (
                floor_sidecar_path(pgm),
                floor_sidecar_path(dst_pgm),
                os.path.splitext(yml)[0] + "_floor.yaml",
                os.path.splitext(dst_yaml)[0] + "_floor.yaml",
            ),
            (
                traversability_sidecar_path(pgm),
                traversability_sidecar_path(dst_pgm),
                os.path.splitext(yml)[0] + "_traversable.yaml",
                os.path.splitext(dst_yaml)[0] + "_traversable.yaml",
            ),
        ]
        for src_extra_pgm, dst_extra_pgm, src_extra_yaml, dst_extra_yaml in extras:
            if not os.path.isfile(src_extra_pgm):
                continue
            shutil.copy2(src_extra_pgm, dst_extra_pgm)
            if os.path.isfile(src_extra_yaml):
                with open(src_extra_yaml, "r", encoding="utf-8") as handle:
                    extra_yaml_text = handle.read()
                extra_yaml_text = rewrite_nav2_yaml_image(extra_yaml_text, os.path.basename(dst_extra_pgm))
                with open(dst_extra_yaml, "w", encoding="utf-8") as handle:
                    handle.write(extra_yaml_text)
            s._log(f"Saved sidecar map: {dst_extra_pgm}", "success")
            if os.path.isfile(dst_extra_yaml):
                s._log(f"Saved sidecar config: {dst_extra_yaml}", "success")
        s._log(f"Saved map: {dst_pgm}", "success")
        s._log(f"Saved map config: {dst_yaml}", "success")

    def _selected_map_source_path(s):
        path = s.e_in.text().strip()
        label = "raw input point cloud"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Raw point cloud not found:\n{path}")
        return path, label

    def _map_world_bounds(s, yaml_path=None):
        if s._map_w <= 0 or s._map_h <= 0:
            return None
        if yaml_path is None:
            _, yaml_path = s._get_pgm()
        if not yaml_path or not os.path.isfile(yaml_path):
            return None
        yd = parse_yaml(yaml_path)
        res = yd["resolution"]
        ox, oy = yd["origin"][0], yd["origin"][1]
        return (ox, ox + s._map_w * res, oy, oy + s._map_h * res)

    def _selection_world_bounds(s, yaml_path=None):
        if not s.mw.sel or s._map_w <= 0 or s._map_h <= 0:
            return None
        if yaml_path is None:
            _, yaml_path = s._get_pgm()
        if not yaml_path or not os.path.isfile(yaml_path):
            return None
        yd = parse_yaml(yaml_path)
        return selection_to_world_bounds(s.mw.sel, s._map_w, s._map_h, yd)

    def _selection_center_world(s, yaml_path=None):
        if not s.mw.sel or s._map_w <= 0 or s._map_h <= 0:
            return None
        if yaml_path is None:
            _, yaml_path = s._get_pgm()
        if not yaml_path or not os.path.isfile(yaml_path):
            return None
        yd = parse_yaml(yaml_path)
        center = selection_center_px(s.mw.sel)
        if center is None:
            return None
        res = yd["resolution"]
        ox, oy = yd["origin"][0], yd["origin"][1]
        cx, cy = center
        return (ox + cx * res, oy + (s._map_h - 1 - cy) * res)

    def _update_actual_start_bounds(s, yaml_path=None):
        if not hasattr(s, "sx_") or not hasattr(s, "sy_"):
            return
        bounds = s._map_world_bounds(yaml_path)
        if bounds is None:
            return
        min_x, max_x, min_y, max_y = bounds
        s.sx_.setRange(min_x, max_x)
        s.sy_.setRange(min_y, max_y)
        s.sx_.setValue(min(max(s.sx_.value(), min_x), max_x))
        s.sy_.setValue(min(max(s.sy_.value(), min_y), max_y))

    def _set_actual_start_from_selection(s):
        if not hasattr(s, "sx_") or not hasattr(s, "sy_"):
            return
        _, yaml_path = s._get_pgm()
        center = s._selection_center_world(yaml_path)
        if center is None:
            bounds = s._map_world_bounds(yaml_path)
            if bounds is None:
                return
            min_x, max_x, min_y, max_y = bounds
            center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
            s._log("No selection active. Actual start set to map center.", "info")
        else:
            s._log(f"Actual start set to selection center: ({center[0]:.2f}, {center[1]:.2f})", "info")
        s.sx_.setValue(center[0])
        s.sy_.setValue(center[1])

    def _coverage_start_note(s, result):
        if not result or not result.get("startAdjusted"):
            return ""
        eff = result.get("effectiveStartWorld")
        if not eff:
            return ""
        reason = result.get("startAdjustmentReason")
        if reason == "tiny_island":
            return (
                "Start was on an isolated reachable pocket. "
                f"Used main connected region at ({eff[0]:.2f}, {eff[1]:.2f})."
            )
        if reason == "blocked":
            return (
                "Start was blocked after inflation. "
                f"Used nearest reachable cell at ({eff[0]:.2f}, {eff[1]:.2f})."
            )
        return f"Used reachable start at ({eff[0]:.2f}, {eff[1]:.2f})."

    def _result_area(s, result):
        if not result:
            return 0.0
        return float(result.get("accessibleArea", result.get("reachableArea", result.get("coveredArea", 0.0))))

    def _result_floor_area(s, result):
        if not result:
            return 0.0
        return float(result.get("totalFloorArea", 0.0))

    def _results_share_map(s):
        if not s.ref_r or not s.act_r:
            return False
        if int(s.ref_r.get("w", -1)) != int(s.act_r.get("w", -1)):
            return False
        if int(s.ref_r.get("h", -1)) != int(s.act_r.get("h", -1)):
            return False
        ref_map = s.ref_r.get("pgm_path")
        act_map = s.act_r.get("pgm_path")
        if ref_map and act_map:
            return os.path.abspath(ref_map) == os.path.abspath(act_map)
        return True

    def _next_semantic_session_token(s):
        s._sem_session_token += 1
        return s._sem_session_token

    def _clear_semantic_progress(s):
        if hasattr(s, "sem_prog_lbl"):
            s.sem_prog_lbl.clear()
            s.sem_prog_lbl.hide()
        if hasattr(s, "sem_prog"):
            s.sem_prog.setValue(0)
            s.sem_prog.hide()

    def _set_semantic_candidate_placeholder(s, message):
        if not hasattr(s, "sem_candidate_list"):
            return
        s.sem_candidate_list.blockSignals(True)
        s.sem_candidate_list.clear()
        placeholder = QListWidgetItem(message)
        placeholder.setFlags(Qt.NoItemFlags)
        s.sem_candidate_list.addItem(placeholder)
        s.sem_candidate_list.blockSignals(False)

    def _update_semantic_ready_state(s):
        ready = (
            not s._sem_load_active
            and not s._sem_analysis_active
            and s.ref_r is not None
            and s.act_r is not None
            and s._sem_pts is not None
            and s._sem_labels is not None
        )
        if hasattr(s, "bsem"):
            s.bsem.setEnabled(ready)
        candidate_ready = bool(s._sem_candidates) and not s._sem_load_active and not s._sem_analysis_active
        if hasattr(s, "sem_candidate_list"):
            s._set_semantic_candidate_controls_enabled(candidate_ready)

    def _invalidate_semantic_state(
        s,
        *,
        keep_loaded_cloud=True,
        candidate_message="Run semantic analysis to populate removable-object candidates.",
        status_message=None,
        status_color="#6b7280",
        clear_progress=True,
    ):
        token = s._next_semantic_session_token()
        s._sem_load_active = False
        s._sem_analysis_active = False
        if not keep_loaded_cloud:
            s._sem_pts = None
            s._sem_labels = None
        s._label_grid = None
        s._sem_analysis = None
        s._sem_candidates = []
        s._sem_improved = None
        s._sem_layered_result = None
        s._sem_focused_candidate_id = None
        if hasattr(s, "sem_filter"):
            s.sem_filter.blockSignals(True)
            s.sem_filter.setCurrentIndex(0)
            s.sem_filter.blockSignals(False)
        if hasattr(s, "sem_layered_status"):
            s.sem_layered_status.clear()
            s.sem_layered_status.hide()
        if hasattr(s, "sem_candidate_status"):
            s.sem_candidate_status.setText(
                "Run semantic analysis to populate removable-object candidates, filter them by fixation, and recompute an Optimised RII score."
            )
            s.sem_candidate_status.setStyleSheet("color:#6b7280;font-size:11px")
            s.sem_candidate_status.setVisible(True)
        s._set_semantic_candidate_placeholder(candidate_message)
        s._hide_semantic_whatif_card()
        s.mw.clear_focus()
        if hasattr(s, 'bsem_3d'):
            s.bsem_3d.setEnabled(keep_loaded_cloud and s._sem_pts is not None and s._sem_labels is not None)
        s._imgs["Semantic"] = make_info_image("Run semantic analysis to view object candidates and Optimised RII changes.")
        if s._is_view_active("Semantic"):
            s._switch_view("Semantic")
        if status_message is not None and hasattr(s, "sem_status"):
            s.sem_status.setText(status_message)
            s.sem_status.setStyleSheet(f"color:{status_color};font-size:11px")
        if clear_progress:
            s._clear_semantic_progress()
        s._update_semantic_ready_state()
        return token

    def _clear_step5_results(s, reason=None):
        s.ref_r = None
        s.act_r = None
        if hasattr(s, "lref"):
            s.lref.setText("")
        if hasattr(s, "lact"):
            s.lact.setText("")
        if hasattr(s, "lref_note"):
            s.lref_note.setText("")
        if hasattr(s, "lact_note"):
            s.lact_note.setText("")
        if hasattr(s, "riif"):
            s.riif.hide()
        if hasattr(s, "sem_riif"):
            s._hide_semantic_whatif_card()
        s._imgs["Reference Coverage"] = make_info_image("Run Step 3 to view reference coverage.")
        s._imgs["Actual Coverage"] = make_info_image("Run Step 3 to view actual coverage.")
        s._imgs["Compare"] = make_info_image("Run both Reference and Actual on the same map to compare coverage.")
        s._invalidate_semantic_state(
            keep_loaded_cloud=True,
            candidate_message="Run Step 3 and then semantic analysis to repopulate removable-object candidates for the current map.",
            clear_progress=True,
        )
        s._update_stc_path_view()
        if reason:
            s._log(reason, "info")

    def _use_stc_mode(s):
        """Return selected planner name, or None if 'Without Path Planner'."""
        if not hasattr(s, "rii_mode") or s.rii_mode.currentIndex() == 0:
            return None
        return s.planner_combo.currentText()

    def _toggle_planner_combo(s):
        s.planner_row.setVisible(s.rii_mode.currentIndex() == 1)

    def _update_stc_path_view(s):
        ref = s.ref_r if s.ref_r and s.ref_r.get("useSTC") else None
        act = s.act_r if s.act_r and s.act_r.get("useSTC") else None
        planner_label = ""
        for r in (act, ref):
            if r and r.get("planner"):
                planner_label = r["planner"]
                break
        if ref is None and act is None:
            s._imgs["Planner Path"] = make_info_image("Run Step 3 with a path planner to view the coverage path.")
        else:
            s._imgs["Planner Path"] = render_stc_path_fast(ref, act, bg_pgm=getattr(s, '_pgm_pixels', None), planner_label=planner_label)
        if s._is_view_active("Planner Path"):
            s._switch_view("Planner Path")

    def _toggle_shape(s, prefix):
        if prefix == 'r':
            c = s.rs.currentIndex() == 0; s.rc.setVisible(c); s.rrc.setVisible(not c)
        else:
            c = s.as_.currentIndex() == 0; s.ac.setVisible(c); s.arc.setVisible(not c)

    def _get_params(s, prefix):
        """Return the robot footprint parameters used for horizontal RII."""
        if prefix == 'r':
            if s.rs.currentIndex() == 0:
                r = s.rr.value()
                return dict(shape='circular', radius=r, halfW=r, halfL=r)
            else:
                w = s.rw.value(); l = s.rl.value()
                return dict(shape='rectangular', radius=math.hypot(w/2, l/2), halfW=w/2, halfL=l/2)
        else:
            if s.as_.currentIndex() == 0:
                r = s.ar.value()
                return dict(shape='circular', radius=r, halfW=r, halfL=r)
            else:
                w = s.aw.value(); l = s.al.value()
                return dict(shape='rectangular', radius=math.hypot(w/2, l/2), halfW=w/2, halfL=l/2)

    def _get_pgm(s):
        p = s.e_pgm.text()
        if not p: p = os.path.join(s.e_save.text(), "map.pgm")
        y = s.e_yaml.text()
        if not y: y = p.replace('.pgm', '.yaml')
        if not os.path.isfile(p):
            QMessageBox.warning(s, "Error", f"PGM not found:\n{p}"); return None, None
        if not os.path.isfile(y):
            QMessageBox.warning(s, "Error", f"YAML not found:\n{y}\n\nPlease browse for the .yaml file in Step 3."); return None, None
        return p, y

    def _switch_view(s, nm):
        # Select the matching tab without re-triggering the signal
        if hasattr(s, 'view_tab_bar'):
            for i in range(s.view_tab_bar.count()):
                if s.view_tab_bar.tabText(i) == nm:
                    s.view_tab_bar.blockSignals(True)
                    s.view_tab_bar.setCurrentIndex(i)
                    s.view_tab_bar.blockSignals(False)
                    break
        is_3d = nm in ("3D Viewer", "Clean Cloud", "Vertical Coverage")
        if hasattr(s, '_zclip_bar'):
            s._zclip_bar.setVisible(is_3d)
        if is_3d:
            s.mw.clear_focus()
            s.view_stack.setCurrentWidget(s.pcw)
            if nm in s._clouds:
                s.pcw.set_cloud(s._clouds[nm])
                if s._zclip_cb.isChecked():
                    s._apply_zclip()
            else:
                s.pcw.clear_cloud(f"No {nm.lower()} loaded")
            return
        s.view_stack.setCurrentWidget(s.mw)
        if nm != "Semantic":
            s.mw.clear_focus()
        if nm in s._imgs:
            s.mw.set_qi(s._imgs[nm])
        else:
            s.mw.set_qi(make_info_image(f"No {nm} image"))

    def _is_view_active(s, nm):
        """Check if a given view tab is currently selected."""
        if hasattr(s, 'view_tab_bar'):
            return s.view_tab_bar.tabText(s.view_tab_bar.currentIndex()) == nm
        return False

    def _active_view_name(s):
        """Return the name of the currently selected view tab."""
        if hasattr(s, 'view_tab_bar'):
            return s.view_tab_bar.tabText(s.view_tab_bar.currentIndex())
        return PRIMARY_SELECTION_VIEW

    def _apply_zclip(s):
        """Clip the current 3D cloud above the chosen Z threshold."""
        if hasattr(s, 'pcw') and hasattr(s.pcw, 'clip_z'):
            s.pcw.clip_z(s._zclip_spin.value())

    # ── Split View ──
    def _toggle_split_view(s):
        """Show or hide the secondary split viewer panel."""
        show = s.btn_split_view.isChecked()
        s._split_panel.setVisible(show)
        if show:
            s._switch_split_view(s._split_tab_bar.tabText(s._split_tab_bar.currentIndex()))
            s._split_splitter.setSizes([1, 1])  # equal widths

    def _switch_split_view(s, nm):
        """Switch the secondary split panel to the named view."""
        for i in range(s._split_tab_bar.count()):
            if s._split_tab_bar.tabText(i) == nm:
                s._split_tab_bar.blockSignals(True)
                s._split_tab_bar.setCurrentIndex(i)
                s._split_tab_bar.blockSignals(False)
                break
        if nm in ("3D Viewer", "Clean Cloud", "Vertical Coverage"):
            s._split_view_stack.setCurrentWidget(s._split_pcw)
            if nm in s._clouds:
                s._split_pcw.set_cloud(s._clouds[nm])
            else:
                s._split_pcw.clear_cloud(f"No {nm.lower()} loaded")
            return
        s._split_view_stack.setCurrentWidget(s._split_mw)
        if nm in s._imgs:
            s._split_mw.set_qi(s._imgs[nm])
        else:
            s._split_mw.set_qi(make_info_image(f"No {nm} image"))

    def _set_img(s, nm, qi):
        s._imgs[nm] = qi; s._switch_view(nm)

    def _set_cloud(s, nm, cloud):
        s._clouds[nm] = cloud
        s._switch_view(nm)

    def _fallback_point_cloud_viewer(s, reason):
        if isinstance(s.pcw, PointCloudPreviewW):
            return
        current = s._active_view_name()
        old = s.pcw
        idx = s.view_stack.indexOf(old)
        preview = PointCloudPreviewW()
        if idx >= 0:
            s.view_stack.removeWidget(old)
            old.hide()
            old.setParent(None)
            s.view_stack.insertWidget(idx, preview)
        s.pcw = preview
        s._log(f"Embedded 3D viewer backend failed; using software preview instead. {reason}", "warn")
        if current in ("3D Viewer", "Clean Cloud"):
            s._switch_view(current)

    def _load_map(s, pgm):
        previous_size = (s._map_w, s._map_h)
        w, h, pixels = parse_pgm(pgm)
        previous_path = s._loaded_map_path
        s._map_w, s._map_h = w, h
        s._pgm_pixels = pixels
        qi = QImage(pixels.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
        s._imgs[BLOCKED_MAP_VIEW] = qi
        s._load_map_sidecars(pgm)
        s._switch_view(PRIMARY_SELECTION_VIEW)
        if previous_path and (
            os.path.abspath(previous_path) != os.path.abspath(pgm) or previous_size != (w, h)
        ):
            s._clear_step5_results("Loaded a different map. Rerun Step 3 on the current map.")
        s._loaded_map_path = pgm
        s._update_statusbar_map(pgm)
        yaml_guess = pgm.replace(".pgm", ".yaml")
        if os.path.isfile(yaml_guess):
            s._update_actual_start_bounds(yaml_guess)
            try:
                yd = parse_yaml(yaml_guess)
                s.mw.set_map_metadata(yd["resolution"], yd["origin"], h, w)
            except Exception:
                pass

    def _show_report_window(s):
        """Open the RII analysis report in a separate window."""
        html = getattr(s, '_sem_report_html', '')
        if not html:
            QMessageBox.warning(s, "No Report", "Run semantic analysis first to generate a report.")
            return
        win = QMainWindow(s)
        win.setWindowTitle("RII Analysis Report")
        win.resize(700, 600)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setStyleSheet("background:#ffffff;color:#1f2937;font-family:monospace;font-size:12px;padding:12px")
        te.setHtml(html)
        win.setCentralWidget(te)
        win.show()

    def _enable_start_pick(s):
        p, _ = s._get_pgm()
        if not p:
            return
        if BLOCKED_MAP_VIEW not in s._imgs:
            s._load_map(p)
        s._switch_view(BLOCKED_MAP_VIEW)
        s.mw.enable_start_pick()
        s._log("Click on the map to set the robot start point.", "info")

    def _on_start_picked(s, px, py, wx, wy):
        s._act_start_world = (wx, wy)
        s._act_start_px = (px, py)
        s.start_label.setText(f"Start: ({wx:.2f}, {wy:.2f}) m")
        s.start_label.setStyleSheet("color:#dc2626;font-size:11px;font-weight:bold")
        s._log(f"Start point set: pixel ({px}, {py}) → world ({wx:.2f}, {wy:.2f}) m", "success")

    def _draw_start_marker(s, qi, result, start_world):
        """Draw a red circle + crosshair at the robot start point on a coverage QImage."""
        from PyQt5.QtGui import QPainter, QPen, QBrush
        yd_path = s.e_yaml.text()
        if not yd_path or not os.path.isfile(yd_path):
            return qi
        yd = parse_yaml(yd_path)
        res = float(yd['resolution'])
        ox, oy = float(yd['origin'][0]), float(yd['origin'][1])
        h = result['h']
        wx, wy = start_world
        px = int((wx - ox) / res)
        py = h - 1 - int((wy - oy) / res)

        img = qi.copy()
        p = QPainter(img)
        p.setPen(QPen(QColor(220, 30, 30), 2))
        p.setBrush(QBrush(QColor(220, 30, 30, 100)))
        r = 8
        p.drawEllipse(px - r, py - r, 2 * r, 2 * r)
        p.drawLine(px - r - 4, py, px + r + 4, py)
        p.drawLine(px, py - r - 4, px, py + r + 4)
        p.setPen(QColor(255, 255, 255))
        p.drawText(px + r + 6, py + 4, "START")
        p.end()
        return img

    def _on_hover_coords(s, px, py, wx, wy):
        s.coord_label.setText(f"pixel ({px}, {py})  |  world ({wx:.3f}, {wy:.3f}) m")

    def _save_current_view(s):
        """Save the currently displayed map view as PNG."""
        view_name = None
        for name, img in s._imgs.items():
            if s.mw._bp and not s.mw._bp.isNull():
                view_name = name
        path, _ = QFileDialog.getSaveFileName(s, "Save View", "view.png", "PNG (*.png)")
        if path and s.mw._bp and not s.mw._bp.isNull():
            s.mw._bp.save(path, "PNG")
            s._log(f"Saved view: {path}", "success")

    def _export_results_csv(s):
        """Export RII results and candidate list as CSV files."""
        import csv
        path, _ = QFileDialog.getSaveFileName(s, "Export CSV", "rii_results.csv", "CSV (*.csv)")
        if not path:
            return
        base = os.path.splitext(path)[0]

        # Export main results
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Metric", "Value"])
            if s.act_r:
                aa = s._result_area(s.act_r)
                fa = s._result_floor_area(s.act_r)
                rii = (aa / fa * 100) if fa > 0 else 0
                w.writerow(["RII_Horizontal_%", f"{rii:.2f}"])
                w.writerow(["Accessible_Area_m2", f"{aa:.4f}"])
                w.writerow(["Total_Floor_Area_m2", f"{fa:.4f}"])
                w.writerow(["Planner", s.act_r.get("planner", "None")])
            if s.ref_r:
                w.writerow(["Ref_Accessible_Area_m2", f"{s._result_area(s.ref_r):.4f}"])
        s._log(f"Exported results: {path}", "success")

        # Export candidates if available
        if s._sem_candidates:
            cand_path = base + "_candidates.csv"
            with open(cand_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["ID", "Name", "Fixation", "Area_m2", "PotentialUnlock_m2",
                             "TrueUnlock_m2", "BottleneckRatio", "IsBottleneck", "ActionType"])
                for c in s._sem_candidates:
                    w.writerow([
                        c["id"], c["name"], c["fixation"], f"{c['area']:.4f}",
                        f"{c['potentialUnlockArea']:.4f}",
                        f"{c.get('trueUnlockArea', 0):.4f}",
                        f"{c.get('bottleneckRatio', 0):.2f}",
                        c.get("isBottleneck", False),
                        c.get("actionType", ""),
                    ])
            s._log(f"Exported candidates: {cand_path}", "success")

    def _load_map_sidecars(s, pgm):
        s._trav_pixels = None
        sidecars = [
            ("Floor Mask", floor_sidecar_path(pgm), "No floor sidecar for this map bundle."),
            ("Traversable Ground", traversability_sidecar_path(pgm), "No traversability sidecar for this map bundle."),
        ]
        for name, path, placeholder in sidecars:
            if os.path.isfile(path):
                try:
                    w, h, pixels = parse_pgm(path)
                    qi = QImage(pixels.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()
                    s._imgs[name] = qi
                    if name == "Traversable Ground":
                        s._trav_pixels = pixels
                except Exception as exc:
                    s._imgs[name] = make_info_image(f"Failed to load {os.path.basename(path)}.\n{exc}")
            else:
                s._imgs[name] = make_info_image(placeholder)

    # ── Build selection mask matching HTML lines 342-344 ──
    def _make_sel_mask(s):
        """Create a selection mask in OccGrid coords from rectangle or freeform input."""
        if not s.mw.sel or s._map_w == 0:
            return None
        return selection_mask_from_display(s.mw.sel, s._map_w, s._map_h)

    # ══════════════════════════════════════════════════════════════════════
    # Project save/load
    # ══════════════════════════════════════════════════════════════════════
    _MRU_MAX = 5
    _MRU_KEY = "recent_projects"

    def _build_menu(s):
        from PyQt5.QtWidgets import QStyle
        st = s.style()
        mb = s.menuBar()
        mb.setStyleSheet(
            f"QMenuBar {{ background: {s._BG}; border-bottom: 1px solid {s._BORDER}; padding: 2px; }}"
            f"QMenuBar::item {{ background: transparent; padding: 6px 12px; color: {s._TEXT}; }}"
            f"QMenuBar::item:selected {{ background: #dbeafe; color: {s._ACCENT}; border-radius: 4px; }}"
            f"QMenu {{ background: {s._BG}; border: 1px solid {s._BORDER}; padding: 4px; }}"
            f"QMenu::item {{ padding: 6px 24px 6px 24px; color: {s._TEXT}; border-radius: 4px; }}"
            f"QMenu::item:selected {{ background: #dbeafe; color: {s._ACCENT}; }}"
            f"QMenu::item:disabled {{ color: {s._TEXT_MUTED}; }}"
            f"QMenu::separator {{ height: 1px; background: {s._BORDER}; margin: 4px 8px; }}"
        )
        m = mb.addMenu("&File")
        a_open = QAction(st.standardIcon(QStyle.SP_DirOpenIcon), "&Open Project…", s)
        a_open.setShortcut(QKeySequence.Open)
        a_open.triggered.connect(s._open_project); m.addAction(a_open)
        s._recent_menu = m.addMenu("Open &Recent")
        s._recent_menu.setIcon(st.standardIcon(QStyle.SP_FileIcon))
        s._rebuild_recent_menu()
        m.addSeparator()
        a_save = QAction(st.standardIcon(QStyle.SP_DialogSaveButton), "&Save Project", s)
        a_save.setShortcut(QKeySequence.Save)
        a_save.triggered.connect(s._save_project); m.addAction(a_save)
        a_save_as = QAction(st.standardIcon(QStyle.SP_DialogSaveButton), "Save Project &As…", s)
        a_save_as.setShortcut("Ctrl+Shift+S")
        a_save_as.triggered.connect(s._save_project_as); m.addAction(a_save_as)
        m.addSeparator()
        a_quit = QAction(st.standardIcon(QStyle.SP_DialogCloseButton), "&Quit", s)
        a_quit.setShortcut(QKeySequence.Quit)
        a_quit.triggered.connect(s.close); m.addAction(a_quit)

        # ── Help menu ────────────────────────────────────────────────────
        h = mb.addMenu("&Help")
        a_guide = QAction(st.standardIcon(QStyle.SP_MessageBoxQuestion), "&User Guide", s)
        a_guide.setShortcut("F1")
        a_guide.triggered.connect(s._show_user_guide); h.addAction(a_guide)
        a_keys = QAction(st.standardIcon(QStyle.SP_FileDialogContentsView), "&Keyboard Shortcuts", s)
        a_keys.triggered.connect(s._show_shortcuts); h.addAction(a_keys)
        h.addSeparator()
        a_update = QAction(st.standardIcon(QStyle.SP_BrowserReload), "Check for &Updates…", s)
        a_update.triggered.connect(s._check_for_updates); h.addAction(a_update)
        h.addSeparator()
        a_about = QAction(st.standardIcon(QStyle.SP_MessageBoxInformation), "&About", s)
        a_about.triggered.connect(s._show_about); h.addAction(a_about)

    def _show_user_guide(s):
        from gui.help_dialog import UserGuideDialog
        UserGuideDialog(s).exec_()

    def _show_shortcuts(s):
        from gui.help_dialog import ShortcutsDialog
        ShortcutsDialog(s).exec_()

    def _show_about(s):
        from gui.help_dialog import AboutDialog
        AboutDialog(s).exec_()

    def _check_for_updates(s, silent=False):
        from gui.update_dialog import check_and_prompt
        check_and_prompt(s, silent=silent)

    def _startup_update_check(s):
        """Silent background check on launch — only notifies if an update exists."""
        try:
            s._check_for_updates(silent=True)
        except Exception as e:  # noqa: BLE001
            s._log(f"[Update] Background check failed: {e}", "warn")

    def _settings(s):
        return QSettings("rii_pipeline", "rii_pipeline")

    def _get_recent(s):
        v = s._settings().value(s._MRU_KEY, [])
        if isinstance(v, str):
            v = [v]
        return [p for p in (v or []) if isinstance(p, str) and p]

    def _push_recent(s, path):
        path = os.path.abspath(path)
        recent = [p for p in s._get_recent() if p != path]
        recent.insert(0, path)
        s._settings().setValue(s._MRU_KEY, recent[:s._MRU_MAX])
        s._rebuild_recent_menu()

    def _rebuild_recent_menu(s):
        if not hasattr(s, "_recent_menu"):
            return
        s._recent_menu.clear()
        recent = s._get_recent()
        if not recent:
            empty = QAction("(none)", s); empty.setEnabled(False)
            s._recent_menu.addAction(empty)
            return
        for path in recent:
            label = os.path.basename(path) + "   " + os.path.dirname(path)
            a = QAction(label, s)
            a.triggered.connect(lambda _checked=False, p=path: s._open_project_path(p))
            s._recent_menu.addAction(a)
        s._recent_menu.addSeparator()
        clear = QAction("Clear Recent", s)
        clear.triggered.connect(s._clear_recent)
        s._recent_menu.addAction(clear)

    def _clear_recent(s):
        s._settings().setValue(s._MRU_KEY, [])
        s._rebuild_recent_menu()

    # ── Status bar ────────────────────────────────────────────────────────
    def _build_statusbar(s):
        sb = s.statusBar()
        sb.setStyleSheet(
            f"QStatusBar {{ background: {s._BG_PANEL}; border-top: 1px solid {s._BORDER}; "
            f"color: {s._TEXT_SECONDARY}; font-size: 11px; font-family: monospace; }}"
            f"QStatusBar QLabel {{ padding: 0 8px; }}"
            f"QStatusBar::item {{ border: none; }}"
        )
        s._sb_map = QLabel("No map loaded")
        s._sb_coords = QLabel("")
        s._sb_coords.setMinimumWidth(280)
        s._sb_worker = QLabel("Idle")
        s._sb_worker.setStyleSheet("color: #6b7280;")
        sb.addWidget(s._sb_map, 1)
        sb.addPermanentWidget(s._sb_coords)
        sb.addPermanentWidget(s._sb_worker)

    def _update_statusbar_map(s, pgm_path):
        if not pgm_path or not os.path.isfile(pgm_path):
            s._sb_map.setText("No map loaded")
            return
        try:
            from core.map_io import parse_pgm, parse_yaml
            w, h, _ = parse_pgm(pgm_path)
            yaml_path = pgm_path.replace(".pgm", ".yaml")
            res = parse_yaml(yaml_path).get("resolution", 0.0) if os.path.isfile(yaml_path) else 0.0
            s._sb_map.setText(f"{os.path.basename(pgm_path)}   {w}×{h} px   {float(res):.3f} m/px")
        except Exception:
            s._sb_map.setText(os.path.basename(pgm_path))

    def _on_hover_coords(s, px, py, wx, wy):
        s._sb_coords.setText(f"px({px}, {py})   world({wx:+.2f}, {wy:+.2f}) m")

    def _set_worker_status(s, text, busy=False):
        s._sb_worker.setText(text)
        color = "#2563eb" if busy else "#6b7280"
        s._sb_worker.setStyleSheet(f"color: {color};")

    # ── Pipeline stepper ──────────────────────────────────────────────────
    _STEP_LABELS = ["View PCD", "Build Map", "RII Horizontal", "Analysis", "RII Vertical"]

    def _build_stepper(s):
        w = StepIndicator(s._STEP_LABELS)
        w.step_clicked.connect(s._scroll_to_step)
        w.set_status(0, "active")
        return w

    def _scroll_to_step(s, idx):
        if 0 <= idx < len(s._group_boxes):
            gb = s._group_boxes[idx]
            if hasattr(gb, "expand"):
                gb.expand()
            area = s._sidebar_scroll
            if area is not None:
                from PyQt5.QtCore import QTimer
                def _do_scroll():
                    y = gb.mapTo(area.widget(), gb.rect().topLeft()).y()
                    area.verticalScrollBar().setValue(max(0, y - 8))
                QTimer.singleShot(0, _do_scroll)
            s._stepper.set_active(idx)
            s._sync_section_states(idx)

    def _sync_section_states(s, active_idx):
        """Mirror the stepper's active/complete/pending states onto the section headers."""
        for i, gb in enumerate(s._group_boxes):
            if not hasattr(gb, "setActive"):
                continue
            if i < active_idx:
                gb.setStatus("complete"); gb.setActive(False)
            elif i == active_idx:
                gb.setStatus("active"); gb.setActive(True)
            else:
                gb.setStatus("pending"); gb.setActive(False)

    # Widget attribute names → (type, getter-hint). Kept in one place so
    # save/load stay in sync when a new parameter is added.
    _PROJECT_PARAM_ATTRS = [
        "e_in", "cb_noise_filter", "filter_radius", "filter_min_nb",
        "oz1", "oz2", "min_pts_cell", "t_slope", "t_step",
        "cb_multi_level",
        "edit_brush_shape", "edit_brush_size", "edit_brush_w", "edit_brush_h",
        "edit_ref_overlay",
        "e_pgm", "e_yaml", "e_sem_pcd",
        "sel_mode", "rii_mode", "planner_combo",
        "rs", "rw", "rl", "as_", "ar", "aw", "al",
        "sem_filter",
        "rv_wall_min_h", "rv_wall_max_h", "rv_voxel", "rv_reach", "rv_angle",
        "rv_paint_w", "rv_paint_vspan", "rv_sweep", "rv_stride",
        "rv_max_samples", "rv_wall_ids", "rv_gamma",
        "_zclip_cb", "_zclip_spin",
    ]

    def _collect_params(s):
        from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox
        out = {}
        for attr in s._PROJECT_PARAM_ATTRS:
            w = getattr(s, attr, None)
            if w is None:
                continue
            if isinstance(w, QCheckBox):
                out[attr] = {"kind": "check", "value": bool(w.isChecked())}
            elif isinstance(w, QComboBox):
                out[attr] = {"kind": "combo", "value": w.currentText()}
            elif isinstance(w, QLineEdit):
                out[attr] = {"kind": "line", "value": w.text()}
            elif isinstance(w, QSpinBox):
                out[attr] = {"kind": "spin_i", "value": int(w.value())}
            elif isinstance(w, QDoubleSpinBox):
                out[attr] = {"kind": "spin_f", "value": float(w.value())}
        return out

    def _apply_params(s, params):
        from PyQt5.QtWidgets import QCheckBox, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox
        for attr, entry in (params or {}).items():
            w = getattr(s, attr, None)
            if w is None:
                continue
            v = entry.get("value")
            if isinstance(w, QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QComboBox):
                idx = w.findText(str(v))
                if idx >= 0:
                    w.setCurrentIndex(idx)
            elif isinstance(w, QLineEdit):
                w.setText(str(v) if v is not None else "")
            elif isinstance(w, QSpinBox):
                try: w.setValue(int(v))
                except (TypeError, ValueError): pass
            elif isinstance(w, QDoubleSpinBox):
                try: w.setValue(float(v))
                except (TypeError, ValueError): pass

    def _collect_artifact_paths(s):
        return {
            "pcd_in": s.pcd_in or "",
            "loaded_map": getattr(s, "_loaded_map_path", "") or "",
            "pgm": s.e_pgm.text() if hasattr(s, "e_pgm") else "",
            "yaml": s.e_yaml.text() if hasattr(s, "e_yaml") else "",
            "sem_pcd": s.e_sem_pcd.text() if hasattr(s, "e_sem_pcd") else "",
        }

    def _open_project(s):
        from core.project_io import PROJECT_EXT
        start = os.path.dirname(s._project_path) if s._project_path else ""
        f, _ = QFileDialog.getOpenFileName(s, "Open Project", start,
                                            f"RII Project (*{PROJECT_EXT})")
        if not f:
            return
        s._open_project_path(f)

    def _open_project_path(s, f):
        from core.project_io import load_project
        if not os.path.isfile(f):
            QMessageBox.warning(s, "Open Project", f"File not found:\n{f}")
            # Drop stale MRU entries
            recent = [p for p in s._get_recent() if p != f]
            s._settings().setValue(s._MRU_KEY, recent)
            s._rebuild_recent_menu()
            return
        try:
            data = load_project(f)
        except Exception as e:
            QMessageBox.warning(s, "Open Project", f"Failed to load:\n{e}")
            return
        s._apply_params(data.get("params", {}))
        paths = data.get("paths", {})
        if paths.get("pcd_in"):
            s.pcd_in = paths["pcd_in"]
            if hasattr(s, "e_in"):
                s.e_in.setText(paths["pcd_in"])
        pgm = paths.get("pgm") or paths.get("loaded_map")
        if pgm and os.path.isfile(pgm) and hasattr(s, "_load_map"):
            s._load_map(pgm)
            if hasattr(s, "e_pgm"): s.e_pgm.setText(pgm)
            if paths.get("yaml") and hasattr(s, "e_yaml"):
                s.e_yaml.setText(paths["yaml"])
        s._manual_ramps = list(data.get("manual_ramps", []))
        for t in s._manual_ramps:
            t._is_manual = True
        s._ground_result = data.get("ground_result")
        if s._ground_result is not None:
            s._on_ground_result(s._ground_result)
        s._project_path = f
        s._push_recent(f)
        s.setWindowTitle(f"Robot Inclusivity Index (RII) — {os.path.basename(f)}")
        s._log(f"Opened project: {f}", "success")

    def _save_project_as(s):
        from core.project_io import PROJECT_EXT
        start = s._project_path or os.path.join(os.path.expanduser("~"), f"untitled{PROJECT_EXT}")
        f, _ = QFileDialog.getSaveFileName(s, "Save Project As", start,
                                            f"RII Project (*{PROJECT_EXT})")
        if not f:
            return
        if not f.endswith(PROJECT_EXT):
            f += PROJECT_EXT
        s._project_path = f
        s._save_project()

    def _save_project(s):
        from core.project_io import build_project_dict, save_project, PROJECT_EXT
        if not s._project_path:
            return s._save_project_as()
        try:
            data = build_project_dict(
                params=s._collect_params(),
                artifact_paths=s._collect_artifact_paths(),
                manual_ramps=getattr(s, "_manual_ramps", []),
                ground_result=getattr(s, "_ground_result", None),
                project_file=s._project_path,
            )
            save_project(s._project_path, data)
        except Exception as e:
            QMessageBox.warning(s, "Save Project", f"Failed to save:\n{e}")
            return
        s._push_recent(s._project_path)
        s.setWindowTitle(f"Robot Inclusivity Index (RII) — {os.path.basename(s._project_path)}")
        s._log(f"Saved project: {s._project_path}", "success")

    # ══════════════════════════════════════════════════════════════════════
    # _build — construct all UI widgets
    # ══════════════════════════════════════════════════════════════════════
    def _build(s):
        s._build_menu()
        s._build_statusbar()
        cw = QWidget(); s.setCentralWidget(cw)
        ml = QVBoxLayout(cw); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)
        s._stepper = s._build_stepper(); ml.addWidget(s._stepper)
        sp = QSplitter(Qt.Horizontal); sp.setChildrenCollapsible(False); sp.setHandleWidth(8); ml.addWidget(sp, 1)

        ls = DragScrollArea()
        ls.setMinimumWidth(360)
        lw = QWidget(); ll = QVBoxLayout(lw)
        ll.setSpacing(8); ll.setContentsMargins(12, 12, 12, 12)

        from PyQt5.QtWidgets import QStyle
        _folder_icon = s.style().standardIcon(QStyle.SP_DirOpenIcon)
        _browse_ss = (
            "QPushButton { background: #ffffff; border: 1px solid #d1d5db;"
            "              border-radius: 6px; padding: 4px; }"
            "QPushButton:hover { background: #eff6ff; border-color: #93c5fd; }"
            "QPushButton:pressed { background: #dbeafe; }"
        )

        def mkbr_dir(le, attr):
            b = QPushButton(_folder_icon, ""); b.setFixedSize(36, 34)
            b.setToolTip("Browse for folder"); b.setStyleSheet(_browse_ss)
            b.clicked.connect(lambda: s._browse_dir(le, attr)); return b

        def mkbr_cloud(le, attr):
            b = QPushButton(_folder_icon, ""); b.setFixedSize(36, 34)
            b.setToolTip("Browse for point cloud (.pcd / .ply)")
            b.setStyleSheet(_browse_ss)
            b.clicked.connect(lambda: s._browse_point_cloud(le, attr)); return b

        # Step 1
        g1 = QWidget(); l1 = QVBoxLayout()
        h1 = QHBoxLayout(); s.e_in = QLineEdit(s.pcd_in)
        s.e_in.setPlaceholderText("Pick a raw .pcd or .ply file")
        h1.addWidget(QLabel("Input File:")); h1.addWidget(s.e_in, 1); h1.addWidget(mkbr_cloud(s.e_in, 'pcd_in'))
        l1.addLayout(h1)
        s.b1 = QPushButton("View Raw Point Cloud"); s.b1.setStyleSheet(s._B())
        s.b1.clicked.connect(s._step1); l1.addWidget(s.b1)
        # Noise filter
        filter_row = QHBoxLayout()
        s.cb_noise_filter = QCheckBox("Apply noise filter")
        s.cb_noise_filter.setToolTip(
            "Remove scattered noise points before map generation.\n"
            "Points with fewer than 'Min neighbors' within 'Radius'\n"
            "are removed. Enable if your map has scattered false obstacles."
        )
        filter_row.addWidget(s.cb_noise_filter)
        filter_row.addWidget(QLabel("Radius (m):"))
        s.filter_radius = QDoubleSpinBox(); s.filter_radius.setRange(0.1, 5.0); s.filter_radius.setValue(0.5); s.filter_radius.setDecimals(1)
        s.filter_radius.setToolTip("Search radius for 2D density check.\n0.5m = typical for LiDAR maps.")
        filter_row.addWidget(s.filter_radius)
        filter_row.addWidget(QLabel("Min density:"))
        s.filter_min_nb = QSpinBox(); s.filter_min_nb.setRange(1, 1000); s.filter_min_nb.setValue(100)
        s.filter_min_nb.setToolTip("Minimum point count within radius to keep.\n50 = light cleaning\n100 = medium (recommended)\n200 = aggressive")
        filter_row.addWidget(s.filter_min_nb)
        s.filter_status = QLabel("")
        s.filter_status.setStyleSheet("color:#059669;font-size:11px")
        filter_row.addWidget(s.filter_status, 1)
        l1.addLayout(filter_row)
        g1.setLayout(l1)
        section1 = CollapsibleSection(1, "View Raw Point Cloud", g1); ll.addWidget(section1)

        # Step 2
        g4 = QWidget(); l4 = QVBoxLayout()
        h4 = QHBoxLayout(); s.e_save = QLineEdit(s.map_dir)
        s.e_save.setReadOnly(True)
        h4.addWidget(QLabel("Cache Dir:")); h4.addWidget(s.e_save, 1)
        l4.addLayout(h4)
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Map Source:"))
        src_label = QLabel("Raw input point cloud")
        src_label.setStyleSheet("color:#1f2937;font-weight:bold")
        src_row.addWidget(src_label, 1)
        l4.addLayout(src_row)
        h4p = QHBoxLayout()
        s.b4preset = QPushButton("Apply Ground-Based Z Preset")
        s.b4preset.setStyleSheet(s._B_secondary())
        s.b4preset.clicked.connect(lambda: s._apply_map_ground_preset())
        h4p.addWidget(s.b4preset)
        h4p.addStretch()
        l4.addLayout(h4p)
        s.map_preset_status = QLabel("")
        s.map_preset_status.setWordWrap(True)
        s.map_preset_status.setStyleSheet("color:#16a34a;font-size:11px")
        l4.addWidget(s.map_preset_status)
        s.map_source_hint = QLabel(
            "Adjust the obstacle max height, max slope and max step below, then click 'Generate 2D Map'. "
            "You can also edit the traversable map manually using the Draw/Erase tools below."
        )
        s.map_source_hint.setWordWrap(True)
        l4.addWidget(s.map_source_hint)

        # ── Floor detection status ──
        s.floor_status = QLabel("")
        s.floor_status.setWordWrap(True)
        s.floor_status.setStyleSheet("color:#2563eb;font-size:11px;font-weight:bold;padding:2px")
        l4.addWidget(s.floor_status)

        if s._v3_mode:
            # V3: min_z linked to max_step — obstacles below step height are ignored
            zh_minz = QHBoxLayout()
            zh_minz.addWidget(QLabel("Min obstacle height (m):"))
            s.oz1 = QDoubleSpinBox(); s.oz1.setRange(0.0, 5.0); s.oz1.setValue(0.25); s.oz1.setDecimals(2)
            s.oz1.setToolTip(
                "V3: Linked to max_step.\n"
                "Obstacles shorter than this are ignored (robot climbs over them).\n"
                "Automatically set when you change max_step below."
            )
            zh_minz.addWidget(s.oz1)
            l4.addLayout(zh_minz)
        else:
            # V1/V2: hidden, fixed at 0.05
            s.oz1 = QDoubleSpinBox(); s.oz1.setValue(0.05); s.oz1.hide()

        zh = QHBoxLayout()
        zh.addWidget(QLabel("Obstacle max height (m):"))
        s.oz2 = QDoubleSpinBox(); s.oz2.setRange(-20, 20); s.oz2.setValue(1.00); s.oz2.setDecimals(2)
        s.oz2.setToolTip(
            "Maximum obstacle height above detected floor.\n"
            "Points above this are ignored (ceiling, pipes).\n"
            "Set to slightly above robot height."
        )
        zh.addWidget(s.oz2)
        zh.addWidget(QLabel("  Min points/cell:"))
        s.min_pts_cell = QSpinBox(); s.min_pts_cell.setRange(1, 20); s.min_pts_cell.setValue(3)
        s.min_pts_cell.setToolTip(
            "Minimum LiDAR points in a cell to count as an obstacle.\n"
            "Increase to reduce noise (scattered false obstacles).\n"
            "  3 = default, 5-8 = cleaner map, 10+ = very aggressive filtering"
        )
        zh.addWidget(s.min_pts_cell)
        l4.addLayout(zh)
        th = QHBoxLayout()
        th.addWidget(QLabel("max_slope (deg):"))
        s.t_slope = QDoubleSpinBox(); s.t_slope.setRange(1, 89); s.t_slope.setValue(35.0); s.t_slope.setDecimals(1)
        s.t_slope.setToolTip(
            "Maximum slope the robot can traverse.\n"
            "Steeper surfaces are marked non-traversable.\n"
            "Tracked: ~30-40 deg | Wheeled: ~15-25 deg"
        )
        th.addWidget(s.t_slope)
        th.addWidget(QLabel("max_step (m):"))
        s.t_step = QDoubleSpinBox(); s.t_step.setRange(0.01, 9999.0); s.t_step.setValue(0.25); s.t_step.setDecimals(2)
        s.t_step.setToolTip(
            "Maximum step height between adjacent cells.\n"
            "Steps taller than this are non-traversable (curbs, stairs).\n"
            "Should match robot's actual step-climbing ability."
        )
        th.addWidget(s.t_step)
        if s._v3_mode:
            # V3: link max_step → oz1 (min obstacle height)
            s.t_step.valueChanged.connect(lambda v: s.oz1.setValue(v))
            s.oz1.setValue(s.t_step.value())  # sync initial value
        l4.addLayout(th)
        if s._v3_mode:
            l4.addWidget(QLabel("V3: max_slope = steepest ramp, max_step = min obstacle height (steps below this are ignored)."))
        else:
            l4.addWidget(QLabel("Terrain thresholds for traversability: max_slope = steepest ramp, max_step = tallest climbable step."))
        s.cb_multi_level = QCheckBox("Multi-floor building (auto-detect)")
        s.cb_multi_level.setToolTip(
            "Tick this if your scan covers more than one floor.\n"
            "Step 2 will auto-detect levels and build one map per floor."
        )
        l4.addWidget(s.cb_multi_level)

        s.b4 = QPushButton("Generate 2D Map"); s.b4.setStyleSheet(s._B("#aa66ff"))
        s.b4.clicked.connect(s._step4); l4.addWidget(s.b4)
        s.b4save = QPushButton("Save Map (.pgm + .yaml) As..."); s.b4save.setStyleSheet(s._B_secondary())
        s.b4save.clicked.connect(s._save_map_bundle); l4.addWidget(s.b4save)
        s.b_ground = QPushButton("Detect Ramps (RANSAC)"); s.b_ground.setStyleSheet(s._B("#059669"))
        s.b_ground.setToolTip(
            "Run RANSAC ground segmentation to detect ramps.\n"
            "Results are overlaid on the obstacle map:\n"
            "  Green = robot can pass (within slope limits)\n"
            "  Red = robot cannot pass (exceeds limits)\n"
            "Labels show measured angle."
        )
        s.b_ground.clicked.connect(s._run_ground_analysis); l4.addWidget(s.b_ground)
        # ── Ramp Management ──
        ramp_row1 = QHBoxLayout()
        s.b_select_ramp = QPushButton("Select Ramp Area"); s.b_select_ramp.setStyleSheet(s._B_secondary())
        s.b_select_ramp.setToolTip("Draw a rectangle on the obstacle map to mark a ramp region.")
        s.b_select_ramp.clicked.connect(s._start_ramp_selection)
        ramp_row1.addWidget(s.b_select_ramp)
        s.b_mark_slope = QPushButton("Add Ramp"); s.b_mark_slope.setStyleSheet(s._B("#059669"))
        s.b_mark_slope.setToolTip("Add the selected area as a ramp — enter its slope angle.")
        s.b_mark_slope.clicked.connect(s._mark_slope_manual)
        ramp_row1.addWidget(s.b_mark_slope)
        s.b_remove_ramp = QPushButton("Remove Last"); s.b_remove_ramp.setStyleSheet(s._B_danger())
        s.b_remove_ramp.setToolTip("Remove the last manually added ramp.")
        s.b_remove_ramp.clicked.connect(s._remove_last_manual_ramp)
        ramp_row1.addWidget(s.b_remove_ramp)
        l4.addLayout(ramp_row1)

        # Ramp list display
        s.ramp_list = QListWidget()
        s.ramp_list.setMaximumHeight(80)
        s.ramp_list.setStyleSheet("QListWidget{font-size:11px;border:1px solid #d1d5db;border-radius:4px}")
        l4.addWidget(s.ramp_list)

        ramp_row2 = QHBoxLayout()
        s.b_toggle_ramps = QPushButton("Show/Hide Ramps"); s.b_toggle_ramps.setCheckable(True); s.b_toggle_ramps.setChecked(True)
        s.b_toggle_ramps.setStyleSheet(
            "QPushButton{background:#f8f9fa;color:#6b7280;border:1px solid #d1d5db;border-radius:4px;padding:4px 10px}"
            "QPushButton:checked{background:#dbeafe;color:#2563eb;border-color:#2563eb}")
        s.b_toggle_ramps.setToolTip("Toggle ramp overlay visibility.")
        s.b_toggle_ramps.toggled.connect(s._toggle_ramp_visibility)
        ramp_row2.addWidget(s.b_toggle_ramps)
        s.b_manual_only = QPushButton("Manual Ramps Only"); s.b_manual_only.setCheckable(True)
        s.b_manual_only.setStyleSheet(s.b_toggle_ramps.styleSheet())
        s.b_manual_only.setToolTip("Show only manually marked ramps. Disable all auto-detected ramps.")
        s.b_manual_only.toggled.connect(s._toggle_manual_only)
        ramp_row2.addWidget(s.b_manual_only)
        l4.addLayout(ramp_row2)

        # Track manual ramps separately
        s._manual_ramps = []
        l4.addWidget(QLabel("The floor is auto-detected. Obstacle max height is relative to the detected floor level."))
        l4.addWidget(QLabel("Outputs are cached in a temporary session folder unless you explicitly save them."))
        g4.setLayout(l4)
        section4 = CollapsibleSection(2, "Generate 2D Map", g4); ll.addWidget(section4)

        # ── Edit Map ──
        edit_title = "Edit Obstacle Map" if s._v3_mode else "Edit Traversable Ground"
        edit_desc = ("Draw or erase on the obstacle map. Apply to save changes." if s._v3_mode
                     else "Draw or erase on the traversable ground map. Apply to save changes for RII computation.")
        g_edit = QGroupBox(edit_title); l_edit = QVBoxLayout(); l_edit.setSpacing(6)
        l_edit.addWidget(QLabel(edit_desc))
        edit_row1 = QHBoxLayout()
        s.btn_edit_draw = QPushButton("Draw"); s.btn_edit_draw.setCheckable(True)
        s.btn_edit_draw.setStyleSheet("QPushButton{background:#ffffff;color:#6b7280;border:1px solid #d1d5db;border-radius:4px;padding:4px 10px}"
                                       "QPushButton:checked{background:#dbeafe;color:#2563eb;border-color:#2563eb}")
        s.btn_edit_erase = QPushButton("Erase"); s.btn_edit_erase.setCheckable(True)
        s.btn_edit_erase.setStyleSheet(s.btn_edit_draw.styleSheet())
        s.btn_edit_draw.clicked.connect(lambda: s._toggle_edit_mode("draw"))
        s.btn_edit_erase.clicked.connect(lambda: s._toggle_edit_mode("erase"))
        edit_row1.addWidget(s.btn_edit_draw); edit_row1.addWidget(s.btn_edit_erase)
        l_edit.addLayout(edit_row1)
        edit_row2 = QHBoxLayout()
        edit_row2.addWidget(QLabel("Brush:"))
        s.edit_brush_shape = QComboBox(); s.edit_brush_shape.addItems(["Circle", "Rectangle", "Free Draw"])
        s.edit_brush_shape.currentTextChanged.connect(s._on_brush_shape_changed)
        edit_row2.addWidget(s.edit_brush_shape)
        # Circle / Free Draw size
        s._brush_size_label = QLabel("Radius (m):")
        edit_row2.addWidget(s._brush_size_label)
        s.edit_brush_size = QDoubleSpinBox(); s.edit_brush_size.setRange(0.01, 9999.0); s.edit_brush_size.setValue(0.50); s.edit_brush_size.setDecimals(2); s.edit_brush_size.setSingleStep(0.1)
        s.edit_brush_size.valueChanged.connect(s._update_brush_size_px)
        edit_row2.addWidget(s.edit_brush_size)
        # Rectangle W/H
        s._brush_w_label = QLabel("W (m):")
        edit_row2.addWidget(s._brush_w_label)
        s.edit_brush_w = QDoubleSpinBox(); s.edit_brush_w.setRange(0.01, 9999.0); s.edit_brush_w.setValue(0.50); s.edit_brush_w.setDecimals(2); s.edit_brush_w.setSingleStep(0.1)
        s.edit_brush_w.valueChanged.connect(s._update_brush_rect_px)
        edit_row2.addWidget(s.edit_brush_w)
        s._brush_h_label = QLabel("H (m):")
        edit_row2.addWidget(s._brush_h_label)
        s.edit_brush_h = QDoubleSpinBox(); s.edit_brush_h.setRange(0.01, 9999.0); s.edit_brush_h.setValue(0.50); s.edit_brush_h.setDecimals(2); s.edit_brush_h.setSingleStep(0.1)
        s.edit_brush_h.valueChanged.connect(s._update_brush_rect_px)
        edit_row2.addWidget(s.edit_brush_h)
        # Initially show circle controls, hide rectangle controls
        s._brush_w_label.hide(); s.edit_brush_w.hide()
        s._brush_h_label.hide(); s.edit_brush_h.hide()
        l_edit.addLayout(edit_row2)
        s.edit_ref_overlay = QCheckBox("Show Obstacle Map Reference")
        s.edit_ref_overlay.setStyleSheet("QCheckBox{color:#6b7280;font-size:11px}")
        s.edit_ref_overlay.setToolTip("Overlay the obstacle map semi-transparently so you can match the traversability map to it")
        s.edit_ref_overlay.toggled.connect(lambda v: s.mw.set_reference_overlay_visible(v))
        l_edit.addWidget(s.edit_ref_overlay)

        edit_row3 = QHBoxLayout()
        s.btn_edit_apply = QPushButton("Apply to Obstacle Map" if s._v3_mode else "Apply to Traversability Map")
        s.btn_edit_apply.setStyleSheet(s._B())
        s.btn_edit_apply.clicked.connect(s._apply_trav_edit)
        s.btn_edit_revert = QPushButton("Revert")
        s.btn_edit_revert.setStyleSheet(s._B_danger())
        s.btn_edit_revert.clicked.connect(s._revert_trav_edit)
        edit_row3.addWidget(s.btn_edit_apply); edit_row3.addWidget(s.btn_edit_revert)
        l_edit.addLayout(edit_row3)
        g_edit.setLayout(l_edit); l4.addWidget(g_edit)

        # Step 3
        g5 = QWidget(); l5 = QVBoxLayout(); l5.setSpacing(8)
        hm = QHBoxLayout()
        s.e_pgm = QLineEdit(""); s.e_pgm.setPlaceholderText("Auto from Step 2 or browse")
        bm = s._mk_browse_btn("Browse for .pgm file", s._browse_pgm)
        hm.addWidget(QLabel(".pgm:")); hm.addWidget(s.e_pgm, 1); hm.addWidget(bm); l5.addLayout(hm)
        hy = QHBoxLayout()
        s.e_yaml = QLineEdit(""); s.e_yaml.setPlaceholderText("Auto from .pgm path or browse")
        by = s._mk_browse_btn("Browse for .yaml file", s._browse_yaml)
        hy.addWidget(QLabel(".yaml:")); hy.addWidget(s.e_yaml, 1); hy.addWidget(by); l5.addLayout(hy)

        sh = QHBoxLayout()
        sh.addWidget(QLabel("Selection:"))
        s.sel_mode = QComboBox()
        s.sel_mode.addItems(["Rectangle", "Spline / Freeform"])
        sh.addWidget(s.sel_mode, 1)
        l5.addLayout(sh)
        s.bsel = QPushButton("✎  Select Area on Floor Mask (optional)")
        s.bsel.setStyleSheet(s._B()); s.bsel.clicked.connect(s._enable_sel); l5.addWidget(s.bsel)
        s.slbl = QLabel(""); s.slbl.setStyleSheet("color:#2563eb;font-size:11px"); l5.addWidget(s.slbl)
        mh = QHBoxLayout()
        mh.addWidget(QLabel("Mode:"))
        s.rii_mode = QComboBox()
        s.rii_mode.addItems(["Without Path Planner", "With Path Planner"])
        s.rii_mode.currentIndexChanged.connect(s._toggle_planner_combo)
        mh.addWidget(s.rii_mode, 1)
        l5.addLayout(mh)
        ph = QHBoxLayout()
        ph.addWidget(QLabel("Planner:"))
        s.planner_combo = QComboBox()
        s.planner_combo.addItems(PLANNER_NAMES)
        ph.addWidget(s.planner_combo, 1)
        s.planner_row = QWidget()
        s.planner_row.setLayout(ph)
        s.planner_row.hide()
        l5.addWidget(s.planner_row)

        # Reference robot
        l5.addWidget(QLabel("─── Reference Robot (comparison only) ───"))
        l5.addWidget(QLabel("Optional benchmark footprint for comparison only. It does not set the RII Horizontal denominator."))
        rsh = QHBoxLayout(); rsh.addWidget(QLabel("Shape:"))
        s.rs = QComboBox(); s.rs.addItems(["circular", "rectangular"])
        s.rs.currentIndexChanged.connect(lambda: s._toggle_shape('r')); rsh.addWidget(s.rs, 1); l5.addLayout(rsh)
        s.rc = QWidget(); rc_ = QHBoxLayout(s.rc); rc_.setContentsMargins(0, 0, 0, 0)
        rc_.addWidget(QLabel("Radius (m):"))
        s.rr = QDoubleSpinBox(); s.rr.setRange(.001, 5); s.rr.setValue(.035); s.rr.setDecimals(3)
        rc_.addWidget(s.rr, 1); l5.addWidget(s.rc)
        s.rrc = QWidget(); rrc_ = QHBoxLayout(s.rrc); rrc_.setContentsMargins(0, 0, 0, 0)
        rrc_.addWidget(QLabel("W (m):")); s.rw = QDoubleSpinBox(); s.rw.setRange(.01, 5); s.rw.setValue(.07); s.rw.setDecimals(3); rrc_.addWidget(s.rw, 1)
        rrc_.addWidget(QLabel("L (m):")); s.rl = QDoubleSpinBox(); s.rl.setRange(.01, 5); s.rl.setValue(.07); s.rl.setDecimals(3); rrc_.addWidget(s.rl, 1)
        s.rrc.hide(); l5.addWidget(s.rrc)
        s.bref = QPushButton("Run Reference"); s.bref.setStyleSheet(s._B())
        s.bref.clicked.connect(s._run_ref); l5.addWidget(s.bref)
        s.lref = QLabel(""); s.lref.setStyleSheet("color:#2563eb;font-size:13px;font-weight:bold"); l5.addWidget(s.lref)
        s.lref_note = QLabel("")
        s.lref_note.setWordWrap(True)
        s.lref_note.setStyleSheet("color:#6b7280;font-size:11px")
        l5.addWidget(s.lref_note)

        # Actual robot
        l5.addWidget(QLabel("─── Actual Robot (your real platform) ───"))
        l5.addWidget(QLabel("RII Horizontal = inflated accessible area / total floor area. 'With Path Planner' keeps only the largest connected inflated-accessible region."))
        ash = QHBoxLayout(); ash.addWidget(QLabel("Shape:"))
        s.as_ = QComboBox(); s.as_.addItems(["circular", "rectangular"]); s.as_.setCurrentIndex(1)
        s.as_.currentIndexChanged.connect(lambda: s._toggle_shape('a')); ash.addWidget(s.as_, 1); l5.addLayout(ash)
        s.ac = QWidget(); ac_ = QHBoxLayout(s.ac); ac_.setContentsMargins(0, 0, 0, 0)
        ac_.addWidget(QLabel("Radius (m):"))
        s.ar = QDoubleSpinBox(); s.ar.setRange(.01, 5); s.ar.setValue(.35); s.ar.setDecimals(3)
        ac_.addWidget(s.ar, 1); s.ac.hide(); l5.addWidget(s.ac)
        s.arc = QWidget(); arc_ = QHBoxLayout(s.arc); arc_.setContentsMargins(0, 0, 0, 0)
        arc_.addWidget(QLabel("W (m):")); s.aw = QDoubleSpinBox(); s.aw.setRange(.01, 5); s.aw.setValue(.6); s.aw.setDecimals(3); arc_.addWidget(s.aw, 1)
        arc_.addWidget(QLabel("L (m):")); s.al = QDoubleSpinBox(); s.al.setRange(.01, 5); s.al.setValue(.4); s.al.setDecimals(3); arc_.addWidget(s.al, 1)
        l5.addWidget(s.arc)
        start_row = QHBoxLayout()
        s.bset_start = QPushButton("Click Map to Set Start Point")
        s.bset_start.setStyleSheet(s._B_secondary())
        s.bset_start.clicked.connect(s._enable_start_pick)
        start_row.addWidget(s.bset_start)
        s.start_label = QLabel("No start point set (will use map center)")
        s.start_label.setStyleSheet("color:#6b7280;font-size:11px")
        start_row.addWidget(s.start_label, 1)
        l5.addLayout(start_row)
        s._act_start_world = None  # (x, y) world coords or None
        s.bact = QPushButton("Run Actual"); s.bact.setStyleSheet(s._B())
        s.bact.clicked.connect(s._run_act); l5.addWidget(s.bact)
        s.lact = QLabel(""); s.lact.setStyleSheet("color:#16a34a;font-size:13px;font-weight:bold"); l5.addWidget(s.lact)
        s.lact_note = QLabel("")
        s.lact_note.setWordWrap(True)
        s.lact_note.setStyleSheet("color:#6b7280;font-size:11px")
        l5.addWidget(s.lact_note)
        cov_hint = QLabel("Area tabs: colored = inflated accessible area, light = floor but inaccessible, dark = blocked after robot inflation.")
        cov_hint.setWordWrap(True)
        l5.addWidget(cov_hint)
        stc_hint = QLabel("Planner Path tab: reference path = blue, actual path = green when a path planner is selected.")
        stc_hint.setWordWrap(True)
        l5.addWidget(stc_hint)

        # RII display
        s.riif = QFrame()
        s.riif.setStyleSheet("QFrame{background:#f0f4ff;border:1px solid #d1d5db;border-radius:8px;padding:12px}")
        rf = QVBoxLayout(s.riif)
        rf.setSpacing(6)
        s.riit = QLabel("RII Horizontal"); s.riit.setAlignment(Qt.AlignCenter)
        s.riit.setStyleSheet("color:#1f2937;font-size:18px;font-weight:bold;letter-spacing:1px")
        rf.addWidget(s.riit)
        s.riiv = QLabel("—"); s.riiv.setAlignment(Qt.AlignCenter)
        s.riiv.setMinimumHeight(58)
        s.riiv.setWordWrap(False)
        s.riiv.setStyleSheet("color:#2563eb;font-size:42px;font-weight:bold;font-family:monospace")
        rf.addWidget(s.riiv)
        s.riis = QLabel(""); s.riis.setAlignment(Qt.AlignCenter)
        s.riis.setStyleSheet("color:#6b7280;font-size:12px"); rf.addWidget(s.riis)
        s.riif.hide(); l5.addWidget(s.riif)

        g5.setLayout(l5)
        section5 = CollapsibleSection(3, "RII Horizontal", g5); ll.addWidget(section5)

        # Step 4 — RII Horizontal Analysis
        g6 = QWidget(); l6 = QVBoxLayout(); l6.setSpacing(8)
        l6.addWidget(QLabel(
            "Load a CloudCompare-labeled PCD or PLY (labels 0-15 from the paper taxonomy)\n"
            "to explain the RII Horizontal gap by fixation group and recommended interventions."
        ))

        hsem = QHBoxLayout()
        s.e_sem_pcd = QLineEdit(""); s.e_sem_pcd.setPlaceholderText("Path to labeled .pcd or .ply file")
        bsem = s._mk_browse_btn("Browse for labelled cloud", s._browse_sem_pcd)
        hsem.addWidget(QLabel("Labeled Cloud:")); hsem.addWidget(s.e_sem_pcd, 1); hsem.addWidget(bsem)
        l6.addLayout(hsem)

        s.sem_status = QLabel(""); s.sem_status.setStyleSheet("color:#6b7280;font-size:11px")
        l6.addWidget(s.sem_status)

        s.bsem_3d = QPushButton("View Semantic Labels in 3D Viewer")
        s.bsem_3d.setStyleSheet(
            "QPushButton{background:#2563eb;color:#ffffff;border:none;border-radius:4px;"
            "padding:8px;font-weight:bold;font-size:11px}"
            "QPushButton:hover{border:2px solid white}"
            "QPushButton:disabled{background:#333;color:#666}")
        s.bsem_3d.setToolTip("Show the loaded semantic point cloud in the 3D Viewer, colored by label class")
        s.bsem_3d.clicked.connect(s._show_semantic_3d)
        s.bsem_3d.setEnabled(False)
        l6.addWidget(s.bsem_3d)

        def _sem_btn_style(color=None):
            return (f"QPushButton {{ background: #ffffff; color: #2563eb; border: 1px solid #d1d5db; border-radius: 4px; "
                    f"padding: 8px; font-weight: bold; font-size: 11px; }}"
                    f"QPushButton:hover {{ background: #f0f4ff; border-color: #2563eb; }}"
                    f"QPushButton:disabled {{ background: #f3f4f6; color: #9ca3af; }}")

        s.bsem = QPushButton("Analyze RII Horizontal"); s.bsem.setStyleSheet(s._B())
        s.bsem.clicked.connect(s._run_semantic_analysis); l6.addWidget(s.bsem)

        s.sem_prog_lbl = QLabel("")
        s.sem_prog_lbl.setStyleSheet("color:#2563eb;font-size:11px")
        s.sem_prog_lbl.hide()
        l6.addWidget(s.sem_prog_lbl)
        s.sem_prog = QProgressBar()
        s.sem_prog.setRange(0, 100)
        s.sem_prog.setValue(0)
        s.sem_prog.setTextVisible(True)
        s.sem_prog.setFormat("%p%")
        s.sem_prog.setMaximumHeight(18)
        s.sem_prog.hide()
        l6.addWidget(s.sem_prog)

        s.sem_candidate_status = QLabel("")
        s.sem_candidate_status.setWordWrap(True)
        s.sem_candidate_status.setStyleSheet("color:#6b7280;font-size:11px")
        s.sem_candidate_status.setText(
            "Run semantic analysis to populate removable-object candidates, filter them by fixation, and recompute an Optimised RII score."
        )
        l6.addWidget(s.sem_candidate_status)

        s.sem_candidate_hdr = QLabel("Object candidates to remove")
        s.sem_candidate_hdr.setStyleSheet("color:#1f2937;font-size:12px;font-weight:bold")
        s.sem_candidate_hdr.setWordWrap(True)
        l6.addWidget(s.sem_candidate_hdr)

        sem_filter_row = QHBoxLayout()
        s.sem_filter_lbl = QLabel("Filter:")
        sem_filter_row.addWidget(s.sem_filter_lbl)
        s.sem_filter = QComboBox()
        s.sem_filter.addItem("All Fixations", None)
        s.sem_filter.addItem("Portable", "Portable")
        s.sem_filter.addItem("Movable", "Movable")
        s.sem_filter.addItem("Semi-Fixed", "Semi-Fixed")
        s.sem_filter.currentIndexChanged.connect(s._apply_semantic_candidate_filter)
        s.sem_filter.setEnabled(False)
        s.sem_filter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sem_filter_row.addWidget(s.sem_filter)
        s.bsem_select_filtered = QPushButton("Select Filtered")
        s.bsem_select_filtered.setStyleSheet(_sem_btn_style())
        s.bsem_select_filtered.clicked.connect(s._select_filtered_semantic_candidates)
        s.bsem_select_filtered.setEnabled(False)
        sem_filter_row.addWidget(s.bsem_select_filtered)
        l6.addLayout(sem_filter_row)

        s.sem_candidate_list = QListWidget()
        s.sem_candidate_list.setMaximumHeight(170)
        s.sem_candidate_list.setStyleSheet(
            "QListWidget{background:#ffffff;border:1px solid #d1d5db;border-radius:6px;color:#1f2937;font-size:11px;}"
            "QListWidget::item{padding:4px;}"
        )
        s.sem_candidate_list.itemChanged.connect(s._semantic_candidate_selection_changed)
        s.sem_candidate_list.currentItemChanged.connect(s._semantic_candidate_current_changed)
        s.sem_candidate_list.itemClicked.connect(lambda item: s._semantic_candidate_current_changed(item, None))
        placeholder = QListWidgetItem("Run semantic analysis to populate removable-object candidates.")
        placeholder.setFlags(Qt.NoItemFlags)
        s.sem_candidate_list.addItem(placeholder)
        s.sem_candidate_list.setEnabled(False)
        l6.addWidget(s.sem_candidate_list)

        s.bsem_select_portable = QPushButton("Portable")
        s.bsem_select_portable.setStyleSheet(_sem_btn_style("#ffc857"))
        s.bsem_select_portable.clicked.connect(lambda: s._set_semantic_candidates_by_fixation({"Portable"}))
        s.bsem_select_portable.setEnabled(False)
        s.bsem_select_movable = QPushButton("Movable")
        s.bsem_select_movable.setStyleSheet(_sem_btn_style("#ff9f43"))
        s.bsem_select_movable.clicked.connect(lambda: s._set_semantic_candidates_by_fixation({"Movable"}))
        s.bsem_select_movable.setEnabled(False)
        s.bsem_select_semi_fixed = QPushButton("Semi-Fixed")
        s.bsem_select_semi_fixed.setStyleSheet(_sem_btn_style("#f368e0"))
        s.bsem_select_semi_fixed.clicked.connect(lambda: s._set_semantic_candidates_by_fixation({"Semi-Fixed"}))
        s.bsem_select_semi_fixed.setEnabled(False)
        s.bsem_select_portable_movable = QPushButton("Portable + Movable")
        s.bsem_select_portable_movable.setStyleSheet(_sem_btn_style("#ffcc66"))
        s.bsem_select_portable_movable.clicked.connect(lambda: s._set_semantic_candidates_by_fixation({"Portable", "Movable"}))
        s.bsem_select_portable_movable.setEnabled(False)
        s.bsem_select_all_candidates = QPushButton("All Removable")
        s.bsem_select_all_candidates.setStyleSheet(_sem_btn_style("#7ad9ff"))
        s.bsem_select_all_candidates.clicked.connect(lambda: s._set_semantic_candidates_by_fixation(set(SEMANTIC_REMOVABLE_FIXATIONS)))
        s.bsem_select_all_candidates.setEnabled(False)
        s.bsem_clear_candidates = QPushButton("Clear Selection")
        s.bsem_clear_candidates.setStyleSheet(_sem_btn_style("#7a8ba3"))
        s.bsem_clear_candidates.clicked.connect(s._clear_semantic_candidates)
        s.bsem_clear_candidates.setEnabled(False)
        for btn in (
            s.bsem_select_portable,
            s.bsem_select_movable,
            s.bsem_select_semi_fixed,
            s.bsem_select_portable_movable,
            s.bsem_select_all_candidates,
            s.bsem_clear_candidates,
        ):
            btn.setMinimumHeight(34)

        sem_btn_grid = QGridLayout()
        sem_btn_grid.setHorizontalSpacing(8)
        sem_btn_grid.setVerticalSpacing(8)
        sem_btn_grid.addWidget(s.bsem_select_portable, 0, 0)
        sem_btn_grid.addWidget(s.bsem_select_movable, 0, 1)
        sem_btn_grid.addWidget(s.bsem_select_semi_fixed, 1, 0)
        sem_btn_grid.addWidget(s.bsem_select_portable_movable, 1, 1)
        sem_btn_grid.addWidget(s.bsem_select_all_candidates, 2, 0)
        sem_btn_grid.addWidget(s.bsem_clear_candidates, 2, 1)
        l6.addLayout(sem_btn_grid)

        s.bsem_recompute = QPushButton("Recompute Optimised RII")
        s.bsem_recompute.setStyleSheet(s._B())
        s.bsem_recompute.clicked.connect(s._recompute_semantic_improvement)
        s.bsem_recompute.setEnabled(False)
        l6.addWidget(s.bsem_recompute)

        s.bsem_bottleneck = QPushButton("Analyse Bottlenecks and Relocation")
        s.bsem_bottleneck.setStyleSheet(s._B_secondary())
        s.bsem_bottleneck.clicked.connect(s._run_bottleneck_analysis)
        s.bsem_bottleneck.setEnabled(False)
        s.bsem_bottleneck.setToolTip(
            "Score top candidates by true area gain (BFS reachability),\n"
            "identify chokepoint objects, and suggest relocation zones."
        )
        l6.addWidget(s.bsem_bottleneck)
        s.bottleneck_status = QLabel("")
        s.bottleneck_status.setWordWrap(True)
        s.bottleneck_status.setStyleSheet("color:#6b7280;font-size:11px")
        l6.addWidget(s.bottleneck_status)

        s.bsem_optimize = QPushButton("Optimize Layout (Multi-Object)")
        s.bsem_optimize.setStyleSheet(s._B())
        s.bsem_optimize.clicked.connect(s._run_optimization)
        s.bsem_optimize.setEnabled(False)
        s.bsem_optimize.setToolTip(
            "Greedy multi-object relocation: iteratively move bottleneck objects\n"
            "to low-traffic spots, re-evaluating after each move."
        )
        l6.addWidget(s.bsem_optimize)
        s.optimization_status = QLabel("")
        s.optimization_status.setWordWrap(True)
        s.optimization_status.setStyleSheet("color:#6b7280;font-size:11px")
        l6.addWidget(s.optimization_status)

        s.sem_riif = QFrame()
        s.sem_riif.setStyleSheet("QFrame{background:#f0f4ff;border:1px solid #d1d5db;border-radius:8px;padding:12px}")
        sem_rf = QVBoxLayout(s.sem_riif)
        sem_rf.setSpacing(6)
        # ── Side-by-side comparison: Current → Optimised ──
        sem_cmp = QHBoxLayout(); sem_cmp.setSpacing(12)
        # Current RII (left)
        cur_col = QVBoxLayout(); cur_col.setSpacing(2)
        s.sem_cur_title = QLabel("Current"); s.sem_cur_title.setAlignment(Qt.AlignCenter)
        s.sem_cur_title.setStyleSheet("color:#6b7280;font-size:12px;font-weight:bold")
        cur_col.addWidget(s.sem_cur_title)
        s.sem_cur_val = QLabel("—"); s.sem_cur_val.setAlignment(Qt.AlignCenter)
        s.sem_cur_val.setStyleSheet("color:#6b7280;font-size:28px;font-weight:bold;font-family:monospace")
        cur_col.addWidget(s.sem_cur_val)
        sem_cmp.addLayout(cur_col)
        # Arrow + Delta (center)
        delta_col = QVBoxLayout(); delta_col.setSpacing(2)
        s.sem_arrow = QLabel("→"); s.sem_arrow.setAlignment(Qt.AlignCenter)
        s.sem_arrow.setStyleSheet("color:#2563eb;font-size:22px;font-weight:bold")
        delta_col.addWidget(s.sem_arrow)
        s.sem_delta = QLabel(""); s.sem_delta.setAlignment(Qt.AlignCenter)
        s.sem_delta.setStyleSheet("color:#16a34a;font-size:14px;font-weight:bold")
        delta_col.addWidget(s.sem_delta)
        sem_cmp.addLayout(delta_col)
        # Optimised RII (right)
        opt_col = QVBoxLayout(); opt_col.setSpacing(2)
        s.sem_opt_title = QLabel("Optimised"); s.sem_opt_title.setAlignment(Qt.AlignCenter)
        s.sem_opt_title.setStyleSheet("color:#2563eb;font-size:12px;font-weight:bold")
        opt_col.addWidget(s.sem_opt_title)
        s.sem_riiv = QLabel("—"); s.sem_riiv.setAlignment(Qt.AlignCenter)
        s.sem_riiv.setStyleSheet("color:#2563eb;font-size:28px;font-weight:bold;font-family:monospace")
        opt_col.addWidget(s.sem_riiv)
        sem_cmp.addLayout(opt_col)
        sem_rf.addLayout(sem_cmp)
        # Subtitle line
        s.sem_riis = QLabel(""); s.sem_riis.setAlignment(Qt.AlignCenter)
        s.sem_riis.setWordWrap(True)
        s.sem_riis.setStyleSheet("color:#6b7280;font-size:11px")
        sem_rf.addWidget(s.sem_riis)
        s.sem_riif.hide()
        l6.addWidget(s.sem_riif)

        s.sem_layered_status = QTextEdit(); s.sem_layered_status.setReadOnly(True)
        s.sem_layered_status.setMaximumHeight(145); s.sem_layered_status.setMinimumHeight(90)
        s.sem_layered_status.setStyleSheet(
            "background:#f8f9fa;border:1px solid #d1d5db;border-radius:6px;"
            "color:#1f2937;font-family:monospace;font-size:11px;padding:8px"
        )
        s.sem_layered_status.hide()
        l6.addWidget(s.sem_layered_status)

        # Report button (opens in separate window)
        s._sem_report_html = ""
        s.btn_show_report = QPushButton("View Full Report")
        s.btn_show_report.setStyleSheet(s._B_secondary())
        s.btn_show_report.clicked.connect(s._show_report_window)
        s.btn_show_report.hide()
        l6.addWidget(s.btn_show_report)
        # Hidden widget kept for backward compatibility — report stored but not displayed inline
        s.sem_report = QTextEdit(); s.sem_report.setReadOnly(True); s.sem_report.hide()

        g6.setLayout(l6)
        section6 = CollapsibleSection(4, "RII Horizontal Analysis", g6); ll.addWidget(section6)

        # Step 5 — RII Vertical (Wall Paint Coverage)
        g7 = QWidget(); l7 = QVBoxLayout(); l7.setSpacing(8)
        l7.addWidget(QLabel(
            "Compute wall surface reachability from accessible floor using STVL raycasting.\n"
            "Requires: labelled point cloud (Step 4) + RII Horizontal (Step 3)."
        ))

        # Wall height band
        whb = QHBoxLayout()
        whb.addWidget(QLabel("Wall min h (m):"))
        s.rv_wall_min_h = QDoubleSpinBox(); s.rv_wall_min_h.setRange(0.0, 10.0); s.rv_wall_min_h.setValue(0.40); s.rv_wall_min_h.setDecimals(2); s.rv_wall_min_h.setSingleStep(0.1)
        whb.addWidget(s.rv_wall_min_h)
        whb.addWidget(QLabel("max h (m):"))
        s.rv_wall_max_h = QDoubleSpinBox(); s.rv_wall_max_h.setRange(0.1, 20.0); s.rv_wall_max_h.setValue(2.00); s.rv_wall_max_h.setDecimals(2); s.rv_wall_max_h.setSingleStep(0.1)
        whb.addWidget(s.rv_wall_max_h)
        l7.addLayout(whb)

        # Raycasting params
        rcp = QHBoxLayout()
        rcp.addWidget(QLabel("Voxel (m):"))
        s.rv_voxel = QDoubleSpinBox(); s.rv_voxel.setRange(0.01, 1.0); s.rv_voxel.setValue(0.05); s.rv_voxel.setDecimals(3); s.rv_voxel.setSingleStep(0.01)
        rcp.addWidget(s.rv_voxel)
        rcp.addWidget(QLabel("Reach (m):"))
        s.rv_reach = QDoubleSpinBox(); s.rv_reach.setRange(0.1, 5.0); s.rv_reach.setValue(1.0); s.rv_reach.setDecimals(2); s.rv_reach.setSingleStep(0.1)
        rcp.addWidget(s.rv_reach)
        rcp.addWidget(QLabel("Angle (°):"))
        s.rv_angle = QDoubleSpinBox(); s.rv_angle.setRange(1.0, 45.0); s.rv_angle.setValue(10.0); s.rv_angle.setDecimals(1); s.rv_angle.setSingleStep(5.0)
        rcp.addWidget(s.rv_angle)
        l7.addLayout(rcp)

        # Paint tool params
        ptp = QHBoxLayout()
        ptp.addWidget(QLabel("Paint width (m):"))
        s.rv_paint_w = QDoubleSpinBox(); s.rv_paint_w.setRange(0.01, 9999.0); s.rv_paint_w.setValue(0.25); s.rv_paint_w.setDecimals(2); s.rv_paint_w.setSingleStep(0.05)
        ptp.addWidget(s.rv_paint_w)
        ptp.addWidget(QLabel("Vertical span (m):"))
        s.rv_paint_vspan = QDoubleSpinBox(); s.rv_paint_vspan.setRange(0.01, 9999.0); s.rv_paint_vspan.setValue(0.30); s.rv_paint_vspan.setDecimals(2); s.rv_paint_vspan.setSingleStep(0.05)
        ptp.addWidget(s.rv_paint_vspan)
        ptp.addWidget(QLabel("Sweep step (m):"))
        s.rv_sweep = QDoubleSpinBox(); s.rv_sweep.setRange(0.01, 9999.0); s.rv_sweep.setValue(0.20); s.rv_sweep.setDecimals(2); s.rv_sweep.setSingleStep(0.05)
        ptp.addWidget(s.rv_sweep)
        l7.addLayout(ptp)

        # Sampling params
        smp = QHBoxLayout()
        smp.addWidget(QLabel("Ground stride (px):"))
        s.rv_stride = QSpinBox(); s.rv_stride.setRange(1, 20); s.rv_stride.setValue(3)
        smp.addWidget(s.rv_stride)
        smp.addWidget(QLabel("Max samples (count):"))
        s.rv_max_samples = QSpinBox(); s.rv_max_samples.setRange(1000, 200000); s.rv_max_samples.setValue(60000); s.rv_max_samples.setSingleStep(5000)
        smp.addWidget(s.rv_max_samples)
        l7.addLayout(smp)

        # Wall label IDs
        wli = QHBoxLayout()
        wli.addWidget(QLabel("Wall label IDs:"))
        s.rv_wall_ids = QLineEdit("1"); s.rv_wall_ids.setPlaceholderText("Comma-separated label IDs treated as wall (e.g. 1)")
        s.rv_wall_ids.setToolTip("Semantic label IDs to treat as wall surface. Default: 1 (Wall)")
        wli.addWidget(s.rv_wall_ids, 1)
        l7.addLayout(wli)

        # ── Wall segment detection & selection ──
        s.brv_detect = QPushButton("🔍  Detect Wall Segments")
        s.brv_detect.setStyleSheet(s._B())
        s.brv_detect.clicked.connect(s._detect_wall_segments)
        l7.addWidget(s.brv_detect)

        s.rv_wall_status = QLabel(""); s.rv_wall_status.setStyleSheet("color:#6b7280;font-size:11px")
        l7.addWidget(s.rv_wall_status)

        l7.addWidget(QLabel("Wall segments (click to visualise in 3D, check to include in RII_V):"))
        s.rv_wall_list = QListWidget()
        s.rv_wall_list.setMaximumHeight(150)
        s.rv_wall_list.setStyleSheet(
            "QListWidget{background:#ffffff;color:#1f2937;border:1px solid #d1d5db;font-size:11px}"
            "QListWidget::item{padding:3px}"
            "QListWidget::item:selected{background:#dbeafe;color:#2563eb}"
        )
        s.rv_wall_list.currentItemChanged.connect(s._rv_wall_current_changed)
        s.rv_wall_list.itemChanged.connect(s._rv_wall_check_changed)
        l7.addWidget(s.rv_wall_list)

        rv_wbtn = QHBoxLayout()
        s.brv_sel_all = QPushButton("Select All"); s.brv_sel_all.setFixedHeight(28)
        s.brv_sel_all.setStyleSheet("QPushButton{background:#2563eb;color:#ffffff;border:none;border-radius:4px;padding:4px 12px;font-weight:bold;font-size:11px}")
        s.brv_sel_all.clicked.connect(lambda: s._rv_wall_select_all(True))
        rv_wbtn.addWidget(s.brv_sel_all)
        s.brv_sel_none = QPushButton("Clear All"); s.brv_sel_none.setFixedHeight(28)
        s.brv_sel_none.setStyleSheet("QPushButton{background:#ffffff;color:#6b7280;border:1px solid #d1d5db;border-radius:4px;padding:4px 12px;font-weight:bold;font-size:11px}")
        s.brv_sel_none.clicked.connect(lambda: s._rv_wall_select_all(False))
        rv_wbtn.addWidget(s.brv_sel_none)
        l7.addLayout(rv_wbtn)

        # Combined RII gamma
        gml = QHBoxLayout()
        gml.addWidget(QLabel("Combined γ:"))
        s.rv_gamma = QDoubleSpinBox(); s.rv_gamma.setRange(0.0, 1.0); s.rv_gamma.setValue(0.50); s.rv_gamma.setDecimals(2); s.rv_gamma.setSingleStep(0.1)
        s.rv_gamma.setToolTip(
            "γ (gamma) controls the balance between Operational Efficiency (OE) and Surface Continuity (SC) "
            "in the combined RII score.\n\n"
            "Formula:  Combined = TCR × (γ · OE + (1-γ) · SC)\n\n"
            "• γ = 1.0  →  Combined depends only on OE (how much floor can reach walls)\n"
            "• γ = 0.0  →  Combined depends only on SC (how contiguous the painted walls are)\n"
            "• γ = 0.5  →  Equal weight to OE and SC (default)\n\n"
            "Where:\n"
            "  TCR = Task Coverage Rate = painted wall area / total wall area\n"
            "  OE  = Operational Efficiency = fraction of floor that can reach walls\n"
            "  SC  = Surface Continuity = largest contiguous painted region / total painted"
        )
        gml.addWidget(s.rv_gamma)
        # Gamma explanation label
        s._gamma_info = QLabel(
            "<span style='color:#6b7a8d;font-size:10px'>"
            "γ balances Operational Efficiency (OE) vs Surface Continuity (SC).<br>"
            "<b>Combined = TCR × (γ · OE + (1-γ) · SC)</b><br>"
            "γ→1: prioritise floor reachability &nbsp;|&nbsp; γ→0: prioritise wall contiguity &nbsp;|&nbsp; γ=0.5: equal weight"
            "</span>"
        )
        s._gamma_info.setWordWrap(True)
        gml.addWidget(s._gamma_info, 1)
        l7.addLayout(gml)

        s.brv = QPushButton("Compute RII Vertical (selected walls)"); s.brv.setStyleSheet(s._B())
        s.brv.clicked.connect(s._run_rii_vertical); l7.addWidget(s.brv)

        s.rv_prog_lbl = QLabel(""); s.rv_prog_lbl.setStyleSheet("color:#2563eb;font-size:11px"); s.rv_prog_lbl.hide()
        l7.addWidget(s.rv_prog_lbl)
        s.rv_prog = QProgressBar(); s.rv_prog.setRange(0, 100); s.rv_prog.setValue(0)
        s.rv_prog.setTextVisible(True); s.rv_prog.setFormat("%p%"); s.rv_prog.setMaximumHeight(18); s.rv_prog.hide()
        l7.addWidget(s.rv_prog)

        # ── RII Vertical result card ──
        s.rv_riif = QFrame()
        s.rv_riif.setStyleSheet("QFrame{background:#f0f4ff;border:1px solid #d1d5db;border-radius:8px;padding:12px}")
        rv_rf = QVBoxLayout(s.rv_riif); rv_rf.setSpacing(6)
        s.rv_riit = QLabel("RII Vertical — Task Coverage Rate (TCR)"); s.rv_riit.setAlignment(Qt.AlignCenter)
        s.rv_riit.setStyleSheet("color:#1f2937;font-size:18px;font-weight:bold;letter-spacing:1px")
        rv_rf.addWidget(s.rv_riit)
        s.rv_riiv = QLabel("—"); s.rv_riiv.setAlignment(Qt.AlignCenter); s.rv_riiv.setMinimumHeight(48)
        s.rv_riiv.setStyleSheet("color:#2563eb;font-size:36px;font-weight:bold;font-family:monospace")
        rv_rf.addWidget(s.rv_riiv)
        s.rv_riis = QLabel(""); s.rv_riis.setAlignment(Qt.AlignCenter); s.rv_riis.setWordWrap(True)
        s.rv_riis.setStyleSheet("color:#6b7280;font-size:11px")
        rv_rf.addWidget(s.rv_riis)
        s.rv_riif.hide()
        l7.addWidget(s.rv_riif)

        # ── Combined RII card ──
        s.rv_combf = QFrame()
        s.rv_combf.setStyleSheet("QFrame{background:#f0f4ff;border:1px solid #d1d5db;border-radius:8px;padding:12px}")
        rv_cf = QVBoxLayout(s.rv_combf); rv_cf.setSpacing(6)

        # Side-by-side: RII_H | RII_V | Combined
        rv_cmp = QHBoxLayout(); rv_cmp.setSpacing(8)
        # RII_H (left)
        hcol = QVBoxLayout(); hcol.setSpacing(2)
        s.rv_ch_title = QLabel("RII Horizontal"); s.rv_ch_title.setAlignment(Qt.AlignCenter)
        s.rv_ch_title.setStyleSheet("color:#1f2937;font-size:11px;font-weight:bold")
        hcol.addWidget(s.rv_ch_title)
        s.rv_ch_val = QLabel("—"); s.rv_ch_val.setAlignment(Qt.AlignCenter)
        s.rv_ch_val.setStyleSheet("color:#2563eb;font-size:22px;font-weight:bold;font-family:monospace")
        hcol.addWidget(s.rv_ch_val)
        rv_cmp.addLayout(hcol)
        # RII_V (middle)
        vcol = QVBoxLayout(); vcol.setSpacing(2)
        s.rv_cv_title = QLabel("RII Vertical"); s.rv_cv_title.setAlignment(Qt.AlignCenter)
        s.rv_cv_title.setStyleSheet("color:#1f2937;font-size:11px;font-weight:bold")
        vcol.addWidget(s.rv_cv_title)
        s.rv_cv_val = QLabel("—"); s.rv_cv_val.setAlignment(Qt.AlignCenter)
        s.rv_cv_val.setStyleSheet("color:#2563eb;font-size:22px;font-weight:bold;font-family:monospace")
        vcol.addWidget(s.rv_cv_val)
        rv_cmp.addLayout(vcol)
        # Combined (right)
        ccol = QVBoxLayout(); ccol.setSpacing(2)
        s.rv_cc_title = QLabel("Combined"); s.rv_cc_title.setAlignment(Qt.AlignCenter)
        s.rv_cc_title.setStyleSheet("color:#1f2937;font-size:11px;font-weight:bold")
        ccol.addWidget(s.rv_cc_title)
        s.rv_cc_val = QLabel("—"); s.rv_cc_val.setAlignment(Qt.AlignCenter)
        s.rv_cc_val.setStyleSheet("color:#2563eb;font-size:22px;font-weight:bold;font-family:monospace")
        ccol.addWidget(s.rv_cc_val)
        rv_cmp.addLayout(ccol)
        rv_cf.addLayout(rv_cmp)

        # Detail line
        s.rv_comb_detail = QLabel(""); s.rv_comb_detail.setAlignment(Qt.AlignCenter)
        s.rv_comb_detail.setWordWrap(True)
        s.rv_comb_detail.setStyleSheet("color:#6b7280;font-size:11px")
        rv_cf.addWidget(s.rv_comb_detail)

        # Formula reminder
        s.rv_comb_formula = QLabel("Combined = Task Coverage Rate (TCR) × (γ · Operational Efficiency (OE) + (1-γ) · Surface Continuity (SC))\nWeighted = 0.5 · RII_H + 0.5 · RII_V")
        s.rv_comb_formula.setAlignment(Qt.AlignCenter); s.rv_comb_formula.setWordWrap(True)
        s.rv_comb_formula.setStyleSheet("color:#444;font-size:10px;font-style:italic")
        rv_cf.addWidget(s.rv_comb_formula)
        s.rv_combf.hide()
        l7.addWidget(s.rv_combf)

        g7.setLayout(l7)
        section7 = CollapsibleSection(5, "RII Vertical — Wall Reachability", g7); ll.addWidget(section7)

        ll.addStretch()
        ls.setWidget(lw); sp.addWidget(ls)
        s._sidebar_scroll = ls
        s._group_boxes = [section1, section4, section5, section6, section7]
        # Initial: Step 1 active, rest pending
        s._sync_section_states(0)

        # RIGHT panel
        rw = QWidget(); rl = QVBoxLayout(rw); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)

        _tab_ss = ("QTabBar{background:transparent;border:none}"
                    "QTabBar::tab{background:#f8f9fa;color:#6b7280;border:1px solid #d1d5db;"
                    "border-radius:5px;padding:4px 12px;margin:2px 2px;font-size:11px}"
                    "QTabBar::tab:selected{background:#dbeafe;color:#2563eb;border-color:#2563eb}")
        if s._v3_mode:
            _tab_names = [PRIMARY_SELECTION_VIEW, "3D Viewer", "Reference Coverage", "Actual Coverage", "Compare", "Planner Path", "Semantic", "Vertical Coverage"]
        else:
            _tab_names = [PRIMARY_SELECTION_VIEW, "Traversable Ground", "3D Viewer", "Reference Coverage", "Actual Coverage", "Compare", "Planner Path", "Semantic", "Vertical Coverage"]
        _tab_tooltips = {
            PRIMARY_SELECTION_VIEW: "2D projection of the raw point cloud.\nBlack = obstacle hit, White = free space.\nUsed as the base map for RII computation and area selection.",
            "Traversable Ground": "Terrain analysis sidecar.\nShows which ground cells pass slope and step-height checks.\nUsed to exclude non-traversable terrain from accessible area.",
        }

        # Primary tab bar row (with split-view toggle)
        tab_row = QHBoxLayout(); tab_row.setContentsMargins(0, 0, 0, 0); tab_row.setSpacing(0)
        s.view_tab_bar = QTabBar()
        s.view_tab_bar.setMovable(True)
        s.view_tab_bar.setExpanding(False)
        s.view_tab_bar.setFixedHeight(32)
        s.view_tab_bar.setStyleSheet(_tab_ss)
        s.vb = {}
        for nm in _tab_names:
            idx = s.view_tab_bar.addTab(nm)
            if nm in _tab_tooltips:
                s.view_tab_bar.setTabToolTip(idx, _tab_tooltips[nm])
            s.vb[nm] = idx
        s.view_tab_bar.setCurrentIndex(0)
        s.view_tab_bar.currentChanged.connect(lambda idx: s._switch_view(s.view_tab_bar.tabText(idx)))
        tab_row.addWidget(s.view_tab_bar, 1)
        s.btn_split_view = QPushButton("Split View")
        s.btn_split_view.setCheckable(True)
        s.btn_split_view.setFixedHeight(28)
        s.btn_split_view.setStyleSheet(
            "QPushButton{background:#ffffff;color:#6b7280;border:1px solid #d1d5db;"
            "border-radius:4px;padding:2px 12px;font-size:11px;font-weight:bold}"
            "QPushButton:checked{background:#dbeafe;color:#2563eb;border-color:#2563eb}")
        s.btn_split_view.setToolTip("Toggle side-by-side split view — drag the divider to resize each panel")
        s.btn_split_view.clicked.connect(s._toggle_split_view)
        tab_row.addWidget(s.btn_split_view)
        rl.addLayout(tab_row)

        s.prog = QProgressBar(); s.prog.setTextVisible(False); s.prog.setMaximumHeight(4)
        rl.addWidget(s.prog)

        # ── Z-clip toolbar (visible only in 3D views) ──
        s._zclip_bar = QWidget()
        zch = QHBoxLayout(s._zclip_bar); zch.setContentsMargins(4, 2, 4, 2); zch.setSpacing(6)
        s._zclip_cb = QCheckBox("Remove roof / ceiling")
        s._zclip_cb.setToolTip("Hide points above a Z threshold to reveal the ground underneath.")
        zch.addWidget(s._zclip_cb)
        zch.addWidget(QLabel("Max Z (m):"))
        s._zclip_spin = QDoubleSpinBox()
        s._zclip_spin.setRange(-50.0, 50.0)
        s._zclip_spin.setDecimals(2)
        s._zclip_spin.setSingleStep(0.1)
        s._zclip_spin.setValue(3.0)
        s._zclip_spin.setToolTip("Absolute Z height above which points are hidden.")
        s._zclip_spin.setEnabled(False)
        zch.addWidget(s._zclip_spin)
        zch.addStretch()
        s._zclip_bar.hide()
        rl.addWidget(s._zclip_bar)

        def _on_zclip_toggle(checked):
            s._zclip_spin.setEnabled(checked)
            if checked:
                s._apply_zclip()
            else:
                # Restore full cloud
                active = s._active_view_name()
                if active in s._clouds:
                    s.pcw.set_cloud(s._clouds[active])
        s._zclip_cb.toggled.connect(_on_zclip_toggle)
        s._zclip_spin.valueChanged.connect(lambda _: s._apply_zclip() if s._zclip_cb.isChecked() else None)

        # Primary viewer (always present)
        s.view_stack = QStackedWidget()
        s.mw = MapW(); s.mw.sel_changed.connect(s._on_sel); s.view_stack.addWidget(s.mw)
        s.mw.hover_coords.connect(s._on_hover_coords)
        s.mw.hover_coords.connect(s._on_hover_coords)
        s.mw.start_picked.connect(s._on_start_picked)
        s.pcw = PointCloudW()
        if hasattr(s.pcw, "gl_failed"):
            s.pcw.gl_failed.connect(s._fallback_point_cloud_viewer)
        s.view_stack.addWidget(s.pcw)

        # Split viewer container (splitter holds primary + secondary)
        s._split_splitter = QSplitter(Qt.Horizontal)
        s._split_splitter.setChildrenCollapsible(False)
        s._split_splitter.setHandleWidth(10)
        s._split_splitter.setStyleSheet(
            "QSplitter::handle{background:#d1d5db;border-radius:3px}"
            "QSplitter::handle:hover{background:#2563eb}"
        )
        s._split_splitter.addWidget(s.view_stack)

        # Secondary viewer panel (built once, shown/hidden)
        s._split_panel = QWidget()
        _sp_layout = QVBoxLayout(s._split_panel); _sp_layout.setContentsMargins(0, 0, 0, 0); _sp_layout.setSpacing(0)
        s._split_tab_bar = QTabBar()
        s._split_tab_bar.setMovable(True)
        s._split_tab_bar.setExpanding(False)
        s._split_tab_bar.setFixedHeight(32)
        s._split_tab_bar.setStyleSheet(_tab_ss)
        for nm in _tab_names:
            idx2 = s._split_tab_bar.addTab(nm)
            if nm in _tab_tooltips:
                s._split_tab_bar.setTabToolTip(idx2, _tab_tooltips[nm])
        s._split_tab_bar.setCurrentIndex(0)
        s._split_tab_bar.currentChanged.connect(lambda idx: s._switch_split_view(s._split_tab_bar.tabText(idx)))
        _sp_layout.addWidget(s._split_tab_bar)
        s._split_view_stack = QStackedWidget()
        s._split_mw = MapW()
        s._split_view_stack.addWidget(s._split_mw)
        s._split_pcw = PointCloudW()
        if hasattr(s._split_pcw, "gl_failed"):
            s._split_pcw.gl_failed.connect(lambda reason: s._log(f"Split 3D viewer: {reason}", "warn"))
        s._split_view_stack.addWidget(s._split_pcw)
        _sp_layout.addWidget(s._split_view_stack, 1)
        s._split_splitter.addWidget(s._split_panel)
        s._split_panel.hide()

        rl.addWidget(s._split_splitter, 1)

        # ── Toolbar: coordinates + export buttons ──
        toolbar = QHBoxLayout()
        s.coord_label = QLabel(""); s.coord_label.setStyleSheet("color:#6b7280;font-family:monospace;font-size:11px")
        toolbar.addWidget(s.coord_label, 1)
        s.btn_save_image = QPushButton("Save View as PNG")
        s.btn_save_image.setStyleSheet(s._B_secondary())
        s.btn_save_image.clicked.connect(s._save_current_view)
        toolbar.addWidget(s.btn_save_image)
        s.btn_export_csv = QPushButton("Export CSV")
        s.btn_export_csv.setStyleSheet(s._B_secondary())
        s.btn_export_csv.clicked.connect(s._export_results_csv)
        toolbar.addWidget(s.btn_export_csv)
        rl.addLayout(toolbar)

        s.log_box = QTextEdit(); s.log_box.setReadOnly(True); s.log_box.setMaximumHeight(140)
        rl.addWidget(s.log_box)
        sp.addWidget(rw); sp.setStretchFactor(0, 0); sp.setStretchFactor(1, 1)
        sp.setSizes([470, 910])

    # ── Selection ──
    def _enable_sel(s):
        p, _ = s._get_pgm()
        if not p: return
        if BLOCKED_MAP_VIEW not in s._imgs: s._load_map(p)
        mode = "freeform" if hasattr(s, "sel_mode") and s.sel_mode.currentIndex() == 1 else "rectangle"
        s._switch_view(PRIMARY_SELECTION_VIEW); s.mw.clear_sel(); s.mw.enable_sel(mode)
        if mode == "freeform":
            s._log(f"Draw a freeform loop on {PRIMARY_SELECTION_VIEW.lower()} to select area.", "gold")
        else:
            s._log(f"Drag on {PRIMARY_SELECTION_VIEW.lower()} to select a rectangular area.", "gold")

    def _on_sel(s, r):
        if not r: return
        p, y = s._get_pgm()
        if not p: return
        yd = parse_yaml(y)
        res = yd['resolution']
        bounds = selection_bounds_px(r)
        if bounds is None:
            return
        x1, y1, x2, y2 = bounds
        wm = (x2 - x1 + 1) * res
        hm = (y2 - y1 + 1) * res
        mask = selection_mask_from_display(r, s._map_w, s._map_h)
        area_m2 = 0.0 if mask is None else float(mask.sum()) * res * res
        if selection_kind(r) == "freeform":
            s.slbl.setText(f"Selected freeform: {area_m2:.1f}m² (bbox {wm:.1f}×{hm:.1f}m)")
            s._log(f"Freeform area: {area_m2:.1f}m²", "gold")
        else:
            s.slbl.setText(f"Selected rectangle: {wm:.1f}×{hm:.1f}m = {area_m2:.1f}m²")
            s._log(f"Rectangle area: {wm:.1f}×{hm:.1f}m", "gold")
        center = s._selection_center_world(y)
        if center is not None:
            s._log(f"Selection center: ({center[0]:.2f}, {center[1]:.2f})", "info")

    # ── Steps 1-4 ──
    def _step1(s):
        p = s.e_in.text().strip()
        if not os.path.isfile(p): QMessageBox.warning(s, "Error", p); return
        s.b1.setEnabled(False)
        s._set_worker_status("Loading point cloud…", busy=True)
        s._stepper.set_active(0); s._sync_section_states(0)
        w = ViewW(p, "3D Viewer")
        w.log.connect(s._log)
        w.loaded.connect(lambda cloud: s._set_cloud("3D Viewer", cloud))
        def _d(ok, _msg):
            s.b1.setEnabled(True)
            s._set_worker_status("Idle")
            if ok:
                s._stepper.set_status(0, "complete")
                s._group_boxes[0].setStatus("complete")
        w.done.connect(_d)
        s._wk.append(w); w.start()

    def _step2(s):
        if not hasattr(s, 'e_out') or not hasattr(s, 'cp') or not hasattr(s, 'b2'):
            s._log("Step 2 (cleanup) is not available in this pipeline version.", "warn")
            return
        pi = s.e_in.text().strip()
        if not os.path.isfile(pi): QMessageBox.warning(s, "Error", pi); return
        od = s.e_out.text(); os.makedirs(od, exist_ok=True)
        po = os.path.join(od, filtered_point_cloud_filename(pi))
        args = " ".join(f"--{k} {v.value()}" for k, v in s.cp.items())
        s.b2.setEnabled(False)
        cmd = (
            f"cd {shlex.quote(PRECLEAN)} && "
            f"python3 pre_map.py --in {shlex.quote(pi)} --out {shlex.quote(po)} {args}"
        )
        w = ShellW(cmd, "Clean", False)
        w.log.connect(s._log); w.done.connect(lambda *_: s.b2.setEnabled(True))
        s._wk.append(w); w.start()

    def _step3(s):
        if not hasattr(s, 'e_out'):
            s._log("Step 3 (view clean cloud) is not available in this pipeline version.", "warn")
            return
        p = resolve_point_cloud_path(
            s.e_out.text(),
            filtered_point_cloud_stem_candidates(s.e_in.text().strip()),
        )
        if not os.path.isfile(p): QMessageBox.warning(s, "Error", p); return
        w = ViewW(p, "Clean Cloud")
        w.log.connect(s._log)
        w.loaded.connect(lambda cloud: s._set_cloud("Clean Cloud", cloud))
        w.done.connect(lambda *_: None)
        s._wk.append(w); w.start()

    def _step4(s):
        try:
            p, src_label = s._selected_map_source_path()
        except FileNotFoundError as exc:
            QMessageBox.warning(s, "Error", str(exc))
            return
        s.b4.setEnabled(False); s.prog.setValue(0); sd = s.e_save.text()
        mz = s.oz1.value()
        xz = s.oz2.value()

        # Show floor detection info before generating
        try:
            pts = load_xyz_points(p)
            preset = estimate_ground_preserving_preset(pts)
            floor_z = preset["floor_anchor_z"]
            # Auto-set a sensible Z-clip ceiling so the user can toggle roof removal
            if hasattr(s, '_zclip_spin') and not s._zclip_cb.isChecked():
                s._zclip_spin.setValue(round(floor_z + max(xz, 1.0), 2))
            if mz >= 0 and xz >= 0:
                abs_min = floor_z + mz
                abs_max = floor_z + xz
                s.floor_status.setText(
                    f"Floor detected at {floor_z:.3f} m  |  "
                    f"Obstacle slice: [{abs_min:.3f}, {abs_max:.3f}] m (absolute)  |  "
                    f"min_z ignores obstacles below {mz:.2f} m above floor (robot climbs over)  |  "
                    f"max_slope={s.t_slope.value():.0f} deg, max_step={s.t_step.value():.2f} m"
                )
            else:
                s.floor_status.setText(
                    f"Floor detected at {floor_z:.3f} m  |  "
                    f"Absolute Z mode: [{mz:.3f}, {xz:.3f}] m"
                )
            s._log(
                f"Floor auto-detected at {floor_z:.3f} m. "
                f"{'Floor-relative' if mz >= 0 and xz >= 0 else 'Absolute'} "
                f"Z slice: [{abs_min if mz >= 0 and xz >= 0 else mz:.3f}, "
                f"{abs_max if mz >= 0 and xz >= 0 else xz:.3f}] m",
                "info",
            )
        except Exception:
            s.floor_status.setText("")

        # Apply noise filter if enabled — 2D density filter
        if s.cb_noise_filter.isChecked():
            radius = s.filter_radius.value()
            min_nb = s.filter_min_nb.value()
            s._log(f"[Filter] 2D density filter (radius={radius}m, min_density={min_nb})...", "info")
            try:
                from src.pcd_package.pcd_package.pcd_tools import (
                    load_xyz_points as _load_pts, write_xyz_pcd,
                )
                from collections import Counter
                raw_pts = _load_pts(p)
                n_before = raw_pts.shape[0]
                # 2D density: count points in XY grid neighborhood
                cell = max(radius / 2.5, 0.1)
                search_r = max(1, int(round(radius / cell)))
                xy = raw_pts[:, :2]
                gx = np.floor(xy[:, 0] / cell).astype(np.int32)
                gy = np.floor(xy[:, 1] / cell).astype(np.int32)
                cell_keys = list(zip(gx.tolist(), gy.tolist()))
                counts = Counter(cell_keys)
                density = np.zeros(n_before, dtype=np.int32)
                for i, key in enumerate(cell_keys):
                    cx, cy = key
                    total = 0
                    for dx in range(-search_r, search_r + 1):
                        for dy in range(-search_r, search_r + 1):
                            total += counts.get((cx + dx, cy + dy), 0)
                    density[i] = total
                keep = density >= min_nb
                filtered = raw_pts[keep]
                n_after = filtered.shape[0]
                n_removed = n_before - n_after
                pct = 100 * n_removed / max(n_before, 1)
                filtered_path = os.path.join(sd, "filtered_cloud.pcd")
                write_xyz_pcd(filtered_path, filtered)
                p = filtered_path
                s.filter_status.setText(f"Removed {n_removed:,} pts ({pct:.1f}%)")
                s._log(f"[Filter] {n_before:,} → {n_after:,} points ({n_removed:,} removed, {pct:.1f}%)", "success")
            except Exception as e:
                s._log(f"[Filter] Failed: {e} — using unfiltered cloud", "warn")
                s.filter_status.setText("Filter failed")

        s._log(
            f"Generating Step 2 map from the {src_label}: {os.path.basename(p)}",
            "info",
        )

        # Branch: multi-level vs. single-level build
        if hasattr(s, "cb_multi_level") and s.cb_multi_level.isChecked():
            s._step4_multilevel(p, sd)
            return

        w = MapBuildW(p, sd, mz, xz, s.t_slope.value(), s.t_step.value(), v3_mode=s._v3_mode, min_points_per_cell=s.min_pts_cell.value())
        w.log.connect(s._log); w.prog.connect(s.prog.setValue)
        s._set_worker_status("Building 2D map…", busy=True)
        s._stepper.set_active(1); s._sync_section_states(1)
        def done(ok, msg):
            s.b4.setEnabled(True)
            s._set_worker_status("Idle")
            if ok:
                pgm = os.path.join(sd, "map.pgm"); s.e_pgm.setText(pgm)
                yml = os.path.join(sd, "map.yaml"); s.e_yaml.setText(yml)
                s._log(f"Map: {pgm}", "success")
                if os.path.isfile(pgm): s._load_map(pgm)
                s._stepper.set_status(1, "complete")
                s._group_boxes[1].setStatus("complete")
            else:
                s._stepper.set_status(1, "pending")
                s._group_boxes[1].setStatus("pending")
        w.done.connect(done); s._wk.append(w); w.start()

    def _step4_multilevel(s, pcd_path, sd):
        """Detect levels, then chain one MapBuildW per level sequentially."""
        from core.level_detection import detect_floor_levels, summarize_levels
        s._set_worker_status("Detecting floor levels…", busy=True)
        try:
            pts = load_xyz_points(pcd_path)
            levels = detect_floor_levels(pts)
        except Exception as e:
            s._log(f"[Levels] Detection failed: {e}", "warn")
            s.b4.setEnabled(True); s._set_worker_status("Idle"); return

        s._log(summarize_levels(levels), "info")
        if len(levels) <= 1:
            s._log("[Levels] Falling back to single-level build.", "warn")
            s.cb_multi_level.setChecked(False)
            s._step4()
            return

        s._multi_level_outputs = []  # list of (level_index, pgm_path)
        s._stepper.set_active(1); s._sync_section_states(1)

        def start_level(i):
            if i >= len(levels):
                _all_done()
                return
            lv = levels[i]
            name = f"map_level{lv.index}"
            s._log(
                f"[Levels] Building {name}: z=[{lv.z_low:+.2f}, {lv.z_high:+.2f}] m "
                f"(anchor {lv.anchor_z:+.2f} m, {lv.point_count:,} pts)",
                "info",
            )
            s._set_worker_status(
                f"Building level {i + 1} of {len(levels)}…", busy=True
            )
            w = MapBuildW(
                pcd_path, sd, lv.z_low, lv.z_high,
                s.t_slope.value(), s.t_step.value(),
                v3_mode=s._v3_mode,
                min_points_per_cell=s.min_pts_cell.value(),
                out_prefix_name=name,
                absolute_z=True,
            )
            w.log.connect(s._log); w.prog.connect(s.prog.setValue)
            def lv_done(ok, msg):
                if ok:
                    pgm = os.path.join(sd, name + ".pgm")
                    s._multi_level_outputs.append((lv.index, pgm))
                    s._log(f"[Levels] Completed {name} → {pgm}", "success")
                else:
                    s._log(f"[Levels] {name} failed: {msg}", "warn")
                start_level(i + 1)
            w.done.connect(lv_done)
            s._wk.append(w)
            w.start()

        def _all_done():
            s.b4.setEnabled(True)
            s._set_worker_status("Idle")
            if not s._multi_level_outputs:
                s._stepper.set_status(1, "pending")
                s._group_boxes[1].setStatus("pending")
                return
            # Load the lowest level into the main view
            s._multi_level_outputs.sort(key=lambda t: t[0])
            first_pgm = s._multi_level_outputs[0][1]
            yml = first_pgm.replace(".pgm", ".yaml")
            s.e_pgm.setText(first_pgm); s.e_yaml.setText(yml)
            if os.path.isfile(first_pgm):
                s._load_map(first_pgm)
            s._log(
                f"[Levels] Loaded level 0 into view. "
                f"Use the .pgm browse button in Step 3 to switch: "
                + ", ".join(os.path.basename(p) for _, p in s._multi_level_outputs),
                "success",
            )
            s._stepper.set_status(1, "complete")
            s._group_boxes[1].setStatus("complete")

        start_level(0)

    # ── Ground Analysis (RANSAC ramp detection) ──
    def _run_ground_analysis(s):
        p = s.e_in.text().strip()
        if not os.path.isfile(p):
            QMessageBox.warning(s, "Error", f"Point cloud not found:\n{p}")
            return
        s.b_ground.setEnabled(False)
        s.prog.setValue(0)
        s._log("Starting RANSAC ground analysis...", "info")

        w = GroundAnalysisW(p, s.t_slope.value(), s.t_step.value())
        w.log.connect(s._log)
        w.prog.connect(s.prog.setValue)
        w.result_ready.connect(s._on_ground_result)
        s._set_worker_status("Detecting ramps (RANSAC)…", busy=True)

        def done(ok, msg):
            s.b_ground.setEnabled(True)
            s._set_worker_status("Idle")
            if ok:
                s._log("Ground analysis complete.", "success")
            else:
                s._log(f"Ground analysis failed: {msg}", "warn")

        w.done.connect(done)
        s._wk.append(w)
        w.start()

    def _on_ground_result(s, result):
        """Handle GroundAnalysisResult — build traversability map from obstacle map + ramps."""
        s._ground_result = result

        pgm = s.e_pgm.text()
        if not pgm or not os.path.isfile(pgm):
            s._log("[Ground] No obstacle map loaded. Generate a 2D map first.", "warn")
            return

        map_w, map_h, pixels = parse_pgm(pgm)
        yaml_path = pgm.replace(".pgm", ".yaml")
        if not os.path.isfile(yaml_path):
            s._log("[Ground] Map YAML not found.", "warn")
            return
        yd = parse_yaml(yaml_path)
        map_res = float(yd["resolution"])
        map_ox = float(yd["origin"][0])
        map_oy = float(yd["origin"][1])

        # ── Step 1: Copy obstacle map as the base traversability map ──
        trav_pixels = pixels.copy()  # exact copy of obstacle map
        trav_2d = trav_pixels.reshape(map_h, map_w)

        # ── Step 2: Draw ramps onto the traversability map ──
        # Non-traversable ramps → mark as blocked (value 0)
        # Traversable ramps → keep as-is (already free in obstacle map)
        analysis_origin = result.grid_origin
        analysis_cell = result.cell_size
        blocked_cells = 0

        for t in result.transitions:
            if t.cells.shape[0] == 0:
                continue
            if t.traversable:
                continue  # Robot can pass — leave obstacle map as-is

            # Block non-traversable ramp cells
            for ci in range(t.cells.shape[0]):
                row_a, col_a = int(t.cells[ci, 0]), int(t.cells[ci, 1])
                wx = analysis_origin[0] + (col_a + 0.5) * analysis_cell
                wy = analysis_origin[1] + (row_a + 0.5) * analysis_cell
                px = int((wx - map_ox) / map_res)
                py = map_h - 1 - int((wy - map_oy) / map_res)
                if 0 <= px < map_w and 0 <= py < map_h:
                    trav_2d[py, px] = 0
                    blocked_cells += 1

        # ── Step 3: Save traversability PGM (same location as obstacle map) ──
        sd = s.e_save.text()
        trav_pgm = os.path.join(sd, "map_traversable.pgm")
        trav_yaml = os.path.join(sd, "map_traversable.yaml")
        # Write PGM
        with open(trav_pgm, "wb") as f:
            header = f"P5\n{map_w} {map_h}\n255\n".encode("ascii")
            f.write(header)
            f.write(trav_pixels.tobytes())
        # Copy YAML
        import shutil
        shutil.copy2(yaml_path, trav_yaml)
        s._log(f"[Ground] Traversability map saved: {trav_pgm}", "success")

        # ── Step 4: Show in Traversable Ground tab ──
        trav_qi = QImage(trav_pixels.data, map_w, map_h, map_w, QImage.Format_Grayscale8)
        trav_qi._np_ref = trav_pixels
        s._set_img("Traversable Ground", trav_qi.copy())

        # ── Step 5: Build RGBA overlay for obstacle map view (ramp visualization) ──
        rgba = np.zeros((map_h, map_w, 4), dtype=np.uint8)
        labels = []

        for t in result.transitions:
            if t.cells.shape[0] == 0:
                continue
            color = (0, 200, 80, 160) if t.traversable else (220, 40, 40, 160)

            # Check if this is a manual ramp (cells are already in pixel coords)
            is_manual = getattr(t, '_is_manual', False)

            for ci in range(t.cells.shape[0]):
                row_a, col_a = int(t.cells[ci, 0]), int(t.cells[ci, 1])
                if is_manual:
                    # Manual ramp: cells are already pixel (py, px)
                    py, px = row_a, col_a
                else:
                    # Auto-detected: convert from analysis grid to map pixels
                    wx = analysis_origin[0] + (col_a + 0.5) * analysis_cell
                    wy = analysis_origin[1] + (row_a + 0.5) * analysis_cell
                    px = int((wx - map_ox) / map_res)
                    py = map_h - 1 - int((wy - map_oy) / map_res)

                if 0 <= px < map_w and 0 <= py < map_h:
                    rgba[py, px] = color
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            nr, nc = py + dr, px + dc
                            if 0 <= nr < map_h and 0 <= nc < map_w:
                                rgba[nr, nc] = color

            # Label position
            if is_manual:
                lpx = float(np.mean(t.cells[:, 1]))
                lpy = float(np.mean(t.cells[:, 0]))
            else:
                cx = analysis_origin[0] + float(np.mean(t.cells[:, 1])) * analysis_cell
                cy = analysis_origin[1] + float(np.mean(t.cells[:, 0])) * analysis_cell
                lpx = (cx - map_ox) / map_res
                lpy = map_h - 1 - (cy - map_oy) / map_res
            text = f"{t.angle_deg:.1f}°"
            label_color = (0, 220, 80) if t.traversable else (255, 60, 60)
            labels.append((lpx, lpy, text, label_color))

        qi = QImage(rgba.data, map_w, map_h, 4 * map_w, QImage.Format_RGBA8888)
        qi._np_ref = rgba
        s.mw.set_transition_overlay(qi.copy(), labels)

        # Switch to obstacle map view to show ramp overlay
        s._switch_view(PRIMARY_SELECTION_VIEW)

        pass_count = sum(1 for t in result.transitions if t.traversable)
        total = len(result.transitions)
        s._log(f"[Ground] Obstacle map + ramps → traversability map", "success")
        s._log(f"[Ground] {pass_count}/{total} ramps passable, "
               f"{blocked_cells} cells blocked", "success")
        s._log(f"[Ground] View 'Traversable Ground' tab for the clean map", "info")

    # ── Ramp Selection & Visibility ──
    def _start_ramp_selection(s):
        """Enable rectangle selection on the obstacle map for ramp marking."""
        p, _ = s._get_pgm()
        if not p:
            QMessageBox.warning(s, "Error", "Generate a 2D map first (Step 2).")
            return
        if BLOCKED_MAP_VIEW not in s._imgs:
            s._load_map(p)
        s._switch_view(PRIMARY_SELECTION_VIEW)
        s.mw.clear_sel()
        s.mw.enable_sel("rectangle")
        s._log("Draw a rectangle over the ramp area, then click 'Add Ramp'.", "gold")

    def _toggle_ramp_visibility(s, visible):
        """Show or hide the ramp overlay on the obstacle map."""
        if visible:
            s._refresh_ramp_overlay()
        else:
            s.mw.clear_transition_overlay()

    def _toggle_manual_only(s, checked):
        """Toggle between showing all ramps vs only manual ramps."""
        if checked:
            s._log("[Ramps] Showing only manually marked ramps.", "info")
        else:
            s._log("[Ramps] Showing all ramps (auto + manual).", "info")
        if s.b_toggle_ramps.isChecked():
            s._refresh_ramp_overlay()

    def _refresh_ramp_overlay(s):
        """Rebuild and display the ramp overlay based on current settings."""
        from core.ground_analysis import GroundAnalysisResult
        if s.b_manual_only.isChecked():
            # Show only manual ramps
            if not s._manual_ramps:
                s.mw.clear_transition_overlay()
                return
            manual_result = GroundAnalysisResult(
                levels=[], transitions=list(s._manual_ramps),
                cell_size=0.05, grid_origin=(0.0, 0.0), grid_shape=(s._map_h, s._map_w),
            )
            s._on_ground_result(manual_result)
        else:
            # Show all (auto + manual)
            if hasattr(s, '_ground_result') and s._ground_result is not None:
                s._on_ground_result(s._ground_result)
            elif s._manual_ramps:
                manual_result = GroundAnalysisResult(
                    levels=[], transitions=list(s._manual_ramps),
                    cell_size=0.05, grid_origin=(0.0, 0.0), grid_shape=(s._map_h, s._map_w),
                )
                s._on_ground_result(manual_result)

    def _update_ramp_list(s):
        """Update the ramp list widget."""
        s.ramp_list.clear()
        for i, t in enumerate(s._manual_ramps):
            status = "PASS" if t.traversable else "FAIL"
            item = QListWidgetItem(f"Ramp {i+1}: {t.angle_deg:.1f}° [{status}] "
                                   f"({int(abs(t.end_xy[0]-t.start_xy[0])/0.05)}x"
                                   f"{int(abs(t.end_xy[1]-t.start_xy[1])/0.05)} px)")
            s.ramp_list.addItem(item)

    # ── Manual Slope Marking ──
    def _mark_slope_manual(s):
        """Add the selected area as a ramp with user-specified angle."""
        from PyQt5.QtWidgets import QInputDialog

        if not s.mw.sel:
            s._log("First click 'Select Ramp Area' and draw a rectangle, then click 'Add Ramp'.", "warn")
            return

        angle, ok = QInputDialog.getDouble(
            s, "Add Ramp", "Enter the slope angle (degrees):",
            value=5.0, min=0.1, max=89.0, decimals=1,
        )
        if not ok:
            return

        sel = s.mw.sel
        if sel["kind"] == "rect":
            x1, y1, x2, y2 = sel["rect"]
        elif sel["kind"] == "freeform":
            pts = sel["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            return

        cells_list = []
        for py in range(int(y1), int(y2) + 1):
            for px in range(int(x1), int(x2) + 1):
                cells_list.append((py, px))
        if not cells_list:
            return

        from core.ground_analysis import TransitionInfo, GroundAnalysisResult

        cells = np.array(cells_list, dtype=np.int32)
        traversable = angle <= s.t_slope.value()

        pgm = s.e_pgm.text()
        yaml_path = pgm.replace(".pgm", ".yaml") if pgm else ""
        map_res = 0.05
        map_ox = map_oy = 0.0
        if os.path.isfile(yaml_path):
            yd = parse_yaml(yaml_path)
            map_res = float(yd["resolution"])
            map_ox = float(yd["origin"][0])
            map_oy = float(yd["origin"][1])
        map_h = s._map_h
        wx1 = map_ox + x1 * map_res
        wy1 = map_oy + (map_h - 1 - y1) * map_res
        wx2 = map_ox + x2 * map_res
        wy2 = map_oy + (map_h - 1 - y2) * map_res

        t = TransitionInfo(
            transition_id=len(s._manual_ramps),
            type="ramp",
            level_from=0, level_to=0,
            start_xy=(wx1, wy1),
            end_xy=(wx2, wy2),
            angle_deg=angle,
            width_m=abs(wx2 - wx1),
            length_m=abs(wy2 - wy1),
            step_height_m=0.0,
            height_from=0.0, height_to=0.0,
            cells=cells,
            traversable=traversable,
        )
        t._is_manual = True  # Flag for overlay rendering (cells are pixel coords)

        s._manual_ramps.append(t)

        # Also add to ground_result for RII integration
        if not hasattr(s, '_ground_result') or s._ground_result is None:
            s._ground_result = GroundAnalysisResult(
                levels=[], transitions=[],
                cell_size=map_res,
                grid_origin=(map_ox, map_oy),
                grid_shape=(s._map_h, s._map_w),
            )
        t.transition_id = len(s._ground_result.transitions)
        s._ground_result.transitions.append(t)

        status = "PASS" if traversable else "FAIL"
        s._log(f"[Manual] [{status}] Added ramp {len(s._manual_ramps)}: {angle:.1f}° "
               f"({int(x2-x1)}x{int(y2-y1)} px)", "success")

        s._update_ramp_list()
        s.mw.clear_sel()

        # Refresh overlay
        if s.b_toggle_ramps.isChecked():
            s._refresh_ramp_overlay()

    def _remove_last_manual_ramp(s):
        """Remove the last manually added ramp."""
        if not s._manual_ramps:
            s._log("No manual ramps to remove.", "info")
            return
        removed = s._manual_ramps.pop()
        # Also remove from ground_result if it's there
        if hasattr(s, '_ground_result') and s._ground_result is not None:
            s._ground_result.transitions = [
                t for t in s._ground_result.transitions
                if not (t.type == "ramp" and t.start_xy == removed.start_xy and t.end_xy == removed.end_xy)
            ]
        s._log(f"[Manual] Removed ramp: {removed.angle_deg:.1f}°", "info")
        s._update_ramp_list()
        if s.b_toggle_ramps.isChecked():
            s._refresh_ramp_overlay()

    # ── Traversability Map Editing ──
    def _toggle_edit_mode(s, mode):
        """Activate draw/erase editing on the map."""
        pgm = s.e_pgm.text()
        if not pgm:
            pgm = os.path.join(s.e_save.text(), "map.pgm")

        if s._v3_mode:
            # V3: edit the obstacle map directly
            if not os.path.isfile(pgm):
                QMessageBox.warning(s, "No Map", "Generate a 2D map first (Step 2).")
                s.btn_edit_draw.setChecked(False)
                s.btn_edit_erase.setChecked(False)
                return
            s._switch_view(PRIMARY_SELECTION_VIEW)
            if not s.mw._edit_active:
                w, h, pixels = parse_pgm(pgm)
                overlay = pixels.reshape(h, w)
                s.mw.enable_edit(overlay)
                s._log("Edit mode enabled — draw or erase on the obstacle map.", "info")
        else:
            # V1/V2: edit traversability sidecar
            trav_path = traversability_sidecar_path(pgm)
            if not os.path.isfile(trav_path):
                QMessageBox.warning(s, "No Traversability Map",
                                    "Generate a 2D map first (Step 2) to create a traversability sidecar.")
                s.btn_edit_draw.setChecked(False)
                s.btn_edit_erase.setChecked(False)
                return
            s._switch_view("Traversable Ground")
            if not s.mw._edit_active:
                w, h, pixels = parse_pgm(trav_path)
                overlay = pixels.reshape(h, w)
                s.mw.enable_edit(overlay)
                if BLOCKED_MAP_VIEW in s._imgs:
                    s.mw.set_reference_overlay(s._imgs[BLOCKED_MAP_VIEW])
                s._log("Edit mode enabled — draw or erase on the traversable ground map.", "info")

        s.mw.set_edit_mode(mode)
        s.btn_edit_draw.setChecked(mode == "draw")
        s.btn_edit_erase.setChecked(mode == "erase")

    def _get_map_resolution(s):
        """Return the map resolution (m/px) from the loaded YAML, or a default."""
        y = s.e_yaml.text()
        if y and os.path.isfile(y):
            try:
                yd = parse_yaml(y)
                return float(yd['resolution'])
            except Exception:
                pass
        return 0.05  # default fallback

    def _update_brush_size_px(s, meters):
        """Convert brush radius from metres to pixels and update MapW."""
        res = s._get_map_resolution()
        px = max(1, int(round(meters / res)))
        s.mw.set_brush_size(px)

    def _update_brush_rect_px(s, _=None):
        """Convert rectangle W/H from metres to pixels and update MapW."""
        res = s._get_map_resolution()
        hw = max(1, int(round(s.edit_brush_w.value() / res)))
        hh = max(1, int(round(s.edit_brush_h.value() / res)))
        s.mw.set_brush_rect_size(hw, hh)

    def _on_brush_shape_changed(s, text):
        """Show/hide the appropriate size controls based on brush shape."""
        s.mw.set_brush_shape("free" if text == "Free Draw" else text.lower())
        is_rect = (text == "Rectangle")
        # Show rectangle W/H controls
        s._brush_w_label.setVisible(is_rect)
        s.edit_brush_w.setVisible(is_rect)
        s._brush_h_label.setVisible(is_rect)
        s.edit_brush_h.setVisible(is_rect)
        # Show circle/free radius control
        s._brush_size_label.setVisible(not is_rect)
        s.edit_brush_size.setVisible(not is_rect)

    def _apply_trav_edit(s):
        """Write the edited overlay back to the map PGM file."""
        if not s.mw._edit_active or s.mw._edit_overlay is None:
            QMessageBox.information(s, "Nothing to Apply", "Enable edit mode first (Draw or Erase).")
            return
        pgm = s.e_pgm.text()
        if not pgm:
            pgm = os.path.join(s.e_save.text(), "map.pgm")

        overlay = s.mw.get_edit_overlay()
        h, w = overlay.shape

        if s._v3_mode:
            # V3: save directly to the obstacle map
            save_path = pgm
            with open(save_path, 'wb') as f:
                f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
                f.write(overlay.tobytes())
            s._log(f"Obstacle map saved: {save_path}", "success")
            s.mw.disable_edit()
            s.btn_edit_draw.setChecked(False)
            s.btn_edit_erase.setChecked(False)
            # Reload the map
            if os.path.isfile(pgm):
                s._load_map(pgm)
            s._switch_view(PRIMARY_SELECTION_VIEW)
        else:
            # V1/V2: save to traversability sidecar
            trav_path = traversability_sidecar_path(pgm)
            with open(trav_path, 'wb') as f:
                f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
                f.write(overlay.tobytes())
            s._log(f"Traversability map saved: {trav_path}", "success")
            s.mw.disable_edit()
            s.mw.set_reference_overlay(None)
            s.mw.set_reference_overlay_visible(False)
            s.btn_edit_draw.setChecked(False)
            s.btn_edit_erase.setChecked(False)
            if hasattr(s, 'edit_ref_overlay'):
                s.edit_ref_overlay.setChecked(False)
            s._load_map_sidecars(pgm)
            s._switch_view("Traversable Ground")

    def _revert_trav_edit(s):
        """Discard edits and reload the original traversability PGM from disk."""
        s.mw.disable_edit()
        s.mw.set_reference_overlay(None)
        s.mw.set_reference_overlay_visible(False)
        s.btn_edit_draw.setChecked(False)
        s.btn_edit_erase.setChecked(False)
        if hasattr(s, 'edit_ref_overlay'):
            s.edit_ref_overlay.setChecked(False)
        pgm = s.e_pgm.text()
        if not pgm:
            pgm = os.path.join(s.e_save.text(), "map.pgm")
        # Reload from disk
        s._load_map_sidecars(pgm)
        s._switch_view("Traversable Ground")
        s._log("Edit reverted — original traversability map restored.", "info")

    # ── Step 5: Coverage ──
    def _run_ref(s):
        pgm, yml = s._get_pgm()
        if not pgm: return
        # Reload map from disk to pick up any edits
        s._load_map(pgm)
        s.bref.setEnabled(False); s._log("Running reference...", "info"); s.prog.setValue(10)
        s.lref_note.setText("")
        params = s._get_params('r')
        sel_mask = s._make_sel_mask()
        trav_sidecar = None if s._v3_mode else traversability_sidecar_path(pgm)
        floor_sidecar = None if s._v3_mode else floor_sidecar_path(pgm)
        planner = s._use_stc_mode()
        yd = parse_yaml(yml)
        res = yd['resolution']; ox = yd['origin'][0]; oy = yd['origin'][1]
        w, h = s._map_w, s._map_h
        # Use manually set start point if available
        if hasattr(s, '_act_start_world') and s._act_start_world is not None:
            cx, cy = s._act_start_world
        else:
            center = s._selection_center_world(yml)
            if center is not None:
                cx, cy = center
            else:
                cx = ox + w * res / 2; cy = oy + h * res / 2

        def _worker():
            try:
                r = run_coverage(
                    pgm,
                    yml,
                    params,
                    cx,
                    cy,
                    sel_mask,
                    "REF",
                    lambda m, c: s.ui_log_sig.emit(m, c),
                    trav_sidecar,
                    floor_sidecar,
                    planner=planner,
                    ground_analysis_result=None,  # Reference robot passes all ramps
                )
                s.ref_result_sig.emit(r, pgm)
            except Exception as e:
                import traceback
                traceback.print_exc()
                s.ref_error_sig.emit(str(e))
        threading.Thread(target=_worker, daemon=True).start()

    def _ref_done(s, r, pgm):
        s.bref.setEnabled(True); s.prog.setValue(100)
        if r:
            r["pgm_path"] = pgm
            s.ref_r = r
            s.lref.setText(f"Ref: {s._result_area(r):.2f} m²")
            s.lref_note.setText(s._coverage_start_note(r))
            # Render coverage — v3 uses obstacle map pixels directly
            if s._v3_mode:
                bg = getattr(s, '_pgm_pixels', None)
            else:
                trav_px = getattr(s, '_trav_pixels', None)
                bg = trav_px if trav_px is not None else getattr(s, '_pgm_pixels', None)
            qi = render_coverage_fast(r, (255, 165, 0), bg_pgm=bg)
            # Draw start point marker on coverage image
            if hasattr(s, '_act_start_world') and s._act_start_world is not None:
                qi = s._draw_start_marker(qi, r, s._act_start_world)
            s._set_img("Reference Coverage", qi)
            s._update_stc_path_view()
            s._check_rii(pgm)
            s._update_semantic_ready_state()
            s._log("Reference accessible area ready.", "success")

    def _ref_failed(s, msg):
        s.bref.setEnabled(True)
        s.prog.setValue(0)
        s._log(f"Reference reachable-area evaluation failed: {msg}", "warn")

    def _run_act(s):
        pgm, yml = s._get_pgm()
        if not pgm: return
        if s._map_w == 0: s._load_map(pgm)
        s.bact.setEnabled(False); s._log("Running actual...", "info"); s.prog.setValue(10)
        s.lact_note.setText("")
        params = s._get_params('a')
        sel_mask = s._make_sel_mask()
        trav_sidecar = None if s._v3_mode else traversability_sidecar_path(pgm)
        floor_sidecar = None if s._v3_mode else floor_sidecar_path(pgm)
        planner = s._use_stc_mode()

        # Use picked start point or selection center or map center
        if s._act_start_world:
            cx, cy = s._act_start_world
            s._log(f"Using picked start point: ({cx:.2f}, {cy:.2f}) m", "info")
        else:
            yd = parse_yaml(yml)
            res = yd['resolution']; ox = yd['origin'][0]; oy = yd['origin'][1]
            center = s._selection_center_world(yml)
            if center is not None:
                cx, cy = center
            else:
                cx = ox + s._map_w * res / 2
                cy = oy + s._map_h * res / 2

        def _worker():
            try:
                r = run_coverage(
                    pgm,
                    yml,
                    params,
                    cx,
                    cy,
                    sel_mask,
                    "ACTUAL",
                    lambda m, c: s.ui_log_sig.emit(m, c),
                    trav_sidecar,
                    floor_sidecar,
                    planner=planner,
                    ground_analysis_result=getattr(s, '_ground_result', None),
                )
                s.act_result_sig.emit(r, pgm)
            except Exception as e:
                import traceback
                traceback.print_exc()
                s.act_error_sig.emit(str(e))
        threading.Thread(target=_worker, daemon=True).start()

    def _act_done(s, r, pgm):
        s.bact.setEnabled(True); s.prog.setValue(100)
        if r:
            r["pgm_path"] = pgm
            s.act_r = r
            s.lact.setText(f"Actual: {s._result_area(r):.2f} m²")
            s.lact_note.setText(s._coverage_start_note(r))
            if s._v3_mode:
                bg = getattr(s, '_pgm_pixels', None)
            else:
                trav_px = getattr(s, '_trav_pixels', None)
                bg = trav_px if trav_px is not None else getattr(s, '_pgm_pixels', None)
            qi = render_coverage_fast(r, (0, 229, 160), bg_pgm=bg)
            if s._act_start_world:
                qi = s._draw_start_marker(qi, r, s._act_start_world)
            s._set_img("Actual Coverage", qi)
            s._update_stc_path_view()
            s._check_rii(pgm)
            s._update_semantic_ready_state()
            s._log("Actual accessible area ready.", "success")

    def _act_failed(s, msg):
        s.bact.setEnabled(True)
        s.prog.setValue(0)
        s._log(f"Actual accessibility evaluation failed: {msg}", "warn")

    def _check_rii(s, pgm):
        if not s.act_r:
            return
        aa = s._result_area(s.act_r)
        floor_area = s._result_floor_area(s.act_r)
        rii = (aa / floor_area * 100) if floor_area > 0 else 0
        planner_name = s.act_r.get("planner", "")
        mode_label = f"With {planner_name}" if planner_name else "Without Path Planner"
        s.riiv.setText(f"{rii:.1f}%")
        s.riis.setText(f"{aa:.2f} / {floor_area:.2f} m² × 100  |  {mode_label}")
        s.riif.show()
        s._log(f"★ RII Horizontal = {rii:.1f}%", "gold")
        if s.ref_r and s._results_share_map():
            bg_cmp = getattr(s, '_trav_pixels', None) if getattr(s, '_trav_pixels', None) is not None else getattr(s, '_pgm_pixels', None)
            s._set_img("Compare", render_compare_fast(s.ref_r, s.act_r, bg_pgm=bg_cmp))
        elif s.ref_r:
            s._set_img("Compare", make_info_image("Reference and Actual are from different maps.\nRerun both Step 3 evaluations on the current map."))
            s._log("Compare view skipped because Reference and Actual are from different maps.", "warn")
        if s.ref_r and s._sem_pts is not None and s._sem_labels is not None:
            s._invalidate_semantic_state(
                keep_loaded_cloud=True,
                candidate_message="Step 3 results changed. Click 'Analyze RII_Horizontal' to re-run semantic analysis.",
                clear_progress=True,
            )
            s._log("Step 3 results changed — semantic analysis reset. Re-run Step 4 when ready.", "warn")

    # ── Step 6: Semantic Analysis ──
    def _browse_sem_pcd(s):
        f, _ = QFileDialog.getOpenFileName(s, "Labeled Point Cloud", "", "Point Cloud (*.pcd *.ply)")
        if f:
            s.e_sem_pcd.setText(f)
            s._log(f"Semantic point cloud: {f}", "info")
            token = s._invalidate_semantic_state(
                keep_loaded_cloud=False,
                candidate_message="Loading labeled cloud...",
                status_message="Loading...",
                status_color="#2563eb",
                clear_progress=True,
            )
            s._sem_load_active = True
            s._update_semantic_ready_state()

            def _load():
                try:
                    pts, labels, field_name = load_semantic_pcd(f)
                    if token != s._sem_session_token:
                        return
                    if pts is None:
                        s.sem_load_error_sig.emit(token, "Failed to load point cloud")
                        return
                    s.sem_loaded_sig.emit(token, pts, labels, field_name)
                except Exception as e:
                    if token == s._sem_session_token:
                        s.sem_load_error_sig.emit(token, f"Error: {e}")

            threading.Thread(target=_load, daemon=True).start()

    def _run_semantic_analysis(s):
        if not s.ref_r or not s.act_r:
            QMessageBox.warning(s, "Error", "Run both Reference and Actual RII Horizontal evaluations first (Step 3).")
            return
        if s._sem_load_active:
            QMessageBox.warning(s, "Error", "Wait for the labeled cloud to finish loading first.")
            return
        if s._sem_analysis_active:
            QMessageBox.warning(s, "Error", "Semantic analysis is already running for the current inputs.")
            return
        if s._sem_pts is None:
            QMessageBox.warning(s, "Error", "Load a labeled PCD or PLY file first.")
            return
        if s._sem_labels is None:
            QMessageBox.warning(s, "Error", "No semantic labels found in the point cloud file.\n"
                                "Ensure it has a scalar field (e.g., 'classification') from CloudCompare.")
            return

        _, yml = s._get_pgm()
        if not yml: return
        yd = parse_yaml(yml)

        token = s._invalidate_semantic_state(
            keep_loaded_cloud=True,
            candidate_message="Semantic analysis in progress...",
            clear_progress=True,
        )
        s._sem_analysis_active = True
        s._update_semantic_ready_state()
        s._log("Running semantic analysis...", "info")
        s._sem_progress(token, 0, "Preparing semantic analysis...")
        pts = s._sem_pts
        labels = s._sem_labels
        map_w = s._map_w
        map_h = s._map_h
        ref_r = s.ref_r
        act_r = s.act_r
        bg_pgm = getattr(s, '_pgm_pixels', None)

        def _analyze():
            try:
                def _is_current():
                    return token == s._sem_session_token

                # Project 3D labels to 2D grid
                s.sem_progress_sig.emit(token, 10, "Projecting semantic labels onto the current map...")
                label_grid = project_labels_to_2d_grid(
                    pts, labels, yd, map_w, map_h)
                if not _is_current():
                    return

                # Analyze
                s.sem_progress_sig.emit(token, 30, "Summarizing semantic accessibility gap...")
                analysis = analyze_semantic_rii(ref_r, act_r, label_grid, yd)
                if not _is_current():
                    return
                s.sem_progress_sig.emit(token, 40, "Computing layered semantic RII scenarios...")
                analysis["layered_rii"] = compute_semantic_layered_rii(
                    act_r,
                    label_grid,
                    logf=lambda m, c="": s.ui_log_sig.emit(m, c),
                    progress_cb=lambda done, total, name: s.sem_progress_sig.emit(
                        token,
                        40 + int(round(25 * done / max(total, 1))),
                        f"Computing layered semantic RII ({done}/{total}): {name}",
                    ),
                )
                if not _is_current():
                    return
                analysis["_label_grid"] = label_grid
                s.ui_log_sig.emit("Finding removable-object candidates...", "info")
                s.sem_progress_sig.emit(token, 70, "Finding removable-object candidates...")
                candidates = identify_semantic_removal_candidates(
                    act_r,
                    label_grid,
                    yd,
                    progress_cb=lambda done, total, fixation, label_id: s.sem_progress_sig.emit(
                        token,
                        70 + int(round(20 * done / max(total, 1))),
                        f"Finding removable-object candidates ({done}/{total}) - {fixation} label {label_id}",
                    ),
                )
                if not _is_current():
                    return
                s.ui_log_sig.emit(
                    f"Found {len(candidates)} removable-object candidate(s).",
                    "info" if candidates else "warn",
                )

                # Render semantic view
                s.sem_progress_sig.emit(token, 95, "Rendering semantic candidate view...")
                sem_img = render_semantic_candidates(ref_r, act_r, label_grid, candidates, selected_ids=[], bg_pgm=bg_pgm)
                if not _is_current():
                    return

                s.sem_result_sig.emit(token, analysis, sem_img, candidates)
            except Exception as e:
                import traceback; traceback.print_exc()
                if token == s._sem_session_token:
                    s.sem_error_sig.emit(token, str(e))

        threading.Thread(target=_analyze, daemon=True).start()

    def _show_semantic_3d(s):
        """Show the semantic point cloud in the 3D viewer, colored by label class."""
        if s._sem_pts is None or s._sem_labels is None:
            QMessageBox.warning(s, "Error", "Load a labeled point cloud first.")
            return
        pts = s._sem_pts
        labels = s._sem_labels
        n = pts.shape[0]
        max_points = 2_000_000 if PYQTGRAPH_GL_AVAILABLE else 250_000
        sampled = n > max_points
        if sampled:
            rng = np.random.default_rng(42)
            keep = rng.choice(n, size=max_points, replace=False)
            pts = pts[keep]
            labels = labels[keep]

        colors = np.full((len(pts), 3), 128, dtype=np.uint8)
        legend = []
        for label_id, color in SEMANTIC_3D_COLORS.items():
            mask = labels == label_id
            if np.any(mask):
                colors[mask] = color
                name = SEMANTIC_LABEL_NAMES.get(label_id, f"Label {label_id}")
                legend.append((name, color))

        cloud = {
            "points": np.ascontiguousarray(pts, dtype=np.float32),
            "colors": colors,
            "legend": legend,
            "label": "Semantic Labels (3D)",
            "path": s.e_sem_pcd.text() if hasattr(s, 'e_sem_pcd') else "",
            "total_points": int(s._sem_pts.shape[0]),
            "display_points": int(pts.shape[0]),
            "sampled": sampled,
        }
        s._set_cloud("3D Viewer", cloud)
        s._log("Showing semantic labels in 3D Viewer — colors indicate label class.", "success")

    def _sem_loaded(s, token, pts, labels, field_name):
        if token != s._sem_session_token:
            return
        s._sem_load_active = False
        s._sem_pts = pts
        s._sem_labels = labels
        n = len(pts)
        if labels is not None:
            unique = np.unique(labels)
            msg = f"Loaded {n:,} pts, field='{field_name}', labels: {list(unique)}"
        else:
            msg = f"Loaded {n:,} pts, no label field found"
        s.sem_status.setText(msg)
        s.sem_status.setStyleSheet("color:#16a34a;font-size:11px")
        s._set_semantic_candidate_placeholder("Run semantic analysis to populate removable-object candidates.")
        s._clear_semantic_progress()
        s._update_semantic_ready_state()
        # Enable 3D semantic view button when labels are available
        if hasattr(s, 'bsem_3d'):
            s.bsem_3d.setEnabled(labels is not None)
        s._log(msg, "success")

    def _sem_load_failed(s, token, msg):
        if token != s._sem_session_token:
            return
        s._sem_load_active = False
        s._sem_pts = None
        s._sem_labels = None
        s.sem_status.setText(msg)
        s.sem_status.setStyleSheet("color:#dc2626;font-size:11px")
        s._set_semantic_candidate_placeholder("Load a labeled cloud, then run semantic analysis to populate removable-object candidates.")
        s._clear_semantic_progress()
        s._update_semantic_ready_state()
        s._log(msg, "warn")

    def _sem_progress(s, token, value, message):
        if token != s._sem_session_token:
            return
        if hasattr(s, "sem_prog_lbl"):
            s.sem_prog_lbl.setText(message)
            s.sem_prog_lbl.show()
        if hasattr(s, "sem_prog"):
            s.sem_prog.setValue(max(0, min(100, int(value))))
            s.sem_prog.show()
        if hasattr(s, "prog"):
            s.prog.setValue(max(0, min(100, int(value))))

    def _sem_failed(s, token, msg):
        if token != s._sem_session_token:
            return
        s._sem_analysis_active = False
        s._update_semantic_ready_state()
        s._sem_progress(token, 0, f"Semantic analysis failed: {msg}")
        s._log(f"Semantic error: {msg}", "warn")

    def _selected_semantic_candidate_ids(s):
        ids = []
        if not hasattr(s, "sem_candidate_list"):
            return ids
        for i in range(s.sem_candidate_list.count()):
            item = s.sem_candidate_list.item(i)
            if item.checkState() == Qt.Checked:
                ids.append(int(item.data(Qt.UserRole)))
        return ids

    def _current_semantic_candidate_filter(s):
        if not hasattr(s, "sem_filter"):
            return None
        return s.sem_filter.currentData()

    def _apply_semantic_candidate_filter(s):
        if not hasattr(s, "sem_candidate_list"):
            return
        wanted = s._current_semantic_candidate_filter()
        for i in range(s.sem_candidate_list.count()):
            item = s.sem_candidate_list.item(i)
            fixation = item.data(Qt.UserRole + 1)
            item.setHidden(bool(wanted and fixation != wanted))
        current = s.sem_candidate_list.currentItem()
        if current is not None and current.isHidden():
            s.sem_candidate_list.setCurrentItem(None)
        s._update_semantic_candidate_status()

    def _select_filtered_semantic_candidates(s):
        if not hasattr(s, "sem_candidate_list"):
            return
        s.sem_candidate_list.blockSignals(True)
        for i in range(s.sem_candidate_list.count()):
            item = s.sem_candidate_list.item(i)
            item.setCheckState(Qt.Checked if not item.isHidden() else Qt.Unchecked)
        s.sem_candidate_list.blockSignals(False)
        s._semantic_candidate_selection_changed()

    def _selected_semantic_fixation_groups(s):
        groups = []
        if hasattr(s, "sem_fix_portable") and s.sem_fix_portable.isChecked():
            groups.append("Portable")
        if hasattr(s, "sem_fix_movable") and s.sem_fix_movable.isChecked():
            groups.append("Movable")
        if hasattr(s, "sem_fix_semi_fixed") and s.sem_fix_semi_fixed.isChecked():
            groups.append("Semi-Fixed")
        return groups

    def _set_semantic_fixation_groups(s, groups, run_recompute=False):
        wanted = set(groups)
        for fixation, cb in (
            ("Portable", getattr(s, "sem_fix_portable", None)),
            ("Movable", getattr(s, "sem_fix_movable", None)),
            ("Semi-Fixed", getattr(s, "sem_fix_semi_fixed", None)),
        ):
            if cb is not None:
                cb.blockSignals(True)
                cb.setChecked(fixation in wanted)
                cb.blockSignals(False)
        s._update_semantic_fixation_status()
        if run_recompute:
            s._recompute_semantic_fixations()

    def _update_semantic_fixation_status(s):
        if not hasattr(s, "sem_fixation_status"):
            return
        groups = s._selected_semantic_fixation_groups()
        has_analysis = bool(s._sem_analysis and s._sem_analysis.get("layered_rii"))
        visible = has_analysis or bool(groups)
        s.sem_fixation_status.setVisible(visible)
        if not visible:
            return
        if groups:
            s.sem_fixation_status.setText(
                "Selected fixation groups for full-group recompute: "
                + ", ".join(groups)
            )
        else:
            s.sem_fixation_status.setText(
                "Use the fixation controls below to remove all Portable, Movable, or Semi-Fixed obstacles and recompute RII."
            )

    def _update_semantic_layered_status(s):
        layered = s._sem_analysis.get("layered_rii") if s._sem_analysis else None
        s._sem_layered_result = layered
        if not hasattr(s, "sem_layered_status"):
            return
        if not layered or not layered.get("layers"):
            s.sem_layered_status.clear()
            s.sem_layered_status.hide()
            return
        lines = ["Layered RII decomposition"]
        for layer in layered["layers"]:
            excluded = ", ".join(layer["excludedFixations"]) if layer["excludedFixations"] else "None"
            lines.append(
                f"{layer['name']}: {layer['riiHorizontal']:.1f}%  "
                f"(excluded: {excluded}; Δ {layer['deltaPts']:+.1f} pts, {layer['deltaArea']:+.2f} m²)"
            )
        lines.append(
            f"Portable {layered['delta_portable']:+.1f} pts | "
            f"Movable {layered['delta_movable']:+.1f} pts | "
            f"Semi-Fixed {layered['delta_semi_fixed']:+.1f} pts"
        )
        lines.append(
            f"Structural max {layered['rii_structural_max']:.1f}% | "
            f"overall potential {layered['improvement_potential']:+.1f} pts"
        )
        s.sem_layered_status.setPlainText("\n".join(lines))
        s._update_semantic_fixation_status()

    def _semantic_candidate_by_id(s, candidate_id):
        if candidate_id is None:
            return None
        return next((c for c in s._sem_candidates if int(c["id"]) == int(candidate_id)), None)

    def _semantic_candidate_bounds_px(s, candidate):
        if not candidate or not s.act_r:
            return None
        w = int(s.act_r["w"])
        h = int(s.act_r["h"])
        flat = np.asarray(candidate.get("indices"), dtype=np.int32)
        if flat.size == 0:
            return None
        rows = flat // w
        cols = flat % w
        disp_rows = h - 1 - rows
        return (
            int(cols.min()),
            int(disp_rows.min()),
            int(cols.max()),
            int(disp_rows.max()),
        )

    def _focus_semantic_candidate(s, candidate_id, switch_view=True):
        candidate = s._semantic_candidate_by_id(candidate_id)
        if candidate is None:
            s._sem_focused_candidate_id = None
            s.mw.clear_focus()
            s._update_semantic_candidate_view()
            return
        already_focused = int(candidate["id"]) == s._sem_focused_candidate_id
        s._sem_focused_candidate_id = int(candidate["id"])
        s._update_semantic_candidate_view()
        if "Bottleneck" in s._imgs:
            s._render_bottleneck_view(focused_id=int(candidate["id"]))
        if switch_view:
            if candidate.get("isBottleneck") and "Bottleneck" in s._imgs:
                s._switch_view("Bottleneck")
            else:
                s._switch_view("Semantic")
        bounds = s._semantic_candidate_bounds_px(candidate)
        if bounds is not None:
            s.mw.focus_rect(
                bounds,
                label=f"#{candidate['id']} {candidate['name']}",
            )
            if not already_focused:
                s._log(
                    f"Focused candidate #{candidate['id']} {candidate['name']} in the Semantic view.",
                    "info",
                )

    def _semantic_candidate_current_changed(s, current, _previous):
        if current is None:
            s._focus_semantic_candidate(None, switch_view=False)
            return
        candidate_id = current.data(Qt.UserRole)
        if candidate_id is None:
            return
        s._focus_semantic_candidate(int(candidate_id), switch_view=True)

    def _populate_semantic_candidates(s, candidates):
        s._sem_candidates = list(candidates)
        s._sem_focused_candidate_id = None
        s.mw.clear_focus()
        if hasattr(s, "sem_filter"):
            s.sem_filter.blockSignals(True)
            s.sem_filter.setCurrentIndex(0)
            s.sem_filter.blockSignals(False)
        s.sem_candidate_list.blockSignals(True)
        s.sem_candidate_list.clear()
        if candidates:
            for candidate in candidates:
                x0, x1, y0, y1 = candidate["bboxWorld"]
                action = candidate.get("actionType", "")
                if action:
                    ratio = candidate.get("bottleneckRatio", 0)
                    true_unlock = candidate.get("trueUnlockArea", candidate["potentialUnlockArea"])
                    tag = f"[{action}] " if action != "Cannot optimize" else ""
                    ratio_str = f"  ratio={ratio:.1f}x" if ratio > 0 else ""
                    text = (
                        f"#{candidate['id']} {tag}{candidate['name']} [{candidate['fixation']}]  "
                        f"unlock={true_unlock:.2f} m²{ratio_str}  "
                        f"object={candidate['area']:.2f} m²"
                    )
                else:
                    text = (
                        f"#{candidate['id']} {candidate['name']} [{candidate['fixation']}]  "
                        f"unlock≈{candidate['potentialUnlockArea']:.2f} m²  "
                        f"object≈{candidate['area']:.2f} m²  "
                        f"bbox=({x0:.1f},{y0:.1f})→({x1:.1f},{y1:.1f})"
                    )
                item = QListWidgetItem(text)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, int(candidate["id"]))
                item.setData(Qt.UserRole + 1, candidate["fixation"])
                item.setToolTip(candidate["recommendation"])
                s.sem_candidate_list.addItem(item)
        else:
            placeholder = QListWidgetItem("No removable candidates were found for the current semantic map.")
            placeholder.setFlags(Qt.NoItemFlags)
            s.sem_candidate_list.addItem(placeholder)
        s.sem_candidate_list.blockSignals(False)

        has_candidates = bool(candidates)
        s._set_semantic_candidate_controls_enabled(has_candidates)
        s._apply_semantic_candidate_filter()
        s._hide_semantic_whatif_card()
        s._update_semantic_candidate_status()

    def _update_semantic_candidate_status(s, message=None, color="#ffcc66"):
        has_candidates = bool(s._sem_candidates)
        selected = len(s._selected_semantic_candidate_ids())
        visible = 0
        total = 0
        if hasattr(s, "sem_candidate_list"):
            total = s.sem_candidate_list.count()
            for i in range(total):
                if not s.sem_candidate_list.item(i).isHidden():
                    visible += 1
        filter_name = s._current_semantic_candidate_filter()
        filter_text = filter_name if filter_name else "All Fixations"
        if message is not None:
            s.sem_candidate_status.setText(message)
            s.sem_candidate_status.setStyleSheet(f"color:{color};font-size:11px")
            s.sem_candidate_status.setVisible(True)
            return
        if has_candidates:
            if selected:
                s.sem_candidate_status.setText(
                    f"{selected} candidate(s) selected. Showing {visible}/{total} under {filter_text}. "
                    "Click 'Recompute Optimised RII' to estimate the updated score. "
                    "Use the filter or the fixation buttons below to select all candidate objects in a fixation group."
                )
                s.sem_candidate_status.setStyleSheet("color:#7ad9ff;font-size:11px")
            else:
                s.sem_candidate_status.setText(
                    f"{visible}/{total} removable-object candidates shown under {filter_text}. "
                    "Amber = candidate, pink = selected candidate, cyan = focused candidate. "
                    "Click a row to center it in the Semantic tab. "
                    "Use the filter or the fixation buttons below to select all candidate objects in a fixation group."
                )
                s.sem_candidate_status.setStyleSheet("color:#6b7280;font-size:11px")
        else:
            s.sem_candidate_status.setText(
                "No removable Portable / Movable / Semi-Fixed object components are currently unlocking additional floor area."
            )
            s.sem_candidate_status.setStyleSheet("color:#6b7280;font-size:11px")
        s.sem_candidate_status.setVisible(True)

    def _hide_semantic_whatif_card(s):
        if hasattr(s, "sem_cur_val"):
            s.sem_cur_val.setText("—")
        if hasattr(s, "sem_riiv"):
            s.sem_riiv.setText("—")
        if hasattr(s, "sem_delta"):
            s.sem_delta.clear()
        if hasattr(s, "sem_riis"):
            s.sem_riis.clear()
        if hasattr(s, "sem_riif"):
            s.sem_riif.hide()

    def _set_semantic_candidate_controls_enabled(s, enabled):
        for widget_name in (
            "sem_filter",
            "bsem_select_filtered",
            "sem_candidate_list",
            "bsem_select_portable",
            "bsem_select_movable",
            "bsem_select_semi_fixed",
            "bsem_select_portable_movable",
            "bsem_select_all_candidates",
            "bsem_clear_candidates",
            "bsem_recompute",
            "bsem_bottleneck",
        ):
            widget = getattr(s, widget_name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    def _update_semantic_candidate_view(s):
        if not (s.ref_r and s.act_r and s._label_grid is not None):
            return
        qi = render_semantic_candidates(
            s.ref_r,
            s.act_r,
            s._label_grid,
            s._sem_candidates,
            selected_ids=s._selected_semantic_candidate_ids(),
            focused_id=s._sem_focused_candidate_id,
            bg_pgm=getattr(s, '_pgm_pixels', None),
        )
        s._imgs["Semantic"] = qi
        if s._is_view_active("Semantic"):
            s._switch_view("Semantic")
        # Also update 3D viewer with semantic overlay
        s._update_semantic_3d_view()

    def _update_semantic_3d_view(s):
        """Build a 3D point cloud colorized with semantic candidate highlights."""
        if not hasattr(s, '_sem_pts') or s._sem_pts is None or len(s._sem_candidates) == 0:
            return
        pts = s._sem_pts
        labels = s._sem_labels
        max_display = 500_000
        step = max(1, pts.shape[0] // max_display)
        pts_sub = pts[::step]
        labels_sub = labels[::step]
        # Base colors: gray for all points
        colors = np.full((pts_sub.shape[0], 3), 140, dtype=np.uint8)
        selected_ids = s._selected_semantic_candidate_ids()
        focused_id = s._sem_focused_candidate_id
        for cand in s._sem_candidates:
            x0, x1, y0, y1 = cand["bboxWorld"]
            lid = cand["label"]
            mask = (
                (labels_sub == lid) &
                (pts_sub[:, 0] >= x0) & (pts_sub[:, 0] <= x1) &
                (pts_sub[:, 1] >= y0) & (pts_sub[:, 1] <= y1)
            )
            if not mask.any():
                continue
            cid = cand["id"]
            if cid == focused_id:
                colors[mask] = [0, 220, 255]    # cyan = focused
            elif cid in selected_ids:
                colors[mask] = [255, 120, 180]   # pink = selected
            else:
                colors[mask] = SEMANTIC_3D_COLORS.get(lid, (255, 180, 50))
        cloud = {"points": pts_sub, "colors": colors}
        s._clouds["3D Viewer"] = cloud
        if s._is_view_active("3D Viewer"):
            s.pcw.set_cloud(cloud)

    def _semantic_candidate_selection_changed(s, *_):
        s._sem_improved = None
        s._hide_semantic_whatif_card()
        s._update_semantic_candidate_status()
        s._update_semantic_candidate_view()

    def _set_semantic_candidates_by_fixation(s, fixations):
        if not s._sem_candidates:
            return
        if hasattr(s, "sem_filter"):
            target = next(iter(fixations)) if len(fixations) == 1 else None
            idx = s.sem_filter.findData(target)
            if idx >= 0:
                s.sem_filter.blockSignals(True)
                s.sem_filter.setCurrentIndex(idx)
                s.sem_filter.blockSignals(False)
                s._apply_semantic_candidate_filter()
        s.sem_candidate_list.blockSignals(True)
        for i in range(s.sem_candidate_list.count()):
            item = s.sem_candidate_list.item(i)
            fixation = item.data(Qt.UserRole + 1)
            item.setCheckState(Qt.Checked if fixation in fixations else Qt.Unchecked)
        s.sem_candidate_list.blockSignals(False)
        s._semantic_candidate_selection_changed()

    def _select_semantic_candidates_portable(s):
        s._set_semantic_candidates_by_fixation({"Portable", "Movable"})

    def _clear_semantic_candidates(s):
        if not s._sem_candidates:
            return
        s.sem_candidate_list.blockSignals(True)
        for i in range(s.sem_candidate_list.count()):
            s.sem_candidate_list.item(i).setCheckState(Qt.Unchecked)
        s.sem_candidate_list.blockSignals(False)
        s._semantic_candidate_selection_changed()

    def _recompute_semantic_improvement(s):
        if s._sem_load_active or s._sem_analysis_active:
            QMessageBox.warning(s, "Error", "Wait for semantic analysis to finish before recomputing the Optimised RII.")
            return
        if not s.act_r or not s._sem_candidates:
            QMessageBox.warning(s, "Error", "Run Step 4 semantic analysis first.")
            return
        selected_ids = s._selected_semantic_candidate_ids()
        if not selected_ids:
            QMessageBox.warning(s, "Error", "Select one or more removable-object candidates first.")
            return
        s.bsem_recompute.setEnabled(False)
        s._log("Recomputing improved RII Horizontal from selected semantic removals...", "info")
        token = s._sem_session_token

        def _recompute():
            try:
                improved = simulate_removed_candidates(
                    s.act_r,
                    s._sem_candidates,
                    selected_ids,
                    label="IMPROVED",
                    logf=lambda m, c: s.ui_log_sig.emit(m, c),
                )
                if token == s._sem_session_token:
                    s.sem_improved_sig.emit(token, improved)
            except Exception as e:
                import traceback; traceback.print_exc()
                if token == s._sem_session_token:
                    s.sem_improved_error_sig.emit(token, str(e))

        threading.Thread(target=_recompute, daemon=True).start()

    def _run_bottleneck_analysis(s):
        if not s.act_r or not s._sem_candidates:
            QMessageBox.warning(s, "Error", "Run Step 4 semantic analysis first.")
            return
        s.bsem_bottleneck.setEnabled(False)
        s.bottleneck_status.setText("Scoring bottlenecks...")
        s._log("Starting bottleneck analysis (BFS reachability for top candidates)...", "info")

        # Ensure act_result has resolution for bottleneck scoring
        act = s.act_r
        if "resolution" not in act:
            yaml_path = s.e_yaml.text()
            if yaml_path and os.path.isfile(yaml_path):
                from core.map_io import parse_yaml
                yd = parse_yaml(yaml_path)
                act["resolution"] = yd["resolution"]
            else:
                act["resolution"] = 0.05

        candidates = list(s._sem_candidates)

        def _analyse():
            try:
                def prog(done, total, name):
                    s.ui_log_sig.emit(f"  Bottleneck {done}/{total}: {name}", "")

                score_bottleneck_candidates(act, candidates, top_n=20, progress_cb=prog)
                s.ui_log_sig.emit(f"Bottleneck scoring done. Finding relocation zones for chokepoints...", "info")

                reloc = {}
                bottlenecks = [c for c in candidates if c.get("isBottleneck")]
                for i, cand in enumerate(bottlenecks[:10]):
                    zones = find_relocation_zones(act, cand)
                    if zones:
                        reloc[cand["id"]] = zones
                    s.ui_log_sig.emit(
                        f"  Relocation {i+1}/{min(len(bottlenecks),10)}: "
                        f"#{cand['id']} {cand.get('name','')} — {len(zones)} zone(s)",
                        "",
                    )

                classify_candidate_actions(candidates, reloc)

                # Summary
                n_bottleneck = sum(1 for c in candidates if c.get("isBottleneck"))
                n_relocate = sum(1 for c in candidates if c.get("actionType") == "Relocate")
                n_remove = sum(1 for c in candidates if c.get("actionType") == "Remove")
                s.ui_log_sig.emit(
                    f"Bottleneck analysis complete: {n_bottleneck} chokepoints, "
                    f"{n_relocate} relocatable, {n_remove} remove-only.",
                    "success",
                )

                # Update GUI on main thread via signal
                s.bottleneck_done_sig.emit()
            except Exception as e:
                import traceback; traceback.print_exc()
                s.ui_log_sig.emit(f"Bottleneck analysis failed: {e}", "warn")
                s.bottleneck_error_sig.emit(str(e))

        threading.Thread(target=_analyse, daemon=True).start()

    def _bottleneck_done(s):
        s.bsem_bottleneck.setEnabled(True)
        n_bottleneck = sum(1 for c in s._sem_candidates if c.get("isBottleneck"))
        n_relocate = sum(1 for c in s._sem_candidates if c.get("actionType") == "Relocate")
        n_remove = sum(1 for c in s._sem_candidates if c.get("actionType") == "Remove")
        s.bottleneck_status.setText(
            f"{n_bottleneck} chokepoint(s) found  |  "
            f"{n_relocate} can be relocated  |  "
            f"{n_remove} should be removed"
        )
        # Re-sort: bottlenecks first, then by true unlock area
        s._sem_candidates.sort(
            key=lambda c: (-int(c.get("isBottleneck", False)), -c.get("trueUnlockArea", 0))
        )
        s._populate_semantic_candidates(s._sem_candidates)
        s._render_bottleneck_view()
        s.bsem_optimize.setEnabled(True)

    def _render_bottleneck_view(s, focused_id=None):
        if not s.act_r or not s._sem_candidates:
            return
        bg = getattr(s, '_trav_pixels', None) if getattr(s, '_trav_pixels', None) is not None else getattr(s, '_pgm_pixels', None)
        qi = render_bottleneck_overlay(s.act_r, s._sem_candidates, bg_pgm=bg, focused_id=focused_id)
        s._set_img("Bottleneck", qi)

    def _run_optimization(s):
        if not s.act_r or not s._sem_candidates:
            QMessageBox.warning(s, "Error", "Run bottleneck analysis first.")
            return
        s.bsem_optimize.setEnabled(False)
        s.optimization_status.setText("Optimizing layout...")
        s._log("Starting multi-object layout optimization...", "info")

        act = s.act_r
        candidates = list(s._sem_candidates)

        def _work():
            try:
                def prog(done, total, name):
                    s.ui_log_sig.emit(f"  Optimization step {done}/{total}: {name}", "")

                result = optimize_multi_object_relocation(act, candidates, max_moves=10, progress_cb=prog)
                n = len(result["moves"])
                gain = result["total_gain"]
                s.ui_log_sig.emit(f"Optimization complete: {n} moves, +{gain:.1f} m² total gain", "success")
                s.optimization_done_sig.emit(result)
            except Exception as e:
                import traceback; traceback.print_exc()
                s.ui_log_sig.emit(f"Optimization failed: {e}", "warn")
                s.optimization_error_sig.emit(str(e))

        threading.Thread(target=_work, daemon=True).start()

    def _optimization_done(s, result):
        s.bsem_optimize.setEnabled(True)
        s._optimization_result = result
        moves = result["moves"]
        gain = result["total_gain"]
        orig = result["original_accessible_area"]
        opt = result["optimized_accessible_area"]
        floor = s._result_floor_area(s.act_r) if s.act_r else opt
        rii_before = (orig / floor * 100) if floor > 0 else 0
        rii_after = (opt / floor * 100) if floor > 0 else 0

        s.optimization_status.setText(
            f"{len(moves)} move(s)  |  +{gain:.1f} m² gained  |  "
            f"RII: {rii_before:.1f}% → {rii_after:.1f}%"
        )

        # Render optimization overlay on map
        bg = getattr(s, '_trav_pixels', None) if getattr(s, '_trav_pixels', None) is not None else getattr(s, '_pgm_pixels', None)
        qi = render_optimization_overlay(s.act_r, result, bg_pgm=bg)
        s._set_img("Optimization", qi)
        s._switch_view("Optimization")

        # Open popup window with details
        s._show_optimization_popup(result)

    def _optimization_failed(s, msg=""):
        s.bsem_optimize.setEnabled(True)
        s.optimization_status.setText(f"Optimization failed: {msg}" if msg else "Optimization failed.")

    def _show_optimization_popup(s, result):
        moves = result["moves"]
        orig = result["original_accessible_area"]
        opt = result["optimized_accessible_area"]
        gain = result["total_gain"]
        floor = s._result_floor_area(s.act_r) if s.act_r else opt

        win = QMainWindow(s)
        win.setWindowTitle("Layout Optimization Report")
        win.resize(600, 400)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setStyleSheet("background:#ffffff;color:#1f2937;font-family:monospace;font-size:12px;padding:12px")

        html = "<h2>Layout Optimization Report</h2>"
        html += f"<p><b>Original accessible area:</b> {orig:.2f} m² ({orig/floor*100:.1f}% RII)</p>"
        html += f"<p><b>Optimized accessible area:</b> {opt:.2f} m² ({opt/floor*100:.1f}% RII)</p>"
        html += f"<p><b>Total gain:</b> +{gain:.1f} m²</p>"
        html += "<hr>"
        html += "<table border='1' cellpadding='4' cellspacing='0' style='border-collapse:collapse'>"
        html += "<tr style='background:#f0f4ff'><th>#</th><th>Object</th><th>Action</th><th>Gain</th><th>Cumulative</th></tr>"
        for i, m in enumerate(moves):
            action = m["action"]
            if m["to_rc"]:
                action += f" to ({m['to_rc'][1]*0.05:.1f}, {m['to_rc'][0]*0.05:.1f}) m"
            html += f"<tr><td>{i+1}</td><td>{m['name']}</td><td>{action}</td>"
            html += f"<td>+{m['step_gain']:.2f} m²</td><td>+{m['cumulative_gain']:.2f} m²</td></tr>"
        html += "</table>"

        if not moves:
            html += "<p>No beneficial moves found. The current layout may already be optimal for the given constraints.</p>"

        te.setHtml(html)
        win.setCentralWidget(te)
        win.show()

    def _bottleneck_failed(s, msg=""):
        s.bsem_bottleneck.setEnabled(True)
        s.bottleneck_status.setText(f"Bottleneck analysis failed: {msg}" if msg else "Bottleneck analysis failed — check log.")

    def _recompute_semantic_fixations(s):
        if not s.act_r or s._label_grid is None:
            QMessageBox.warning(s, "Error", "Run Step 4 semantic analysis first.")
            return
        fixations = s._selected_semantic_fixation_groups()
        if not fixations:
            QMessageBox.warning(s, "Error", "Select one or more fixation groups first.")
            return
        if hasattr(s, "bsem_fix_recompute"):
            s.bsem_fix_recompute.setEnabled(False)
        s._log(
            "Recomputing RII from fixation groups: " + ", ".join(fixations),
            "info",
        )
        token = s._sem_session_token

        def _recompute():
            try:
                improved = simulate_removed_fixations(
                    s.act_r,
                    s._label_grid,
                    fixations,
                    label="FIXATION",
                    logf=lambda m, c: s.ui_log_sig.emit(m, c),
                )
                if token == s._sem_session_token:
                    s.sem_improved_sig.emit(token, improved)
            except Exception as e:
                import traceback; traceback.print_exc()
                if token == s._sem_session_token:
                    s.sem_improved_error_sig.emit(token, str(e))

        threading.Thread(target=_recompute, daemon=True).start()

    def _sem_improved_done(s, token, improved):
        if token != s._sem_session_token:
            return
        s.bsem_recompute.setEnabled(True)
        if hasattr(s, "bsem_fix_recompute"):
            s.bsem_fix_recompute.setEnabled(True)
        s._sem_improved = improved
        current_rii = (s._result_area(s.act_r) / max(s._result_floor_area(s.act_r), 1e-9)) * 100.0
        improved_rii = (s._result_area(improved) / max(s._result_floor_area(improved), 1e-9)) * 100.0
        gain = improved_rii - current_rii
        area_gain = s._result_area(improved) - s._result_area(s.act_r)
        if improved.get("removedMode") == "fixation":
            groups = ", ".join(improved.get("excludedFixations", [])) or "None"
            scenario = f"{groups} removed"
        else:
            ids = s._selected_semantic_candidate_ids()
            scenario = f"{len(ids)} selected object(s) removed"
        floor_area = s._result_floor_area(improved)
        aa = s._result_area(improved)
        s.sem_cur_val.setText(f"{current_rii:.1f}%")
        s.sem_riiv.setText(f"{improved_rii:.1f}%")
        delta_color = "#16a34a" if gain >= 0 else "#dc2626"
        s.sem_delta.setText(f"{gain:+.1f} pts")
        s.sem_delta.setStyleSheet(f"color:{delta_color};font-size:14px;font-weight:bold")
        s.sem_riis.setText(
            f"{scenario}  |  {area_gain:+.2f} m² ({aa:.2f} / {floor_area:.2f} m²)"
        )
        s.sem_riif.show()
        s._update_semantic_candidate_status(
            f"Optimised RII recompute complete. Removed {improved.get('removedArea', 0.0):.2f} m² of blocked footprint.",
            color="#16a34a",
        )
        s._log(
            f"Optimised RII Horizontal = {improved_rii:.1f}% "
            f"(+{gain:.1f} pts, +{area_gain:.2f} m² accessible)",
            "success",
        )

    def _sem_improved_failed(s, token, msg):
        if token != s._sem_session_token:
            return
        s.bsem_recompute.setEnabled(True)
        if hasattr(s, "bsem_fix_recompute"):
            s.bsem_fix_recompute.setEnabled(True)
        s._hide_semantic_whatif_card()
        s._update_semantic_candidate_status(f"Optimised RII recompute failed: {msg}", color="#dc2626")
        s._log(f"Optimised RII recompute failed: {msg}", "warn")

    def _sem_done(s, token, analysis, sem_img, candidates):
        if token != s._sem_session_token:
            return
        s._sem_analysis_active = False
        s._label_grid = analysis.pop("_label_grid", None)
        s._sem_progress(token, 100, f"Semantic analysis complete. {len(candidates)} removable-object candidate(s) ready.")
        s._set_img("Semantic", sem_img)
        s._sem_analysis = analysis
        s._populate_semantic_candidates(candidates)
        s._update_semantic_layered_status()
        s._hide_semantic_whatif_card()
        s._update_semantic_ready_state()

        # Build report HTML
        html = '<b style="color:#1f2937;font-size:14px">RII Horizontal — Gap Analysis</b><br><br>'

        floor_area = s._result_floor_area(s.act_r)
        aa = s._result_area(s.act_r)
        rii = (aa / floor_area * 100) if floor_area > 0 else 0
        html += f'<b>RII Horizontal:</b> <span style="color:#2563eb">{rii:.1f}%</span><br>'
        html += f'<b>Total missed area:</b> {analysis["total_missed_area"]:.2f} m²<br><br>'

        if analysis.get('fixation_breakdown'):
            html += '<b>Gap by fixation group:</b><br>'
            for item in analysis['fixation_breakdown']:
                html += (
                    f'<span style="color:#1f2937">{item["fixation"]}: '
                    f'{item["area"]:.2f} m² ({item["pct"]:.1f}%)</span><br>'
                )
            html += '<br>'

        layered = analysis.get("layered_rii")
        if layered and layered.get("layers"):
            html += '<b style="color:#7ad9ff">Layered RII decomposition:</b><br>'
            html += '<table style="width:100%;font-size:11px;border-collapse:collapse">'
            html += '<tr style="color:#6b7280"><td>Scenario</td><td>Excluded</td><td>RII_H</td><td>Δ pts</td><td>Δ area</td></tr>'
            for layer in layered["layers"]:
                excluded = ", ".join(layer["excludedFixations"]) if layer["excludedFixations"] else "None"
                color = "#2563eb" if layer["layer"] == len(layered["layers"]) - 1 else "#c5cdd8"
                html += (
                    f'<tr style="color:{color}">'
                    f'<td>{layer["name"]}</td>'
                    f'<td>{excluded}</td>'
                    f'<td>{layer["riiHorizontal"]:.1f}%</td>'
                    f'<td>{layer["deltaPts"]:+.1f}</td>'
                    f'<td>{layer["deltaArea"]:+.2f} m²</td>'
                    f'</tr>'
                )
            html += '</table><br>'
            html += (
                f'<span style="color:#1f2937">Portable contribution: '
                f'<span style="color:#2563eb">{layered["delta_portable"]:+.1f} pts</span> '
                f'({layered["delta_portable_area"]:+.2f} m²)</span><br>'
            )
            html += (
                f'<span style="color:#1f2937">Movable contribution: '
                f'<span style="color:#2563eb">{layered["delta_movable"]:+.1f} pts</span> '
                f'({layered["delta_movable_area"]:+.2f} m²)</span><br>'
            )
            html += (
                f'<span style="color:#1f2937">Semi-fixed contribution: '
                f'<span style="color:#2563eb">{layered["delta_semi_fixed"]:+.1f} pts</span> '
                f'({layered["delta_semi_fixed_area"]:+.2f} m²)</span><br>'
            )
            html += (
                f'<span style="color:#1f2937">Structural maximum: '
                f'<span style="color:#7ad9ff">{layered["rii_structural_max"]:.1f}%</span> '
                f'(overall improvement potential {layered["improvement_potential"]:+.1f} pts)</span><br><br>'
            )

        html += '<b>Accessibility gap by surface type:</b><br>'
        html += '<table style="width:100%;font-size:11px;border-collapse:collapse">'
        html += '<tr style="color:#6b7280"><td>Label</td><td>Fixation</td><td>Area (m²)</td><td>% of Gap</td></tr>'

        for b in analysis['label_breakdown']:
            color = "#1f2937"
            if b['pct'] > 20: color = "#dc2626"
            elif b['pct'] > 10: color = "#d97706"
            elif b['pct'] > 5: color = "#ff9940"
            html += (f'<tr style="color:{color}">'
                     f'<td>{b["label"]}: {b["name"]}</td>'
                     f'<td>{b["fixation"]}</td>'
                     f'<td>{b["area"]:.2f}</td>'
                     f'<td>{b["pct"]:.1f}%</td></tr>')

        html += '</table><br>'

        if analysis['top_recommendations']:
            html += '<b style="color:#16a34a">Top Recommendations:</b><br>'
            for rec in analysis['top_recommendations']:
                html += f'<span style="color:#1f2937">{rec}</span><br>'

        if candidates:
            html += '<br><b style="color:#ffcc66">Removable-object candidates:</b><br>'
            for candidate in candidates[:6]:
                html += (
                    f'<span style="color:#1f2937">#{candidate["id"]} {candidate["name"]} '
                    f'[{candidate["fixation"]}] unlock≈{candidate["potentialUnlockArea"]:.2f} m², '
                    f'object≈{candidate["area"]:.2f} m²</span><br>'
                )

        s._sem_report_html = html
        s.sem_report.setHtml(html)
        s.btn_show_report.show()
        if layered:
            s._log(
                "Layered semantic RII: "
                f"portable {layered['delta_portable']:+.1f} pts, "
                f"movable {layered['delta_movable']:+.1f} pts, "
                f"semi-fixed {layered['delta_semi_fixed']:+.1f} pts, "
                f"structural max {layered['rii_structural_max']:.1f}%.",
                "info",
            )
        s._log(f"Semantic analysis complete — {len(analysis['label_breakdown'])} categories in gap", "success")

    # ════ RII Vertical — Wall Segments ════
    def _detect_wall_segments(s):
        if s._sem_pts is None or s._sem_labels is None:
            QMessageBox.warning(s, "Error", "Load a labelled point cloud first (Step 4).")
            return
        try:
            wall_ids = {int(x.strip()) for x in s.rv_wall_ids.text().split(",") if x.strip()}
        except ValueError:
            QMessageBox.warning(s, "Error", "Invalid wall label IDs.")
            return
        if not wall_ids:
            QMessageBox.warning(s, "Error", "Wall label IDs cannot be empty.")
            return

        s._log("Detecting wall segments...", "info")
        s._rv_wall_segments = identify_wall_segments(
            s._sem_pts, s._sem_labels, wall_label_ids=wall_ids,
            voxel_size=0.20, min_area_m2=0.5,
        )
        s._rv_focused_wall_id = None
        s.rv_wall_list.blockSignals(True)
        s.rv_wall_list.clear()
        for seg in s._rv_wall_segments:
            text = (f"Wall #{seg['id']}  area={seg['area_m2']:.1f}m²  "
                    f"h={seg['height_span']:.1f}m  w={seg['width_span']:.1f}m  "
                    f"pts={seg['num_points']}")
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, seg["id"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            s.rv_wall_list.addItem(item)
        s.rv_wall_list.blockSignals(False)
        s.rv_wall_status.setText(f"{len(s._rv_wall_segments)} wall segment(s) found. Click to view in 3D.")
        s._log(f"Found {len(s._rv_wall_segments)} wall segments.", "success")
        # Show the full wall cloud in 3D
        s._rv_update_3d_wall_view()

    def _rv_wall_current_changed(s, current, _previous):
        if current is None:
            s._rv_focused_wall_id = None
            s._rv_update_3d_wall_view()
            return
        wall_id = current.data(Qt.UserRole)
        if wall_id is None:
            return
        s._rv_focused_wall_id = int(wall_id)
        s._rv_update_3d_wall_view()
        s._log(f"Focused wall #{wall_id} in 3D viewer.", "info")

    def _rv_wall_check_changed(s, item):
        # Update the 3D view to reflect selection state
        s._rv_update_3d_wall_view()

    def _rv_wall_select_all(s, checked):
        s.rv_wall_list.blockSignals(True)
        for i in range(s.rv_wall_list.count()):
            s.rv_wall_list.item(i).setCheckState(Qt.Checked if checked else Qt.Unchecked)
        s.rv_wall_list.blockSignals(False)
        s._rv_update_3d_wall_view()

    def _rv_selected_wall_ids(s):
        """Return set of checked wall segment IDs."""
        ids = set()
        for i in range(s.rv_wall_list.count()):
            item = s.rv_wall_list.item(i)
            if item.checkState() == Qt.Checked:
                wid = item.data(Qt.UserRole)
                if wid is not None:
                    ids.add(int(wid))
        return ids

    def _rv_update_3d_wall_view(s):
        """Show the labelled point cloud in the 3D viewer with walls highlighted."""
        if s._sem_pts is None or s._sem_labels is None or not s._rv_wall_segments:
            return
        try:
            wall_ids = {int(x.strip()) for x in s.rv_wall_ids.text().split(",") if x.strip()}
        except ValueError:
            wall_ids = {1}

        selected = s._rv_selected_wall_ids()
        focused = getattr(s, '_rv_focused_wall_id', None)

        all_pts = s._sem_pts
        all_labels = s._sem_labels
        max_display = 600_000

        # Keep ALL wall points, subsample only non-wall points
        wall_mask = np.isin(all_labels, np.array(sorted(wall_ids), dtype=np.int32))
        wall_indices = np.where(wall_mask)[0]
        nonwall_indices = np.where(~wall_mask)[0]

        if len(wall_indices) + len(nonwall_indices) > max_display and len(nonwall_indices) > 0:
            budget = max(max_display - len(wall_indices), max_display // 4)
            step = max(1, len(nonwall_indices) // budget)
            nonwall_sub = nonwall_indices[::step]
        else:
            nonwall_sub = nonwall_indices

        # Combine: wall points first, then subsampled non-wall
        keep_indices = np.concatenate([wall_indices, nonwall_sub])
        keep_indices.sort()
        pts = all_pts[keep_indices]
        labels = all_labels[keep_indices]

        # Remap segment point_indices to the new array
        idx_remap = np.full(all_pts.shape[0], -1, dtype=np.int64)
        idx_remap[keep_indices] = np.arange(len(keep_indices))

        segments_mapped = []
        for seg in s._rv_wall_segments:
            new_idx = idx_remap[seg["point_indices"]]
            new_idx = new_idx[new_idx >= 0]  # drop any that weren't kept
            if len(new_idx) > 0:
                segments_mapped.append(dict(seg, point_indices=new_idx))

        colors = colorize_cloud_with_walls(
            pts, labels, segments_mapped,
            selected_wall_ids=selected,
            focused_wall_id=focused,
            wall_label_ids=wall_ids,
        )

        cloud = dict(
            points=pts,
            colors=colors,
            label="Wall Segments (3D)",
            path=s.e_sem_pcd.text() if hasattr(s, 'e_sem_pcd') else "",
            total_points=all_pts.shape[0],
            display_points=pts.shape[0],
            sampled=pts.shape[0] < all_pts.shape[0],
        )
        # Show wall segments in the main 3D Viewer tab
        s._clouds["3D Viewer"] = cloud
        if s._is_view_active("3D Viewer"):
            s.pcw.set_cloud(cloud)
        else:
            s._set_cloud("3D Viewer", cloud)

    # ════ RII Vertical — Computation ════
    def _run_rii_vertical(s):
        if not s.act_r:
            QMessageBox.warning(s, "Error", "Run Actual RII Horizontal first (Step 3).")
            return
        if s._sem_pts is None or s._sem_labels is None:
            QMessageBox.warning(s, "Error", "Load a labelled point cloud first (Step 4).")
            return
        if s._rv_active:
            QMessageBox.warning(s, "Error", "RII Vertical is already running.")
            return

        # Parse wall IDs
        try:
            wall_ids = {int(x.strip()) for x in s.rv_wall_ids.text().split(",") if x.strip()}
        except ValueError:
            QMessageBox.warning(s, "Error", "Invalid wall label IDs. Use comma-separated integers (e.g. '1' or '1,7').")
            return
        if not wall_ids:
            QMessageBox.warning(s, "Error", "Wall label IDs cannot be empty.")
            return

        s._rv_active = True
        s.brv.setEnabled(False)
        s.rv_prog.show(); s.rv_prog.setValue(0)
        s.rv_prog_lbl.show(); s.rv_prog_lbl.setText("Starting RII Vertical computation...")
        s.rv_riif.hide(); s.rv_combf.hide()
        s._log("Running RII Vertical (wall reachability via raycasting)...", "info")

        pts = s._sem_pts.copy()
        labels = s._sem_labels.copy()
        act_r = dict(s.act_r)
        # Copy numpy arrays to avoid threading issues
        for k in ("covPx", "floorPx", "blocked", "sourceBlocked"):
            if k in act_r and isinstance(act_r[k], np.ndarray):
                act_r[k] = act_r[k].copy()

        params = dict(
            wall_label_ids=wall_ids,
            voxel_size=s.rv_voxel.value(),
            max_reach=s.rv_reach.value(),
            angle_step_deg=s.rv_angle.value(),
            wall_min_h=s.rv_wall_min_h.value(),
            wall_max_h=s.rv_wall_max_h.value(),
            ground_stride=s.rv_stride.value(),
            max_ground_samples=s.rv_max_samples.value(),
            paint_width=s.rv_paint_w.value(),
            paint_vertical_span=s.rv_paint_vspan.value(),
            sweep_step=s.rv_sweep.value(),
        )
        gamma = s.rv_gamma.value()

        def _compute():
            try:
                rv = compute_rii_vertical(
                    pts, labels, act_r,
                    logf=lambda m, c="": s.ui_log_sig.emit(m, c),
                    progress_cb=lambda v: s.rv_progress_sig.emit(v, ""),
                    **params,
                )
                rii_h = float(act_r.get("riiHorizontal", 0.0))
                combined = compute_combined_rii(rii_h, rv, gamma=gamma)
                rv["combined"] = combined
                s.rv_result_sig.emit(rv)
            except Exception as exc:
                import traceback
                s.rv_error_sig.emit(f"{exc}\n{traceback.format_exc()}")

        threading.Thread(target=_compute, daemon=True).start()

    def _rv_progress(s, value, message):
        s.rv_prog.setValue(value)
        if message:
            s.rv_prog_lbl.setText(message)

    def _show_rv_painted_cloud(s, rv):
        """After RII Vertical, show full point cloud with wall coverage overlay in the Vertical Coverage tab."""
        wall_band = rv.get("wall_band", set())
        painted = rv.get("painted_voxels", set())
        voxel_origin = rv.get("voxel_origin")
        vs = rv.get("voxel_size", 0.05)
        if not wall_band or voxel_origin is None:
            return

        # -- Build wall voxel points with green/red colors --
        wall_pts_list = []
        wall_colors_list = []
        for k in wall_band:
            center = voxel_origin + (np.array(k, dtype=np.float32) + 0.5) * vs
            wall_pts_list.append(center)
            if k in painted:
                wall_colors_list.append([0, 200, 100])   # green = reachable
            else:
                wall_colors_list.append([255, 70, 50])    # red = unreachable
        wall_pts = np.array(wall_pts_list, dtype=np.float32)
        wall_colors = np.array(wall_colors_list, dtype=np.uint8)

        # -- Combine with the full semantic point cloud colored by label --
        if s._sem_pts is not None and s._sem_labels is not None:
            bg_pts = s._sem_pts
            bg_labels = s._sem_labels
            # Exclude wall-label points (they're shown as voxels already)
            wall_label_ids = set()
            for item in s.rv_wall_list.findItems("*", Qt.MatchWildcard):
                if item.checkState() == Qt.Checked:
                    try:
                        wall_label_ids.add(int(item.data(Qt.UserRole)))
                    except Exception:
                        pass
            if not wall_label_ids:
                wall_label_ids = {1}
            non_wall_mask = ~np.isin(bg_labels, np.array(sorted(wall_label_ids), dtype=np.int32))
            bg_pts = bg_pts[non_wall_mask]
            bg_labels = bg_labels[non_wall_mask]
            # Color by semantic label, dimmed to 40% brightness
            bg_colors = np.full((bg_pts.shape[0], 3), 50, dtype=np.uint8)
            legend = [("Reachable wall", (0, 200, 100)), ("Unreachable wall", (255, 70, 50))]
            for label_id, color in SEMANTIC_3D_COLORS.items():
                mask = bg_labels == label_id
                if np.any(mask):
                    bg_colors[mask] = (np.array(color, dtype=np.float32) * 0.4).astype(np.uint8)
                    if label_id not in (0, 1):  # skip unlabelled and wall
                        name = SEMANTIC_LABEL_NAMES.get(label_id, f"Label {label_id}")
                        legend.append((name, color))
            pts = np.concatenate([bg_pts, wall_pts], axis=0)
            colors = np.concatenate([bg_colors, wall_colors], axis=0)
            total = bg_pts.shape[0] + len(wall_band)
        else:
            pts = wall_pts
            colors = wall_colors
            legend = [("Reachable wall", (0, 200, 100)), ("Unreachable wall", (255, 70, 50))]
            total = len(wall_band)

        painted_pct = len(painted) / max(1, len(wall_band)) * 100
        label = (f"Vertical Coverage  |  Wall voxels: {len(wall_band):,}  |  "
                 f"Painted: {len(painted):,} ({painted_pct:.1f}%)")
        cloud = dict(
            points=pts,
            colors=colors,
            legend=legend,
            label=label,
            total_points=total,
            display_points=len(pts),
            sampled=False,
        )
        s._set_cloud("Vertical Coverage", cloud)

    def _rv_done(s, rv):
        s._rv_active = False
        s._rv_result = rv
        s.brv.setEnabled(True)
        s.rv_prog.setValue(100)
        s.rv_prog_lbl.setText("RII Vertical complete.")

        tcr_pct = rv.get("riiVertical", 0.0)
        sc_pct = rv.get("sc", 0.0) * 100.0
        painted = rv.get("painted_area_m2", 0.0)
        total_wall = rv.get("total_wall_area_m2", 0.0)
        rays_w = rv.get("rays_wall", 0)
        rays_o = rv.get("rays_obstacle", 0)
        rays_m = rv.get("rays_miss", 0)

        s.rv_riiv.setText(f"{tcr_pct:.1f}%")
        s.rv_riis.setText(
            f"{painted:.2f} / {total_wall:.2f} m²  |  "
            f"Surface Continuity (SC)={sc_pct:.0f}%  |  "
            f"Rays: {rays_w} wall, {rays_o} obstacle, {rays_m} miss  |  "
            f"{rv.get('ground_samples', 0)} ground samples"
        )
        s.rv_riif.show()

        # Combined card
        comb = rv.get("combined", {})
        s.rv_ch_val.setText(f"{comb.get('rii_h', 0):.1f}%")
        s.rv_cv_val.setText(f"{comb.get('rii_v', 0):.1f}%")
        s.rv_cc_val.setText(f"{comb.get('combined_paint', 0):.1f}%")
        s.rv_comb_detail.setText(
            f"Task Coverage Rate (TCR)={comb.get('tcr', 0):.1f}%  |  "
            f"Operational Efficiency (OE)={comb.get('oe', 0):.1f}%  |  "
            f"Surface Continuity (SC)={comb.get('sc', 0):.1f}%  |  "
            f"γ={comb.get('gamma', 0.5):.2f}  |  "
            f"Weighted avg={comb.get('weighted_avg', 0):.1f}%"
        )
        s.rv_combf.show()
        s._log(f"RII Vertical = {tcr_pct:.1f}%, Combined = {comb.get('combined_paint', 0):.1f}%", "success")
        # Show painted vs unpainted walls in the Vertical Coverage 3D window
        s._show_rv_painted_cloud(rv)

    def _rv_failed(s, msg):
        s._rv_active = False
        s.brv.setEnabled(True)
        s.rv_prog_lbl.setText("RII Vertical failed.")
        s.rv_prog_lbl.setStyleSheet("color:#dc2626;font-size:11px")
        s._log(f"RII Vertical failed: {msg}", "warn")

    def closeEvent(s, e):
        for w in s._wk:
            if hasattr(w, 'cancel'): w.cancel()
        for w in s._wk:
            if isinstance(w, QThread) and w.isRunning():
                w.wait(250)
        if s._cache_root and os.path.isdir(s._cache_root):
            shutil.rmtree(s._cache_root, ignore_errors=True)
        e.accept()
