"""Interactive editor for `core.map_merge.merge_two_maps`.

Shows Map A as a static background and Map B as a movable overlay.
The user can drag B around, rotate it about its centre, and brush-erase
any cells they don't want included before committing the merge.

Scene coordinate system: 1 scene unit = 1 pixel of Map A (so A is drawn
1:1). All world<->scene conversions use A's resolution / origin.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QTimer
from PyQt5.QtGui import (
    QBrush, QColor, QImage, QPainter, QPen, QPixmap, QTransform,
)
from PyQt5.QtWidgets import (
    QDialog, QDoubleSpinBox, QFileDialog, QGraphicsEllipseItem,
    QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QRadioButton,
    QSlider, QVBoxLayout,
)

from core.map_io import parse_pgm, parse_yaml
from core.map_merge import merge_two_maps


_NAV2_UNKNOWN = 205


def _pgm_to_rgba(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> QImage:
    """uint8 PGM (h, w) → RGBA QImage.

    mask: optional (h, w) uint8 with 1=keep, 0=erased. Erased cells become
    fully transparent; unknown cells become semi-transparent so the layer
    underneath shows through.
    """
    h, w = arr.shape
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = arr
    rgba[..., 1] = arr
    rgba[..., 2] = arr
    rgba[..., 3] = 255
    unknown = (arr == _NAV2_UNKNOWN)
    rgba[unknown, 3] = 70  # see-through grey
    if mask is not None:
        rgba[mask == 0, 3] = 0  # erased → invisible
    qimg = QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
    return qimg.copy()  # detach from the temporary numpy buffer


class _MergeView(QGraphicsView):
    """QGraphicsView with wheel-zoom, middle-mouse pan, and parent-delegated edit events."""

    def __init__(self, dialog, scene):
        super().__init__(scene)
        self._dialog = dialog
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

    def wheelEvent(self, e):
        zoom = 1.25 if e.angleDelta().y() > 0 else 0.8
        self.scale(zoom, zoom)

    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(e)
            return
        scene_pos = self.mapToScene(e.pos())
        self._dialog._on_mouse_press(e.button(), scene_pos)

    def mouseMoveEvent(self, e):
        scene_pos = self.mapToScene(e.pos())
        self._dialog._on_mouse_move(e.buttons(), scene_pos)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            super().mouseReleaseEvent(e)
            self.setDragMode(QGraphicsView.NoDrag)
            return
        self._dialog._on_mouse_release(e.button())


class MapMergeDialog(QDialog):
    """Drag / rotate / erase Map B over Map A, then commit a merged map."""

    MODE_MOVE = 0
    MODE_ROTATE = 1
    MODE_ERASE = 2

    def __init__(self, parent, pgm_a: str, yaml_a: str, pgm_b: str, yaml_b: str):
        super().__init__(parent)
        self.setWindowTitle("Merge Two Maps")
        self.resize(1300, 850)

        # ── Load inputs ───────────────────────────────────────
        wA, hA, pixA = parse_pgm(pgm_a)
        wB, hB, pixB = parse_pgm(pgm_b)
        yA = parse_yaml(yaml_a)
        yB = parse_yaml(yaml_b)

        self._pgm_a, self._yaml_a = pgm_a, yaml_a
        self._pgm_b, self._yaml_b = pgm_b, yaml_b
        self._wA, self._hA = wA, hA
        self._wB, self._hB = wB, hB
        self._arrA = pixA.reshape(hA, wA).copy()
        self._arrB = pixB.reshape(hB, wB).copy()
        self._maskB = np.ones((hB, wB), dtype=np.uint8)  # 1 = keep

        self._resA = float(yA["resolution"])
        self._resB = float(yB["resolution"])
        self._oxA, self._oyA = float(yA["origin"][0]), float(yA["origin"][1])
        self._oxB, self._oyB = float(yB["origin"][0]), float(yB["origin"][1])
        self._bcx0 = self._oxB + wB * self._resB / 2.0
        self._bcy0 = self._oyB + hB * self._resB / 2.0

        # User-controlled transform
        self._tx_m = 0.0   # world-frame translate
        self._ty_m = 0.0
        self._theta_deg = 0.0  # CCW-positive in world frame
        self._brush_radius_m = 0.30
        self._mode = self.MODE_MOVE

        # Drag state
        self._drag_origin: Optional[QPointF] = None
        self._drag_tx0 = 0.0
        self._drag_ty0 = 0.0
        self._rot_pivot_scene: Optional[QPointF] = None
        self._rot_start_angle = 0.0
        self._rot_theta0 = 0.0

        # Output paths (set on accept)
        self._out_pgm: Optional[str] = None
        self._out_yaml: Optional[str] = None

        # ── Scene + items ─────────────────────────────────────
        self._scene = QGraphicsScene(self)
        self._scene.setBackgroundBrush(QColor(48, 48, 48))

        self._a_item = QGraphicsPixmapItem(QPixmap.fromImage(_pgm_to_rgba(self._arrA)))
        self._a_item.setZValue(0)
        self._scene.addItem(self._a_item)

        self._b_item = QGraphicsPixmapItem(QPixmap.fromImage(_pgm_to_rgba(self._arrB, self._maskB)))
        self._b_item.setZValue(1)
        self._b_item.setOpacity(0.90)
        self._scene.addItem(self._b_item)

        self._brush_item = QGraphicsEllipseItem()
        self._brush_item.setPen(QPen(QColor(255, 80, 80), 2))
        self._brush_item.setBrush(QBrush(QColor(255, 80, 80, 60)))
        self._brush_item.setZValue(2)
        self._brush_item.hide()
        self._scene.addItem(self._brush_item)

        self._view = _MergeView(self, self._scene)

        # ── Side controls ─────────────────────────────────────
        controls = QGroupBox("Tools")
        cv = QVBoxLayout(controls)

        self._b_move = QRadioButton("Move Map B")
        self._b_rot = QRadioButton("Rotate Map B")
        self._b_erase = QRadioButton("Erase Map B")
        self._b_move.setChecked(True)
        self._b_move.toggled.connect(lambda v: v and self._set_mode(self.MODE_MOVE))
        self._b_rot.toggled.connect(lambda v: v and self._set_mode(self.MODE_ROTATE))
        self._b_erase.toggled.connect(lambda v: v and self._set_mode(self.MODE_ERASE))
        for w in (self._b_move, self._b_rot, self._b_erase):
            cv.addWidget(w)

        cv.addSpacing(8)
        cv.addWidget(QLabel("<b>Rotation</b>"))
        rotrow = QHBoxLayout()
        self._rot_slider = QSlider(Qt.Horizontal)
        self._rot_slider.setRange(-180, 180)
        self._rot_slider.setValue(0)
        self._rot_slider.valueChanged.connect(self._on_rot_slider)
        self._rot_spin = QDoubleSpinBox()
        self._rot_spin.setRange(-180.0, 180.0)
        self._rot_spin.setSuffix(" °")
        self._rot_spin.setSingleStep(0.5)
        self._rot_spin.setDecimals(1)
        self._rot_spin.valueChanged.connect(self._on_rot_spin)
        rotrow.addWidget(self._rot_slider, 1)
        rotrow.addWidget(self._rot_spin)
        cv.addLayout(rotrow)

        cv.addSpacing(8)
        cv.addWidget(QLabel("<b>Eraser brush</b>"))
        self._brush_spin = QDoubleSpinBox()
        self._brush_spin.setRange(0.05, 5.0)
        self._brush_spin.setSuffix(" m")
        self._brush_spin.setSingleStep(0.05)
        self._brush_spin.setDecimals(2)
        self._brush_spin.setValue(self._brush_radius_m)
        self._brush_spin.valueChanged.connect(lambda v: setattr(self, "_brush_radius_m", float(v)))
        cv.addWidget(self._brush_spin)

        cv.addSpacing(8)
        btn_reset_xform = QPushButton("Reset position / rotation")
        btn_reset_xform.clicked.connect(self._reset_transform)
        cv.addWidget(btn_reset_xform)

        btn_restore = QPushButton("Restore erased cells")
        btn_restore.clicked.connect(self._restore_mask)
        cv.addWidget(btn_restore)

        btn_fit = QPushButton("Fit view")
        btn_fit.clicked.connect(self._fit_view)
        cv.addWidget(btn_fit)

        cv.addStretch()

        # ── Bottom buttons ────────────────────────────────────
        self._status = QLabel("Drag Map B • wheel = zoom • middle-drag = pan")
        self._status.setStyleSheet("color: #555;")

        btn_save = QPushButton("Save merged map…")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self._save_merged)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addStretch(); btn_row.addWidget(btn_cancel); btn_row.addWidget(btn_save)

        # ── Root layout ───────────────────────────────────────
        left = QVBoxLayout()
        left.addWidget(controls)

        right = QVBoxLayout()
        right.addWidget(self._view, 1)
        right.addWidget(self._status)
        right.addLayout(btn_row)

        root = QHBoxLayout(self)
        root.addLayout(left, 0)
        root.addLayout(right, 1)

        self._refresh_b_transform()
        QTimer.singleShot(0, self._fit_view)

    # ── Public ────────────────────────────────────────────────
    def output_paths(self) -> Tuple[Optional[str], Optional[str]]:
        return self._out_pgm, self._out_yaml

    # ── Coordinate helpers ────────────────────────────────────
    def _scene_to_world(self, pt: QPointF) -> Tuple[float, float]:
        wx = self._oxA + pt.x() * self._resA
        wy = (self._oyA + self._hA * self._resA) - pt.y() * self._resA
        return wx, wy

    def _world_to_scene(self, wx: float, wy: float) -> Tuple[float, float]:
        sx = (wx - self._oxA) / self._resA
        sy = ((self._oyA + self._hA * self._resA) - wy) / self._resA
        return sx, sy

    # ── Render ────────────────────────────────────────────────
    def _refresh_b_transform(self):
        cx_world = self._bcx0 + self._tx_m
        cy_world = self._bcy0 + self._ty_m
        scx, scy = self._world_to_scene(cx_world, cy_world)
        scale = self._resB / self._resA  # A-pixels per B-pixel
        t = QTransform()
        t.translate(scx, scy)
        # Qt rotate() is CW in Y-down; theta is CCW in world (Y-up) → flip sign.
        t.rotate(-self._theta_deg)
        t.scale(scale, scale)
        t.translate(-self._wB / 2.0, -self._hB / 2.0)
        self._b_item.setTransform(t)

    def _refresh_b_pixmap(self):
        self._b_item.setPixmap(QPixmap.fromImage(_pgm_to_rgba(self._arrB, self._maskB)))

    def _fit_view(self):
        rect = self._scene.itemsBoundingRect()
        if not rect.isEmpty():
            self._view.fitInView(rect, Qt.KeepAspectRatio)

    # ── Mode + widgets ────────────────────────────────────────
    def _set_mode(self, m: int):
        self._mode = m
        self._brush_item.setVisible(m == self.MODE_ERASE)
        if m == self.MODE_MOVE:
            self._status.setText("MOVE — drag Map B to reposition it")
        elif m == self.MODE_ROTATE:
            self._status.setText("ROTATE — drag to rotate Map B about its centre")
        else:
            self._status.setText("ERASE — drag to remove cells from Map B")

    def _on_rot_slider(self, v: int):
        if abs(self._theta_deg - float(v)) < 1e-6:
            return
        self._theta_deg = float(v)
        self._rot_spin.blockSignals(True); self._rot_spin.setValue(float(v)); self._rot_spin.blockSignals(False)
        self._refresh_b_transform()

    def _on_rot_spin(self, v: float):
        if abs(self._theta_deg - float(v)) < 1e-6:
            return
        self._theta_deg = float(v)
        self._rot_slider.blockSignals(True); self._rot_slider.setValue(int(round(v))); self._rot_slider.blockSignals(False)
        self._refresh_b_transform()

    def _reset_transform(self):
        self._tx_m = 0.0
        self._ty_m = 0.0
        self._theta_deg = 0.0
        self._rot_slider.blockSignals(True); self._rot_slider.setValue(0); self._rot_slider.blockSignals(False)
        self._rot_spin.blockSignals(True); self._rot_spin.setValue(0.0); self._rot_spin.blockSignals(False)
        self._refresh_b_transform()

    def _restore_mask(self):
        self._maskB.fill(1)
        self._refresh_b_pixmap()

    # ── Mouse events ──────────────────────────────────────────
    def _on_mouse_press(self, button, scene_pos: QPointF):
        if button != Qt.LeftButton:
            return
        if self._mode == self.MODE_MOVE:
            self._drag_origin = scene_pos
            self._drag_tx0 = self._tx_m
            self._drag_ty0 = self._ty_m
        elif self._mode == self.MODE_ROTATE:
            cx, cy = self._world_to_scene(self._bcx0 + self._tx_m, self._bcy0 + self._ty_m)
            self._rot_pivot_scene = QPointF(cx, cy)
            self._rot_start_angle = math.degrees(math.atan2(
                scene_pos.y() - cy, scene_pos.x() - cx))
            self._rot_theta0 = self._theta_deg
        elif self._mode == self.MODE_ERASE:
            self._apply_eraser(scene_pos)

    def _on_mouse_move(self, buttons, scene_pos: QPointF):
        if self._mode == self.MODE_ERASE:
            r_scene = self._brush_radius_m / self._resA
            self._brush_item.setRect(
                scene_pos.x() - r_scene, scene_pos.y() - r_scene,
                2 * r_scene, 2 * r_scene)
            self._brush_item.show()

        if not (buttons & Qt.LeftButton):
            return

        if self._mode == self.MODE_MOVE and self._drag_origin is not None:
            dx_scene = scene_pos.x() - self._drag_origin.x()
            dy_scene = scene_pos.y() - self._drag_origin.y()
            self._tx_m = self._drag_tx0 + dx_scene * self._resA
            self._ty_m = self._drag_ty0 + (-dy_scene) * self._resA  # screen Y is inverted
            self._refresh_b_transform()
            self._status.setText(
                f"Map B offset: Δx={self._tx_m:+.2f} m, Δy={self._ty_m:+.2f} m, θ={self._theta_deg:+.1f}°")
        elif self._mode == self.MODE_ROTATE and self._rot_pivot_scene is not None:
            curr = math.degrees(math.atan2(
                scene_pos.y() - self._rot_pivot_scene.y(),
                scene_pos.x() - self._rot_pivot_scene.x()))
            # screen atan2 grows clockwise; world θ is CCW → subtract delta
            delta_screen = curr - self._rot_start_angle
            self._theta_deg = self._rot_theta0 - delta_screen
            while self._theta_deg > 180:
                self._theta_deg -= 360
            while self._theta_deg < -180:
                self._theta_deg += 360
            self._rot_slider.blockSignals(True); self._rot_slider.setValue(int(round(self._theta_deg))); self._rot_slider.blockSignals(False)
            self._rot_spin.blockSignals(True); self._rot_spin.setValue(self._theta_deg); self._rot_spin.blockSignals(False)
            self._refresh_b_transform()
            self._status.setText(
                f"Map B rotation: θ={self._theta_deg:+.1f}°  (Δx={self._tx_m:+.2f} m, Δy={self._ty_m:+.2f} m)")
        elif self._mode == self.MODE_ERASE:
            self._apply_eraser(scene_pos)

    def _on_mouse_release(self, button):
        self._drag_origin = None
        self._rot_pivot_scene = None

    # ── Eraser ────────────────────────────────────────────────
    def _apply_eraser(self, scene_pos: QPointF):
        wx, wy = self._scene_to_world(scene_pos)
        # Un-transform: relative to current world centre, rotate by -theta.
        cx_world = self._bcx0 + self._tx_m
        cy_world = self._bcy0 + self._ty_m
        dx = wx - cx_world
        dy = wy - cy_world
        theta = math.radians(self._theta_deg)
        c, s = math.cos(theta), math.sin(theta)
        # R(-theta) · (dx, dy) = (c·dx + s·dy, -s·dx + c·dy)
        rx = c * dx + s * dy
        ry = -s * dx + c * dy
        P_x = self._bcx0 + rx
        P_y = self._bcy0 + ry
        col0 = int(round((P_x - self._oxB) / self._resB))
        row0 = int(round((self._oyB + self._hB * self._resB - P_y) / self._resB))
        radius_px = max(1, int(round(self._brush_radius_m / self._resB)))

        rr_min = max(0, row0 - radius_px)
        rr_max = min(self._hB, row0 + radius_px + 1)
        cc_min = max(0, col0 - radius_px)
        cc_max = min(self._wB, col0 + radius_px + 1)
        if rr_min >= rr_max or cc_min >= cc_max:
            return

        rows = np.arange(rr_min, rr_max)[:, np.newaxis]
        cols = np.arange(cc_min, cc_max)[np.newaxis, :]
        circle = (rows - row0) ** 2 + (cols - col0) ** 2 <= radius_px ** 2
        self._maskB[rr_min:rr_max, cc_min:cc_max][circle] = 0
        self._refresh_b_pixmap()

    # ── Save ──────────────────────────────────────────────────
    def _save_merged(self):
        default_out = os.path.join(os.path.dirname(self._pgm_a), "merged_map.pgm")
        out_pgm, _ = QFileDialog.getSaveFileName(
            self, "Save merged map as…", default_out, "PGM (*.pgm)")
        if not out_pgm:
            return
        if not out_pgm.lower().endswith(".pgm"):
            out_pgm += ".pgm"
        out_yaml = os.path.splitext(out_pgm)[0] + ".yaml"

        try:
            stats = merge_two_maps(
                self._pgm_a, self._yaml_a,
                self._pgm_b, self._yaml_b,
                out_pgm, out_yaml,
                b_translate_world=(self._tx_m, self._ty_m),
                b_rotation_deg=self._theta_deg,
                b_mask=self._maskB,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Merge failed", str(e))
            return

        self._out_pgm = out_pgm
        self._out_yaml = out_yaml
        QMessageBox.information(
            self, "Merged",
            f"Wrote {os.path.basename(out_pgm)}\n"
            f"{stats['width']}×{stats['height']} @ {stats['resolution']:.3f} m\n"
            f"Occupied: {stats['occupied_area_m2']:.1f} m²\n"
            f"Free:     {stats['free_area_m2']:.1f} m²"
        )
        self.accept()
