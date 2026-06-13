"""Custom GUI widgets — MapW, DragScrollArea, PointCloudPreviewW, PointCloudW."""

import os
import math
import numpy as np
from PyQt5.QtWidgets import QWidget, QScrollArea
from PyQt5.QtCore import Qt, QRect, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPainterPath, QPen, QColor, QBrush, QPolygonF, QFont,
)

from core.semantic_selection import (
    selection_bounds_px,
    selection_to_screen_path,
)

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as pgl
    from pyqtgraph import Vector
    PYQTGRAPH_GL_AVAILABLE = True
except Exception:
    pg = None
    pgl = None
    Vector = None
    PYQTGRAPH_GL_AVAILABLE = False


def _height_colormap(z_vals: np.ndarray) -> np.ndarray:
    """Map Z values to a blue→cyan→green→yellow→red heatmap. Returns (N, 3) float32 in [0,1]."""
    z_lo, z_hi = np.percentile(z_vals, [5, 95])
    if z_hi <= z_lo:
        z_hi = z_lo + 1.0
    t = np.clip((z_vals - z_lo) / (z_hi - z_lo), 0.0, 1.0).astype(np.float32)
    # 4-segment linear interpolation: blue→cyan→green→yellow→red
    r = np.where(t < 0.25, 0.0,
        np.where(t < 0.50, (t - 0.25) * 4.0,
        np.where(t < 0.75, 1.0,
                           1.0))).astype(np.float32)
    g = np.where(t < 0.25, t * 4.0,
        np.where(t < 0.50, 1.0,
        np.where(t < 0.75, 1.0 - (t - 0.50) * 4.0,
                           0.0))).astype(np.float32)
    b = np.where(t < 0.25, 1.0,
        np.where(t < 0.50, 1.0 - (t - 0.25) * 4.0,
        np.where(t < 0.75, 0.0,
                           0.0))).astype(np.float32)
    return np.column_stack((r, g, b)).astype(np.float32)


class MapW(QWidget):
    sel_changed = pyqtSignal(object)
    hover_coords = pyqtSignal(int, int, float, float)  # pixel_x, pixel_y, world_x, world_y
    start_picked = pyqtSignal(int, int, float, float)  # pixel_x, pixel_y, world_x, world_y
    def __init__(s):
        super().__init__()
        s.setMinimumSize(200, 200)
        s.setMouseTracking(True)
        s._map_resolution = 0.05
        s._map_origin = (0.0, 0.0)
        s._map_hw = (0, 0)
        s._pick_start_mode = False
        s._start_point = None  # (px, py) in image coords
        s.setFocusPolicy(Qt.StrongFocus)
        s._bp = None
        s._sm = False
        s._drag_mode = None
        s._ds = s._de = None
        s._poly = []
        s._last_pos = None
        s.sel = None
        s._selection_mode = "rectangle"
        s._zoom = 1.0
        s._pan = np.array([0.0, 0.0], dtype=np.float32)
        s._focus_rect = None
        s._focus_label = ""
        s._edit_active = False
        s._edit_mode = "draw"
        s._brush_shape = "circle"
        s._brush_size = 10
        s._brush_rect_w = 10
        s._brush_rect_h = 10
        s._edit_overlay = None
        s._edit_strokes = []
        s._last_edit_pt = None
        # Undo / redo for brush edits (snapshot of _edit_overlay per stroke)
        s._edit_undo = []
        s._edit_redo = []
        s._edit_undo_cap = 20
        # Manual relocation mode (Step 4 drag-and-drop). When active, the
        # owner installs callbacks that accept (col, row) image-coords on
        # press / move / release. Press returns True if it picked up an
        # object; on True we route subsequent moves and the release to the
        # owner's handlers instead of starting a pan/select drag.
        s.manual_relocate_active  = False
        s.manual_relocate_press   = None
        s.manual_relocate_move    = None
        s.manual_relocate_release = None
        # Drag-ghost overlay: a small RGBA QImage of the picked-up object,
        # drawn translated by offset_yx (in BASE map pixels). Updated on
        # every mouse-move so the drag feels live, independent of BFS.
        s._drag_ghost_img        = None     # QImage of bbox-sized RGBA blob
        s._drag_ghost_origin_yx  = None     # (row, col) of QImage top-left in base map coords
        s._drag_ghost_offset_yx  = (0, 0)   # (dy, dx) — added to origin each paint
        s._ref_overlay = None
        s._ref_overlay_visible = False
        s._transition_overlay = None   # QImage for ramp/step overlay
        s._transition_labels = []      # list of (screen_x, screen_y, text, color)
        s._update_cursor()
    def set_qi(s, qi):
        old_size = s._bp.size() if s._bp is not None else None
        s._bp = QPixmap.fromImage(qi)
        if old_size is None or old_size != s._bp.size():
            s.reset_view()
        else:
            s.update()
    def set_selection_mode(s, mode):
        s._selection_mode = "freeform" if mode == "freeform" else "rectangle"
        s.update()
    def enable_sel(s, mode=None):
        if mode is not None:
            s.set_selection_mode(mode)
        s._sm = True
        s._poly = []
        s._update_cursor()
    def clear_sel(s):
        s.sel = None
        s._ds = s._de = None
        s._poly = []
        s.update()
    def clear_focus(s):
        s._focus_rect = None
        s._focus_label = ""
        s.update()
    def reset_view(s):
        s._zoom = 1.0
        s._pan[:] = 0.0
        s.update()
    def focus_rect(s, rect, label=""):
        if not rect or not s._bp or s._bp.isNull():
            return
        x1, y1, x2, y2 = rect
        x1 = max(0.0, min(float(x1), s._bp.width() - 1.0))
        x2 = max(0.0, min(float(x2), s._bp.width() - 1.0))
        y1 = max(0.0, min(float(y1), s._bp.height() - 1.0))
        y2 = max(0.0, min(float(y2), s._bp.height() - 1.0))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        bw = max(6.0, x2 - x1 + 1.0)
        bh = max(6.0, y2 - y1 + 1.0)
        fit = min(s.width() / max(1, s._bp.width()), s.height() / max(1, s._bp.height()))
        fit = max(fit, 1e-6)
        target_scale = min(s.width() / (bw * 2.8), s.height() / (bh * 2.8))
        s._zoom = max(0.4, min(20.0, target_scale / fit))
        scale = fit * s._zoom
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        ox = (s.width() - s._bp.width() * scale) * 0.5
        oy = (s.height() - s._bp.height() * scale) * 0.5
        s._pan[0] = (s.width() * 0.5) - (ox + cx * scale)
        s._pan[1] = (s.height() * 0.5) - (oy + cy * scale)
        s._focus_rect = (x1, y1, x2, y2)
        s._focus_label = label
        s.update()
    def enable_edit(s, overlay_array=None):
        if overlay_array is not None:
            s._edit_overlay = np.asarray(overlay_array, dtype=np.uint8).copy()
        elif s._bp and not s._bp.isNull():
            h, w = s._bp.height(), s._bp.width()
            s._edit_overlay = np.full((h, w), 127, dtype=np.uint8)
        s._edit_active = True
        s._edit_undo.clear(); s._edit_redo.clear()
        # Hide / clear any blue overlay that could otherwise paint on top of
        # brush strokes and look like a bug (selection polygons, focus
        # rectangles around ramps, in-progress drag previews, start-point
        # marker, in-progress polygons).
        s._sm = False
        s._poly = []
        s._ds = None
        s._de = None
        s.sel = None
        s._focus_rect = None
        s._focus_label = ""
        s._start_point = None
        s._update_cursor()
        s.update()

    def disable_edit(s):
        s._edit_active = False
        s._edit_undo.clear(); s._edit_redo.clear()
        s._update_cursor()
        s.update()

    # ── Undo / redo for brush edits ─────────────────────────────────────
    def _snapshot_edit(s):
        """Push the current overlay onto the undo stack (called before each stroke)."""
        if s._edit_overlay is None:
            return
        s._edit_undo.append(s._edit_overlay.copy())
        if len(s._edit_undo) > s._edit_undo_cap:
            s._edit_undo.pop(0)
        s._edit_redo.clear()

    def undo_edit(s) -> bool:
        """Restore the previous edit-overlay snapshot. Returns True on success."""
        if not s._edit_active or s._edit_overlay is None or not s._edit_undo:
            return False
        s._edit_redo.append(s._edit_overlay.copy())
        s._edit_overlay = s._edit_undo.pop()
        s.update()
        return True

    def redo_edit(s) -> bool:
        """Re-apply the last undone stroke."""
        if not s._edit_active or s._edit_overlay is None or not s._edit_redo:
            return False
        s._edit_undo.append(s._edit_overlay.copy())
        s._edit_overlay = s._edit_redo.pop()
        s.update()
        return True

    def can_undo_edit(s) -> bool:
        return s._edit_active and bool(s._edit_undo)

    def can_redo_edit(s) -> bool:
        return s._edit_active and bool(s._edit_redo)
    def get_edit_overlay(s):
        return s._edit_overlay
    def set_edit_mode(s, mode):
        s._edit_mode = mode
        s._update_cursor()
    def set_brush_shape(s, shape):
        s._brush_shape = shape
        s._update_cursor()
    def set_brush_size(s, size):
        s._brush_size = max(1, int(size))
        s._update_cursor()
    def set_brush_rect_size(s, half_w, half_h):
        s._brush_rect_w = max(1, int(half_w))
        s._brush_rect_h = max(1, int(half_h))
        s._update_cursor()
    def set_transition_overlay(s, qi, labels=None):
        """Set a semi-transparent RGBA QImage overlay showing ramp/step regions."""
        s._transition_overlay = qi
        s._transition_labels = labels or []
        s.update()

    def clear_transition_overlay(s):
        s._transition_overlay = None
        s._transition_labels = []
        s.update()

    def set_reference_overlay(s, qi):
        s._ref_overlay = qi
        s.update()
    def set_reference_overlay_visible(s, visible):
        s._ref_overlay_visible = bool(visible)
        s.update()
    def _paint_brush(s, ix, iy):
        if s._edit_overlay is None:
            return
        h, w = s._edit_overlay.shape
        r = s._brush_size
        val = 0 if s._edit_mode == "draw" else 254
        if s._brush_shape == "circle":
            yy, xx = np.ogrid[-r:r+1, -r:r+1]
            mask = xx*xx + yy*yy <= r*r
            y0, x0 = iy - r, ix - r
            for dy in range(mask.shape[0]):
                for dx in range(mask.shape[1]):
                    py, px = y0 + dy, x0 + dx
                    if mask[dy, dx] and 0 <= py < h and 0 <= px < w:
                        s._edit_overlay[py, px] = val
        elif s._brush_shape == "rectangle":
            rw = s._brush_rect_w
            rh = s._brush_rect_h
            y0 = max(0, iy - rh); y1 = min(h, iy + rh + 1)
            x0 = max(0, ix - rw); x1 = min(w, ix + rw + 1)
            s._edit_overlay[y0:y1, x0:x1] = val
        else:
            if 0 <= iy < h and 0 <= ix < w:
                s._edit_overlay[iy, ix] = val
        s.update()
    def _update_cursor(s):
        if s._edit_active:
            # Show brush size as a circle cursor
            m = s._metrics()
            if m:
                scale = m[0]
                if s._brush_shape == "rectangle":
                    sz = max(int(max(s._brush_rect_w, s._brush_rect_h) * 2 * scale), 8)
                else:
                    sz = max(int(s._brush_size * 2 * scale), 8)
                sz = min(sz, 128)  # cap cursor size
                from PyQt5.QtGui import QPixmap, QCursor
                pm = QPixmap(sz, sz)
                pm.fill(QColor(0, 0, 0, 0))
                p = QPainter(pm)
                p.setPen(QPen(QColor(37, 99, 235, 180), 2))
                p.drawEllipse(1, 1, sz - 2, sz - 2)
                p.drawLine(sz // 2 - 3, sz // 2, sz // 2 + 3, sz // 2)
                p.drawLine(sz // 2, sz // 2 - 3, sz // 2, sz // 2 + 3)
                p.end()
                s.setCursor(QCursor(pm, sz // 2, sz // 2))
            else:
                s.setCursor(Qt.CrossCursor)
        elif s._sm:
            s.setCursor(Qt.CrossCursor)
        elif s._bp:
            s.setCursor(Qt.OpenHandCursor)
        else:
            s.setCursor(Qt.ArrowCursor)
    def _metrics(s):
        if not s._bp or s._bp.isNull():
            return None
        fit = min(s.width() / max(1, s._bp.width()), s.height() / max(1, s._bp.height()))
        fit = max(fit, 1e-6)
        scale = fit * s._zoom
        sw = s._bp.width() * scale
        sh = s._bp.height() * scale
        ox = (s.width() - sw) * 0.5 + float(s._pan[0])
        oy = (s.height() - sh) * 0.5 + float(s._pan[1])
        return scale, ox, oy, sw, sh
    def _image_xy_float(s, pos, clamp=False):
        m = s._metrics()
        if not m:
            return None
        scale, ox, oy, _, _ = m
        x = (pos.x() - ox) / scale
        y = (pos.y() - oy) / scale
        if clamp:
            x = max(0.0, min(x, s._bp.width() - 1))
            y = max(0.0, min(y, s._bp.height() - 1))
        elif not (0.0 <= x < s._bp.width() and 0.0 <= y < s._bp.height()):
            return None
        return x, y
    def _ic(s, pos, clamp=False):
        xy = s._image_xy_float(pos, clamp=clamp)
        if xy is None:
            return None
        return (
            int(round(xy[0])),
            int(round(xy[1])),
        )
    def enable_start_pick(s):
        s._pick_start_mode = True
        s.setCursor(Qt.CrossCursor)

    def mousePressEvent(s, e):
        if not s._bp:
            return
        # Manual relocation: pick up a candidate (if cursor is over one).
        # The owner-supplied callback returns True when an object was picked.
        if (s.manual_relocate_active and e.button() == Qt.LeftButton
                and callable(s.manual_relocate_press)):
            pt = s._ic(e.pos(), clamp=True)
            if pt is not None and s.manual_relocate_press(pt):
                s._drag_mode = "manual_relocate"
                s.setCursor(Qt.ClosedHandCursor)
                s.update()
                return
        # Start point picking mode
        if s._pick_start_mode and e.button() == Qt.LeftButton:
            pt = s._ic(e.pos(), clamp=True)
            if pt is not None:
                px, py = pt
                s._start_point = (px, py)
                s._pick_start_mode = False
                s.setCursor(Qt.ArrowCursor)
                h, w_map = s._map_hw
                if h > 0:
                    world_x = s._map_origin[0] + px * s._map_resolution
                    world_y = s._map_origin[1] + (h - 1 - py) * s._map_resolution
                    s.start_picked.emit(px, py, world_x, world_y)
                s.update()
            return
        if s._edit_active and e.button() == Qt.LeftButton:
            pt = s._ic(e.pos(), clamp=True)
            if pt:
                s._drag_mode = "edit"
                s._last_edit_pt = pt
                s._snapshot_edit()   # push pre-stroke snapshot onto undo stack
                s._paint_brush(pt[0], pt[1])
            return
        if s._sm and e.button() == Qt.LeftButton:
            pt = s._ic(e.pos(), clamp=False)
            if pt is None:
                return
            s._drag_mode = "select"
            if s._selection_mode == "freeform":
                s._poly = [pt]
                s._ds = s._de = None
            else:
                s._ds = pt
                s._de = pt
            return
        if e.button() in (Qt.LeftButton, Qt.MiddleButton, Qt.RightButton):
            s._drag_mode = "pan"
            s._last_pos = e.pos()
            s.setCursor(Qt.ClosedHandCursor)
    def set_map_metadata(s, resolution, origin, height, width):
        s._map_resolution = resolution
        s._map_origin = (origin[0], origin[1])
        s._map_hw = (height, width)

    # ── Drag-ghost API (Step 4 manual relocation) ──
    def begin_drag_ghost(s, ghost_img, origin_yx):
        """Install a translucent ghost image for the dragged object.

        ghost_img : QImage (bbox-sized RGBA) of the cells being moved
        origin_yx : (row, col) of the QImage top-left in base map coords
        """
        s._drag_ghost_img       = ghost_img
        s._drag_ghost_origin_yx = origin_yx
        s._drag_ghost_offset_yx = (0, 0)
        s.update()

    def update_drag_ghost_offset(s, offset_yx):
        s._drag_ghost_offset_yx = offset_yx
        s.update()

    def clear_drag_ghost(s):
        s._drag_ghost_img       = None
        s._drag_ghost_origin_yx = None
        s._drag_ghost_offset_yx = (0, 0)
        s.update()

    def mouseMoveEvent(s, e):
        # Emit world coordinates on hover
        pt = s._ic(e.pos())
        if pt is not None and s._map_hw[0] > 0:
            px, py = pt
            h, w = s._map_hw
            world_x = s._map_origin[0] + px * s._map_resolution
            world_y = s._map_origin[1] + (h - 1 - py) * s._map_resolution
            s.hover_coords.emit(px, py, world_x, world_y)

        if s._drag_mode == "manual_relocate":
            if not (e.buttons() & Qt.LeftButton):
                # Lost mouse — treat as release at last known position
                s._drag_mode = None
                s.setCursor(Qt.ArrowCursor)
                if callable(s.manual_relocate_release):
                    pt2 = s._ic(e.pos(), clamp=True)
                    if pt2 is not None:
                        s.manual_relocate_release(pt2)
                return
            pt2 = s._ic(e.pos(), clamp=True)
            if pt2 is not None and callable(s.manual_relocate_move):
                s.manual_relocate_move(pt2)
            s.update()
            return
        if s._drag_mode == "edit":
            # Defensive: if the mouse button is no longer pressed (e.g. user
            # released outside the widget), end the stroke instead of
            # interpolating a long straight line back to the stale anchor.
            if not (e.buttons() & Qt.LeftButton):
                s._drag_mode = None
                s._last_edit_pt = None
                return
            pt2 = s._ic(e.pos(), clamp=True)
            if pt2:
                # Interpolate between last edit point and current for continuous stroke
                last = getattr(s, '_last_edit_pt', None)
                if last and last != pt2:
                    x0, y0 = last
                    x1, y1 = pt2
                    dx = abs(x1 - x0)
                    dy = abs(y1 - y0)
                    steps = max(dx, dy)
                    # Cap interpolation length: if the gap is huge (lost-mouse,
                    # window switch, etc.), just stamp the new position rather
                    # than drawing a line across the whole map.
                    MAX_INTERPOLATE_PX = 64
                    if steps > MAX_INTERPOLATE_PX:
                        s._paint_brush(pt2[0], pt2[1])
                    elif steps > 0:
                        for i in range(1, steps + 1):
                            t = i / steps
                            ix = int(round(x0 + t * (x1 - x0)))
                            iy = int(round(y0 + t * (y1 - y0)))
                            s._paint_brush(ix, iy)
                    else:
                        s._paint_brush(pt2[0], pt2[1])
                else:
                    s._paint_brush(pt2[0], pt2[1])
                s._last_edit_pt = pt2
            return
        if s._drag_mode == "select":
            # Defensive: if the mouse button isn't actually held any more
            # (user released the mouse outside the widget), end the drag
            # before tacking another point onto the freeform polygon —
            # otherwise every later hover-move appends a vertex, drawing
            # a long straight edge across the map.
            if not (e.buttons() & Qt.LeftButton):
                s._drag_mode = None
                return
            if s._selection_mode == "freeform":
                pt2 = s._ic(e.pos(), clamp=True)
                if pt2 is not None:
                    last = s._poly[-1] if s._poly else None
                    if last is None:
                        s._poly.append(pt2)
                    else:
                        gap = max(abs(pt2[0] - last[0]), abs(pt2[1] - last[1]))
                        # Skip giant jumps (focus loss / cursor warp). A real
                        # freehand sweep produces dense small steps; a 200-px
                        # jump in one event means we've lost mouse tracking.
                        SELECT_GAP_MAX = 64
                        if gap >= 2 and gap <= SELECT_GAP_MAX:
                            s._poly.append(pt2)
            else:
                s._de = s._ic(e.pos(), clamp=True)
            s.update()
            return
        if s._drag_mode == "pan" and s._last_pos is not None:
            delta = e.pos() - s._last_pos
            s._pan[0] += delta.x()
            s._pan[1] += delta.y()
            s._last_pos = e.pos()
            s.update()
    def mouseReleaseEvent(s, e):
        if s._drag_mode == "manual_relocate":
            s._drag_mode = None
            s.setCursor(Qt.ArrowCursor)
            if callable(s.manual_relocate_release):
                pt = s._ic(e.pos(), clamp=True)
                if pt is not None:
                    s.manual_relocate_release(pt)
            s.update()
            return
        if s._drag_mode == "edit":
            s._drag_mode = None
            s._last_edit_pt = None
            return
        if s._drag_mode == "select":
            if s._selection_mode == "freeform":
                pt = s._ic(e.pos(), clamp=True)
                if pt is not None and (not s._poly or pt != s._poly[-1]):
                    s._poly.append(pt)
                if len(s._poly) >= 3:
                    bounds = selection_bounds_px({"kind": "freeform", "points": s._poly})
                    if bounds is not None:
                        x1, y1, x2, y2 = bounds
                        if x2 - x1 > 5 and y2 - y1 > 5:
                            s.sel = {"kind": "freeform", "points": list(s._poly)}
                            s.sel_changed.emit(s.sel)
                s._poly = []
            else:
                s._de = s._ic(e.pos(), clamp=True)
                if s._ds and s._de:
                    x1 = min(s._ds[0], s._de[0]); y1 = min(s._ds[1], s._de[1])
                    x2 = max(s._ds[0], s._de[0]); y2 = max(s._ds[1], s._de[1])
                    if x2 - x1 > 5 and y2 - y1 > 5:
                        s.sel = {"kind": "rect", "rect": (x1, y1, x2, y2)}
                        s.sel_changed.emit(s.sel)
            s._sm = False
            s._drag_mode = None
            s._update_cursor()
            s.update()
            return
        if s._drag_mode == "pan":
            s._drag_mode = None
            s._last_pos = None
            s._update_cursor()
            s.update()
    def paintEvent(s, e):
        p = QPainter(s)
        p.fillRect(s.rect(), QColor("#f0f0f0"))
        if not s._bp or s._bp.isNull():
            p.setPen(QColor("#6b7280"))
            p.drawText(s.rect(), Qt.AlignCenter, "No image")
            p.end()
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        m = s._metrics()
        if not m:
            p.end()
            return
        scale, ox, oy, sw, sh = m
        target = QRect(int(round(ox)), int(round(oy)), int(round(sw)), int(round(sh)))
        p.drawPixmap(target, s._bp)
        # ── Transition overlay (ramps/steps) ──
        if s._transition_overlay is not None:
            p.setOpacity(0.55)
            p.drawImage(target, s._transition_overlay)
            p.setOpacity(1.0)
            # Draw labels
            if s._transition_labels:
                font = QFont("monospace", 9)
                font.setBold(True)
                p.setFont(font)
                for lx, ly, text, clr in s._transition_labels:
                    sx = ox + lx * scale
                    sy = oy + ly * scale
                    p.setPen(QColor(0, 0, 0, 180))
                    p.drawText(int(sx) + 1, int(sy) + 1, text)
                    p.setPen(QColor(*clr) if isinstance(clr, (tuple, list)) else QColor(clr))
                    p.drawText(int(sx), int(sy), text)

        if s._edit_active and s._ref_overlay is not None and s._ref_overlay_visible:
            p.setOpacity(0.35)
            p.drawImage(target, s._ref_overlay)
            p.setOpacity(1.0)
        if s._edit_active and s._edit_overlay is not None:
            oh, ow = s._edit_overlay.shape
            ov = s._edit_overlay
            rgba = np.zeros((oh, ow, 4), dtype=np.uint8)
            draw_mask = ov >= 200
            erase_mask = ov < 50
            rgba[draw_mask] = [0, 200, 100, 100]
            rgba[erase_mask] = [255, 60, 40, 100]
            qi = QImage(rgba.data, ow, oh, 4 * ow, QImage.Format_RGBA8888)
            qi._np_ref = rgba
            p.drawImage(target, qi)
        preview = None
        closed = True
        if s._drag_mode == "select":
            if s._selection_mode == "freeform" and len(s._poly) >= 2:
                preview = {"kind": "freeform", "points": list(s._poly)}
                closed = False
            elif s._ds and s._de:
                preview = {
                    "kind": "rect",
                    "rect": (
                        min(s._ds[0], s._de[0]),
                        min(s._ds[1], s._de[1]),
                        max(s._ds[0], s._de[0]),
                        max(s._ds[1], s._de[1]),
                    ),
                }
        elif s.sel:
            preview = s.sel
        if preview:
            sel_path = selection_to_screen_path(preview, scale, ox, oy, close_path=closed)
            if closed:
                target_path = QPainterPath()
                target_path.addRect(QRectF(target))
                p.fillPath(target_path.subtracted(sel_path), QColor(0, 0, 0, 80))
                p.save()
                p.setClipPath(sel_path)
                p.drawPixmap(target, s._bp)
                p.restore()
                p.setPen(QPen(QColor(37, 99, 235), 2, Qt.DashLine))
                p.drawPath(sel_path)
            else:
                p.setPen(QPen(QColor(37, 99, 235), 2))
                p.drawPath(sel_path)
        if s._focus_rect is not None:
            x1, y1, x2, y2 = s._focus_rect
            focus_rect = QRectF(
                ox + x1 * scale,
                oy + y1 * scale,
                max(3.0, (x2 - x1 + 1.0) * scale),
                max(3.0, (y2 - y1 + 1.0) * scale),
            )
            p.setPen(QPen(QColor(37, 99, 235), 2))
            p.setBrush(QColor(37, 99, 235, 28))
            p.drawRect(focus_rect)
            if s._focus_label:
                tag_rect = QRectF(focus_rect.left(), max(target.top() + 8, focus_rect.top() - 24), 260, 20)
                p.fillRect(tag_rect, QColor(255, 255, 255, 220))
                p.setPen(QColor(37, 99, 235))
                p.drawText(tag_rect.adjusted(6, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, s._focus_label)
        # Drag-ghost overlay (Step 4 manual relocation). Painted after the
        # heatmap so the ghost sits on top, and before the start-point
        # marker so the seed pin always stays visible.
        if s._drag_ghost_img is not None and s._drag_ghost_origin_yx is not None:
            try:
                oy_img, ox_img = s._drag_ghost_origin_yx
                dy, dx = s._drag_ghost_offset_yx
                gx_img = ox_img + dx
                gy_img = oy_img + dy
                gw_img = s._drag_ghost_img.width()
                gh_img = s._drag_ghost_img.height()
                ghost_rect = QRectF(
                    ox + gx_img * scale,
                    oy + gy_img * scale,
                    gw_img * scale,
                    gh_img * scale,
                )
                p.setOpacity(0.65)
                p.drawImage(ghost_rect, s._drag_ghost_img)
                p.setOpacity(1.0)
                # Outline the ghost for clarity
                p.setPen(QPen(QColor(255, 140, 0), 2, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(ghost_rect)
            except Exception:
                pass
        # Draw start point marker
        if s._start_point is not None:
            sx = ox + s._start_point[0] * scale
            sy = oy + s._start_point[1] * scale
            r = max(4.0, 6.0 * scale)
            p.setPen(QPen(QColor(220, 40, 40), 2))
            p.setBrush(QColor(220, 40, 40, 100))
            p.drawEllipse(QPointF(sx, sy), r, r)
            p.setPen(QPen(QColor(220, 40, 40), 2))
            p.drawLine(QPointF(sx - r - 2, sy), QPointF(sx + r + 2, sy))
            p.drawLine(QPointF(sx, sy - r - 2), QPointF(sx, sy + r + 2))
        if s._pick_start_mode:
            p.setPen(QColor(220, 40, 40))
            p.setFont(p.font())
            p.drawText(max(12, target.left() + 12), max(18, target.top() + 18), "Click to set robot start point")

        p.setPen(QColor("#1f2937"))
        overlay = []
        if s._zoom != 1.0:
            overlay.append(f"zoom: {s._zoom:.2f}x")
        if s._sm:
            if s._selection_mode == "freeform":
                overlay.append("wheel: zoom | drag: pan | double-click: reset | draw a freeform loop to select")
            else:
                overlay.append("wheel: zoom | drag: pan | double-click: reset | drag a rectangle to select")
        else:
            overlay.append("wheel: zoom | drag: pan | double-click: reset | select via Step 3 button")
        y = max(18, target.top() + 18)
        for line in overlay:
            p.drawText(max(12, target.left() + 12), y, line)
            y += 16
        p.end()
    def resizeEvent(s, e):
        super().resizeEvent(e)
        s.update()
    def mouseDoubleClickEvent(s, e):
        s.reset_view()
    def wheelEvent(s, e):
        if not s._bp:
            return
        anchor = s._image_xy_float(e.pos(), clamp=False)
        old_zoom = s._zoom
        factor = 1.15 if e.angleDelta().y() > 0 else (1 / 1.15)
        s._zoom = max(0.2, min(20.0, s._zoom * factor))
        if anchor is not None and abs(s._zoom - old_zoom) > 1e-9:
            fit = min(s.width() / max(1, s._bp.width()), s.height() / max(1, s._bp.height()))
            scale = max(fit, 1e-6) * s._zoom
            s._pan[0] = e.pos().x() - ((s.width() - s._bp.width() * scale) * 0.5) - anchor[0] * scale
            s._pan[1] = e.pos().y() - ((s.height() - s._bp.height() * scale) * 0.5) - anchor[1] * scale
        s.update()
        e.accept()


class DragScrollArea(QScrollArea):
    def __init__(s):
        super().__init__()
        s._dragging = False
        s._last_pos = None
        s.setWidgetResizable(True)
        s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        s.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        s.viewport().setCursor(Qt.OpenHandCursor)

    def mousePressEvent(s, e):
        if e.button() == Qt.LeftButton:
            s._dragging = True
            s._last_pos = e.pos()
            s.viewport().setCursor(Qt.ClosedHandCursor)
            e.accept()
            return
        super().mousePressEvent(e)

    def mouseMoveEvent(s, e):
        if s._dragging and s._last_pos is not None:
            delta = e.pos() - s._last_pos
            s.verticalScrollBar().setValue(s.verticalScrollBar().value() - delta.y())
            s._last_pos = e.pos()
            e.accept()
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(s, e):
        if e.button() == Qt.LeftButton and s._dragging:
            s._dragging = False
            s._last_pos = None
            s.viewport().setCursor(Qt.OpenHandCursor)
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def setWidget(s, widget):
        super().setWidget(widget)
        if widget is not None:
            widget.setMinimumWidth(s.viewport().width())
        s.horizontalScrollBar().setValue(0)

    def resizeEvent(s, e):
        super().resizeEvent(e)
        widget = s.widget()
        if widget is not None:
            widget.setMinimumWidth(s.viewport().width())
        s.horizontalScrollBar().setValue(0)


class PointCloudPreviewW(QWidget):
    def __init__(s):
        super().__init__()
        s.setMinimumSize(200, 200)
        s.setMouseTracking(True)
        s.setFocusPolicy(Qt.StrongFocus)
        s._msg = "No point cloud"
        s._img = None
        s._points = None
        s._z_vals = None
        s._path = ""
        s._label = ""
        s._total_points = 0
        s._display_points = 0
        s._sampled = False
        s._yaw = -45.0
        s._pitch = 28.0
        s._zoom = 1.0
        s._pan = np.array([0.0, 0.0], dtype=np.float32)
        s._drag_btn = None
        s._last_pos = None
        s._raw_pts = None
        s._raw_cloud = None

    def clear_cloud(s, message="No point cloud"):
        s._msg = message
        s._img = None
        s._points = None
        s._z_vals = None
        s._raw_pts = None
        s._raw_cloud = None
        s.update()

    def set_cloud(s, cloud):
        pts = np.asarray(cloud["points"], dtype=np.float32)
        if pts.size == 0:
            s.clear_cloud("Point cloud is empty")
            return
        s._raw_pts = pts.copy()
        s._raw_cloud = cloud
        s._apply_preview_cloud(pts, cloud)

    def clip_z(s, max_z):
        """Re-render keeping only points with Z <= max_z."""
        if s._raw_pts is None:
            return
        mask = s._raw_pts[:, 2] <= max_z
        pts = s._raw_pts[mask]
        if pts.size == 0:
            s.clear_cloud("All points clipped")
            return
        cloud = dict(s._raw_cloud)
        cloud["points"] = pts
        if "colors" in cloud and cloud["colors"] is not None:
            cloud["colors"] = np.asarray(cloud["colors"])[mask]
        cloud["display_points"] = int(pts.shape[0])
        s._apply_preview_cloud(pts, cloud)

    def _apply_preview_cloud(s, pts, cloud):
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) * 0.5
        s._points = np.ascontiguousarray(pts - center, dtype=np.float32)
        s._z_vals = pts[:, 2].astype(np.float32, copy=False)
        s._colors = np.asarray(cloud["colors"], dtype=np.uint8) if "colors" in cloud and cloud["colors"] is not None else None
        s._path = cloud.get("path", "")
        s._label = cloud.get("label", "Point Cloud")
        s._total_points = int(cloud.get("total_points", pts.shape[0]))
        s._display_points = int(cloud.get("display_points", pts.shape[0]))
        s._sampled = bool(cloud.get("sampled", False))
        s.reset_view(render=False)
        s._msg = ""
        s._render()

    def reset_view(s, render=True):
        s._yaw = -45.0
        s._pitch = 28.0
        s._zoom = 1.0
        s._pan[:] = 0.0
        if render:
            s._render()

    def _render(s):
        if s._points is None or s.width() <= 2 or s.height() <= 2:
            s._img = None
            s.update()
            return

        pts = s._points
        w = max(2, s.width())
        h = max(2, s.height())

        yaw = math.radians(s._yaw)
        pitch = math.radians(s._pitch)
        cz, sz = math.cos(yaw), math.sin(yaw)
        cx, sx = math.cos(pitch), math.sin(pitch)

        x1 = pts[:, 0] * cz - pts[:, 1] * sz
        y1 = pts[:, 0] * sz + pts[:, 1] * cz
        z1 = pts[:, 2]
        y2 = y1 * cx - z1 * sx
        z2 = y1 * sx + z1 * cx

        span_x = max(float(np.ptp(x1)), 1e-3)
        span_z = max(float(np.ptp(z2)), 1e-3)
        scale = 0.9 * min(w / span_x, h / span_z) * s._zoom

        xs = np.rint(x1 * scale + (w * 0.5) + s._pan[0]).astype(np.int32)
        ys = np.rint(-z2 * scale + (h * 0.5) + s._pan[1]).astype(np.int32)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not np.any(inside):
            s._img = None
            s.update()
            return

        xs = xs[inside]
        ys = ys[inside]
        depth = y2[inside]
        z_vals = s._z_vals[inside]

        if s._colors is not None and s._colors.shape[0] == s._points.shape[0]:
            colors = s._colors[inside]
        else:
            rgb_f = _height_colormap(z_vals)
            colors = (rgb_f * 255).astype(np.uint8)

        order = np.argsort(depth)
        xs = xs[order]
        ys = ys[order]
        colors = colors[order]

        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = [6, 10, 16]
        img[ys, xs] = colors
        if xs.shape[0] < 120_000:
            img[np.clip(ys + 1, 0, h - 1), xs] = colors
            img[ys, np.clip(xs + 1, 0, w - 1)] = colors

        s._img = QImage(img.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
        s.update()

    def paintEvent(s, e):
        p = QPainter(s)
        p.fillRect(s.rect(), QColor("#f0f0f0"))
        if s._img is not None:
            p.drawImage(0, 0, s._img)
        else:
            p.setPen(QColor("#6b7280"))
            p.drawText(s.rect(), Qt.AlignCenter, s._msg or "No point cloud")

        if s._points is not None:
            p.setPen(QColor("#1f2937"))
            overlay = [
                s._label,
                f"file: {os.path.basename(s._path)}",
                f"points: {s._display_points:,}/{s._total_points:,}" + (" sampled" if s._sampled else ""),
                "left-drag: rotate | right-drag: pan | wheel: zoom | double-click: reset",
            ]
            y = 22
            for line in overlay:
                p.drawText(12, y, line)
                y += 16

    def resizeEvent(s, e):
        super().resizeEvent(e)
        s._render()

    def mousePressEvent(s, e):
        s._drag_btn = e.button()
        s._last_pos = e.pos()
        s.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(s, e):
        if s._points is None or s._last_pos is None:
            return
        dx = e.x() - s._last_pos.x()
        dy = e.y() - s._last_pos.y()
        if s._drag_btn == Qt.LeftButton:
            s._yaw += dx * 0.5
            s._pitch = max(-89.0, min(89.0, s._pitch + dy * 0.35))
        elif s._drag_btn == Qt.RightButton:
            s._pan[0] += dx
            s._pan[1] += dy
        s._last_pos = e.pos()
        s._render()

    def mouseReleaseEvent(s, e):
        s._drag_btn = None
        s._last_pos = None
        s.setCursor(Qt.ArrowCursor)

    def mouseDoubleClickEvent(s, e):
        s.reset_view()

    def wheelEvent(s, e):
        if s._points is None:
            return
        delta = e.angleDelta().y()
        factor = 1.1 if delta > 0 else 1 / 1.1
        s._zoom = max(0.1, min(20.0, s._zoom * factor))
        s._render()


if PYQTGRAPH_GL_AVAILABLE:
    class PointCloudW(pgl.GLViewWidget):
        gl_failed = pyqtSignal(str)

        def __init__(s):
            super().__init__()
            s.setBackgroundColor((6, 10, 16))
            s.setMinimumSize(200, 200)
            s._msg = "No point cloud"
            s._path = ""
            s._label = ""
            s._total_points = 0
            s._display_points = 0
            s._sampled = False
            s._radius = 1.0
            s._count = 0
            s._last_pos = None
            s._raw_pts = None
            s._raw_cloud = None
            s._scatter = pgl.GLScatterPlotItem(
                pos=np.zeros((0, 3), dtype=np.float32),
                color=np.zeros((0, 4), dtype=np.float32),
                size=2.0,
                pxMode=True,
            )
            s._scatter.setGLOptions("translucent")
            s.addItem(s._scatter)
            s._legend = []
            s.reset_view()

        def clear_cloud(s, message="No point cloud"):
            s._msg = message
            s._count = 0
            s._legend = []
            s._raw_pts = None
            s._raw_cloud = None
            try:
                s._scatter.setData(
                    pos=np.zeros((0, 3), dtype=np.float32),
                    color=np.zeros((0, 4), dtype=np.float32),
                    size=1.0,
                    pxMode=True,
                )
            except Exception:
                pass
            s.update()

        def set_cloud(s, cloud):
            try:
                pts = np.asarray(cloud["points"], dtype=np.float32)
                if pts.size == 0:
                    s.clear_cloud("Point cloud is empty")
                    return

                # Store raw data for Z-clipping
                s._raw_pts = pts.copy()
                s._raw_cloud = cloud

                s._apply_cloud(pts, cloud)
            except Exception as e:
                s.gl_failed.emit(f"pyqtgraph.opengl failed: {e}")

        def clip_z(s, max_z):
            """Re-render the cloud keeping only points with Z <= max_z."""
            if s._raw_pts is None:
                return
            mask = s._raw_pts[:, 2] <= max_z
            pts = s._raw_pts[mask]
            if pts.size == 0:
                s.clear_cloud("All points clipped")
                return
            cloud = dict(s._raw_cloud)
            cloud["points"] = pts
            if "colors" in cloud and cloud["colors"] is not None:
                cloud["colors"] = np.asarray(cloud["colors"])[mask]
            cloud["display_points"] = int(pts.shape[0])
            s._apply_cloud(pts, cloud, keep_camera=True)

        def _apply_cloud(s, pts, cloud, keep_camera=False):
            mins = pts.min(axis=0)
            maxs = pts.max(axis=0)
            center = (mins + maxs) * 0.5
            centered = np.ascontiguousarray(pts - center, dtype=np.float32)
            s._radius = float(0.5 * np.linalg.norm(maxs - mins))
            if s._radius <= 1e-6:
                s._radius = 1.0

            if "colors" in cloud and cloud["colors"] is not None:
                rgb = np.asarray(cloud["colors"], dtype=np.float32)
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                n = rgb.shape[0]
                colors = np.column_stack((
                    rgb[:, 0], rgb[:, 1], rgb[:, 2],
                    np.full(n, 0.96, dtype=np.float32),
                )).astype(np.float32)
            else:
                rgb = _height_colormap(pts[:, 2])
                colors = np.column_stack((
                    rgb, np.full(rgb.shape[0], 0.96, dtype=np.float32),
                )).astype(np.float32)

            point_px = max(1.4, min(2.8, 2400.0 / max(centered.shape[0], 1) ** 0.33))
            s._scatter.setData(pos=centered, color=colors, size=float(point_px), pxMode=True)
            s._path = cloud.get("path", "")
            s._label = cloud.get("label", "Point Cloud")
            s._total_points = int(cloud.get("total_points", pts.shape[0]))
            s._display_points = int(cloud.get("display_points", pts.shape[0]))
            s._sampled = bool(cloud.get("sampled", False))
            s._count = centered.shape[0]
            s._center_offset = center.copy()
            s._msg = ""
            # Store legend entries for 2D overlay
            s._legend = cloud.get("legend", [])
            if not keep_camera:
                s.reset_view()
            else:
                s.update()

        def reset_view(s):
            s.opts["center"] = Vector(0.0, 0.0, 0.0)
            s.setCameraPosition(distance=max(2.2 * s._radius, 1.0), elevation=28.0, azimuth=-45.0)
            s.update()

        def paintEvent(s, e):
            try:
                super().paintEvent(e)
            except Exception as exc:
                s.gl_failed.emit(f"pyqtgraph paint failed: {exc}")
                return
            p = QPainter(s)
            p.setRenderHint(QPainter.TextAntialiasing)
            if s._count == 0:
                p.setPen(QColor("#6b7280"))
                p.drawText(s.rect(), Qt.AlignCenter, s._msg or "No point cloud")
            else:
                p.setPen(QColor("#1f2937"))
                overlay = [
                    s._label,
                    f"file: {os.path.basename(s._path)}",
                    f"points: {s._display_points:,}/{s._total_points:,}" + (" sampled" if s._sampled else ""),
                    "pyqtgraph.opengl | left-drag: rotate | right-drag: pan | wheel: zoom | double-click: reset",
                ]
                y = 22
                for line in overlay:
                    p.drawText(12, y, line)
                    y += 16
                # Draw legend in bottom-right
                if s._legend:
                    font = QFont("monospace", 8)
                    p.setFont(font)
                    row_h = 16
                    swatch = 10
                    pad = 8
                    n = len(s._legend)
                    max_text_w = max(p.fontMetrics().horizontalAdvance(txt) for txt, _ in s._legend)
                    box_w = swatch + 6 + max_text_w + pad * 2
                    box_h = n * row_h + pad * 2
                    bx = s.width() - box_w - 12
                    by = s.height() - box_h - 12
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(10, 14, 20, 180))
                    p.drawRoundedRect(bx, by, box_w, box_h, 6, 6)
                    ly = by + pad
                    for txt, clr in s._legend:
                        p.setPen(Qt.NoPen)
                        p.setBrush(QColor(*clr) if isinstance(clr, (tuple, list)) else QColor(clr))
                        p.drawRect(bx + pad, ly + 2, swatch, swatch)
                        p.setPen(QColor("#1f2937"))
                        p.drawText(bx + pad + swatch + 6, ly + row_h - 4, txt)
                        ly += row_h
            p.end()

        def mousePressEvent(s, e):
            s._last_pos = e.pos()
            super().mousePressEvent(e)

        def mouseMoveEvent(s, e):
            if s._last_pos is not None and (e.buttons() & Qt.RightButton) and not (e.buttons() & Qt.LeftButton):
                dx = e.x() - s._last_pos.x()
                dy = e.y() - s._last_pos.y()
                pan_scale = max(s.opts.get("distance", 1.0), 1.0) * 0.002
                try:
                    s.pan(-dx * pan_scale, dy * pan_scale, 0, relative="view")
                except TypeError:
                    s.pan(-dx * pan_scale, dy * pan_scale, 0)
                s._last_pos = e.pos()
                s.update()
                return
            s._last_pos = e.pos()
            super().mouseMoveEvent(e)

        def mouseReleaseEvent(s, e):
            s._last_pos = None
            super().mouseReleaseEvent(e)

        def mouseDoubleClickEvent(s, e):
            s.reset_view()
            e.accept()
else:
    PointCloudW = PointCloudPreviewW
