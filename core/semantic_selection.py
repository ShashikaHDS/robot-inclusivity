"""Selection geometry utilities — polygon, rectangle, lasso operations."""

import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QImage, QPainter, QPainterPath, QPolygonF, QColor


def _normalize_selection(selection):
    if not selection:
        return None
    if isinstance(selection, tuple) and len(selection) == 4:
        x1, y1, x2, y2 = selection
        return {"kind": "rect", "rect": (int(x1), int(y1), int(x2), int(y2))}
    return selection


def selection_kind(selection):
    selection = _normalize_selection(selection)
    if not selection:
        return None
    return selection.get("kind", "rect")


def selection_bounds_px(selection):
    selection = _normalize_selection(selection)
    if not selection:
        return None
    if selection_kind(selection) == "rect":
        return tuple(int(v) for v in selection["rect"])
    pts = selection.get("points") or []
    if not pts:
        return None
    xs = [int(round(pt[0])) for pt in pts]
    ys = [int(round(pt[1])) for pt in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _polygon_area_and_centroid(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        if pts.shape[0] == 0:
            return 0.0, None
        center = pts.mean(axis=0)
        return 0.0, (float(center[0]), float(center[1]))
    x = pts[:, 0]
    y = pts[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cross = x * y2 - x2 * y
    area = 0.5 * cross.sum()
    if abs(area) < 1e-6:
        center = pts.mean(axis=0)
        return 0.0, (float(center[0]), float(center[1]))
    cx = ((x + x2) * cross).sum() / (6.0 * area)
    cy = ((y + y2) * cross).sum() / (6.0 * area)
    return abs(float(area)), (float(cx), float(cy))


def selection_center_px(selection):
    selection = _normalize_selection(selection)
    if not selection:
        return None
    if selection_kind(selection) == "rect":
        x1, y1, x2, y2 = selection["rect"]
        return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)
    _area, centroid = _polygon_area_and_centroid(selection.get("points") or [])
    if centroid is not None:
        return centroid
    bounds = selection_bounds_px(selection)
    if bounds is None:
        return None
    x1, y1, x2, y2 = bounds
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


def selection_mask_from_display(selection, map_width, map_height):
    selection = _normalize_selection(selection)
    if not selection or map_width <= 0 or map_height <= 0:
        return None

    if selection_kind(selection) == "rect":
        px1, py1, px2, py2 = selection["rect"]
        oy1 = map_height - 1 - py2
        oy2 = map_height - 1 - py1
        mask = np.zeros(map_width * map_height, dtype=np.uint8)
        mask_2d = mask.reshape(map_height, map_width)
        mask_2d[oy1:oy2 + 1, px1:px2 + 1] = 1
        return mask

    points = selection.get("points") or []
    if len(points) < 3:
        return None
    img = QImage(map_width, map_height, QImage.Format_Grayscale8)
    img.fill(0)
    painter = QPainter(img)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(255, 255, 255))
    poly = QPolygonF(
        [QPointF(float(x), float(map_height - 1 - y)) for x, y in points]
    )
    painter.drawPolygon(poly)
    painter.end()

    bits = img.bits()
    bits.setsize(img.bytesPerLine() * img.height())
    arr = np.frombuffer(bits, dtype=np.uint8).reshape(map_height, img.bytesPerLine())[:, :map_width].copy()
    return (arr > 0).astype(np.uint8).reshape(-1)


def selection_to_world_bounds(selection, map_width, map_height, yaml_data):
    """Convert a display-space selection into world-coordinate bounds."""
    bounds = selection_bounds_px(selection)
    if bounds is None:
        return None
    px1, py1, px2, py2 = bounds
    res = yaml_data["resolution"]
    ox, oy = yaml_data["origin"][0], yaml_data["origin"][1]
    oy1 = map_height - 1 - py2
    oy2 = map_height - 1 - py1
    x1 = ox + px1 * res
    x2 = ox + px2 * res
    y1 = oy + oy1 * res
    y2 = oy + oy2 * res
    return (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))


def selection_to_screen_path(selection, scale, ox, oy, close_path=True):
    selection = _normalize_selection(selection)
    path = QPainterPath()
    if not selection:
        return path
    if selection_kind(selection) == "rect":
        x1, y1, x2, y2 = selection["rect"]
        path.addRect(
            QRectF(
                ox + x1 * scale,
                oy + y1 * scale,
                max(scale, (x2 - x1 + 1) * scale),
                max(scale, (y2 - y1 + 1) * scale),
            )
        )
        return path
    points = selection.get("points") or []
    if not points:
        return path
    x0, y0 = points[0]
    path.moveTo(ox + x0 * scale, oy + y0 * scale)
    for x, y in points[1:]:
        path.lineTo(ox + x * scale, oy + y * scale)
    if close_path and len(points) >= 3:
        path.closeSubpath()
    return path
