"""2D image rendering — obstacle maps, coverage overlays, STC paths."""

import os
import numpy as np
from PyQt5.QtCore import Qt, QRect, QPointF
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QFont, QBrush


def render_coverage(result, pgm_path, color=(0, 200, 130)):
    w, h = result['w'], result['h']
    buf = np.zeros((h, w, 3), dtype=np.uint8)
    covPx = result['covPx'].reshape(h, w)
    blk = result['blocked'].reshape(h, w)
    floor = result.get('floorPx')
    floor2d = floor.reshape(h, w) if isinstance(floor, np.ndarray) else None
    for y in range(h):
        pr = h - 1 - y
        for x in range(w):
            if covPx[y, x]:
                buf[pr, x] = color
            elif floor2d is not None and floor2d[y, x]:
                buf[pr, x] = [236, 240, 245]
            elif blk[y, x]:
                buf[pr, x] = [25, 25, 30]
            else:
                buf[pr, x] = [25, 25, 30]
    return QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


def _extra_inflation_halo(result, h, w):
    """Cells that are blocked ONLY because of the extra inflation_radius.

    Total halo = blocked & ~sourceBlocked  (includes the physical footprint
    inflation, which can be very thick — e.g. a 0.6 m robot adds 6 px on
    each side even before any safety margin).

    To isolate just the safety-margin contribution we re-dilate
    sourceBlocked with the PHYSICAL footprint only, then subtract that
    physical-only inflation from `blocked`. What's left is exactly the
    band the user's inflation_radius slider added.

    Returns a (h, w) bool mask in display orientation (rows flipped), or
    None if the result doesn't carry enough metadata.
    """
    import math as _math
    params = result.get('params', {}) or {}
    if float(params.get('inflation_radius', 0.0)) <= 0.0:
        return None
    blocked = result.get('blocked')
    src_blocked = result.get('sourceBlocked')
    if blocked is None or src_blocked is None:
        return None
    try:
        from scipy.ndimage import binary_dilation as _binary_dilation
    except ImportError:
        return None
    res = float(result.get('resolution', 0.05))
    shape = params.get('shape', 'rectangular')
    phys_halfW = float(params.get(
        'physical_halfW',
        params.get('physical_radius', params.get('halfW', 0.0))))
    phys_halfL = float(params.get(
        'physical_halfL',
        params.get('physical_radius', params.get('halfL', 0.0))))
    iX = max(0, int(_math.ceil(phys_halfW / res)))
    iY = max(0, int(_math.ceil(phys_halfL / res)))
    sb = np.asarray(src_blocked, dtype=np.uint8).reshape(h, w)
    if iX == 0 and iY == 0:
        phys_inflated = sb.astype(bool)
    elif shape == 'circular':
        r_phys = max(iX, iY)
        yy, xx = np.ogrid[-r_phys:r_phys + 1, -r_phys:r_phys + 1]
        struct = (xx * xx + yy * yy) <= (r_phys * r_phys)
        phys_inflated = _binary_dilation(sb.astype(bool), structure=struct)
    else:
        struct = np.ones((2 * iY + 1, 2 * iX + 1), dtype=bool)
        phys_inflated = _binary_dilation(sb.astype(bool), structure=struct)
    blk = np.asarray(blocked, dtype=np.uint8).reshape(h, w)
    halo = (blk == 1) & (~phys_inflated)
    return halo[::-1, :]


def _build_bg(h, w, result, bg_pgm=None):
    """Build an RGB background: blocked-cells-map grayscale or synthetic dark/light."""
    if bg_pgm is not None:
        g = bg_pgm.reshape(h, w)
        return np.stack([g, g, g], axis=-1).copy()
    blk = result['blocked'].reshape(h, w)[::-1, :]
    floor = result.get('floorPx')
    floor2d = floor.reshape(h, w)[::-1, :] if isinstance(floor, np.ndarray) else None
    buf = np.full((h, w, 3), [25, 25, 30], dtype=np.uint8)
    if floor2d is not None:
        buf[floor2d == 1] = [236, 240, 245]
    else:
        buf[blk == 0] = [236, 240, 245]
    return buf


def render_coverage_fast(result, color=(0, 200, 130), bg_pgm=None,
                         show_inflation=False):
    """Vectorized version of render_coverage.

    When ``show_inflation`` is True, cells that are floor and blocked-by-
    inflation but free in the raw obstacle map are painted pink first, so
    the safety-margin halo around walls is visible underneath the
    coverage colour. Used by the Actual robot view when its
    ``inflation_radius`` param is > 0.
    """
    w, h = result['w'], result['h']
    covPx = result['covPx'].reshape(h, w)[::-1, :]
    buf = _build_bg(h, w, result, bg_pgm)

    if show_inflation:
        halo_disp = _extra_inflation_halo(result, h, w)
        if halo_disp is not None:
            buf[halo_disp] = [255, 105, 180]  # hot pink, solid

    mask = covPx == 1
    color_f = np.array(color, dtype=np.float32)
    if bg_pgm is not None:
        buf[mask] = (0.45 * color_f + 0.55 * buf[mask].astype(np.float32)).astype(np.uint8)
    else:
        buf[mask] = color
    return QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


def render_compare_fast(ref, act, bg_pgm=None, show_inflation=False):
    """Side-by-side comparison overlay.

    When ``show_inflation`` is True, the Actual robot's inflation halo
    (cells blocked by inflation but free in the raw obstacle map) is
    painted solid hot pink on top of the green/orange overlay so the
    safety buffer added by the Actual's inflation_radius is visible
    against the Reference's wider coverage.
    """
    if not ref or not act:
        return make_info_image("Run both Reference and Actual on the same map to compare coverage.")
    if int(ref.get("w", -1)) != int(act.get("w", -1)) or int(ref.get("h", -1)) != int(act.get("h", -1)):
        return make_info_image(
            "Reference and Actual results are from different map sizes.\n"
            "Rerun both Step 3 evaluations on the current map."
        )
    w, h = ref['w'], ref['h']
    rc = ref['covPx'].reshape(h, w)[::-1, :]
    ac = act['covPx'].reshape(h, w)[::-1, :]
    buf = _build_bg(h, w, act, bg_pgm)
    green = np.array([0, 200, 130], dtype=np.float32)
    orange = np.array([255, 165, 0], dtype=np.float32)
    ac_mask = ac == 1
    ref_only = (ac == 0) & (rc == 1)
    if bg_pgm is not None:
        buf[ac_mask] = (0.45 * green + 0.55 * buf[ac_mask].astype(np.float32)).astype(np.uint8)
        buf[ref_only] = (0.45 * orange + 0.55 * buf[ref_only].astype(np.float32)).astype(np.uint8)
    else:
        buf[ac_mask] = [0, 200, 130]
        buf[ref_only] = [255, 165, 0]

    if show_inflation:
        halo_disp = _extra_inflation_halo(act, h, w)
        if halo_disp is not None:
            buf[halo_disp] = [255, 105, 180]  # hot pink, solid — overrides ref-only orange

    return QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


def make_info_image(text: str, width: int = 960, height: int = 720):
    img = QImage(width, height, QImage.Format_RGB888)
    img.fill(QColor("#05080d"))
    p = QPainter(img)
    p.setPen(QColor("#c5cdd8"))
    p.drawText(QRect(0, 0, width, height), Qt.AlignCenter | Qt.TextWordWrap, text)
    p.end()
    return img


def _stc_display_points(result):
    if not result or not result.get("useSTC"):
        return []
    path = result.get("stcPath") or []
    step = max(1, int(result.get("stcStep", 1)))
    height = int(result["h"])
    pts = []
    for row, col in path:
        x = (float(col) + 0.5) * step
        y = height - ((float(row) + 0.5) * step)
        pts.append(QPointF(x, y))
    return pts


def render_stc_path_fast(ref=None, act=None, bg_pgm=None, planner_label=""):
    base = act if act is not None else ref
    if base is None:
        return make_info_image("Run Step 3 with a path planner to view the coverage path.")

    w, h = base["w"], base["h"]
    buf = _build_bg(h, w, base, bg_pgm)

    if ref is not None:
        rc = ref["covPx"].reshape(h, w)[::-1, :]
        if bg_pgm is not None:
            mask = rc == 1
            buf[mask] = (0.35 * np.array([205, 228, 255], dtype=np.float32) + 0.65 * buf[mask].astype(np.float32)).astype(np.uint8)
        else:
            buf[rc == 1] = [205, 228, 255]
    if act is not None:
        ac = act["covPx"].reshape(h, w)[::-1, :]
        if bg_pgm is not None:
            mask = ac == 1
            buf[mask] = (0.35 * np.array([204, 247, 230], dtype=np.float32) + 0.65 * buf[mask].astype(np.float32)).astype(np.uint8)
        else:
            buf[ac == 1] = [204, 247, 230]

    img = QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)

    if ref is not None and ref.get("useSTC"):
        pts = _stc_display_points(ref)
        if len(pts) >= 2:
            p.setPen(QPen(QColor(0, 140, 255), max(2, int(ref.get("stcStep", 1) * 0.6))))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)

    if act is not None and act.get("useSTC"):
        pts = _stc_display_points(act)
        if len(pts) >= 2:
            p.setPen(QPen(QColor(0, 229, 160), max(2, int(act.get("stcStep", 1) * 0.6))))
            for a, b in zip(pts, pts[1:]):
                p.drawLine(a, b)

    # ── Legend panel (bottom-right) ──
    tag = planner_label or "Planner"
    entries = []
    if ref is not None and ref.get("useSTC"):
        entries.append(("Reference path", QColor(0, 140, 255)))
        entries.append(("Reference accessible area", QColor(205, 228, 255)))
    if act is not None and act.get("useSTC"):
        entries.append(("Actual path", QColor(0, 229, 160)))
        entries.append(("Actual accessible area", QColor(204, 247, 230)))
    entries.append(("Inaccessible floor", QColor(60, 60, 60)))

    font = QFont("monospace", 8)
    p.setFont(font)
    fm = p.fontMetrics()
    row_h = 16
    swatch = 10
    pad = 8
    title = f"{tag} Path"
    title_h = 18
    max_text_w = max(fm.horizontalAdvance(txt) for txt, _ in entries)
    max_text_w = max(max_text_w, fm.horizontalAdvance(title))
    box_w = swatch + 6 + max_text_w + pad * 2
    box_h = title_h + len(entries) * row_h + pad * 2

    # Detect map content bounding box and anchor legend inside it
    bg_color = np.array([25, 25, 30], dtype=np.uint8)
    content_mask = np.any(buf != bg_color, axis=-1)
    if bg_pgm is not None:
        # With PGM background the whole image has content; fall back to
        # non-black pixels (grayscale > 30 counts as map content).
        content_mask = np.any(buf > 30, axis=-1)
    rows = np.where(content_mask.any(axis=1))[0]
    cols = np.where(content_mask.any(axis=0))[0]
    if rows.size and cols.size:
        margin = 6
        bx = int(cols[0]) + margin
        by = int(rows[0]) + margin
    else:
        bx, by = 12, 12
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(10, 14, 20, 180))
    p.drawRoundedRect(bx, by, box_w, box_h, 6, 6)

    # Title
    p.setPen(QColor(220, 225, 232))
    bold = QFont("monospace", 8)
    bold.setBold(True)
    p.setFont(bold)
    p.drawText(bx + pad, by + pad + fm.ascent(), title)
    p.setFont(font)

    # Entries
    y0 = by + pad + title_h
    for idx_e, (txt, color) in enumerate(entries):
        cy = y0 + idx_e * row_h
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawRect(bx + pad, cy + (row_h - swatch) // 2, swatch, swatch)
        p.setPen(QColor(200, 205, 216))
        p.drawText(bx + pad + swatch + 6, cy + (row_h + fm.ascent()) // 2 - 1, txt)

    p.end()
    return img


