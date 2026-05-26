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
        blocked = result.get('blocked')
        src_blocked = result.get('sourceBlocked')
        floor = result.get('floorPx')
        if (blocked is not None and src_blocked is not None
                and floor is not None):
            blk = np.asarray(blocked, dtype=np.uint8).reshape(h, w)[::-1, :]
            sb = np.asarray(src_blocked, dtype=np.uint8).reshape(h, w)[::-1, :]
            fl = np.asarray(floor, dtype=np.uint8).reshape(h, w)[::-1, :]
            halo = (blk == 1) & (sb == 0) & (fl == 1)
            pink = np.array([255, 105, 180], dtype=np.float32)
            if bg_pgm is not None:
                buf[halo] = (0.55 * pink + 0.45 * buf[halo].astype(np.float32)).astype(np.uint8)
            else:
                buf[halo] = pink.astype(np.uint8)

    mask = covPx == 1
    color_f = np.array(color, dtype=np.float32)
    if bg_pgm is not None:
        buf[mask] = (0.45 * color_f + 0.55 * buf[mask].astype(np.float32)).astype(np.uint8)
    else:
        buf[mask] = color
    return QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()


def render_compare_fast(ref, act, bg_pgm=None):
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


def render_bottleneck_overlay(act_result, candidates, bg_pgm=None, focused_id=None):
    """Render bottleneck candidates with relocation zones and arrows.

    Bottleneck objects shown in red, relocation zones in green rectangles,
    arrows from object to best zone. Non-bottleneck candidates shown dimmed.
    """
    import math
    from PyQt5.QtGui import QPolygon
    from PyQt5.QtCore import QPoint

    w, h = int(act_result["w"]), int(act_result["h"])
    buf = _build_bg(h, w, act_result, bg_pgm)

    # Show accessible area as light tint
    cov = act_result["covPx"].reshape(h, w)[::-1, :]
    cov_mask = cov == 1
    if bg_pgm is not None:
        buf[cov_mask] = (0.2 * np.array([37, 99, 235], dtype=np.float32) + 0.8 * buf[cov_mask].astype(np.float32)).astype(np.uint8)
    else:
        buf[cov_mask] = [220, 230, 245]

    # Draw non-bottleneck candidates dimmed
    for cand in candidates:
        if cand.get("isBottleneck"):
            continue
        flat = cand["indices"]
        rows, cols = flat // w, flat % w
        disp_rows = h - 1 - rows
        buf[disp_rows, cols] = [160, 160, 160]

    # Draw bottleneck candidates in red
    for cand in candidates:
        if not cand.get("isBottleneck"):
            continue
        flat = cand["indices"]
        rows, cols = flat // w, flat % w
        disp_rows = h - 1 - rows
        is_focused = (focused_id is not None and int(cand["id"]) == focused_id)

        buf[disp_rows, cols] = [255, 50, 50] if is_focused else [220, 60, 60]

        # Draw relocation zones as green rectangles
        for zone in cand.get("relocationZones", []):
            zr, zc = zone["top_left_rc"]
            obj_h, obj_w = zone["footprint_hw"]
            dr0 = max(0, h - 1 - (zr + obj_h))
            dr1 = min(h - 1, h - 1 - zr)
            dc0, dc1 = max(0, zc), min(w - 1, zc + obj_w)
            green = [50, 200, 100] if is_focused else [100, 180, 100]
            if dr1 > dr0 and dc1 > dc0:
                zone_area = buf[dr0:dr1, dc0:dc1].astype(np.float32)
                buf[dr0:dr1, dc0:dc1] = (0.35 * np.array(green, dtype=np.float32) + 0.65 * zone_area).astype(np.uint8)
                buf[dr0, dc0:dc1] = green
                buf[min(dr1, h - 1), dc0:dc1] = green
                buf[dr0:dr1, dc0] = green
                buf[dr0:dr1, min(dc1, w - 1)] = green

        # Bounding box for focused
        if is_focused:
            r0 = max(0, int(disp_rows.min()) - 3)
            r1 = min(h - 1, int(disp_rows.max()) + 3)
            c0 = max(0, int(cols.min()) - 3)
            c1 = min(w - 1, int(cols.max()) + 3)
            for edge_r in range(r0, r1 + 1):
                buf[edge_r, c0] = buf[edge_r, c1] = [255, 50, 50]
            buf[r0, c0:c1 + 1] = buf[r1, c0:c1 + 1] = [255, 50, 50]

    img = QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()

    # Draw arrows from focused bottleneck to best relocation zone
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    for cand in candidates:
        if not cand.get("isBottleneck") or not cand.get("bestZone"):
            continue
        if focused_id is not None and int(cand["id"]) != focused_id:
            continue

        flat = cand["indices"]
        rows, cols = flat // w, flat % w
        cr, cc = int(h - 1 - rows.mean()), int(cols.mean())
        best = cand["bestZone"]
        zr, zc = best["top_left_rc"]
        obj_h, obj_w = best["footprint_hw"]
        zr_d, zc_d = int(h - 1 - (zr + obj_h / 2)), int(zc + obj_w / 2)

        pen = QPen(QColor(255, 200, 50), 2)
        pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.drawLine(cc, cr, zc_d, zr_d)

        dx, dy = zc_d - cc, zr_d - cr
        length = math.sqrt(dx * dx + dy * dy)
        if length > 10:
            ux, uy = dx / length, dy / length
            ax, ay = zc_d - ux * 10, zr_d - uy * 10
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 200, 50))
            p.drawPolygon(QPolygon([
                QPoint(zc_d, zr_d),
                QPoint(int(ax + uy * 5), int(ay - ux * 5)),
                QPoint(int(ax - uy * 5), int(ay + ux * 5)),
            ]))

        gain = best.get("net_area_gain", 0)
        ratio = cand.get("bottleneckRatio", 0)
        p.setPen(QColor(255, 200, 50))
        p.setFont(QFont("monospace", 9, QFont.Bold))
        mid_x, mid_y = (cc + zc_d) // 2, (cr + zr_d) // 2
        p.drawText(mid_x + 5, mid_y - 5, f"+{gain:.1f} m² ({ratio:.0f}x)")

    p.end()
    return img


def render_optimization_overlay(act_result, optimization_result, bg_pgm=None):
    """Render multi-object optimization moves on the map.

    Shows: red numbered objects → green numbered zones, yellow arrows,
    blue tint for newly unlocked area, grey for original accessible.
    """
    import math
    from PyQt5.QtGui import QPolygon
    from PyQt5.QtCore import QPoint

    w, h = int(act_result["w"]), int(act_result["h"])
    moves = optimization_result.get("moves", [])
    optimized_blocked = optimization_result.get("optimized_blocked")
    buf = _build_bg(h, w, act_result, bg_pgm)

    # Original accessible area in grey
    orig_cov = act_result.get("covPx")
    if orig_cov is not None:
        cov2d = np.asarray(orig_cov, dtype=np.uint8).reshape(h, w)[::-1, :]
        buf[cov2d == 1] = (0.3 * np.array([180, 180, 180], dtype=np.float32) + 0.7 * buf[cov2d == 1].astype(np.float32)).astype(np.uint8)

    # Newly unlocked area in blue (optimized minus original)
    if optimized_blocked is not None:
        from core.RII_horizontal import _dilate_binary_mask, _footprint_inflation_pixels
        from core.semantic_analysis import _quick_reachable_area
        params = act_result.get("params", {})
        res = float(act_result.get("resolution", 0.05))
        floor2d = np.asarray(act_result.get("floorPx"), dtype=np.uint8).reshape(h, w)
        _, new_acc = _quick_reachable_area(optimized_blocked.reshape(h, w), floor2d, params, res)
        new_acc_disp = new_acc[::-1, :]
        orig_disp = cov2d if orig_cov is not None else np.zeros((h, w), dtype=np.uint8)
        unlocked = (new_acc_disp == 1) & (orig_disp == 0)
        buf[unlocked] = (0.5 * np.array([50, 150, 255], dtype=np.float32) + 0.5 * buf[unlocked].astype(np.float32)).astype(np.uint8)

    # Draw moves: red source, green destination
    for move in moves:
        flat = move["from_indices"]
        rows, cols = flat // w, flat % w
        disp_rows = h - 1 - rows
        buf[disp_rows, cols] = [220, 50, 50]  # red source

        if move["to_rc"] is not None:
            zr, zc = move["to_rc"]
            fh, fw = move["footprint_hw"]
            dr0 = max(0, h - 1 - (zr + fh))
            dr1 = min(h - 1, h - 1 - zr)
            dc0, dc1 = max(0, zc), min(w - 1, zc + fw)
            if dr1 > dr0 and dc1 > dc0:
                zone = buf[dr0:dr1, dc0:dc1].astype(np.float32)
                buf[dr0:dr1, dc0:dc1] = (0.4 * np.array([50, 200, 100], dtype=np.float32) + 0.6 * zone).astype(np.uint8)
                buf[dr0, dc0:dc1] = [50, 200, 100]
                buf[min(dr1, h-1), dc0:dc1] = [50, 200, 100]
                buf[dr0:dr1, dc0] = [50, 200, 100]
                buf[dr0:dr1, min(dc1, w-1)] = [50, 200, 100]

    img = QImage(buf.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()

    # Draw numbered arrows with QPainter
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)

    for i, move in enumerate(moves):
        flat = move["from_indices"]
        rows, cols = flat // w, flat % w
        cr = int(h - 1 - rows.mean())
        cc = int(cols.mean())

        # Number label on source
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(220, 50, 50))
        p.drawEllipse(QPointF(cc, cr), 12, 12)
        p.setPen(QColor(255, 255, 255))
        p.setFont(QFont("monospace", 9, QFont.Bold))
        p.drawText(cc - 5, cr + 4, f"{i+1}")

        if move["to_rc"] is not None:
            zr, zc = move["to_rc"]
            fh, fw = move["footprint_hw"]
            zr_d = int(h - 1 - (zr + fh / 2))
            zc_d = int(zc + fw / 2)

            # Arrow
            pen = QPen(QColor(255, 200, 50), 2)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(cc, cr, zc_d, zr_d)

            # Arrowhead
            dx, dy = zc_d - cc, zr_d - cr
            length = math.sqrt(dx * dx + dy * dy)
            if length > 10:
                ux, uy = dx / length, dy / length
                ax, ay = zc_d - ux * 10, zr_d - uy * 10
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 200, 50))
                p.drawPolygon(QPolygon([
                    QPoint(zc_d, zr_d),
                    QPoint(int(ax + uy * 5), int(ay - ux * 5)),
                    QPoint(int(ax - uy * 5), int(ay + ux * 5)),
                ]))

            # Number on destination
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(50, 200, 100))
            p.drawEllipse(QPointF(zc_d, zr_d), 12, 12)
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("monospace", 9, QFont.Bold))
            p.drawText(zc_d - 5, zr_d + 4, f"{i+1}")

            # Gain label
            p.setPen(QColor(255, 200, 50))
            p.setFont(QFont("monospace", 8, QFont.Bold))
            mid_x, mid_y = (cc + zc_d) // 2, (cr + zr_d) // 2
            p.drawText(mid_x + 8, mid_y - 3, f"+{move['step_gain']:.1f} m²")
        else:
            # Remove only — X mark
            p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawLine(cc - 6, cr - 6, cc + 6, cr + 6)
            p.drawLine(cc - 6, cr + 6, cc + 6, cr - 6)

    # Summary text at top
    total = optimization_result.get("total_gain", 0)
    n = len(moves)
    p.setPen(QColor(255, 255, 255))
    p.setFont(QFont("monospace", 11, QFont.Bold))
    p.fillRect(QRect(5, 5, 350, 22), QColor(0, 0, 0, 160))
    p.drawText(10, 20, f"Optimization: {n} moves, +{total:.1f} m² total gain")

    p.end()
    return img
