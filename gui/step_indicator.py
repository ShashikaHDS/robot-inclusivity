"""Horizontal pipeline step indicator — numbered circles with connector lines.

Emits `step_clicked(int)` when the user clicks a step chip. The parent
wires that to scroll the sidebar to the matching group box.
"""

from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QFontMetrics
from PyQt5.QtWidgets import QWidget


STATUS_PENDING = "pending"
STATUS_ACTIVE = "active"
STATUS_COMPLETE = "complete"


class StepIndicator(QWidget):
    step_clicked = pyqtSignal(int)  # 0-based index

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self._labels = list(labels)
        self._status = [STATUS_PENDING] * len(self._labels)
        self._hover = -1
        self.setMouseTracking(True)
        self.setMinimumHeight(68)

    # ── Theme (can be overridden by parent) ──────────────────────────────
    C_BG = QColor("#f8f9fa")
    C_BORDER_BOTTOM = QColor("#d1d5db")
    C_CONNECTOR = QColor("#e5e7eb")
    C_CONNECTOR_DONE = QColor("#10b981")
    C_PENDING_BG = QColor("#f3f4f6")
    C_PENDING_BORDER = QColor("#d1d5db")
    C_PENDING_TEXT = QColor("#6b7280")
    C_ACTIVE_BG = QColor("#2563eb")
    C_ACTIVE_BORDER = QColor("#1d4ed8")
    C_ACTIVE_TEXT = QColor("#ffffff")
    C_COMPLETE_BG = QColor("#10b981")
    C_COMPLETE_BORDER = QColor("#059669")
    C_COMPLETE_TEXT = QColor("#ffffff")
    C_LABEL = QColor("#1f2937")
    C_LABEL_MUTED = QColor("#6b7280")
    C_HOVER_RING = QColor("#93c5fd")

    def set_status(self, index, status):
        if 0 <= index < len(self._status) and status in (STATUS_PENDING, STATUS_ACTIVE, STATUS_COMPLETE):
            self._status[index] = status
            self.update()

    def set_active(self, index):
        """Mark `index` as active, everything before as complete, after as pending."""
        for i in range(len(self._status)):
            if i < index:
                self._status[i] = STATUS_COMPLETE
            elif i == index:
                self._status[i] = STATUS_ACTIVE
            else:
                self._status[i] = STATUS_PENDING
        self.update()

    # ── Layout helpers ───────────────────────────────────────────────────
    def _chip_centers(self):
        n = len(self._labels)
        if n == 0:
            return []
        w = self.width()
        margin = 40
        if n == 1:
            return [w // 2]
        step = (w - 2 * margin) / (n - 1)
        return [int(margin + i * step) for i in range(n)]

    CHIP_R = 16

    def _chip_at(self, pos):
        cy = 22
        for i, cx in enumerate(self._chip_centers()):
            if (pos.x() - cx) ** 2 + (pos.y() - cy) ** 2 <= (self.CHIP_R + 4) ** 2:
                return i
        return -1

    # ── Events ───────────────────────────────────────────────────────────
    def mouseMoveEvent(self, e):
        idx = self._chip_at(e.pos())
        if idx != self._hover:
            self._hover = idx
            self.setCursor(Qt.PointingHandCursor if idx >= 0 else Qt.ArrowCursor)
            self.update()

    def leaveEvent(self, _e):
        if self._hover != -1:
            self._hover = -1
            self.setCursor(Qt.ArrowCursor)
            self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        idx = self._chip_at(e.pos())
        if idx >= 0:
            self.step_clicked.emit(idx)

    # ── Paint ────────────────────────────────────────────────────────────
    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), self.C_BG)
        # Bottom border
        p.setPen(QPen(self.C_BORDER_BOTTOM, 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        centers = self._chip_centers()
        cy = 22
        r = self.CHIP_R

        # Connector lines between chips
        for i in range(len(centers) - 1):
            done = self._status[i] == STATUS_COMPLETE
            pen = QPen(self.C_CONNECTOR_DONE if done else self.C_CONNECTOR, 2)
            p.setPen(pen)
            p.drawLine(centers[i] + r, cy, centers[i + 1] - r, cy)

        # Chips + labels
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSize(10)
        fm = QFontMetrics(font)

        label_font = QFont(self.font())
        label_font.setPointSize(9)
        label_fm = QFontMetrics(label_font)

        for i, cx in enumerate(centers):
            status = self._status[i]
            if status == STATUS_ACTIVE:
                bg, border, text = self.C_ACTIVE_BG, self.C_ACTIVE_BORDER, self.C_ACTIVE_TEXT
            elif status == STATUS_COMPLETE:
                bg, border, text = self.C_COMPLETE_BG, self.C_COMPLETE_BORDER, self.C_COMPLETE_TEXT
            else:
                bg, border, text = self.C_PENDING_BG, self.C_PENDING_BORDER, self.C_PENDING_TEXT

            if i == self._hover:
                p.setPen(QPen(self.C_HOVER_RING, 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QRectF(cx - r - 3, cy - r - 3, 2 * (r + 3), 2 * (r + 3)))

            p.setPen(QPen(border, 2))
            p.setBrush(QBrush(bg))
            p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))

            p.setPen(text)
            p.setFont(font)
            if status == STATUS_COMPLETE:
                # Draw a checkmark
                path_pen = QPen(text, 2.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                p.setPen(path_pen)
                p.drawLine(cx - 6, cy + 1, cx - 1, cy + 6)
                p.drawLine(cx - 1, cy + 6, cx + 7, cy - 4)
            else:
                label = str(i + 1)
                tw = fm.horizontalAdvance(label)
                p.drawText(cx - tw // 2, cy + fm.ascent() // 2 - 2, label)

            # Label below
            p.setFont(label_font)
            p.setPen(self.C_LABEL if status == STATUS_ACTIVE else self.C_LABEL_MUTED)
            label = self._labels[i]
            lw = label_fm.horizontalAdvance(label)
            p.drawText(cx - lw // 2, cy + r + 18, label)

        p.end()

    def sizeHint(self):
        return QSize(800, 68)
