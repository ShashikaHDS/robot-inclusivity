"""Horizontal pipeline step indicator — numbered circles with connector lines.

Emits `step_clicked(int)` when the user clicks a step chip. The parent
wires that to scroll the sidebar to the matching group box.
"""

from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QSize
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QFontMetrics, QLinearGradient
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
        self.setMinimumHeight(86)

    # ── Theme (can be overridden by parent) ──────────────────────────────
    C_BG_TOP = QColor("#ffffff")
    C_BG_BOTTOM = QColor("#f8f9fa")
    C_BORDER_BOTTOM = QColor("#e5e7eb")
    C_CONNECTOR = QColor("#e5e7eb")
    C_CONNECTOR_DONE = QColor("#10b981")
    C_PENDING_BG = QColor("#ffffff")
    C_PENDING_BORDER = QColor("#d1d5db")
    C_PENDING_TEXT = QColor("#9ca3af")
    C_ACTIVE_BG = QColor("#2563eb")
    C_ACTIVE_BORDER = QColor("#1d4ed8")
    C_ACTIVE_GLOW = QColor(37, 99, 235, 60)
    C_ACTIVE_TEXT = QColor("#ffffff")
    C_COMPLETE_BG = QColor("#10b981")
    C_COMPLETE_BORDER = QColor("#059669")
    C_COMPLETE_TEXT = QColor("#ffffff")
    C_LABEL_ACTIVE = QColor("#111827")
    C_LABEL_DONE = QColor("#059669")
    C_LABEL_MUTED = QColor("#9ca3af")
    C_HOVER_RING = QColor(37, 99, 235, 90)

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
        margin = 60
        if n == 1:
            return [w // 2]
        step = (w - 2 * margin) / (n - 1)
        return [int(margin + i * step) for i in range(n)]

    CHIP_R = 19
    CHIP_CY = 30

    def _chip_at(self, pos):
        for i, cx in enumerate(self._chip_centers()):
            if (pos.x() - cx) ** 2 + (pos.y() - self.CHIP_CY) ** 2 <= (self.CHIP_R + 6) ** 2:
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

        # Subtle vertical gradient background
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, self.C_BG_TOP)
        grad.setColorAt(1.0, self.C_BG_BOTTOM)
        p.fillRect(self.rect(), QBrush(grad))

        # Bottom border
        p.setPen(QPen(self.C_BORDER_BOTTOM, 1))
        p.drawLine(0, self.height() - 1, self.width(), self.height() - 1)

        centers = self._chip_centers()
        cy = self.CHIP_CY
        r = self.CHIP_R

        # Connector lines between chips — gradient fill for completed halves
        for i in range(len(centers) - 1):
            x1 = centers[i] + r
            x2 = centers[i + 1] - r
            done = self._status[i] == STATUS_COMPLETE
            next_active = self._status[i + 1] == STATUS_ACTIVE
            if done and next_active:
                # Gradient from green → blue
                lg = QLinearGradient(x1, cy, x2, cy)
                lg.setColorAt(0.0, self.C_CONNECTOR_DONE)
                lg.setColorAt(1.0, self.C_ACTIVE_BG)
                p.setPen(QPen(QBrush(lg), 2.5))
            elif done:
                p.setPen(QPen(self.C_CONNECTOR_DONE, 2.5))
            else:
                p.setPen(QPen(self.C_CONNECTOR, 2))
            p.drawLine(x1, cy, x2, cy)

        # Chips + labels
        num_font = QFont(self.font())
        num_font.setBold(True)
        num_font.setPointSize(11)
        num_fm = QFontMetrics(num_font)

        label_font = QFont(self.font())
        label_font.setPointSize(9)
        label_font.setWeight(QFont.DemiBold)
        label_fm = QFontMetrics(label_font)

        for i, cx in enumerate(centers):
            status = self._status[i]
            if status == STATUS_ACTIVE:
                bg, border, text = self.C_ACTIVE_BG, self.C_ACTIVE_BORDER, self.C_ACTIVE_TEXT
                # Outer glow for active
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(self.C_ACTIVE_GLOW))
                p.drawEllipse(QRectF(cx - r - 6, cy - r - 6, 2 * (r + 6), 2 * (r + 6)))
            elif status == STATUS_COMPLETE:
                bg, border, text = self.C_COMPLETE_BG, self.C_COMPLETE_BORDER, self.C_COMPLETE_TEXT
            else:
                bg, border, text = self.C_PENDING_BG, self.C_PENDING_BORDER, self.C_PENDING_TEXT

            if i == self._hover:
                p.setPen(QPen(self.C_HOVER_RING, 2.5))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(QRectF(cx - r - 4, cy - r - 4, 2 * (r + 4), 2 * (r + 4)))

            p.setPen(QPen(border, 2))
            p.setBrush(QBrush(bg))
            p.drawEllipse(QRectF(cx - r, cy - r, 2 * r, 2 * r))

            p.setPen(text)
            p.setFont(num_font)
            if status == STATUS_COMPLETE:
                pen = QPen(text, 2.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                p.setPen(pen)
                p.drawLine(cx - 7, cy + 1, cx - 1, cy + 7)
                p.drawLine(cx - 1, cy + 7, cx + 8, cy - 5)
            else:
                label = str(i + 1)
                tw = num_fm.horizontalAdvance(label)
                p.drawText(cx - tw // 2, cy + num_fm.ascent() // 2 - 3, label)

            # Label below
            p.setFont(label_font)
            if status == STATUS_ACTIVE:
                p.setPen(self.C_LABEL_ACTIVE)
            elif status == STATUS_COMPLETE:
                p.setPen(self.C_LABEL_DONE)
            else:
                p.setPen(self.C_LABEL_MUTED)
            label = self._labels[i]
            lw = label_fm.horizontalAdvance(label)
            p.drawText(cx - lw // 2, cy + r + 22, label)

        p.end()

    def sizeHint(self):
        return QSize(800, 86)
