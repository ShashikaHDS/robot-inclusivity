"""Collapsible section widget — a clickable header above a body widget.

Used to turn the left sidebar's flat list of QGroupBoxes into an
expand/contract tree so the user can focus on the active step.
"""

from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPoint
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPolygon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFrame, QGraphicsDropShadowEffect


class _SectionHeader(QWidget):
    clicked = pyqtSignal()

    def __init__(self, step_number, title, parent=None):
        super().__init__(parent)
        self._step = step_number
        self._title = title
        self._expanded = True
        self._active = False
        self._hover = False
        self._status = "pending"  # pending | active | complete
        self.setCursor(Qt.PointingHandCursor)
        self.setMouseTracking(True)
        self.setFixedHeight(48)
        self.setAttribute(Qt.WA_StyledBackground, False)

    def setExpanded(self, v):
        self._expanded = v
        self.update()

    def setActive(self, v):
        self._active = v
        self.update()

    def setStatus(self, status):
        if status in ("pending", "active", "complete"):
            self._status = status
            self.update()

    def enterEvent(self, _e):
        self._hover = True; self.update()

    def leaveEvent(self, _e):
        self._hover = False; self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()

    # Colors
    C_BG = QColor("#ffffff")
    C_BG_HOVER = QColor("#f9fafb")
    C_BG_ACTIVE = QColor("#eff6ff")
    C_BORDER = QColor("#e5e7eb")
    C_CHEVRON = QColor("#6b7280")
    C_TITLE = QColor("#111827")
    C_TITLE_MUTED = QColor("#6b7280")
    C_ACCENT = QColor("#2563eb")
    C_BADGE_PENDING_BG = QColor("#f3f4f6")
    C_BADGE_PENDING_BORDER = QColor("#d1d5db")
    C_BADGE_PENDING_TEXT = QColor("#6b7280")
    C_BADGE_ACTIVE_BG = QColor("#2563eb")
    C_BADGE_ACTIVE_TEXT = QColor("#ffffff")
    C_BADGE_COMPLETE_BG = QColor("#10b981")
    C_BADGE_COMPLETE_TEXT = QColor("#ffffff")

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        rect = self.rect()

        # Background
        if self._active:
            bg = self.C_BG_ACTIVE
        elif self._hover:
            bg = self.C_BG_HOVER
        else:
            bg = self.C_BG
        p.setPen(Qt.NoPen)
        p.setBrush(bg)
        p.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 8, 8)

        # Active left-accent bar
        if self._active:
            p.setBrush(self.C_ACCENT)
            p.drawRoundedRect(0, 8, 3, rect.height() - 16, 1.5, 1.5)

        # Bottom separator when expanded
        if self._expanded:
            p.setPen(QPen(self.C_BORDER, 1))
            y = rect.height() - 1
            p.drawLine(14, y, rect.width() - 14, y)

        # Chevron triangle
        cx = 18
        cy = rect.height() // 2
        chevron_color = self.C_ACCENT if self._active else self.C_CHEVRON
        p.setBrush(QBrush(chevron_color))
        p.setPen(Qt.NoPen)
        if self._expanded:
            pts = [(cx - 5, cy - 2), (cx + 5, cy - 2), (cx, cy + 4)]
        else:
            pts = [(cx - 2, cy - 5), (cx - 2, cy + 5), (cx + 4, cy)]
        p.drawPolygon(QPolygon([QPoint(x, y) for x, y in pts]))

        # Step number badge
        badge_r = 13
        bx = 42
        by = cy
        if self._status == "complete":
            bg_c, text_c, border_c = self.C_BADGE_COMPLETE_BG, self.C_BADGE_COMPLETE_TEXT, None
        elif self._status == "active" or self._active:
            bg_c, text_c, border_c = self.C_BADGE_ACTIVE_BG, self.C_BADGE_ACTIVE_TEXT, None
        else:
            bg_c, text_c, border_c = self.C_BADGE_PENDING_BG, self.C_BADGE_PENDING_TEXT, self.C_BADGE_PENDING_BORDER

        p.setBrush(QBrush(bg_c))
        if border_c is not None:
            p.setPen(QPen(border_c, 1.2))
        else:
            p.setPen(Qt.NoPen)
        p.drawEllipse(bx - badge_r, by - badge_r, 2 * badge_r, 2 * badge_r)

        if self._status == "complete":
            # Draw checkmark
            p.setPen(QPen(text_c, 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(bx - 5, by + 1, bx - 1, by + 5)
            p.drawLine(bx - 1, by + 5, bx + 6, by - 3)
        else:
            p.setPen(text_c)
            badge_font = QFont(self.font()); badge_font.setBold(True); badge_font.setPointSize(9)
            p.setFont(badge_font)
            p.drawText(bx - badge_r, by - badge_r, 2 * badge_r, 2 * badge_r,
                       Qt.AlignCenter, str(self._step))

        # Title
        title_color = self.C_TITLE if (self._active or self._status == "complete") else self.C_TITLE
        p.setPen(title_color)
        title_font = QFont(self.font())
        title_font.setPointSize(11)
        title_font.setWeight(QFont.DemiBold)
        title_font.setLetterSpacing(QFont.AbsoluteSpacing, 0.2)
        p.setFont(title_font)
        title_x = bx + badge_r + 12
        p.drawText(title_x, 0, rect.width() - title_x - 14, rect.height(),
                   Qt.AlignVCenter | Qt.AlignLeft, self._title)

        p.end()

    def sizeHint(self):
        return QSize(320, 48)


class CollapsibleSection(QFrame):
    """A section with a clickable header and a collapsible body."""
    toggled = pyqtSignal(bool)

    def __init__(self, step_number, title, body, parent=None):
        super().__init__(parent)
        self.setObjectName("CollapsibleSection")
        self.setStyleSheet(
            "#CollapsibleSection {"
            "  background: #ffffff;"
            "  border: 1px solid #e5e7eb;"
            "  border-radius: 8px;"
            "}"
        )
        # Soft drop shadow — commercial-grade depth
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(14)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(17, 24, 39, 18))
        self.setGraphicsEffect(shadow)

        self._expanded = True
        self._body = body
        self._header = _SectionHeader(step_number, title, self)
        self._header.clicked.connect(self.toggle)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._header)
        self._body_container = QWidget()
        cl = QVBoxLayout(self._body_container)
        cl.setContentsMargins(14, 12, 14, 14)
        cl.setSpacing(8)
        cl.addWidget(body)
        root.addWidget(self._body_container)

    def toggle(self):
        self.setExpanded(not self._expanded)

    def setExpanded(self, v):
        if self._expanded == v:
            return
        self._expanded = v
        self._body_container.setVisible(v)
        self._header.setExpanded(v)
        self.toggled.emit(v)

    def setActive(self, v):
        self._header.setActive(v)

    def setStatus(self, status):
        self._header.setStatus(status)

    def expand(self):
        self.setExpanded(True)

    def collapse(self):
        self.setExpanded(False)

    def isExpanded(self):
        return self._expanded
