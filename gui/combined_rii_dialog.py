"""Combined Building RII dialog.

Summarises Step-3 coverage results across all floors built by the
multi-floor pipeline and reports a building-wide RII computed as
sum(actual) / sum(reference).
"""

from __future__ import annotations

import os
from typing import Dict

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialogButtonBox,
    QFrame, QAbstractItemView,
)


_STYLE = """
QDialog { background: #ffffff; }
QLabel#title { color: #111827; font-size: 18px; font-weight: 600; letter-spacing: 0.2px; }
QLabel#subtitle { color: #6b7280; font-size: 12px; }
QLabel#total_label { color: #6b7280; font-size: 11px; letter-spacing: 0.4px; text-transform: uppercase; }
QLabel#total_value { color: #2563eb; font-size: 32px; font-weight: 700;
                     font-family: "JetBrains Mono", "Consolas", monospace; }
QLabel#total_subvalue { color: #6b7280; font-size: 12px;
                        font-family: "JetBrains Mono", "Consolas", monospace; }
QFrame#card { background: #f8f9fa; border: 1px solid #e5e7eb; border-radius: 8px; }
QFrame#sep { background: #e5e7eb; max-height: 1px; min-height: 1px; }
QTableWidget { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px;
               gridline-color: #e5e7eb; font-size: 12px; }
QTableWidget::item { padding: 6px; color: #1f2937; }
QTableWidget::item:selected { background: #dbeafe; color: #1d4ed8; }
QHeaderView::section { background: #f9fafb; color: #374151; font-weight: 600;
                       padding: 8px; border: none; border-bottom: 1px solid #e5e7eb;
                       border-right: 1px solid #e5e7eb; font-size: 11px; }
QPushButton#close { background: #2563eb; color: #ffffff; border: none;
                    border-radius: 6px; padding: 9px 22px; font-weight: 600; font-size: 13px; }
QPushButton#close:hover { background: #1d4ed8; }
"""


class CombinedRiiDialog(QDialog):
    """level_pgm_map: {level_idx: pgm_path}
       level_results: {level_idx: {'ref': coverage_dict, 'act': coverage_dict}}
    """

    def __init__(self, level_pgm_map: Dict[int, str],
                 level_results: Dict[int, Dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combined Building RII")
        self.setModal(True)
        self.resize(720, 520)
        self.setStyleSheet(_STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18); root.setSpacing(14)

        # Header
        title = QLabel("Combined Building RII"); title.setObjectName("title")
        root.addWidget(title)
        sub = QLabel("Per-floor accessible area and an area-weighted "
                     "building-wide RII score.")
        sub.setObjectName("subtitle"); sub.setWordWrap(True)
        root.addWidget(sub)

        # Headline card
        card = QFrame(); card.setObjectName("card")
        card_l = QHBoxLayout(card); card_l.setContentsMargins(20, 16, 20, 16); card_l.setSpacing(24)
        combined_col = QVBoxLayout(); combined_col.setSpacing(2)
        self._headline_label = QLabel("Building RII"); self._headline_label.setObjectName("total_label")
        self._headline_value = QLabel("—"); self._headline_value.setObjectName("total_value")
        self._headline_sub = QLabel(""); self._headline_sub.setObjectName("total_subvalue")
        combined_col.addWidget(self._headline_label)
        combined_col.addWidget(self._headline_value)
        combined_col.addWidget(self._headline_sub)
        card_l.addLayout(combined_col, 1)

        area_col = QVBoxLayout(); area_col.setSpacing(2)
        self._area_label = QLabel("Accessible / Reachable"); self._area_label.setObjectName("total_label")
        self._area_value = QLabel("—"); self._area_value.setObjectName("total_subvalue")
        self._area_value.setStyleSheet("font-size: 16px; color:#1f2937; font-weight:600;")
        area_col.addWidget(self._area_label)
        area_col.addWidget(self._area_value)
        card_l.addLayout(area_col, 1)
        root.addWidget(card)

        # Table
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            "Floor", "Reference (m²)", "Actual (m²)", "RII", "Status"
        ])
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        root.addWidget(self._table, 1)

        # Close
        btns = QDialogButtonBox()
        close_btn = QPushButton("Close"); close_btn.setObjectName("close")
        close_btn.clicked.connect(self.accept); close_btn.setDefault(True)
        btns.addButton(close_btn, QDialogButtonBox.AcceptRole)
        root.addWidget(btns)

        self._populate(level_pgm_map, level_results)

    @staticmethod
    def _cell_area(r) -> float:
        """Accessible / reachable area in m² from a coverage result dict."""
        if not r:
            return 0.0
        # Prefer 'area' if present; otherwise derive from mask + resolution
        if "area" in r:
            try:
                return float(r["area"])
            except (TypeError, ValueError):
                pass
        mask = r.get("mask")
        res = r.get("res") or r.get("resolution")
        try:
            if mask is not None and res is not None:
                import numpy as np
                return float(np.count_nonzero(mask)) * float(res) * float(res)
        except Exception:
            pass
        return 0.0

    def _populate(self, level_pgm_map, level_results):
        rows = sorted(set(level_pgm_map.keys()) | set(level_results.keys()))

        total_ref = 0.0
        total_act = 0.0
        floor_ready = 0

        self._table.setRowCount(len(rows))
        for row_i, idx in enumerate(rows):
            bucket = level_results.get(idx, {}) or {}
            ref_r = bucket.get("ref")
            act_r = bucket.get("act")
            ref_area = self._cell_area(ref_r)
            act_area = self._cell_area(act_r)
            rii = (act_area / ref_area) if ref_area > 0 else 0.0

            status_parts = []
            if not ref_r:
                status_parts.append("missing reference")
            if not act_r:
                status_parts.append("missing actual")
            if not status_parts:
                status_parts.append("complete")
                total_ref += ref_area
                total_act += act_area
                floor_ready += 1
            status = " / ".join(status_parts)

            name = os.path.basename(level_pgm_map.get(idx, f"level {idx}"))
            cells = [
                QTableWidgetItem(f"L{idx}  {name}"),
                QTableWidgetItem(f"{ref_area:,.2f}" if ref_r else "—"),
                QTableWidgetItem(f"{act_area:,.2f}" if act_r else "—"),
                QTableWidgetItem(f"{rii * 100:.1f} %" if ref_r and act_r else "—"),
                QTableWidgetItem(status),
            ]
            for c in cells[1:4]:
                c.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            # Colour the status
            if status_parts == ["complete"]:
                cells[4].setForeground(Qt.darkGreen)
            else:
                cells[4].setForeground(Qt.darkRed)
            for col_i, cell in enumerate(cells):
                self._table.setItem(row_i, col_i, cell)

        if floor_ready == 0:
            self._headline_value.setText("—")
            self._headline_sub.setText(
                "Run Reference + Actual on at least one floor to populate the "
                "combined score."
            )
            self._area_value.setText("—")
            return

        combined = total_act / total_ref if total_ref > 0 else 0.0
        self._headline_value.setText(f"{combined * 100:.1f} %")
        self._headline_sub.setText(
            f"weighted over {floor_ready} floor{'s' if floor_ready != 1 else ''}"
        )
        self._area_value.setText(f"{total_act:,.2f}  /  {total_ref:,.2f}  m²")
