#!/usr/bin/env python3
"""RII Pipeline — Horizontal-only build.

Trimmed GUI for clients whose scope ends at Step 3 (RII Horizontal).
Same windows, buttons and Tools menu as the regular pipeline, but
Step 4 (Analysis / semantic / layout optimisation) and Step 5
(RII Vertical) are stripped from the sidebar, the view-bar tabs, and
the stepper. Uses its own copy of the GUI class
(``gui.main_window_horizontal.MainWin``) so the regular pipeline at
``rii_pipeline.py`` / ``rii_pipeline_v3.py`` is unaffected.

Pipeline:

    1. Select Point Cloud
    2. Generate 2D Map
    3. RII Horizontal — reference / actual robot reachability + RII

The Tools menu (Merge Two 2D Maps, Export Maps for Validation) is
kept since both items operate on Step-3 outputs.

Usage:
    python rii_pipeline_horizontal.py
"""

from __future__ import annotations

import os
import signal
import sys

# V3 single-map flow is the default for this build — it's the simplest
# pipeline that produces the Step-3 outputs the client needs.
os.environ.setdefault("RII_PIPELINE_VERSION", "3")


def _run_gui() -> None:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    from gui.main_window_horizontal import MainWin

    sigint_state = {"pending": False}

    def _handle_sigint(signum, frame):  # noqa: ARG001
        sigint_state["pending"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWin()
    win.setWindowTitle("Robot Accessibility")
    win.show()
    sig_timer = QTimer()
    sig_timer.timeout.connect(
        lambda: win.close() if sigint_state.pop("pending", False) else None
    )
    sig_timer.start(100)
    app._sigint_timer = sig_timer  # noqa: SLF001 — keep timer alive
    sys.exit(app.exec_())


if __name__ == "__main__":
    _run_gui()
