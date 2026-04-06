#!/usr/bin/env python3
"""
RII Pipeline GUI — entry point.
Requires: pip install PyQt5 numpy Pillow pyqtgraph PyOpenGL --break-system-packages
"""
import sys
import signal
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from gui.main_window import MainWin

if __name__ == "__main__":
    sigint_state = {"pending": False}

    def _handle_sigint(signum, frame):
        sigint_state["pending"] = True

    signal.signal(signal.SIGINT, _handle_sigint)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWin()
    w.show()
    sig_timer = QTimer()
    sig_timer.timeout.connect(lambda: w.close() if sigint_state.pop("pending", False) else None)
    sig_timer.start(100)
    app._sigint_timer = sig_timer
    sys.exit(app.exec_())
