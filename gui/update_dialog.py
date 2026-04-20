"""Update dialogs — commercial-grade check/download/install flow.

Entry points:
    check_and_prompt(parent, silent=False)
        Starts a background check. If update found, shows the
        UpdateAvailableDialog. If silent=True, stays quiet when
        there's no update or an error (used for startup auto-check).
"""

from __future__ import annotations

import webbrowser

from PyQt5.QtCore import Qt, QThread, QUrl, pyqtSignal, QTimer
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QProgressBar, QDialogButtonBox, QWidget, QFrame,
    QMessageBox, QSizePolicy,
)

from core.updater import (
    ReleaseInfo, UpdateDownloader, check_for_update,
    launch_installer, human_size,
)


# ── Shared styled base ───────────────────────────────────────────────────────
_BASE_QSS = """
QDialog { background: #ffffff; }
QLabel { color: #1f2937; }
QLabel#title { color: #111827; font-size: 18px; font-weight: 600; letter-spacing: 0.2px; }
QLabel#subtitle { color: #6b7280; font-size: 12px; }
QLabel#version { font-family: "JetBrains Mono", "Consolas", monospace;
                 font-size: 13px; padding: 4px 10px; border-radius: 12px;
                 background: #f3f4f6; color: #374151; }
QLabel#version[role="current"] { background: #f3f4f6; color: #6b7280; }
QLabel#version[role="latest"] { background: #dbeafe; color: #1d4ed8; }
QLabel#arrow { color: #9ca3af; font-size: 16px; font-weight: 600; padding: 0 2px; }
QTextBrowser { background: #fafbfc; border: 1px solid #e5e7eb; border-radius: 8px;
               padding: 10px 12px; font-size: 12px; color: #1f2937; }
QPushButton#primary { background: #2563eb; color: #ffffff; border: none;
                      border-radius: 6px; padding: 9px 20px; font-weight: 600; font-size: 13px; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton#primary:pressed { background: #1e40af; }
QPushButton#primary:disabled { background: #93c5fd; color: #e0e7ff; }
QPushButton#secondary { background: #ffffff; color: #374151;
                        border: 1px solid #d1d5db; border-radius: 6px;
                        padding: 9px 16px; font-weight: 600; font-size: 13px; }
QPushButton#secondary:hover { background: #f9fafb; border-color: #9ca3af; }
QPushButton#secondary:pressed { background: #f3f4f6; }
QFrame#sep { background: #e5e7eb; max-height: 1px; min-height: 1px; }
QProgressBar { background: #f3f4f6; border: none; border-radius: 4px;
               height: 8px; text-align: center; }
QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                      stop:0 #3b82f6, stop:1 #2563eb); border-radius: 4px; }
"""


class _StyledDialog(QDialog):
    def __init__(self, parent, title, w=560, h=520):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(w, h)
        self.setModal(True)
        self.setStyleSheet(_BASE_QSS)


# ── Check dialog (spinner) ───────────────────────────────────────────────────
class _CheckWorker(QThread):
    done = pyqtSignal(object)  # ReleaseInfo

    def run(self):
        info = check_for_update()
        self.done.emit(info)


class CheckingDialog(_StyledDialog):
    """Small modal shown while we hit the GitHub API."""
    def __init__(self, parent=None):
        super().__init__(parent, "Checking for Updates", w=420, h=170)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 20); root.setSpacing(10)

        title = QLabel("Checking for updates…"); title.setObjectName("title")
        root.addWidget(title)

        sub = QLabel("Contacting GitHub to find the latest release."); sub.setObjectName("subtitle")
        sub.setWordWrap(True)
        root.addWidget(sub)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)  # indeterminate
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        root.addWidget(self._bar)

        root.addStretch()

        btns = QHBoxLayout(); btns.addStretch()
        cancel = QPushButton("Cancel"); cancel.setObjectName("secondary")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        root.addLayout(btns)


# ── Up-to-date confirmation ──────────────────────────────────────────────────
class UpToDateDialog(_StyledDialog):
    def __init__(self, current_version: str, parent=None):
        super().__init__(parent, "Up to Date", w=400, h=180)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 16); root.setSpacing(8)

        title = QLabel("You're up to date"); title.setObjectName("title")
        root.addWidget(title)

        sub = QLabel(f"You are running the latest version "
                     f"(<b>v{current_version}</b>)."); sub.setObjectName("subtitle")
        sub.setTextFormat(Qt.RichText); sub.setWordWrap(True)
        root.addWidget(sub)

        root.addStretch()

        btns = QHBoxLayout(); btns.addStretch()
        ok = QPushButton("OK"); ok.setObjectName("primary")
        ok.clicked.connect(self.accept)
        ok.setDefault(True)
        btns.addWidget(ok)
        root.addLayout(btns)


# ── Update available ─────────────────────────────────────────────────────────
class UpdateAvailableDialog(_StyledDialog):
    """Shows release notes + [Download & Install] [View Release] [Later]."""

    DOWNLOAD = 1
    VIEW_PAGE = 2
    LATER = 0

    def __init__(self, info: ReleaseInfo, parent=None):
        super().__init__(parent, "Update Available", w=600, h=540)
        self._choice = self.LATER
        self._info = info

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18); root.setSpacing(14)

        # Header
        header = QVBoxLayout(); header.setSpacing(4)
        title = QLabel("A new version is available"); title.setObjectName("title")
        header.addWidget(title)
        sub = QLabel("Update to get the latest features, fixes, and improvements.")
        sub.setObjectName("subtitle"); sub.setWordWrap(True)
        header.addWidget(sub)
        root.addLayout(header)

        # Version chips
        vrow = QHBoxLayout(); vrow.setSpacing(6); vrow.setContentsMargins(0, 4, 0, 4)
        cur = QLabel(f"v{info.current_version}"); cur.setObjectName("version")
        cur.setProperty("role", "current"); cur.setAlignment(Qt.AlignCenter)
        cur.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        arrow = QLabel("→"); arrow.setObjectName("arrow")
        latest = QLabel(info.latest_version or "—"); latest.setObjectName("version")
        latest.setProperty("role", "latest"); latest.setAlignment(Qt.AlignCenter)
        latest.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        size_label = QLabel("")
        if info.asset_size:
            size_label.setText(f"   Download size: {human_size(info.asset_size)}")
            size_label.setStyleSheet("color:#6b7280; font-size: 11px;")
        vrow.addWidget(cur); vrow.addWidget(arrow); vrow.addWidget(latest)
        vrow.addWidget(size_label); vrow.addStretch()
        root.addLayout(vrow)

        # Separator
        sep = QFrame(); sep.setObjectName("sep"); sep.setFrameShape(QFrame.HLine)
        root.addWidget(sep)

        # Release notes
        notes_label = QLabel("What's new:")
        notes_label.setStyleSheet("color:#1f2937; font-size: 12px; font-weight: 600;")
        root.addWidget(notes_label)

        self._notes = QTextBrowser()
        self._notes.setOpenExternalLinks(True)
        self._notes.setHtml(self._render_notes(info.release_notes))
        root.addWidget(self._notes, 1)

        # Buttons
        btns = QHBoxLayout(); btns.setSpacing(8)
        later = QPushButton("Remind Me Later"); later.setObjectName("secondary")
        later.clicked.connect(self._later); btns.addWidget(later)
        btns.addStretch()
        view = QPushButton("View Release"); view.setObjectName("secondary")
        view.clicked.connect(self._view); btns.addWidget(view)
        if info.download_url:
            dl = QPushButton("Download && Install"); dl.setObjectName("primary")
            dl.setDefault(True); dl.clicked.connect(self._download)
            btns.addWidget(dl)
        root.addLayout(btns)

    @staticmethod
    def _render_notes(md: str) -> str:
        """Very small markdown-ish → HTML renderer (headings, bullets, code, links)."""
        import html
        text = html.escape(md or "").replace("\r\n", "\n")
        lines = text.split("\n")
        out, in_list = [], False
        for ln in lines:
            s = ln.rstrip()
            if s.startswith(("- ", "* ")):
                if not in_list:
                    out.append("<ul style='margin:4px 0 4px 18px;padding:0;'>"); in_list = True
                out.append(f"<li>{s[2:]}</li>")
                continue
            if in_list:
                out.append("</ul>"); in_list = False
            if s.startswith("### "):
                out.append(f"<h4 style='margin:10px 0 4px 0;color:#1f2937;'>{s[4:]}</h4>")
            elif s.startswith("## "):
                out.append(f"<h3 style='margin:12px 0 4px 0;color:#1f2937;'>{s[3:]}</h3>")
            elif s.startswith("# "):
                out.append(f"<h3 style='margin:12px 0 4px 0;color:#1f2937;'>{s[2:]}</h3>")
            elif not s:
                out.append("<br>")
            else:
                out.append(f"<div style='margin:2px 0;color:#374151;'>{s}</div>")
        if in_list:
            out.append("</ul>")
        return "".join(out)

    def _later(self):
        self._choice = self.LATER
        self.reject()

    def _view(self):
        QDesktopServices.openUrl(QUrl(self._info.release_url))
        self._choice = self.VIEW_PAGE
        self.accept()

    def _download(self):
        self._choice = self.DOWNLOAD
        self.accept()

    def choice(self) -> int:
        return self._choice


# ── Download progress ────────────────────────────────────────────────────────
class DownloadDialog(_StyledDialog):
    """Shows a progress bar while streaming the installer from GitHub."""

    def __init__(self, info: ReleaseInfo, parent=None):
        super().__init__(parent, "Downloading Update", w=480, h=220)
        self._info = info
        self._installer_path = ""
        self._error = ""
        self._done = False

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18); root.setSpacing(10)

        title = QLabel(f"Downloading v{info.latest_version}"); title.setObjectName("title")
        root.addWidget(title)

        sub = QLabel(f"Fetching {info.asset_name or 'installer'} from GitHub…")
        sub.setObjectName("subtitle"); sub.setWordWrap(True)
        root.addWidget(sub)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100); self._bar.setValue(0)
        self._bar.setFixedHeight(10); self._bar.setTextVisible(False)
        root.addWidget(self._bar)

        self._status = QLabel("Starting…")
        self._status.setStyleSheet("color:#6b7280; font-size: 11px; font-family: monospace;")
        root.addWidget(self._status)

        root.addStretch()

        btns = QHBoxLayout(); btns.addStretch()
        self._cancel_btn = QPushButton("Cancel"); self._cancel_btn.setObjectName("secondary")
        self._cancel_btn.clicked.connect(self._cancel)
        btns.addWidget(self._cancel_btn)
        root.addLayout(btns)

        # Start the download
        self._dl = UpdateDownloader(info.download_url, info.asset_name, info.asset_size, self)
        self._dl.progress.connect(self._on_progress)
        self._dl.finished_ok.connect(self._on_finished)
        self._dl.failed.connect(self._on_failed)
        self._dl.start()

    def _on_progress(self, done: int, total: int):
        if total > 0:
            pct = int(done / total * 100)
            self._bar.setValue(pct)
            self._status.setText(
                f"{human_size(done)} of {human_size(total)}  ·  {pct}%"
            )
        else:
            self._status.setText(f"{human_size(done)} downloaded")

    def _on_finished(self, path: str):
        self._done = True
        self._installer_path = path
        self._bar.setValue(100)
        self._status.setText("Download complete.")
        self._cancel_btn.setText("Close")
        self.accept()

    def _on_failed(self, err: str):
        self._done = True
        self._error = err
        self.reject()

    def _cancel(self):
        if self._dl.isRunning():
            self._dl.cancel()
            self._dl.wait(1500)
        self.reject()

    def installer_path(self) -> str:
        return self._installer_path

    def error(self) -> str:
        return self._error


# ── Public entry point ───────────────────────────────────────────────────────
def check_and_prompt(parent, silent: bool = False) -> None:
    """Full flow: check → prompt → download → launch installer.

    silent=True  — used by startup auto-check; stays quiet if no update or error.
    silent=False — used by the menu action; always shows something.
    """
    dlg = None if silent else CheckingDialog(parent)
    worker = _CheckWorker(parent)

    def on_done(info: ReleaseInfo):
        if dlg and dlg.isVisible():
            dlg.accept()
        _handle_result(parent, info, silent)

    worker.done.connect(on_done)
    worker.start()
    if dlg:
        dlg.exec_()  # blocks until worker finishes (or user cancels)
    else:
        # Silent: poll until worker finishes so we don't return early
        while worker.isRunning():
            QApplication.processEvents()
            worker.wait(50)


def _handle_result(parent, info: ReleaseInfo, silent: bool):
    if info.error:
        if not silent:
            QMessageBox.warning(parent, "Update Check Failed", info.error)
        return
    if not info.has_update:
        if not silent:
            UpToDateDialog(info.current_version, parent).exec_()
        return
    dlg = UpdateAvailableDialog(info, parent)
    dlg.exec_()
    if dlg.choice() != UpdateAvailableDialog.DOWNLOAD:
        return
    if not info.download_url:
        QDesktopServices.openUrl(QUrl(info.release_url))
        return
    dl = DownloadDialog(info, parent)
    if dl.exec_() != QDialog.Accepted:
        if dl.error() and not silent:
            QMessageBox.warning(parent, "Download Failed", dl.error())
        return
    path = dl.installer_path()
    if not path:
        return
    # Confirm — then launch installer and quit app
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Question)
    msg.setWindowTitle("Install Update")
    msg.setText(f"Update v{info.latest_version} downloaded.\n\n"
                f"The app will close and the installer will start now.")
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    msg.setDefaultButton(QMessageBox.Ok)
    if msg.exec_() != QMessageBox.Ok:
        return
    if launch_installer(path):
        QApplication.quit()
    else:
        QMessageBox.warning(parent, "Install Failed",
                            f"Could not launch the installer.\nFile: {path}")
