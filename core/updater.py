"""GitHub Releases-based update checker and downloader.

Pure-stdlib (urllib, json) — no new dependencies. The UI lives in
gui/update_dialog.py; this module returns plain data structures and
runs the network I/O so the UI stays testable.

Workflow:
    info = check_for_update()              # query GitHub API
    if info.has_update:
        downloader = UpdateDownloader(info.download_url, total=info.asset_size)
        downloader.progress.connect(...)
        downloader.finished.connect(...)
        downloader.start()                 # QThread
    launch_installer(path)                 # run installer, quit app
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from PyQt5.QtCore import QThread, pyqtSignal

from config import APP_VERSION, GITHUB_REPO, UPDATE_ASSET_PREFIX


# ── Version comparison ───────────────────────────────────────────────────────
_VERSION_RE = re.compile(r"^[vV]?(\d+(?:\.\d+)*)")


def _parse_version(s: str) -> tuple[int, ...]:
    """Parse '2.1.0' / 'v2.1' / '2.1.0-beta' → (2, 1, 0). Returns () if unparseable."""
    if not s:
        return ()
    m = _VERSION_RE.match(s.strip())
    if not m:
        return ()
    try:
        return tuple(int(p) for p in m.group(1).split("."))
    except ValueError:
        return ()


def _is_newer(remote: str, local: str) -> bool:
    r = _parse_version(remote)
    l = _parse_version(local)
    if not r or not l:
        return False
    # Pad to same length for tuple comparison
    n = max(len(r), len(l))
    r = r + (0,) * (n - len(r))
    l = l + (0,) * (n - len(l))
    return r > l


# ── Release info ─────────────────────────────────────────────────────────────
@dataclass
class ReleaseInfo:
    """Result of check_for_update()."""
    has_update: bool
    current_version: str
    latest_version: str = ""
    release_notes: str = ""
    download_url: str = ""
    asset_size: int = 0
    asset_name: str = ""
    release_url: str = ""
    error: str = ""


def check_for_update(timeout: float = 8.0) -> ReleaseInfo:
    """Query GitHub for the latest release. Returns ReleaseInfo.

    Never raises — network / API errors are reported via .error.
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"RII-Pipeline/{APP_VERSION}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return ReleaseInfo(False, APP_VERSION, error="No releases published yet.")
        return ReleaseInfo(False, APP_VERSION, error=f"GitHub API error: HTTP {e.code}")
    except urllib.error.URLError as e:
        return ReleaseInfo(False, APP_VERSION, error=f"Network error: {e.reason}")
    except (json.JSONDecodeError, TimeoutError) as e:
        return ReleaseInfo(False, APP_VERSION, error=f"Bad response: {e}")

    tag = data.get("tag_name") or data.get("name") or ""
    notes = data.get("body") or "(no release notes provided)"
    release_url = data.get("html_url") or f"https://github.com/{GITHUB_REPO}/releases"

    # Find the installer asset
    download_url = ""
    asset_size = 0
    asset_name = ""
    for asset in data.get("assets") or []:
        name = asset.get("name", "")
        if name.startswith(UPDATE_ASSET_PREFIX) and name.endswith(".exe"):
            download_url = asset.get("browser_download_url", "")
            asset_size = int(asset.get("size") or 0)
            asset_name = name
            break

    has_update = _is_newer(tag, APP_VERSION)
    return ReleaseInfo(
        has_update=has_update,
        current_version=APP_VERSION,
        latest_version=tag,
        release_notes=notes,
        download_url=download_url,
        asset_size=asset_size,
        asset_name=asset_name,
        release_url=release_url,
    )


# ── Downloader (QThread) ─────────────────────────────────────────────────────
class UpdateDownloader(QThread):
    """Download the installer in a background thread.

    Signals:
        progress(int bytes_done, int bytes_total)
        finished_ok(str path)
        failed(str error)
    """
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    CHUNK = 65536  # 64 KB

    def __init__(self, url: str, asset_name: str, total: int = 0, parent=None):
        super().__init__(parent)
        self._url = url
        self._total = int(total)
        self._asset_name = asset_name or "RII_Pipeline_Setup.exe"
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        out_dir = tempfile.mkdtemp(prefix="rii_update_")
        out_path = os.path.join(out_dir, self._asset_name)
        try:
            req = urllib.request.Request(
                self._url,
                headers={"User-Agent": f"RII-Pipeline/{APP_VERSION}"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp, open(out_path, "wb") as f:
                total = int(resp.headers.get("Content-Length") or self._total or 0)
                done = 0
                self.progress.emit(0, total)
                while True:
                    if self._cancel:
                        self.failed.emit("Cancelled")
                        return
                    chunk = resp.read(self.CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    self.progress.emit(done, total)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Download failed: {e}")
            return
        self.finished_ok.emit(out_path)


# ── Launch the installer ─────────────────────────────────────────────────────
def launch_installer(path: str) -> bool:
    """Launch the downloaded installer and return True on success.

    On Windows, uses ShellExecute via os.startfile so UAC elevation works.
    The caller should quit the app immediately after this returns True.
    """
    if not os.path.isfile(path):
        return False
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            # Dev/testing on Linux — just mark it executable and open
            os.chmod(path, 0o755)
            import subprocess
            subprocess.Popen([path])
        return True
    except Exception:  # noqa: BLE001
        return False


def human_size(n: int) -> str:
    """42_123_456 → '40.2 MB'."""
    if n <= 0:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"
