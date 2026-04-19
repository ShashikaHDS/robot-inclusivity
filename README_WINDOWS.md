# Building the RII Pipeline for Windows 10 / 11

This guide produces a **single `RII_Pipeline_Setup_<version>.exe` installer** that end users can double-click to install the app — no Python, no command line, no dependencies required on the target machine.

The pipeline is two stages:

```
Python source ──► [build_windows.py + PyInstaller] ──► dist\RII_Pipeline\  (folder with .exe)
                                                                │
dist\RII_Pipeline\ ──► [Inno Setup + installer.iss] ──► Output\RII_Pipeline_Setup_2.0.exe
```

---

## 1. One-time setup on the build machine

You only need a **Windows 10 or 11 (x64) machine** with admin rights. No ROS, no WSL.

### 1a. Install Python 3.10 or 3.11

Download from [python.org](https://www.python.org/downloads/windows/). Pick the **64-bit installer** and tick **"Add Python to PATH"** during install.

Verify in PowerShell:
```powershell
python --version   # → Python 3.11.x
```

> **Note:** Python 3.12+ also works, but 3.10/3.11 have the best PyQt5 wheel support and smaller installer sizes.

### 1b. Clone the repo and install dependencies

```powershell
git clone https://github.com/ShashikaHDS/Teal-Robot.git
cd Teal-Robot\src\pcd_package\rii_pipeline
pip install -r requirements.txt pyinstaller
```

### 1c. Install Inno Setup 6

Download and install: [https://jrsoftware.org/isdl.php](https://jrsoftware.org/isdl.php). Accept all defaults.

---

## 2. Build the application folder

From the `rii_pipeline` directory:

```powershell
python build_windows.py
```

This will:

1. Auto-generate `icon.ico` from `icon.png` if needed (via Pillow).
2. Invoke PyInstaller with the correct hidden imports, data dirs, and icon.
3. Produce **`dist\RII_Pipeline\RII_Pipeline.exe`** plus a tree of DLLs and Python runtime.

Quick smoke test — launch the folder-mode app directly to confirm it works before packaging:

```powershell
dist\RII_Pipeline\RII_Pipeline.exe
```

You should see the main window. Close it when satisfied.

> If the app crashes with "failed to execute script":
> - Temporarily change `--windowed` to `--console` in `build_windows.py`, rebuild, and watch the stderr output for the missing module.
> - Add the missing module to the `hidden` list in `build_windows.py`.

---

## 3. Build the single-file installer

Open `installer.iss` in the **Inno Setup Compiler** and press **F9** (Build → Compile), or run from PowerShell:

```powershell
& "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
```

Output:

```
Output\RII_Pipeline_Setup_2.0.exe
```

That single file is everything you ship. Upload it to GitHub Releases, email it, host on a shared drive — whatever fits.

---

## 4. What the end user sees

A user downloads `RII_Pipeline_Setup_2.0.exe` and double-clicks it. They get:

1. A standard Windows setup wizard (Welcome → License → Install dir → Shortcuts → Install → Finish).
2. The app installed to `C:\Program Files\RII_Pipeline\`.
3. A **Start Menu entry** under `RII Pipeline`.
4. An **optional desktop shortcut** (checkbox in wizard).
5. `.riiproj` **file association** — double-clicking a saved project opens it directly.
6. An **uninstaller** registered in *Settings → Apps → Installed apps*.

On uninstall, the session cache under `%TEMP%\rii_pipeline_cache` is cleaned up too.

---

## 5. Version bumps

When you ship a new version, edit **two** constants in `installer.iss`:

```
#define MyAppVersion    "2.1"    ; ← bump this
```

Keep the `AppId` GUID the same — that's what tells Windows it's an upgrade of the existing app, not a parallel install.

---

## 6. Tradeoffs & optional polish

### Current install size
Roughly **200–250 MB** on disk. Biggest contributors: PyQt5 (~60 MB), numpy + numba + scipy (~80 MB), PyOpenGL + pyqtgraph (~30 MB).

### SmartScreen warning
Unsigned installers trigger *"Windows protected your PC"* on first run. Users have to click *More info → Run anyway*. To remove this:

- **Short term:** document the workaround in release notes.
- **Long term:** buy a code-signing certificate (~$100/year from Sectigo, DigiCert, SSL.com) and sign both `RII_Pipeline.exe` and the setup .exe with `signtool`. After ~a few hundred downloads, SmartScreen learns to trust the certificate and the warning disappears permanently.

### Per-user vs. all-users install
Current script installs **for all users** (needs admin). To allow non-admin install, change `installer.iss`:

```
PrivilegesRequired=lowest
DefaultDirName={userpf}\RII_Pipeline
```

### Portable zip distribution (no installer)
If you prefer a no-install version, just zip `dist\RII_Pipeline\` after step 2 and ship that. Users unzip anywhere and double-click `RII_Pipeline.exe`. They lose Start Menu / file association / uninstaller but gain zero-trust portability.

### CI build
Both steps run on GitHub Actions' `windows-latest` runners. A rough outline:

```yaml
# .github/workflows/windows-build.yml
on: [push]
jobs:
  windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r src/pcd_package/rii_pipeline/requirements.txt pyinstaller
      - run: python src/pcd_package/rii_pipeline/build_windows.py
      - name: Install Inno Setup
        run: choco install innosetup -y
      - run: iscc src/pcd_package/rii_pipeline/installer.iss
      - uses: actions/upload-artifact@v4
        with: { path: src/pcd_package/rii_pipeline/Output/*.exe }
```

Flip this on once you have a release cadence — every push produces a downloadable installer artifact.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` when exe runs | PyInstaller missed a hidden import | Add module to `hidden` list in `build_windows.py` |
| App opens but shows blank 3D viewer | PyOpenGL / pyqtgraph.opengl not bundled | Confirm `--collect-all pyqtgraph` is in `cmd` and PyOpenGL is installed |
| Setup fails with "access denied" | Running from Program Files without admin | Right-click installer → Run as administrator, OR switch to per-user install (see §6) |
| `icon.ico` generation fails | Pillow not installed on build machine | `pip install Pillow` then re-run `build_windows.py` |
| `.riiproj` double-click opens wrong app | Another app stole the association | Right-click a `.riiproj` → Open with → Choose another app → RII Pipeline → check "Always use this app" |
| Installer is ~700 MB | `open3d` still in requirements | Confirm it was removed from `requirements.txt` and uninstall it in the build venv (`pip uninstall open3d`) |
