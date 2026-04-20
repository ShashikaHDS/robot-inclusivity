RII Pipeline - Windows Build Kit
================================

This folder contains everything needed to build and install the RII
Pipeline on Windows 10 or 11. No ROS, no Linux tools required.


QUICK START (one-time setup)
----------------------------

1. Install Python 3.10 or 3.11 (64-bit):
   https://www.python.org/downloads/windows/
   -> IMPORTANT: tick "Add Python to PATH" during install.

2. Install Inno Setup 6 (needed only for the installer step):
   https://jrsoftware.org/isdl.php
   Accept defaults.

3. Double-click BUILD.bat in this folder.
   It will:
     - install Python dependencies (numpy, PyQt5, etc.)
     - build the standalone app via PyInstaller
     - build the Windows installer (RII_Pipeline_Setup_2.0.exe)

   The first run takes 5-10 minutes. Re-runs are faster.


BUILD OUTPUT
------------

After BUILD.bat completes:

  dist\RII_Pipeline\RII_Pipeline.exe
     --> The portable app. Zip this folder to distribute without an
         installer. Users unzip and double-click the .exe.

  Output\RII_Pipeline_Setup_2.0.exe
     --> The proper installer. Users double-click this, click Next a
         few times, and the app is installed to Program Files with a
         Start Menu shortcut and .riiproj file association.


OTHER BATCH FILES
-----------------

  INSTALL_DEPS.bat
     Install just the Python dependencies. Useful if you only want to
     run from source without building an installer.

  RUN_DEV.bat
     Launch the app directly from source (no build step). Requires
     INSTALL_DEPS.bat to have been run first. Good for development
     and quick smoke tests.


FULL DOCUMENTATION
------------------

See README_WINDOWS.md in this folder for:
  - detailed build troubleshooting
  - code-signing guidance (SmartScreen)
  - per-user vs. all-users install
  - GitHub Actions CI workflow
  - distribution options


FIRST-TIME SMARTSCREEN WARNING
-------------------------------

The installer is unsigned. When a user runs it for the first time,
Windows SmartScreen may show a blue screen saying:

   "Windows protected your PC"

Tell the user to click "More info" then "Run anyway". This is only
on the first download of each version. For production deployments,
obtain a code-signing certificate (~$100/year) - see README_WINDOWS.md.
