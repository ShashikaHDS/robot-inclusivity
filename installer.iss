; Inno Setup script for the RII Pipeline — Windows 10/11 installer.
;
; Prerequisites:
;   1. Run `python build_windows.py` first — it populates dist\RII_Pipeline\
;   2. Install Inno Setup 6: https://jrsoftware.org/isdl.php
;   3. Open this .iss in Inno Setup Compiler and press F9 (Build → Compile),
;      or run from the command line:
;          "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss
;
; Output:
;   Output\RII_Pipeline_Setup_2.0.exe  — single-file installer
;
; Users run that .exe, click Next a few times, and the app is installed to
;   %ProgramFiles%\RII_Pipeline\
; with a Start Menu entry, optional Desktop shortcut, and an uninstaller
; registered under "Add or Remove Programs".

#define MyAppName       "RII Pipeline"
#define MyAppVersion    "2.0"
#define MyAppPublisher  "Teal Robotics"
#define MyAppURL        "https://github.com/ShashikaHDS/Teal-Robot"
#define MyAppExeName    "RII_Pipeline.exe"
#define MyAppFolderName "RII_Pipeline"

[Setup]
; A unique AppId — keep this constant across version updates so upgrades
; replace old installs instead of making parallel ones.
AppId={{C7A3B1D2-5F4E-4B61-9E8C-7D3F2A1B9C0D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppFolderName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=RII_Pipeline_Setup_{#MyAppVersion}
SetupIconFile=icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Require Windows 10 or newer (10.0 covers both Win10 and Win11)
MinVersion=10.0
; Install for all users by default; flip to commonappdata→userappdata for per-user
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
; Uninstaller icon
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional shortcuts:"; Flags: unchecked
Name: "quicklaunchicon"; Description: "Create a &Quick Launch icon"; GroupDescription: "Additional shortcuts:"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; Pull in the entire PyInstaller output folder. `recursesubdirs` walks into
; subfolders; `createallsubdirs` recreates the tree under {app}.
Source: "dist\{#MyAppFolderName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Start Menu
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"
Name: "{autoprograms}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
; Desktop (optional, via task checkbox)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; IconFilename: "{app}\{#MyAppExeName}"
; Quick Launch (legacy; only pre-Win7)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
; Offer to launch after install (checkbox on last wizard page)
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Registry]
; File association for .riiproj — double-clicking a project opens it
Root: HKLM; Subkey: "Software\Classes\.riiproj"; ValueType: string; ValueName: ""; ValueData: "RIIPipeline.Project"; Flags: uninsdeletevalue
Root: HKLM; Subkey: "Software\Classes\RIIPipeline.Project"; ValueType: string; ValueName: ""; ValueData: "RII Pipeline Project"; Flags: uninsdeletekey
Root: HKLM; Subkey: "Software\Classes\RIIPipeline.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKLM; Subkey: "Software\Classes\RIIPipeline.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[UninstallDelete]
; Clean the session cache left behind in %TEMP%
Type: filesandordirs; Name: "{localappdata}\Temp\rii_pipeline_cache"
