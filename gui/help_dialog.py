"""Help dialogs — User Guide, Keyboard Shortcuts, About."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QDialogButtonBox, QListWidget, QListWidgetItem,
    QStackedWidget, QWidget,
)


from config import APP_VERSION

APP_NAME = "Robot Inclusivity Index (RII)"


_GUIDE_SECTIONS = [
    ("Quick Start", """
        <h2>Quick Start</h2>
        <ol>
          <li><b>Pick a point cloud</b> — use the <i>Input File</i> field in Step 1 (or drag a .pcd/.ply file onto the window).</li>
          <li><b>Click <i>View Raw Point Cloud</i></b> to inspect it in the 3D Viewer tab.</li>
          <li><b>Set parameters</b> in Step 2 (robot's slope / step limits, Z-bounds) then click <i>Generate 2D Map</i>.</li>
          <li><b>Detect ramps</b> with the <i>Detect Ramps (RANSAC)</i> button. Traversable ramps stay free on the map; non-traversable ones are blocked.</li>
          <li><b>Optional:</b> draw ramps manually if RANSAC misses any — use the <i>Select Ramp Area</i> / <i>Add Ramp</i> buttons.</li>
          <li><b>Analyse</b> — Steps 3, 4, 5 evaluate RII Horizontal, Semantic Analysis, and RII Vertical respectively.</li>
          <li><b>Save your work</b> — <i>File → Save Project</i> writes a .riiproj file so you can reopen without re-processing.</li>
        </ol>
    """),

    ("Step 1 — View Point Cloud", """
        <h2>Step 1 — View Raw Point Cloud</h2>
        <p>Load and inspect the raw 3D data before processing.</p>
        <h3>Inputs</h3>
        <ul>
          <li><b>Input File</b> — path to a .pcd or .ply file (hint: drag-and-drop also works).</li>
        </ul>
        <h3>Optional noise filter</h3>
        <p>Enable <b>Apply noise filter</b> to remove scattered points before the map is built. The filter keeps points with at least <i>Min density</i> neighbours inside <i>Radius</i>.</p>
        <ul>
          <li><i>Radius</i> ≈ 0.5 m works for most LiDAR maps.</li>
          <li><i>Min density</i>: 50 light / 100 medium (default) / 200 aggressive.</li>
        </ul>
    """),

    ("Step 2 — Build 2D Map", """
        <h2>Step 2 — Generate 2D Map</h2>
        <p>Project the point cloud into a 2D occupancy grid plus traversability/floor sidecars.</p>
        <h3>Robot capability thresholds</h3>
        <ul>
          <li><b>max_slope (deg)</b> — steepest ramp the robot can climb.</li>
          <li><b>max_step (m)</b> — tallest curb/step the robot can clear.</li>
        </ul>
        <h3>Z-range</h3>
        <ul>
          <li><b>min_z / max_z</b> — height window for the obstacle layer. Points outside are ignored.</li>
          <li><b>min_points_per_cell</b> — noise gate: cells with fewer points are treated as empty.</li>
        </ul>
        <h3>Outputs</h3>
        <p>Produces <code>map.pgm</code>, <code>map.yaml</code>, and traversability/floor sidecars in the map output directory.</p>
    """),

    ("Ramp Detection & Editing", """
        <h2>Ramps</h2>
        <p>The pipeline treats ramps as traversable terrain the robot can roll up. Non-traversable ramps (too steep / too tall) are blocked on the 2D map.</p>
        <h3>Auto-detect (RANSAC)</h3>
        <ol>
          <li>After Step 2, click <b>Detect Ramps (RANSAC)</b>.</li>
          <li>Results are overlaid on the obstacle map — <span style="color:#10b981">green</span> = passable, <span style="color:#dc2626">red</span> = blocked.</li>
        </ol>
        <h3>Manual ramp editing</h3>
        <ol>
          <li>Click <b>Select Ramp Area</b>, then drag a rectangle (or freeform polygon) on the map.</li>
          <li>Click <b>Add Ramp</b> and enter the measured slope angle in degrees.</li>
          <li>The new ramp joins the detected set and the traversability map updates immediately.</li>
          <li>Use <b>Remove Last Ramp</b> to undo the most recent addition.</li>
        </ol>
        <h3>Brush editing</h3>
        <p>Use the brush shape/size controls to paint-in or paint-out map regions that need manual cleanup.</p>
    """),

    ("Step 3 — RII Horizontal", """
        <h2>Step 3 — RII Horizontal (Coverage)</h2>
        <p>Evaluate how much of the map the robot can actually reach.</p>
        <h3>Inputs</h3>
        <ul>
          <li><b>PGM / YAML</b> — auto-filled from Step 2. Browse to load a different map.</li>
          <li><b>Robot shape</b> — circular (radius) or rectangular (width × length).</li>
          <li><b>Anchor shape</b> — the starting footprint. Select <i>Freeform</i> mode to draw a custom start region on the map.</li>
          <li><b>Selection mode</b> — Rectangle or Freeform (polygon).</li>
        </ul>
        <h3>Run</h3>
        <p>Click <b>Reference</b> to compute the theoretical reachable area, then <b>Actual</b> to compute after current obstacles. The <i>Compare</i> tab overlays both.</p>
        <h3>Path planner</h3>
        <p>Pick a planner from the drop-down (e.g. STC, RRT) and click <b>Plan Path</b> to view a coverage route.</p>
    """),

    ("Step 4 — Semantic Analysis", """
        <h2>Step 4 — RII Horizontal Analysis (Semantic)</h2>
        <p>Load a labelled point cloud and identify removable objects that open up coverage.</p>
        <h3>Workflow</h3>
        <ol>
          <li>Pick the labelled PCD file and click <b>Load</b>.</li>
          <li>Click <b>Analyse</b> — the system scores obstacles as <i>removable candidates</i>.</li>
          <li>Select a candidate row to see its 2D overlay and predicted gain (m²).</li>
          <li>Use <b>Simulate Removal</b> to preview the RII lift if the object is removed.</li>
          <li><b>Optimise Relocation</b> searches combinations that maximise reclaimed area.</li>
        </ol>
    """),

    ("Step 5 — RII Vertical", """
        <h2>Step 5 — RII Vertical (Wall Reachability)</h2>
        <p>Compute the fraction of vertical wall surface the robot can paint/clean/inspect from a reachable floor location.</p>
        <h3>Key parameters</h3>
        <ul>
          <li><b>Wall min/max height</b> — which Z-slab counts as "wall".</li>
          <li><b>Reach</b> — the robot's horizontal arm reach (m).</li>
          <li><b>Angle</b> — the nozzle / tool cone angle.</li>
          <li><b>Paint w / vspan / sweep</b> — effective coverage geometry per stroke.</li>
          <li><b>γ (gamma)</b> — how much RII Vertical contributes to the combined score vs. RII Horizontal.</li>
        </ul>
        <p>Results appear in the <i>Vertical Coverage</i> tab plus a combined RII card.</p>
    """),

    ("Projects", """
        <h2>Projects</h2>
        <p>A project (<code>.riiproj</code>) bundles your input paths, all parameter values, detected/manual ramps, and generated map artifacts so you can pick up where you left off.</p>
        <h3>Actions</h3>
        <ul>
          <li><b>File → Open Project…</b> (Ctrl+O) — restore a saved session.</li>
          <li><b>File → Open Recent</b> — last 5 projects.</li>
          <li><b>File → Save Project</b> (Ctrl+S) / <b>Save Project As…</b> (Ctrl+Shift+S).</li>
        </ul>
        <h3>What gets saved</h3>
        <ul>
          <li>Input point cloud path.</li>
          <li>All Step 1-5 parameters.</li>
          <li>Generated map paths (.pgm / .yaml).</li>
          <li>Manual ramps and RANSAC detection result.</li>
        </ul>
        <h3>What does <i>not</i> get saved (re-run required)</h3>
        <ul>
          <li>Step 3 coverage results.</li>
          <li>Step 4 semantic analysis and candidates.</li>
          <li>Step 5 vertical coverage.</li>
        </ul>
    """),

    ("Tips & Troubleshooting", """
        <h2>Tips & Troubleshooting</h2>
        <ul>
          <li><b>Pipeline stepper</b> (top banner) — click any step to jump the sidebar to its panel.</li>
          <li><b>Status bar</b> (bottom) — shows loaded map, cursor world-coords, and the running worker.</li>
          <li><b>Log panel</b> (below the map) — colour-coded: <span style="color:#2563eb">blue=info</span> / <span style="color:#16a34a">green=success</span> / <span style="color:#dc2626">red=error</span>.</li>
          <li><b>If ramps look wrong</b>, tune <i>max_slope</i> and <i>max_step</i> first, then re-run Detect Ramps.</li>
          <li><b>Map looks noisy</b> — enable the Step 1 noise filter with higher <i>Min density</i>, or raise <i>min_points_per_cell</i> in Step 2.</li>
          <li><b>Map is empty</b> — your Z-range is probably too narrow. Widen min_z / max_z.</li>
          <li><b>Labelled PCD won't load</b> — it must include an integer <code>label</code> channel. Check the console log for parse errors.</li>
        </ul>
    """),
]


_SHORTCUTS = [
    ("File", [
        ("Ctrl+O", "Open Project"),
        ("Ctrl+S", "Save Project"),
        ("Ctrl+Shift+S", "Save Project As"),
        ("Ctrl+Q", "Quit"),
    ]),
    ("Map View", [
        ("Left-drag", "Pan the map"),
        ("Mouse wheel", "Zoom in / out"),
        ("Drag rectangle", "Select ramp / region (when a Select mode is active)"),
        ("Shift + drag", "Freeform polygon selection (when enabled)"),
    ]),
    ("Help", [
        ("F1", "Open User Guide"),
    ]),
]


class _StyledDialog(QDialog):
    """Shared base — applies the app's theme and sets a sensible default size."""
    def __init__(self, parent, title, w=820, h=620):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(w, h)
        self.setStyleSheet(
            "QDialog { background: #ffffff; }"
            "QLabel { color: #1f2937; }"
            "QListWidget { background: #f8f9fa; border: 1px solid #e5e7eb;"
            "              border-radius: 6px; font-size: 12px; padding: 4px; }"
            "QListWidget::item { padding: 8px 12px; border-radius: 4px; color: #374151; }"
            "QListWidget::item:selected { background: #dbeafe; color: #1d4ed8; }"
            "QListWidget::item:hover { background: #eff6ff; }"
            "QTextBrowser { background: #ffffff; border: 1px solid #e5e7eb;"
            "               border-radius: 6px; font-size: 13px; padding: 12px; color: #1f2937; }"
            "QPushButton { background: #2563eb; color: #ffffff; border: 1px solid #1d4ed8;"
            "              border-radius: 6px; padding: 8px 20px; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background: #1d4ed8; }"
        )


class UserGuideDialog(_StyledDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "User Guide — " + APP_NAME, w=900, h=640)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12); root.setSpacing(10)

        header = QLabel(f"<h2 style='margin:0;color:#1f2937;'>How to use {APP_NAME}</h2>"
                        "<p style='margin:2px 0 0 0;color:#6b7280;font-size:12px;'>"
                        "Select a topic on the left to see step-by-step instructions.</p>")
        header.setTextFormat(Qt.RichText)
        root.addWidget(header)

        body = QHBoxLayout(); body.setSpacing(12); root.addLayout(body, 1)

        nav = QListWidget()
        nav.setFixedWidth(220)
        for title, _html in _GUIDE_SECTIONS:
            nav.addItem(QListWidgetItem(title))
        body.addWidget(nav)

        stack = QStackedWidget()
        for _title, html in _GUIDE_SECTIONS:
            tb = QTextBrowser()
            tb.setOpenExternalLinks(True)
            tb.setHtml(html)
            stack.addWidget(tb)
        body.addWidget(stack, 1)

        nav.currentRowChanged.connect(stack.setCurrentIndex)
        nav.setCurrentRow(0)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)


class ShortcutsDialog(_StyledDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "Keyboard Shortcuts", w=480, h=420)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12); root.setSpacing(10)
        root.addWidget(QLabel("<h2 style='margin:0;color:#1f2937;'>Keyboard & Mouse Shortcuts</h2>"))
        tb = QTextBrowser()
        parts = []
        for group, rows in _SHORTCUTS:
            parts.append(f"<h3 style='color:#2563eb;margin-top:14px;'>{group}</h3>")
            parts.append("<table cellspacing='0' cellpadding='6' style='border-collapse:collapse;width:100%;'>")
            for key, desc in rows:
                parts.append(
                    "<tr>"
                    f"<td style='font-family:monospace;background:#f3f4f6;border:1px solid #e5e7eb;"
                    f"border-radius:4px;padding:4px 8px;width:160px;color:#1f2937;'><b>{key}</b></td>"
                    f"<td style='padding:4px 12px;color:#374151;'>{desc}</td>"
                    "</tr>"
                )
            parts.append("</table>")
        tb.setHtml("".join(parts))
        root.addWidget(tb, 1)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject); btns.accepted.connect(self.accept)
        root.addWidget(btns)


class AboutDialog(_StyledDialog):
    def __init__(self, parent=None):
        super().__init__(parent, "About " + APP_NAME, w=460, h=300)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 16); root.setSpacing(12)

        title = QLabel(f"<h2 style='margin:0;color:#1f2937;'>{APP_NAME}</h2>")
        root.addWidget(title)
        ver = QLabel(f"<span style='color:#6b7280;font-size:12px;'>Version {APP_VERSION}</span>")
        root.addWidget(ver)

        body = QLabel(
            "<p style='color:#374151;font-size:13px;line-height:1.6;'>"
            "A pipeline for evaluating how accessible an indoor environment is to a given "
            "robot. Takes a 3D point cloud, produces a 2D traversability map, detects ramps, "
            "and computes Robot Inclusivity Index metrics (horizontal coverage + vertical "
            "reachability)."
            "</p>"
        )
        body.setWordWrap(True)
        root.addWidget(body)

        root.addStretch()
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(self.accept)
        root.addWidget(btns)
