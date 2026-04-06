# Robot Inclusivity Index (RII) Pipeline

A PyQt5 desktop application for evaluating how accessible an indoor environment is
to a mobile robot, using 3D point cloud data and 2D occupancy maps.

## Quick Start

```bash
# Install all dependencies (PyQt5, numpy, Pillow, pyqtgraph, PyOpenGL, scipy, numba)
./bootstrap.sh

# Launch
./launch.sh
# or
python3 rii_pipeline.py
```

A ROS 2 installation (Jazzy, Humble, Iron, or Rolling) is required for the
map-generation shell workers.

## Project Structure

```
rii_pipeline/
├── rii_pipeline.py              # Entry point (~30 lines)
├── config.py                    # Paths, ROS distro detection
├── launch.sh                    # Convenience launcher (sources ROS)
├── color_scale                  # CloudCompare color scale for semantic labels
├── core/                        # Pure computation (no Qt widgets)
│   ├── RII_horizontal.py        # Inflation, BFS reachability, STC, coverage
│   ├── RII_vertical.py          # Voxel raycasting, TCR / OE / SC metrics
│   ├── map_io.py                # PGM / YAML map I/O
│   ├── rendering.py             # 2D QImage rendering (coverage, STC paths)
│   ├── semantic_analysis.py     # Semantic label analysis, layered RII
│   └── semantic_selection.py    # Polygon / rectangle selection geometry
├── gui/                         # Qt UI layer
│   ├── main_window.py           # MainWin (QMainWindow)
│   ├── widgets.py               # MapW, DragScrollArea, PointCloudW
│   └── workers.py               # QThread workers (shell, viewer, map build)
└── src/pcd_package/             # Point cloud tools (ROS package)
```

## Pipeline Steps

The GUI is organized into five sequential steps:

| Step | Purpose |
|------|---------|
| **1. Select Point Cloud** | Choose a `.pcd` or `.ply` file; optionally pre-clean it |
| **2. Generate 2D Map** | Project the point cloud into a 2D occupancy grid (obstacle, traversability, floor) |
| **3. RII Horizontal** | Compute horizontal accessibility via inflation + BFS flood fill |
| **4. RII Horizontal Analysis** | Semantic gap analysis — identify which object classes block accessibility |
| **5. RII Vertical & Combined** | Compute vertical (wall) accessibility via 3D raycasting, then combine |

### Step 1 — Select Point Cloud

Browse for a `.pcd` or `.ply` file. An optional pre-clean step can filter
noise before map generation.

### Step 2 — Generate 2D Map

Projects the 3D point cloud into a 2D ROS Nav2 occupancy grid. Three map
layers are produced: obstacle, traversability, and floor.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `pt_min_z` | 0.05 m | -20 – 20 | Minimum z height for obstacle projection (offset above detected floor) |
| `pt_max_z` | 1.00 m | -20 – 20 | Maximum z height for obstacle projection |
| `max_slope` | 35.0° | 1 – 89 | Traversability: maximum ground slope before marking non-traversable |
| `max_step` | 0.25 m | 0.01 – 9999 | Traversability: maximum step height between adjacent cells |
| `max_rough` | 0.15 m | 0.01 – 9999 | Traversability: maximum surface roughness |

**Tuning the traversable ground map** — The goal is to make the traversable
ground map match the obstacle map as closely as possible. Start by adjusting
`pt_min_z` and `pt_max_z`, then click **Clean Map** to regenerate. Once the
z-range is close, fine-tune with `max_step`, `max_slope`, and `max_rough` to
capture stairs, ramps, or uneven surfaces.

**Edit Traversable Ground** — After map generation, the traversable ground
mask can be manually painted using circle, rectangle, or free-draw brushes
(size in metres). An obstacle map overlay toggle helps align edits.

### Step 3 — RII Horizontal

Computes horizontal accessibility by inflating obstacles by the robot footprint
and flood-filling reachable floor from a user-selected start point.

**Robot footprint parameters** (separate Reference and Actual):

| Parameter | Default (Ref) | Default (Actual) | Range | Description |
|-----------|---------------|-------------------|-------|-------------|
| Shape | circular | rectangular | circular / rectangular | Robot footprint geometry |
| Radius | 0.035 m | 0.35 m | 0.001 – 5 | Footprint radius (circular mode) |
| Width (W) | 0.07 m | 0.60 m | 0.01 – 5 | Footprint width (rectangular mode) |
| Length (L) | 0.07 m | 0.40 m | 0.01 – 5 | Footprint length (rectangular mode) |

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| Selection mode | Rectangle | Rectangle / Spline | How the evaluation region is drawn on the map |
| Mode | Without Path Planner | Without Path Planner / With Path Planner | Enable coverage path planning |
| Planner | STC | STC / BCD / Wavefront / Morse / Frontier | Path planner algorithm (visible when mode = With Path Planner) |

**Available path planners:**

| Planner | Full Name | Description | Reference |
|---------|-----------|-------------|-----------|
| STC | Spanning-Tree Coverage | Coarsens the free space into a grid, builds a spanning tree over the largest connected component, then follows an Euler tour to cover every cell exactly once. | Gabriely & Rimon, "Spanning-Tree Based Coverage of Continuous Areas by a Mobile Robot", *Annals of Mathematics and Artificial Intelligence*, 2001. |
| BCD | Boustrophedon Cellular Decomposition | Decomposes free space into trapezoidal cells using vertical slice lines at obstacle boundaries; each cell is swept in an ox-plough (boustrophedon) pattern. | Choset & Pignon, "Coverage Path Planning: The Boustrophedon Cellular Decomposition", *Field and Service Robotics*, 1998. |
| Wavefront | Wavefront Coverage | Propagates a BFS distance wavefront from the region centroid; cells are then visited farthest-first via greedy nearest-neighbour, producing a spiral-inward trajectory. | Zelinsky et al., "Planning Paths of Complete Coverage of an Unstructured Environment by a Mobile Robot", *Proc. Int. Conf. Advanced Robotics*, 1993. |
| Morse | Morse-based Cellular Decomposition | Slices free space into vertical segments (Morse cells) at each column, builds an adjacency graph between overlapping segments in neighbouring columns, then traverses cells via DFS with alternating sweep directions. | Acar & Choset, "Sensor-Based Coverage of Unknown Environments: Incremental Construction of Morse Decompositions", *Int. Journal of Robotics Research*, 2002. |
| Frontier | Frontier-based Exploration | Iteratively moves to the nearest unvisited free cell (the frontier between covered and uncovered space), using BFS shortest-path navigation. Naturally prioritises nearby uncovered regions. | Yamauchi, "A Frontier-Based Approach for Autonomous Exploration", *Proc. IEEE Int. Symp. Computational Intelligence in Robotics and Automation*, 1997. |

**Planner Path visualiser legend:**

| Colour | Meaning |
|--------|---------|
| Blue line | Reference robot coverage path |
| Light blue fill | Reference robot accessible area |
| Green line | Actual robot coverage path |
| Light green fill | Actual robot accessible area |
| Dark grey | Inaccessible floor (blocked after inflation) |

### Step 4 — RII Horizontal Analysis

Loads a semantically labelled point cloud (CloudCompare-annotated `.pcd` or
`.ply`) and identifies which object classes contribute to the accessibility gap.

**`color_scale` file** — The `color_scale` file in the repository root is a
CloudCompare color scale definition (XML) that maps each semantic label ID
(0–18) to a distinct colour. Import this file into CloudCompare
(`Edit > Scalar Fields > Color Scale > Import`) before annotating the point
cloud so that label colours are consistent between CloudCompare and the
pipeline. The labels are:

| ID | Label | ID | Label |
|----|-------|----|-------|
| 0 | Unlabelled | 10 | Large Materials |
| 1 | Wall | 11 | Stored Equipment |
| 2 | Drains / Canals | 12 | Mobile Machines & Vehicles |
| 3 | Staircase | 13 | Movable Objects |
| 4 | Fixed Obstacles | 14 | Containers & Pallets |
| 5 | Temporary Ramps | 15 | Small Tools |
| 6 | Safety Barriers & Signs | 16 | Debris & Loose Packaging |
| 7 | Temporary Utilities | 17 | Portable Objects |
| 8 | Scaffold Structure | 18 | Unclassified Items |
| 9 | Semi-Fixed Obstacles | | |

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| Filter | All Fixations | All Fixations / Portable / Movable / Semi-Fixed | Filter candidates by fixation group |

Outputs a layered RII decomposition (progressively removing fixation groups)
and a ranked list of individual removal candidates with estimated area gain.

### Step 5 — RII Vertical & Combined

Computes wall-surface reachability via 3D voxel raycasting from accessible
floor positions, then combines with RII Horizontal.

**Wall height band:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `Wall min h` | 0.40 m | 0 – 10 | Minimum height above ground for the wall band |
| `Wall max h` | 2.00 m | 0.1 – 20 | Maximum height above ground for the wall band |
| `Wall label IDs` | 1 | comma-separated | Semantic label IDs treated as wall surface |

**Raycasting parameters:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `Voxel` | 0.05 m | 0.01 – 1.0 | Voxel grid resolution for the 3D occupancy grid |
| `Reach` | 1.0 m | 0.1 – 5.0 | Maximum ray distance from ground to wall |
| `Angle` | 10.0° | 1 – 45 | Angular step for the horizontal ray fan (smaller = more rays) |

**Paint tool parameters** (models the physical painting tool):

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `Paint width` | 0.25 m | 0.01+ | Width of the paint roller head |
| `Vertical span` | 0.30 m | 0.01+ | Vertical coverage per stroke |
| `Sweep step` | 0.20 m | 0.01+ | Vertical spacing between successive tool heights |

**Sampling parameters:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `Ground stride` | 3 px | 1 – 20 | Sample every Nth accessible floor cell as a ray origin |
| `Max samples` | 60 000 | 1 000 – 200 000 | Cap on total ground sample points |

**Combined RII:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `Combined γ` | 0.50 | 0 – 1 | Balance factor: γ=1 prioritises floor reachability (OE), γ=0 prioritises wall contiguity (SC) |

## RII Mathematics

### RII Horizontal

Measures what fraction of the floor a robot can physically reach.

1. **Obstacle inflation** — The 2D occupancy grid is dilated by the robot's
   footprint (rectangle or circle) so that cells too close to walls become
   blocked.
2. **BFS reachability** — A breadth-first search floods outward from a start
   position over all unblocked floor cells, optionally constrained by a
   traversability sidecar mask (slope, step height, roughness thresholds).
3. **Coverage ratio**:

```
RII_H = Accessible Floor Area / Total Floor Area × 100%
```

**Path planner mode** — When enabled, the accessible region is coarsened into
a grid of larger cells and only the largest connected component is kept. A
coverage path is then computed over that component using the selected planner
algorithm (STC, BCD, Wavefront, Morse, or Frontier). See the Step 3 parameter
table above for algorithm descriptions and references.

### RII Vertical

Measures what fraction of wall surface a robot's tool can reach from accessible
floor positions. Uses voxelized 3D raycasting (Amanatides & Woo algorithm).

1. **Voxelization** — The labelled point cloud is discretized into a 3D voxel
   grid. Voxels are classified as *wall* (within a configurable height band
   above ground) or *obstacle*.
2. **Ground sampling** — Reachable floor cells from the RII Horizontal result
   are sampled as ray origins.
3. **Raycasting** — For each ground sample, at each tool height in a vertical
   sweep, horizontal rays are cast in all directions. When a ray hits a wall
   voxel, the surrounding voxels (within the tool's paint width and vertical
   span) are marked as *painted*. Accelerated with Numba JIT when available;
   falls back to pure Python otherwise.
4. **Metrics**:

| Metric | Definition |
|--------|------------|
| **TCR** (Task Coverage Rate) | Painted wall voxels / Total wall-band voxels |
| **OE** (Operational Efficiency) | Ground samples that reached at least one wall / Total ground samples |
| **SC** (Surface Continuity) | Largest contiguous painted component / Total painted voxels |

### Combined RII

```
RII_Combined = TCR × (γ · OE + (1 - γ) · SC)
```

where `γ` (default 0.5) balances operational efficiency against surface
continuity.

The composite RII adopts a multiplicative coupling structure inspired by the
Cobb-Douglas production function [Cobb & Douglas, 1928], which enforces the
conjunctive requirement that task coverage is a necessary condition for a
meaningful deployment score — analogous to series system reliability where all
subsystems must be operational [Høyland & Rausand, 1994]. The quality modifiers
OE and SC are combined via convex scalarisation [Zadeh, 1963; Marler & Arora,
2010], permitting application-specific tuning of the efficiency-continuity
trade-off through the parameter γ.

A simple weighted average is also reported:

```
RII_Weighted = 0.5 · RII_H + 0.5 · RII_V
```

### Semantic Analysis

When a labelled point cloud is available (with per-point semantic class IDs),
the pipeline can:

- **Project labels to 2D** and identify which semantic categories (walls,
  furniture, columns, etc.) contribute to inaccessible areas.
- **Layered RII** — Progressively remove fixation groups (Portable, Movable,
  Semi-Fixed, Fixed) and recompute RII to quantify each group's impact.
- **Removal candidates** — Identify individual movable/portable objects whose
  removal would most improve accessibility, ranked by area gain.

Objects are classified into fixation groups:
- **Fixed** — walls, floors, ceilings, structural columns
- **Semi-Fixed** — doors, windows, built-in fixtures
- **Movable** — tables, chairs, boards, bookcases
- **Portable** — clutter, small objects

## 2D Map Format

Maps use the ROS Nav2 occupancy grid format:

- **PGM** — 8-bit grayscale image where 0 = occupied, 254 = free, 205 = unknown
- **YAML** — metadata file specifying `resolution` (m/pixel), `origin` `[x, y, θ]`,
  and thresholds

Three map layers are generated from the point cloud:
1. **Obstacle map** (`map.pgm`) — binary occupied/free
2. **Traversability sidecar** (`map_traversable.pgm`) — slope, step, roughness filtered
3. **Floor sidecar** (`map_floor.pgm`) — ground-plane cells only

## Dependencies

- Python 3.10+
- PyQt5
- NumPy
- Pillow
- pyqtgraph + PyOpenGL (optional, for hardware-accelerated 3D point cloud viewer)
- Numba (optional, for fast raycasting in RII Vertical)
- SciPy (optional, for connected-component labelling in Surface Continuity)
- ROS 2 (Jazzy / Humble / Iron / Rolling) for map generation workers
