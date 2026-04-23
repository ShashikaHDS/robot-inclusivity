"""Save/load .riiproj project bundles.

A project file is a **ZIP archive** with the following layout:

    project.json          — metadata: schema, params, paths, ramps, analyses
    data/
        input.pcd         — the source point cloud (optional)
        map.pgm / .yaml   — the 2D obstacle map (+ sidecars + per-level maps)
        ...
    results/
        level<N>_ref.npz  — Step 3 Reference coverage result for level N
        level<N>_act.npz  — Step 3 Actual coverage result for level N

Inside `project.json`, all path fields point into the archive's data/
subtree. On load we extract the archive into a temp directory and
rewrite the paths so the rest of the pipeline reads from there.

Backward compatibility: a file that starts with a JSON "{" is loaded
as a legacy schema-v1 .riiproj (paths refer to original filesystem
locations; no bundled data).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np


PROJECT_SCHEMA_VERSION = 2  # v2 = zip bundle format; v1 = legacy JSON
PROJECT_EXT = ".riiproj"
_PROJECT_JSON = "project.json"


# ── Path helpers for legacy JSON ─────────────────────────────────────────────
def _rel(path: str, base_dir: str) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return os.path.abspath(path)


def _abs(path: str, base_dir: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


# ── Transition / ground_result conversion (unchanged from v1) ────────────────
def _transition_to_dict(t) -> Dict[str, Any]:
    return {
        "transition_id": int(t.transition_id),
        "type": t.type,
        "level_from": int(t.level_from),
        "level_to": int(t.level_to),
        "start_xy": [float(t.start_xy[0]), float(t.start_xy[1])],
        "end_xy": [float(t.end_xy[0]), float(t.end_xy[1])],
        "angle_deg": float(t.angle_deg),
        "width_m": float(t.width_m),
        "length_m": float(t.length_m),
        "step_height_m": float(t.step_height_m),
        "height_from": float(t.height_from),
        "height_to": float(t.height_to),
        "cells": t.cells.astype(np.int32).tolist() if t.cells is not None else [],
        "traversable": bool(t.traversable),
    }


def _dict_to_transition(d: Dict[str, Any]):
    from core.ground_analysis import TransitionInfo
    cells = np.asarray(d.get("cells", []), dtype=np.int32)
    if cells.ndim == 1:
        cells = cells.reshape(0, 2)
    return TransitionInfo(
        transition_id=int(d["transition_id"]),
        type=d.get("type", "ramp"),
        level_from=int(d.get("level_from", 0)),
        level_to=int(d.get("level_to", 0)),
        start_xy=(float(d["start_xy"][0]), float(d["start_xy"][1])),
        end_xy=(float(d["end_xy"][0]), float(d["end_xy"][1])),
        angle_deg=float(d["angle_deg"]),
        width_m=float(d.get("width_m", 0.0)),
        length_m=float(d.get("length_m", 0.0)),
        step_height_m=float(d.get("step_height_m", 0.0)),
        height_from=float(d.get("height_from", 0.0)),
        height_to=float(d.get("height_to", 0.0)),
        cells=cells,
        traversable=bool(d.get("traversable", False)),
    )


def _ground_result_to_dict(result) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "transitions": [_transition_to_dict(t) for t in result.transitions],
        "cell_size": float(result.cell_size),
        "grid_origin": [float(result.grid_origin[0]), float(result.grid_origin[1])],
        "grid_shape": [int(result.grid_shape[0]), int(result.grid_shape[1])],
    }


def _dict_to_ground_result(d: Optional[Dict[str, Any]]):
    if not d:
        return None
    from core.ground_analysis import GroundAnalysisResult
    return GroundAnalysisResult(
        levels=[],
        transitions=[_dict_to_transition(td) for td in d.get("transitions", [])],
        cell_size=float(d["cell_size"]),
        grid_origin=(float(d["grid_origin"][0]), float(d["grid_origin"][1])),
        grid_shape=(int(d["grid_shape"][0]), int(d["grid_shape"][1])),
        level_grid=None,
    )


# ── Step 3 result serialisation ──────────────────────────────────────────────
_COVERAGE_SCALAR_KEYS = (
    "coveredArea", "reachableArea", "accessibleArea", "accessibleCells",
    "totalFloorArea", "totalFloorCells", "riiHorizontal",
    "useSTC", "planner", "stcComponents", "stcLargestTiles", "stcStep",
    "reachableCells", "waypoints", "resolution", "step", "cw", "ch", "w", "h",
    "accurateFootprint", "footprintMotion",
    "startAdjusted", "startAdjustmentReason",
    "startComponentSize", "largestComponentSize",
)
_COVERAGE_ARRAY_KEYS = ("blocked", "sourceBlocked", "floorPx", "covPx",
                        "trafficHeatmap")
_COVERAGE_TUPLE_KEYS = ("origin", "requestedStartWorld", "effectiveStartWorld",
                        "requestedStartCell", "effectiveStartCell")


def _coverage_to_npz_bytes(result: Dict[str, Any]) -> bytes:
    """Serialise a Step-3 coverage result dict to a portable .npz blob."""
    arrays: Dict[str, np.ndarray] = {}
    scalars: Dict[str, Any] = {}
    for k in _COVERAGE_ARRAY_KEYS:
        if k in result and result[k] is not None:
            arr = np.asarray(result[k])
            if arr.size > 0:
                arrays[k] = arr
    for k in _COVERAGE_SCALAR_KEYS:
        if k in result:
            scalars[k] = result[k]
    for k in _COVERAGE_TUPLE_KEYS:
        if k in result and result[k] is not None:
            scalars[k] = [float(v) for v in result[k]]
    if "params" in result:
        scalars["params"] = dict(result["params"])
    if "pgm_path" in result:
        # basename only; the path will be resolved against the bundle's data dir
        scalars["pgm_path_basename"] = os.path.basename(result["pgm_path"])
    scalars["__meta__"] = "rii_coverage_result_v1"

    buf = io.BytesIO()
    np.savez_compressed(buf, scalars=json.dumps(scalars), **arrays)
    return buf.getvalue()


def _coverage_from_npz_bytes(data: bytes, bundle_data_dir: str) -> Dict[str, Any]:
    """Recover a Step-3 coverage result dict from an .npz blob."""
    buf = io.BytesIO(data)
    with np.load(buf, allow_pickle=False) as z:
        result: Dict[str, Any] = {}
        for k in z.files:
            if k == "scalars":
                scalars = json.loads(str(z["scalars"]))
                for sk, sv in scalars.items():
                    if sk == "__meta__":
                        continue
                    if sk.endswith("StartCell") or sk.endswith("StartWorld") or sk == "origin":
                        result[sk.replace("StartCell", "StartCell") if sk.endswith("StartCell") else sk] = tuple(sv) if isinstance(sv, list) else sv
                    elif sk == "pgm_path_basename":
                        result["pgm_path"] = os.path.join(bundle_data_dir, sv)
                    else:
                        result[sk] = sv
            else:
                result[k] = z[k]
    return result


# ── Core API ─────────────────────────────────────────────────────────────────
def build_project_dict(params: Dict[str, Any],
                       artifact_paths: Dict[str, str],
                       manual_ramps: List,
                       ground_result,
                       level_pgm_map: Optional[Dict[int, str]] = None,
                       level_results: Optional[Dict[int, Dict[str, Any]]] = None,
                       active_level: Optional[int] = None) -> Dict[str, Any]:
    """Build the in-memory project structure. The artifact_paths dict is
    later walked by save_project to pull actual files into the zip."""
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "params": params,
        "paths": dict(artifact_paths),
        "manual_ramps": [_transition_to_dict(t) for t in (manual_ramps or [])],
        "ground_result": _ground_result_to_dict(ground_result),
        "level_pgm_map": {str(k): v for k, v in (level_pgm_map or {}).items()},
        "level_results_meta": {
            str(k): {kind: (v.get(kind) is not None)
                     for kind in ("ref", "act")}
            for k, v in (level_results or {}).items()
        },
        "active_level": active_level,
    }


def save_project(project_file: str,
                 data: Dict[str, Any],
                 level_results: Optional[Dict[int, Dict[str, Any]]] = None) -> None:
    """Write a .riiproj ZIP bundle.

    level_results: optional {level_idx: {"ref": result_dict, "act": result_dict}}
    — each non-None result is stored as an .npz blob in results/.
    """
    if not project_file.endswith(PROJECT_EXT):
        project_file += PROJECT_EXT

    # Take a snapshot of the absolute paths BEFORE we rewrite them to be
    # relative (inside the zip).
    absolute_paths = dict(data.get("paths", {}))
    level_pgm_map = {int(k): v for k, v in (data.get("level_pgm_map") or {}).items()}

    # Plan the files that go into the zip and the in-archive names we'll
    # rewrite the JSON paths to.
    bundle_plan: Dict[str, str] = {}   # absolute source path → in-archive path
    for key, src in absolute_paths.items():
        if not src or not os.path.isfile(src):
            continue
        arc = "data/" + os.path.basename(src)
        bundle_plan.setdefault(src, arc)
    for idx, src in level_pgm_map.items():
        if src and os.path.isfile(src):
            arc = "data/" + os.path.basename(src)
            bundle_plan.setdefault(src, arc)

    # Pull in any sibling files that belong to a bundled PGM:
    #   *.yaml, *_floor.pgm, *_floor.yaml, *_traversable.pgm, *_traversable.yaml
    for src in list(bundle_plan.keys()):
        if src.endswith(".pgm"):
            stem = os.path.splitext(src)[0]
            for suffix in (".yaml", "_floor.pgm", "_floor.yaml",
                           "_traversable.pgm", "_traversable.yaml"):
                sib = stem + suffix
                if os.path.isfile(sib) and sib not in bundle_plan:
                    bundle_plan[sib] = "data/" + os.path.basename(sib)

    # Rewrite the JSON paths to point inside the archive
    out_paths = {}
    for key, src in absolute_paths.items():
        if src and src in bundle_plan:
            out_paths[key] = bundle_plan[src]
        elif src:
            out_paths[key] = src   # keep the original path if we can't bundle it
    data["paths"] = out_paths
    data["level_pgm_map"] = {
        str(idx): bundle_plan.get(pgm, pgm)
        for idx, pgm in level_pgm_map.items() if pgm
    }

    # Write the zip
    tmp_file = project_file + ".tmp"
    with zipfile.ZipFile(tmp_file, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(_PROJECT_JSON, json.dumps(data, indent=2))
        for src, arc in bundle_plan.items():
            zf.write(src, arc)
        for idx, results in (level_results or {}).items():
            for kind in ("ref", "act"):
                r = results.get(kind) if results else None
                if r is None:
                    continue
                blob = _coverage_to_npz_bytes(r)
                zf.writestr(f"results/level{int(idx)}_{kind}.npz", blob)
    os.replace(tmp_file, project_file)


def _load_legacy_json(project_file: str) -> Dict[str, Any]:
    """Load a pre-v2 (raw JSON) project file."""
    with open(project_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(project_file))
    data["paths"] = {k: _abs(v, base_dir) for k, v in data.get("paths", {}).items()}
    data["manual_ramps"] = [_dict_to_transition(d) for d in data.get("manual_ramps", [])]
    data["ground_result"] = _dict_to_ground_result(data.get("ground_result"))
    return data


def load_project(project_file: str) -> Dict[str, Any]:
    """Load a .riiproj file.

    Result dict has the same shape the GUI already expects, PLUS:
        data["_bundle_dir"]    — temp dir where bundled files were extracted
        data["level_results"]  — {level_idx: {"ref": result, "act": result}}
    """
    # Peek first bytes to choose format
    with open(project_file, "rb") as f:
        head = f.read(4)
    if head[:2] == b"PK":
        return _load_bundle(project_file)
    if head.lstrip().startswith(b"{"):
        return _load_legacy_json(project_file)
    # Fallback: try both
    try:
        return _load_bundle(project_file)
    except Exception:
        return _load_legacy_json(project_file)


def _load_bundle(project_file: str) -> Dict[str, Any]:
    out_dir = tempfile.mkdtemp(prefix="rii_proj_")
    with zipfile.ZipFile(project_file, "r") as zf:
        zf.extractall(out_dir)

    json_path = os.path.join(out_dir, _PROJECT_JSON)
    if not os.path.isfile(json_path):
        # Old v2 iterations might have written the JSON with a different name
        raise ValueError("Project bundle is missing project.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Rewrite in-archive paths to absolute temp paths
    data_dir = os.path.join(out_dir, "data")
    def _resolve_arc(p):
        if not p:
            return ""
        if p.startswith("data/"):
            return os.path.normpath(os.path.join(out_dir, p))
        # legacy absolute path — leave as-is
        return p

    data["paths"] = {k: _resolve_arc(v) for k, v in data.get("paths", {}).items()}
    data["level_pgm_map"] = {int(k): _resolve_arc(v)
                             for k, v in (data.get("level_pgm_map") or {}).items()}
    data["manual_ramps"] = [_dict_to_transition(d) for d in data.get("manual_ramps", [])]
    data["ground_result"] = _dict_to_ground_result(data.get("ground_result"))

    # Load step-3 results if present
    level_results: Dict[int, Dict[str, Any]] = {}
    results_dir = os.path.join(out_dir, "results")
    if os.path.isdir(results_dir):
        for fn in sorted(os.listdir(results_dir)):
            if not fn.endswith(".npz"):
                continue
            # Filename format: level{N}_{kind}.npz
            try:
                stem = fn[:-4]   # strip .npz
                parts = stem.split("_")
                level_idx = int(parts[0].replace("level", ""))
                kind = parts[1]
            except (ValueError, IndexError):
                continue
            with open(os.path.join(results_dir, fn), "rb") as fh:
                blob = fh.read()
            try:
                r = _coverage_from_npz_bytes(blob, data_dir)
            except Exception:
                continue
            level_results.setdefault(level_idx, {})[kind] = r
    data["level_results"] = level_results
    data["_bundle_dir"] = out_dir
    return data
