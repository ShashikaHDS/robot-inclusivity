"""Save/load .riiproj project files.

A project captures the input PCD, all Step 1-5 parameter values, paths to
generated artifacts (map.pgm/.yaml), and manual ramp edits. Paths are stored
relative to the project file when possible so the project is portable across
machines that share the same data layout.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


PROJECT_SCHEMA_VERSION = 1
PROJECT_EXT = ".riiproj"


def _rel(path: str, base_dir: str) -> str:
    """Return path relative to base_dir when possible, else absolute."""
    if not path:
        return ""
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return os.path.abspath(path)


def _abs(path: str, base_dir: str) -> str:
    """Resolve a stored path (relative or absolute) against base_dir."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


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


def build_project_dict(params: Dict[str, Any],
                       artifact_paths: Dict[str, str],
                       manual_ramps: List,
                       ground_result,
                       project_file: str) -> Dict[str, Any]:
    """Build a JSON-serializable dict from current UI state."""
    base_dir = os.path.dirname(os.path.abspath(project_file))
    paths = {k: _rel(v, base_dir) for k, v in artifact_paths.items() if v}
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "params": params,
        "paths": paths,
        "manual_ramps": [_transition_to_dict(t) for t in (manual_ramps or [])],
        "ground_result": _ground_result_to_dict(ground_result),
    }


def save_project(project_file: str, data: Dict[str, Any]) -> None:
    if not project_file.endswith(PROJECT_EXT):
        project_file += PROJECT_EXT
    with open(project_file, "w") as f:
        json.dump(data, f, indent=2)


def load_project(project_file: str) -> Dict[str, Any]:
    with open(project_file, "r") as f:
        data = json.load(f)
    version = data.get("schema_version", 0)
    if version > PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Project schema v{version} is newer than supported v{PROJECT_SCHEMA_VERSION}. "
            "Upgrade the pipeline."
        )
    base_dir = os.path.dirname(os.path.abspath(project_file))
    data["paths"] = {k: _abs(v, base_dir) for k, v in data.get("paths", {}).items()}
    data["manual_ramps"] = [_dict_to_transition(d) for d in data.get("manual_ramps", [])]
    data["ground_result"] = _dict_to_ground_result(data.get("ground_result"))
    return data
