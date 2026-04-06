from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcd_package.pcd_tools import _resolve_terrain_min_points_threshold


def test_terrain_threshold_stays_requested_for_dense_counts() -> None:
    counts = np.array([1, 2, 3, 3, 4, 4, 5, 8, 10, 12], dtype=np.int32)
    applied, stats = _resolve_terrain_min_points_threshold(counts, 3)
    assert applied == 3
    assert stats["threshold_mode"] == "requested"


def test_terrain_threshold_relaxes_for_sparse_counts() -> None:
    counts = np.array([1, 2, 2, 2, 2, 2, 2, 3, 6, 9], dtype=np.int32)
    applied, stats = _resolve_terrain_min_points_threshold(counts, 3)
    assert applied == 2
    assert stats["threshold_mode"] == "adaptive"
    assert stats["applied_keep_fraction"] >= 0.75


def test_terrain_threshold_can_relax_to_one_for_very_sparse_counts() -> None:
    counts = np.array([1, 1, 1, 2, 2, 2, 2, 3, 6, 9], dtype=np.int32)
    applied, stats = _resolve_terrain_min_points_threshold(counts, 3)
    assert applied == 1
    assert stats["threshold_mode"] == "adaptive"
    assert stats["applied_keep_fraction"] == 1.0
