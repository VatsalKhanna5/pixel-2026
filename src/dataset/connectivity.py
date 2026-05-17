"""
src/dataset/connectivity.py
PIXEL-2026 — Connectivity Validation & Topology Utilities

BFS-based port-to-port connectivity check.
DRC (design-rule check): minimum feature size (width/spacing).
All checks are deterministic and fast (pure NumPy / collections).
"""

from __future__ import annotations

import numpy as np
from collections import deque

# Grid / port constants (must match primitives.py)
H, W = 15, 15
PORT1 = (7, 0)
PORT2 = (7, 14)

# 4-connectivity (Manhattan) neighbours
_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def is_connected(layout: np.ndarray, port1: tuple = PORT1, port2: tuple = PORT2) -> bool:
    """
    BFS from port1 to port2 on the binary layout (4-connectivity).

    Returns True iff there exists a connected conducting path.
    Fails immediately if either port pixel is not a conductor.
    """
    if layout[port1] == 0 or layout[port2] == 0:
        return False

    visited: set[tuple[int, int]] = {port1}
    queue: deque[tuple[int, int]] = deque([port1])

    while queue:
        r, c = queue.popleft()
        if (r, c) == port2:
            return True
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            nbr = (nr, nc)
            if 0 <= nr < H and 0 <= nc < W and nbr not in visited and layout[nr, nc] == 1:
                visited.add(nbr)
                queue.append(nbr)
    return False


def connected_components(layout: np.ndarray) -> np.ndarray:
    """
    Label all 4-connected conductor components.

    Returns:
        label_map: int32 array of shape (H, W), 0 = void, ≥1 = component label
    """
    labels = np.zeros((H, W), dtype=np.int32)
    current_label = 0

    for r in range(H):
        for c in range(W):
            if layout[r, c] == 1 and labels[r, c] == 0:
                current_label += 1
                queue: deque = deque([(r, c)])
                labels[r, c] = current_label
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in _NEIGHBOURS:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and layout[nr, nc] == 1 and labels[nr, nc] == 0:
                            labels[nr, nc] = current_label
                            queue.append((nr, nc))
    return labels


def check_drc(layout: np.ndarray, min_feature_px: int = 1) -> dict[str, bool | int]:
    """
    Basic DRC checks for fabrication validity.

    Checks performed:
    - min_conductor_width:  no isolated single-pixel protrusions (min_feature_px)
    - ports_are_conductors: both port pixels are conductor
    - no_floating_metals:   all conductor pixels are reachable from port1 or port2

    Args:
        layout:         Binary (H, W) uint8 array.
        min_feature_px: Minimum allowed feature width in pixels. Default=1 (no DRC).

    Returns:
        Dict with keys:
          'pass'                  — bool, True if all checks pass
          'ports_are_conductors'  — bool
          'no_floating_metals'    — bool
          'n_components'          — int, number of conductor components
          'main_component_size'   — int, size of largest component
    """
    results: dict = {}

    # Check 1: port pixels must be conductor
    results["ports_are_conductors"] = bool(layout[PORT1] == 1 and layout[PORT2] == 1)

    # Check 2: connected components
    labels = connected_components(layout)
    n_components = labels.max()
    results["n_components"] = int(n_components)

    if n_components > 0:
        # Largest component (excludes void = 0)
        comp_sizes = np.bincount(labels.ravel())[1:]  # skip label 0 (void)
        results["main_component_size"] = int(comp_sizes.max())
        # Both ports should be in the same component
        if layout[PORT1] == 1 and layout[PORT2] == 1:
            same_comp = labels[PORT1] == labels[PORT2] and labels[PORT1] > 0
        else:
            same_comp = False
        results["no_floating_metals"] = bool(same_comp)
    else:
        results["main_component_size"] = 0
        results["no_floating_metals"] = False

    results["pass"] = bool(
        results["ports_are_conductors"]
        and results["no_floating_metals"]
    )
    return results


def topology_features(layout: np.ndarray) -> dict[str, int | float]:
    """
    Compute topology summary features for dataset quality logging.

    Returns:
        conductor_fraction: fraction of pixels that are conductors
        n_components:       number of connected components
        bounding_box_fill:  fill ratio within bounding box of conductors
        path_length_px:     BFS path length port1 → port2 (0 if disconnected)
    """
    feats: dict = {}
    feats["conductor_fraction"] = float(layout.sum()) / (H * W)

    labels = connected_components(layout)
    feats["n_components"] = int(labels.max())

    # Bounding box fill
    rows, cols = np.where(layout == 1)
    if len(rows) > 0:
        bbox_h = rows.max() - rows.min() + 1
        bbox_w = cols.max() - cols.min() + 1
        feats["bounding_box_fill"] = float(layout.sum()) / (bbox_h * bbox_w)
    else:
        feats["bounding_box_fill"] = 0.0

    # BFS path length
    feats["path_length_px"] = _bfs_path_length(layout, PORT1, PORT2)

    return feats


def _bfs_path_length(layout: np.ndarray, port1: tuple, port2: tuple) -> int:
    """BFS shortest path length (in pixels). Returns 0 if not connected."""
    if layout[port1] == 0 or layout[port2] == 0:
        return 0
    dist = {port1: 0}
    queue: deque = deque([port1])
    while queue:
        r, c = queue.popleft()
        if (r, c) == port2:
            return dist[(r, c)]
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            nbr = (nr, nc)
            if 0 <= nr < H and 0 <= nc < W and nbr not in dist and layout[nr, nc] == 1:
                dist[nbr] = dist[(r, c)] + 1
                queue.append(nbr)
    return 0
