"""Compatibility facade for deterministic AP02 graph initialization."""
from __future__ import annotations

from .initialize_graph import (
    Node,
    OBS_CSV,
    _finite_float,
    _node_text,
    best_observations,
    build_graph,
    deterministic_breadth_first_tree,
    edge_metadata,
    edge_quality,
    filter_mode,
    maximum_frontier_tree,
    observation_score,
    marker_node,
    maximum_bottleneck_tree,
    observation_pnp_rmse,
    observer_node,
)
from .initialize_poses import (
    _camera_path_diagnostics,
    _path_to_node,
    _rotation_difference_deg,
    _tree_path_metrics,
    initialize_from_tree,
)
from .initialize_runner import main


if __name__ == "__main__":
    main()
