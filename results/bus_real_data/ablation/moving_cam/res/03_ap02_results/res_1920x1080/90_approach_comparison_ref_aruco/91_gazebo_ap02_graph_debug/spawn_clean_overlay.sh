#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

SDF="$REPO_ROOT/results/bus_real_data/90_approach_comparison_ref_aruco/91_gazebo_ap02_graph_debug/ap02_graph_clean_overlay_model.sdf"

echo "[INFO] SDF overlay:"
echo "$SDF"

if ! command -v ign >/dev/null 2>&1; then
  echo "[ERROR] ign command not found. Run inside the ROS/Gazebo container."
  exit 1
fi

CREATE_SERVICE=$(ign service -l | grep -E '/world/.*/create$' | head -n 1 || true)

if [ -z "$CREATE_SERVICE" ]; then
  echo "[ERROR] Could not find Gazebo create service."
  echo "Start your bus world first, then run this script again."
  echo
  echo "Expected something like:"
  echo "  /world/<world_name>/create"
  echo
  echo "Available services:"
  ign service -l | head -50
  exit 1
fi

echo "[INFO] Using create service: $CREATE_SERVICE"

ign service -s "$CREATE_SERVICE" \
  --reqtype ignition.msgs.EntityFactory \
  --reptype ignition.msgs.Boolean \
  --timeout 5000 \
  --req "sdf_filename: \"$SDF\" name: \"$(basename "$SDF" .sdf)\" allow_renaming: true"

echo "[OK] Spawn request sent."
echo
echo "In Gazebo, look for:"
echo "- yellow sphere/plane = Ref-ArUco 14"
echo "- blue nodes/frustums = estimated static cameras"
echo "- green nodes/planes = estimated ArUco markers"
echo "- purple nodes/line = selected moving trajectory, full overlay only"
echo "- orange lines = observation graph edges, full overlay only"
echo "- red lines = GT error vectors, full overlay only"
