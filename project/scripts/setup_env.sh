#!/usr/bin/env bash

cd /workspaces/project || exit 1

source /opt/ros/humble/setup.bash

if [ -f /workspaces/project/install/setup.bash ]; then
  source /workspaces/project/install/setup.bash
fi

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export IGN_GAZEBO_RESOURCE_PATH=/workspaces/project/src/calib_lab/models:$IGN_GAZEBO_RESOURCE_PATH

echo "ROS environment ready"
echo "RMW_IMPLEMENTATION=$RMW_IMPLEMENTATION"
echo "IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH"
