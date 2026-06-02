#!/bin/bash

cd /mnt/c/Users/PopyH/Desktop/APP-RAS/Maxim/camera-rig-calibration/project || exit 1

source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null

export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export IGN_GAZEBO_RESOURCE_PATH=$PWD/src/calib_lab/models:$IGN_GAZEBO_RESOURCE_PATH

echo "Starting Gazebo..."
ign gazebo src/calib_lab/worlds/minimal_calib_world.sdf -r -v 4 &
GAZEBO_PID=$!

sleep 5

echo "Starting image bridge..."
ros2 run ros_gz_image image_bridge /camera_1/image /camera_2/image &
IMAGE_BRIDGE_PID=$!

sleep 2

echo "Starting camera_info + clock bridge..."
ros2 run ros_gz_bridge parameter_bridge \
  /camera_1/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /camera_2/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo \
  /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock &
PARAM_BRIDGE_PID=$!

sleep 3

echo "ROS topics:"
ros2 topic list | grep -E "camera|clock"

echo ""
echo "Pipeline läuft."
echo "Zum Beenden: Ctrl+C"

trap "kill $GAZEBO_PID $IMAGE_BRIDGE_PID $PARAM_BRIDGE_PID 2>/dev/null" EXIT

wait
