from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ply_points(path: Path) -> list[tuple[float, float, float, int, int, int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    end = lines.index("end_header")
    result = []
    for line in lines[end + 1 :]:
        fields = line.split()
        if len(fields) >= 6:
            result.append(
                (
                    float(fields[0]),
                    float(fields[1]),
                    float(fields[2]),
                    int(fields[3]),
                    int(fields[4]),
                    int(fields[5]),
                )
            )
    return result


def _color(index: int) -> tuple[float, float, float]:
    palette = (
        (0.95, 0.25, 0.25),
        (0.20, 0.75, 0.30),
        (0.20, 0.45, 0.95),
        (0.95, 0.70, 0.15),
        (0.70, 0.30, 0.90),
        (0.15, 0.80, 0.80),
    )
    return palette[index % len(palette)]


def run(visualization: Path) -> None:
    # ROS imports remain deliberately local: normal rigcal use and scene
    # generation do not require a sourced ROS 2 environment.
    import rclpy
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Point, TransformStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray

    manifest = _read_json(visualization / "visualization_manifest.json")
    poses = _read_json(visualization / "poses_anchor_frame.json")
    frustums = _read_json(visualization / "camera_frustums.json")
    fixed_frame = str(manifest["fixed_frame"])
    qos = QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    rclpy.init()
    node = rclpy.create_node("rigcal_result_scene")
    point_publisher = node.create_publisher(
        PointCloud2, "/rigcal/scene/points", qos
    )
    anchor_publisher = node.create_publisher(
        MarkerArray, "/rigcal/scene/anchor", qos
    )
    marker_publishers = {}
    transform_broadcaster = StaticTransformBroadcaster(node)

    points = _ply_points(visualization / str(manifest["point_cloud"]))
    cloud = PointCloud2()
    cloud.header = Header(frame_id=fixed_frame)
    cloud.height = 1
    cloud.width = len(points)
    cloud.is_bigendian = False
    cloud.is_dense = True
    cloud.point_step = 16
    cloud.row_step = cloud.point_step * cloud.width
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    data = bytearray()
    for x, y, z, red, green, blue in points:
        rgb_int = (red << 16) | (green << 8) | blue
        rgb_float = struct.unpack("f", struct.pack("I", rgb_int))[0]
        data.extend(struct.pack("ffff", x, y, z, rgb_float))
    cloud.data = bytes(data)

    anchor = Marker()
    anchor.header.frame_id = fixed_frame
    anchor.ns = "common_anchor"
    anchor.id = 0
    anchor.type = Marker.CUBE
    anchor.action = Marker.ADD
    anchor.pose.orientation.w = 1.0
    anchor.scale.x = 0.17
    anchor.scale.y = 0.17
    anchor.scale.z = 0.005
    anchor.color.r = 0.95
    anchor.color.g = 0.95
    anchor.color.b = 0.95
    anchor.color.a = 0.8
    anchor_markers = [anchor]
    for axis_id, (vector, color) in enumerate(
        (
            ((0.30, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ((0.0, 0.30, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, 0.30), (0.0, 0.0, 1.0)),
        ),
        1,
    ):
        axis = Marker()
        axis.header.frame_id = fixed_frame
        axis.ns = "common_anchor_axes"
        axis.id = axis_id
        axis.type = Marker.ARROW
        axis.action = Marker.ADD
        axis.pose.orientation.w = 1.0
        axis.points = [
            Point(x=0.0, y=0.0, z=0.0),
            Point(x=vector[0], y=vector[1], z=vector[2]),
        ]
        axis.scale.x = 0.018
        axis.scale.y = 0.035
        axis.scale.z = 0.055
        axis.color.r, axis.color.g, axis.color.b = color
        axis.color.a = 1.0
        anchor_markers.append(axis)
    anchor_array = MarkerArray(markers=anchor_markers)

    frustum_by_variant = {
        (item["method"], item["label"]): item["frustums"]
        for item in frustums["variants"]
    }
    transforms = []
    marker_arrays = {}
    for variant_index, variant in enumerate(poses["variants"]):
        method = variant["method"]
        label = variant["label"]
        topic = str(variant["topic"])
        marker_publishers[topic] = node.create_publisher(
            MarkerArray, topic, qos
        )
        red, green, blue = _color(variant_index)
        markers = []
        frustum_map = {
            item["camera_id"]: item["points"]
            for item in frustum_by_variant.get((method, label), [])
        }
        for camera_index, camera in enumerate(variant["cameras"]):
            child = f"{method}/{label}/{camera['camera_id']}_optical_frame"
            transform = TransformStamped()
            transform.header.frame_id = fixed_frame
            transform.child_frame_id = child
            transform.transform.translation.x = float(camera["x_m"])
            transform.transform.translation.y = float(camera["y_m"])
            transform.transform.translation.z = float(camera["z_m"])
            transform.transform.rotation.x = float(camera["qx"])
            transform.transform.rotation.y = float(camera["qy"])
            transform.transform.rotation.z = float(camera["qz"])
            transform.transform.rotation.w = float(camera["qw"])
            transforms.append(transform)
            points_for_camera = frustum_map.get(camera["camera_id"])
            if not points_for_camera:
                continue
            marker = Marker()
            marker.header.frame_id = fixed_frame
            marker.ns = f"{method}/{label}"
            marker.id = camera_index * 3
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale.x = 0.012
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = 1.0
            origin, *corners = points_for_camera
            segments = []
            for corner in corners:
                segments.extend((origin, corner))
            for first, second in ((0, 1), (1, 2), (2, 3), (3, 0)):
                segments.extend((corners[first], corners[second]))
            marker.points = [
                Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                for point in segments
            ]
            markers.append(marker)
            camera_origin = points_for_camera[0]
            edge = Marker()
            edge.header.frame_id = fixed_frame
            edge.ns = f"{method}/{label}/anchor_edges"
            edge.id = camera_index * 3 + 1
            edge.type = Marker.LINE_LIST
            edge.action = Marker.ADD
            edge.scale.x = 0.006
            edge.color.r = red
            edge.color.g = green
            edge.color.b = blue
            edge.color.a = 0.55
            edge.points = [
                Point(x=0.0, y=0.0, z=0.0),
                Point(
                    x=float(camera_origin[0]),
                    y=float(camera_origin[1]),
                    z=float(camera_origin[2]),
                ),
            ]
            markers.append(edge)
            label_marker = Marker()
            label_marker.header.frame_id = fixed_frame
            label_marker.ns = f"{method}/{label}/labels"
            label_marker.id = camera_index * 3 + 2
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.pose.position.x = float(camera_origin[0])
            label_marker.pose.position.y = float(camera_origin[1])
            label_marker.pose.position.z = float(camera_origin[2]) + 0.10
            label_marker.pose.orientation.w = 1.0
            label_marker.scale.z = 0.10
            label_marker.color.r = red
            label_marker.color.g = green
            label_marker.color.b = blue
            label_marker.color.a = 1.0
            label_marker.text = (
                f"{method}/{label}/{camera['camera_id']}"
            )
            markers.append(label_marker)
        marker_arrays[topic] = MarkerArray(markers=markers)
    transform_broadcaster.sendTransform(transforms)

    def publish() -> None:
        stamp = node.get_clock().now().to_msg()
        cloud.header.stamp = stamp
        anchor.header.stamp = stamp
        point_publisher.publish(cloud)
        anchor_publisher.publish(anchor_array)
        for topic, marker_array in marker_arrays.items():
            for marker in marker_array.markers:
                marker.header.stamp = stamp
                marker.lifetime = Duration(sec=0)
            marker_publishers[topic].publish(marker_array)

    node.create_timer(1.0, publish)
    publish()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization", required=True)
    args = parser.parse_args()
    run(Path(args.visualization).resolve())


if __name__ == "__main__":
    main()
