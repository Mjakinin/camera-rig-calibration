#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import PointField
from sensor_msgs_py import point_cloud2


def rgb_to_float(r, g, b):
    rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
    return struct.unpack("f", struct.pack("I", rgb_uint32))[0]


def read_ascii_ply(path: Path):
    with path.open() as f:
        lines = [line.rstrip("\n") for line in f]

    if not lines or lines[0].strip() != "ply":
        raise RuntimeError(f"Not a PLY file: {path}")

    vertex_count = None
    header_end = None

    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        if line.strip() == "end_header":
            header_end = i + 1
            break

    if vertex_count is None or header_end is None:
        raise RuntimeError("Could not parse PLY header.")

    points = []
    for line in lines[header_end:header_end + vertex_count]:
        parts = line.split()
        if len(parts) < 6:
            continue

        x, y, z = map(float, parts[:3])
        r, g, b = map(int, parts[3:6])
        rgb = rgb_to_float(r, g, b)
        points.append((x, y, z, rgb))

    return points


class PlyPublisher(Node):
    def __init__(self, ply_path, topic, frame_id, rate_hz):
        super().__init__("ply_pointcloud_publisher")

        self.frame_id = frame_id
        self.points = read_ascii_ply(Path(ply_path))

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.pub = self.create_publisher(point_cloud2.PointCloud2, topic, qos)

        self.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.timer = self.create_timer(1.0 / rate_hz, self.publish_cloud)

        self.get_logger().info(f"Loaded PLY: {ply_path}")
        self.get_logger().info(f"Points: {len(self.points)}")
        self.get_logger().info(f"Publishing topic: {topic}")
        self.get_logger().info(f"Frame: {frame_id}")

    def publish_cloud(self):
        msg = point_cloud2.create_cloud(
            header=self._make_header(),
            fields=self.fields,
            points=self.points,
        )
        self.pub.publish(msg)

    def _make_header(self):
        from std_msgs.msg import Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        return header


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--topic", default="/colmap_sparse_cloud")
    parser.add_argument("--frame_id", default="map")
    parser.add_argument("--rate_hz", type=float, default=1.0)
    args = parser.parse_args()

    rclpy.init()
    node = PlyPublisher(args.ply, args.topic, args.frame_id, args.rate_hz)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
