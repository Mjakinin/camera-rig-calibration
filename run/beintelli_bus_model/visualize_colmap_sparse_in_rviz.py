#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2


def read_points3d(path: Path, point_stride: int = 1):
    points = []
    colors = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 8:
                continue

            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK...
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            r = int(parts[4])
            g = int(parts[5])
            b = int(parts[6])

            points.append((x, y, z))
            colors.append((r, g, b))

    if point_stride > 1:
        points = points[::point_stride]
        colors = colors[::point_stride]

    return points, colors


def qvec_to_rotmat(qvec):
    # COLMAP qvec order: qw, qx, qy, qz
    qw, qx, qy, qz = qvec

    return np.array([
        [
            1 - 2 * qy * qy - 2 * qz * qz,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
        ],
        [
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx * qx - 2 * qz * qz,
            2 * qy * qz - 2 * qx * qw,
        ],
        [
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx * qx - 2 * qy * qy,
        ],
    ])


def read_camera_centers_from_images_txt(path: Path, camera_stride: int = 1):
    cameras = []

    with path.open() as f:
        lines = list(f)

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        parts = line.split()
        if len(parts) >= 10:
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            image_id = int(parts[0])
            qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
            name = parts[9]

            R = qvec_to_rotmat(qvec)
            C = -R.T @ tvec

            cameras.append({
                "image_id": image_id,
                "name": name,
                "center": C,
            })

            # Next line is POINTS2D, skip it.
            i += 2
        else:
            i += 1

    cameras = sorted(cameras, key=lambda x: x["name"])

    if camera_stride > 1:
        cameras = cameras[::camera_stride]

    return cameras


def rgb_to_float(r, g, b):
    # PointCloud2 rgb packed as float32-compatible uint32.
    rgb_uint32 = (int(r) << 16) | (int(g) << 8) | int(b)
    return rgb_uint32


class ColmapSparsePublisher(Node):
    def __init__(self, sparse_txt: Path, point_stride: int, camera_stride: int):
        super().__init__("colmap_sparse_rviz_publisher")

        points_path = sparse_txt / "points3D.txt"
        images_path = sparse_txt / "images.txt"

        if not points_path.exists():
            raise FileNotFoundError(points_path)
        if not images_path.exists():
            raise FileNotFoundError(images_path)

        self.points, self.colors = read_points3d(points_path, point_stride=point_stride)
        self.cameras = read_camera_centers_from_images_txt(images_path, camera_stride=camera_stride)

        self.get_logger().info(f"Loaded points:  {len(self.points)}")
        self.get_logger().info(f"Loaded cameras: {len(self.cameras)}")
        self.get_logger().info("RViz Fixed Frame should be: colmap")

        self.points_pub = self.create_publisher(PointCloud2, "/colmap/points", 1)
        self.markers_pub = self.create_publisher(MarkerArray, "/colmap/camera_markers", 1)

        self.timer = self.create_timer(1.0, self.publish_all)

    def make_header(self):
        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        h.frame_id = "colmap"
        return h

    def make_pointcloud(self):
        header = self.make_header()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
        ]

        cloud_points = []
        for (x, y, z), (r, g, b) in zip(self.points, self.colors):
            cloud_points.append((float(x), float(y), float(z), rgb_to_float(r, g, b)))

        return pc2.create_cloud(header, fields, cloud_points)

    def make_camera_markers(self):
        header = self.make_header()
        arr = MarkerArray()

        # Camera centers as sphere list.
        centers = Marker()
        centers.header = header
        centers.ns = "colmap_camera_centers"
        centers.id = 0
        centers.type = Marker.SPHERE_LIST
        centers.action = Marker.ADD
        centers.pose.orientation.w = 1.0
        centers.scale.x = 0.04
        centers.scale.y = 0.04
        centers.scale.z = 0.04
        centers.color = ColorRGBA(r=1.0, g=0.2, b=0.0, a=1.0)

        for cam in self.cameras:
            c = cam["center"]
            centers.points.append(Point(x=float(c[0]), y=float(c[1]), z=float(c[2])))

        arr.markers.append(centers)

        # Trajectory as line strip.
        path = Marker()
        path.header = header
        path.ns = "colmap_camera_path"
        path.id = 1
        path.type = Marker.LINE_STRIP
        path.action = Marker.ADD
        path.pose.orientation.w = 1.0
        path.scale.x = 0.015
        path.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0)

        for cam in self.cameras:
            c = cam["center"]
            path.points.append(Point(x=float(c[0]), y=float(c[1]), z=float(c[2])))

        arr.markers.append(path)

        # Start marker.
        if self.cameras:
            start = Marker()
            start.header = header
            start.ns = "colmap_start"
            start.id = 2
            start.type = Marker.SPHERE
            start.action = Marker.ADD
            start.pose.orientation.w = 1.0
            c = self.cameras[0]["center"]
            start.pose.position.x = float(c[0])
            start.pose.position.y = float(c[1])
            start.pose.position.z = float(c[2])
            start.scale.x = 0.15
            start.scale.y = 0.15
            start.scale.z = 0.15
            start.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            arr.markers.append(start)

            end = Marker()
            end.header = header
            end.ns = "colmap_end"
            end.id = 3
            end.type = Marker.SPHERE
            end.action = Marker.ADD
            end.pose.orientation.w = 1.0
            c = self.cameras[-1]["center"]
            end.pose.position.x = float(c[0])
            end.pose.position.y = float(c[1])
            end.pose.position.z = float(c[2])
            end.scale.x = 0.15
            end.scale.y = 0.15
            end.scale.z = 0.15
            end.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            arr.markers.append(end)

        return arr

    def publish_all(self):
        self.points_pub.publish(self.make_pointcloud())
        self.markers_pub.publish(self.make_camera_markers())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse_txt", required=True)
    parser.add_argument("--point_stride", type=int, default=1)
    parser.add_argument("--camera_stride", type=int, default=1)
    args = parser.parse_args()

    rclpy.init()
    node = ColmapSparsePublisher(
        Path(args.sparse_txt),
        point_stride=max(1, args.point_stride),
        camera_stride=max(1, args.camera_stride),
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
