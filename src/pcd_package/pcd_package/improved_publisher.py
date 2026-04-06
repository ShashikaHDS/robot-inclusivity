#!/usr/bin/env python3
"""Publish a static PCD or PLY file as a latched PointCloud2 message."""

from __future__ import annotations

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from pcd_package.pcd_tools import load_xyz_points


class PCDPublisher(Node):
    def __init__(self) -> None:
        super().__init__("pcd_publisher_1")

        self.declare_parameter("pcd_path", "Filtered_GlobalMap.pcd")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("topic", "/cloud_fixed")
        self.declare_parameter("publish_rate_hz", 0.0)

        self.pcd_path = self.get_parameter("pcd_path").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.rate_hz = self.get_parameter("publish_rate_hz").get_parameter_value().double_value

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.publisher = self.create_publisher(PointCloud2, self.topic, qos)

        if not os.path.isfile(self.pcd_path):
            raise FileNotFoundError(f"Point cloud not found: {self.pcd_path}")

        self.get_logger().info(f"Loading point cloud: {self.pcd_path}")
        points = load_xyz_points(self.pcd_path)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        self.cloud_msg = point_cloud2.create_cloud(header, fields, points.tolist())

        self.get_logger().info(
            f"Prepared cloud with {points.shape[0]:,} points. "
            f"Publishing on {self.topic} (frame_id={self.frame_id})."
        )

        if self.rate_hz <= 0.0:
            self.publisher.publish(self.cloud_msg)
            self.get_logger().info("Published once with TRANSIENT_LOCAL QoS (latched).")
        else:
            period = 1.0 / self.rate_hz
            self.timer = self.create_timer(period, self._publish_timer_cb)

    def _publish_timer_cb(self) -> None:
        self.cloud_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.cloud_msg)


def main() -> None:
    rclpy.init()
    node = PCDPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
