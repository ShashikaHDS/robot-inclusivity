#!/usr/bin/env python3
"""Replay keyframe point clouds plus poses for an OctoMap pipeline."""

import glob
import math
import os
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import TransformBroadcaster

from pcd_package.pcd_tools import filter_non_finite, load_xyz_points, slice_points_by_z, voxel_downsample


def load_tum(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] == '#':
                continue
            ts, tx, ty, tz, qx, qy, qz, qw = s.split()
            poses.append({
                "t": float(ts),
                "p": np.array([float(tx), float(ty), float(tz)], dtype=np.float64),
                "q": np.array([float(qx), float(qy), float(qz), float(qw)], dtype=np.float64)
            })
    return poses


def numeric_sort_key(p):
    # "000123.pcd" -> 123; fallback to name
    base = os.path.basename(p)
    num = ''.join(ch for ch in base if ch.isdigit())
    return int(num) if num.isdigit() else base


class KeyframeReplay(Node):
    def __init__(self):
        super().__init__("keyframe_replay_to_octomap")

        # ---- parameters ----
        self.declare_parameter("poses_path", "")
        self.declare_parameter("keyframes_dir", "")
        self.declare_parameter("sensor_frame", "os0_sensor")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("topic", "/keyframes_cloud")
        self.declare_parameter("rate_hz", 5.0)     # playback speed
        self.declare_parameter("min_z", 0.05)      # slice to your robot height
        self.declare_parameter("max_z", 0.80)
        self.declare_parameter("voxel", 0.05)      # optional downsample; 0.0 to disable
        self.declare_parameter("repeat", False)

        self.poses_path  = self.get_parameter("poses_path").get_parameter_value().string_value
        self.kf_dir      = self.get_parameter("keyframes_dir").get_parameter_value().string_value
        self.sensor_frame= self.get_parameter("sensor_frame").get_parameter_value().string_value
        self.map_frame   = self.get_parameter("map_frame").get_parameter_value().string_value
        self.topic       = self.get_parameter("topic").get_parameter_value().string_value
        self.rate_hz     = float(self.get_parameter("rate_hz").value)
        self.min_z       = float(self.get_parameter("min_z").value)
        self.max_z       = float(self.get_parameter("max_z").value)
        self.voxel       = float(self.get_parameter("voxel").value)
        self.repeat      = bool(self.get_parameter("repeat").value)

        if not os.path.isfile(self.poses_path):
            raise FileNotFoundError(f"poses_path not found: {self.poses_path}")
        if not os.path.isdir(self.kf_dir):
            raise FileNotFoundError(f"keyframes_dir not found: {self.kf_dir}")

        self.poses = load_tum(self.poses_path)
        pcd_files = sorted(glob.glob(os.path.join(self.kf_dir, "*.pcd")), key=numeric_sort_key)

        n = min(len(self.poses), len(pcd_files))
        if n == 0:
            raise RuntimeError("No keyframes found. Check keyframes_dir and poses_path.")
        if len(self.poses) != len(pcd_files):
            self.get_logger().warn(f"Pose/PCD count mismatch: poses={len(self.poses)} files={len(pcd_files)}; using {n} matched pairs.")

        self.poses = self.poses[:n]
        self.pcd_files = pcd_files[:n]
        self.get_logger().info(f"Replaying {n} keyframes @ {self.rate_hz} Hz")

        # Preload PCDs -> Nx3 float32 arrays with z-slice + optional voxel
        self.clouds = []
        for i, path in enumerate(self.pcd_files):
            pts = load_xyz_points(path)
            if pts.size == 0:
                self.get_logger().warn(f"[{i}] empty PCD: {path}")
                self.clouds.append(None)
                continue

            pts = filter_non_finite(pts)
            pts = slice_points_by_z(pts, self.min_z, self.max_z)
            if self.voxel > 0.0:
                pts = voxel_downsample(pts, self.voxel)
            self.clouds.append(pts if pts.size else None)

        # publisher (streaming QoS)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(PointCloud2, self.topic, qos)
        self.tfb = TransformBroadcaster(self)

        self.idx = 0
        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1e-3), self._step)

    def _step(self):
        if self.idx >= len(self.clouds):
            if self.repeat:
                self.idx = 0
            else:
                self.get_logger().info("Done.")
                rclpy.shutdown()
                return

        pts = self.clouds[self.idx]
        pose = self.poses[self.idx]
        self.idx += 1

        if pts is None or pts.shape[0] == 0:
            return

        # 1) broadcast TF: map -> sensor_frame
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.sensor_frame
        t.transform.translation.x = float(pose["p"][0])
        t.transform.translation.y = float(pose["p"][1])
        t.transform.translation.z = float(pose["p"][2])
        qx, qy, qz, qw = pose["q"]
        # normalize quaternion
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        t.transform.rotation.x = qx / norm
        t.transform.rotation.y = qy / norm
        t.transform.rotation.z = qz / norm
        t.transform.rotation.w = qw / norm
        self.tfb.sendTransform(t)

        # 2) publish cloud in sensor_frame with SAME stamp
        header = Header()
        header.stamp = now
        header.frame_id = self.sensor_frame
        fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg = pc2.create_cloud(header, fields, pts.tolist())
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = KeyframeReplay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
