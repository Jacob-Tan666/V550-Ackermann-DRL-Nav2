#!/usr/bin/env python3
import copy
import math
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def quaternion_yaw(orientation):
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


class TrajectoryHistory(Node):
    def __init__(self):
        super().__init__("trajectory_history")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("reset_topic", "initialpose")
        self.declare_parameter("path_topic", "trajectory_history")
        self.declare_parameter("sample_distance", 0.03)
        self.declare_parameter("sample_yaw", 0.05)
        self.declare_parameter("reset_tolerance", 0.10)
        self.declare_parameter("max_points", 10000)
        self.declare_parameter("z_offset", 0.08)

        odom_topic = str(self.get_parameter("odom_topic").value)
        reset_topic = str(self.get_parameter("reset_topic").value)
        path_topic = str(self.get_parameter("path_topic").value)
        self.sample_distance = max(float(self.get_parameter("sample_distance").value), 0.001)
        self.sample_yaw = max(float(self.get_parameter("sample_yaw").value), 0.001)
        self.reset_tolerance = max(float(self.get_parameter("reset_tolerance").value), 0.01)
        self.z_offset = float(self.get_parameter("z_offset").value)
        max_points = max(int(self.get_parameter("max_points").value), 2)

        path_qos = QoSProfile(depth=1)
        path_qos.reliability = ReliabilityPolicy.RELIABLE
        path_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.publisher = self.create_publisher(Path, path_topic, path_qos)
        self.odom_subscription = self.create_subscription(Odometry, odom_topic, self.on_odom, 50)
        self.reset_subscription = self.create_subscription(PoseWithCovarianceStamped, reset_topic, self.on_reset, 10)

        self.poses = deque(maxlen=max_points)
        self.frame_id = "odom"
        self.last_xy_yaw = None
        self.reset_target = None
        self.get_logger().info(
            f"Recording actual trajectory {odom_topic} -> {path_topic}; "
            f"sample_distance={self.sample_distance:.3f} m, max_points={max_points}"
        )

    def on_reset(self, msg):
        pose = msg.pose.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().warn("Ignored non-finite trajectory reset pose")
            return

        self.poses.clear()
        self.last_xy_yaw = None
        self.reset_target = (float(pose.position.x), float(pose.position.y))
        self.publish_path(self.get_clock().now().to_msg())

    def on_odom(self, msg):
        pose = msg.pose.pose
        x = float(pose.position.x)
        y = float(pose.position.y)
        yaw = quaternion_yaw(pose.orientation)
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            return

        if self.reset_target is not None:
            if math.hypot(x - self.reset_target[0], y - self.reset_target[1]) > self.reset_tolerance:
                return
            self.reset_target = None

        frame_id = msg.header.frame_id or "odom"
        if frame_id != self.frame_id:
            self.frame_id = frame_id
            self.poses.clear()
            self.last_xy_yaw = None

        if self.last_xy_yaw is not None:
            last_x, last_y, last_yaw = self.last_xy_yaw
            distance = math.hypot(x - last_x, y - last_y)
            yaw_delta = abs(math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw)))
            if distance < self.sample_distance and yaw_delta < self.sample_yaw:
                return

        stamped_pose = PoseStamped()
        stamped_pose.header = copy.deepcopy(msg.header)
        stamped_pose.header.frame_id = self.frame_id
        stamped_pose.pose = copy.deepcopy(pose)
        stamped_pose.pose.position.z += self.z_offset
        self.poses.append(stamped_pose)
        self.last_xy_yaw = (x, y, yaw)
        self.publish_path(msg.header.stamp)

    def publish_path(self, stamp):
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = self.frame_id
        path.poses = list(self.poses)
        self.publisher.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryHistory()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
