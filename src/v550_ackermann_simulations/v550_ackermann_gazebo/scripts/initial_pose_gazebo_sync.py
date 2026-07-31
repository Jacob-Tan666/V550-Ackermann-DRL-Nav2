#!/usr/bin/env python3
import copy
import math
import time

import rclpy
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def quaternion_yaw(orientation):
    return math.atan2(
        2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
        1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
    )


class InitialPoseGazeboSync(Node):
    def __init__(self):
        super().__init__("initial_pose_gazebo_sync")
        self.declare_parameter("input_topic", "initialpose")
        self.declare_parameter("amcl_topic", "initialpose_amcl")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("robot_name", "wheeltec_v550_ackermann")
        self.declare_parameter("set_entity_state_service", "auto")
        self.declare_parameter("position_tolerance", 0.05)
        self.declare_parameter("yaw_tolerance", 0.08)
        self.declare_parameter("sync_timeout", 5.0)

        input_topic = str(self.get_parameter("input_topic").value)
        amcl_topic = str(self.get_parameter("amcl_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        self.robot_name = str(self.get_parameter("robot_name").value)
        self.requested_service = str(self.get_parameter("set_entity_state_service").value)
        self.position_tolerance = max(float(self.get_parameter("position_tolerance").value), 0.01)
        self.yaw_tolerance = max(float(self.get_parameter("yaw_tolerance").value), 0.01)
        self.sync_timeout = max(float(self.get_parameter("sync_timeout").value), 1.0)

        self.amcl_publisher = self.create_publisher(PoseWithCovarianceStamped, amcl_topic, 10)
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped, input_topic, self.on_initial_pose, 10
        )
        self.odom_subscription = self.create_subscription(Odometry, odom_topic, self.on_odom, 20)
        self.timer = self.create_timer(0.05, self.on_timer)

        self.client = None
        self.service_name = None
        self.latest_odom = None
        self.queued_pose = None
        self.service_future = None
        self.awaiting_odom_pose = None
        self.awaiting_since = None
        self.last_wait_log = 0.0
        self.get_logger().info(
            f"Synchronizing RViz {input_topic} with Gazebo model {self.robot_name} "
            f"before forwarding to AMCL on {amcl_topic}"
        )

    def on_initial_pose(self, msg):
        if msg.header.frame_id not in ("", "map"):
            self.get_logger().error(f"Initial pose frame must be map, received {msg.header.frame_id!r}")
            return
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
            self.get_logger().error("Ignored non-finite RViz initial pose")
            return

        self.queued_pose = copy.deepcopy(msg)
        self.awaiting_odom_pose = None
        self.awaiting_since = None
        self.try_send_pose()

    def on_odom(self, msg):
        self.latest_odom = msg

    def on_timer(self):
        if self.client is None:
            self.try_connect_service()
        self.try_send_pose()
        self.try_forward_to_amcl()

    def try_connect_service(self):
        service_name = self.resolve_service_name()
        if service_name is None:
            now = time.monotonic()
            if now - self.last_wait_log >= 2.0:
                self.last_wait_log = now
                self.get_logger().info("Waiting for Gazebo SetEntityState service")
            return
        self.service_name = service_name
        self.client = self.create_client(SetEntityState, service_name)
        self.get_logger().info(f"Using Gazebo state service {service_name}")

    def resolve_service_name(self):
        requested = self.requested_service.strip()
        if requested and requested.lower() != "auto":
            return requested

        matches = sorted(
            name for name, types in self.get_service_names_and_types() if "gazebo_msgs/srv/SetEntityState" in types
        )
        for preferred in ("/gazebo/set_entity_state", "/set_entity_state"):
            if preferred in matches:
                return preferred
        return matches[0] if matches else None

    def try_send_pose(self):
        if self.queued_pose is None or self.service_future is not None:
            return
        if self.client is None or not self.client.wait_for_service(timeout_sec=0.0):
            return

        msg = self.queued_pose
        self.queued_pose = None
        source_pose = msg.pose.pose
        yaw = quaternion_yaw(source_pose.orientation)

        request = SetEntityState.Request()
        request.state.name = self.robot_name
        request.state.reference_frame = "world"
        request.state.pose.position.x = float(source_pose.position.x)
        request.state.pose.position.y = float(source_pose.position.y)
        request.state.pose.position.z = 0.0
        request.state.pose.orientation.z = math.sin(yaw * 0.5)
        request.state.pose.orientation.w = math.cos(yaw * 0.5)
        request.state.twist = Twist()

        self.service_future = self.client.call_async(request)
        self.service_future.add_done_callback(lambda future, pose_msg=msg: self.on_pose_set(future, pose_msg))

    def on_pose_set(self, future, msg):
        self.service_future = None
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"Gazebo initial-pose request failed: {exc}")
            return
        if not result.success:
            self.get_logger().error("Gazebo rejected initial pose")
            return

        self.awaiting_odom_pose = msg
        self.awaiting_since = time.monotonic()
        pose = msg.pose.pose
        self.get_logger().info(
            f"Gazebo model moved to x={pose.position.x:.3f}, "
            f"y={pose.position.y:.3f}; waiting for synchronized odometry"
        )

    def try_forward_to_amcl(self):
        if self.awaiting_odom_pose is None or self.latest_odom is None:
            return

        target = self.awaiting_odom_pose.pose.pose
        actual = self.latest_odom.pose.pose
        position_error = math.hypot(
            float(actual.position.x - target.position.x),
            float(actual.position.y - target.position.y),
        )
        yaw_error = abs(
            math.atan2(
                math.sin(quaternion_yaw(actual.orientation) - quaternion_yaw(target.orientation)),
                math.cos(quaternion_yaw(actual.orientation) - quaternion_yaw(target.orientation)),
            )
        )
        timed_out = self.awaiting_since is not None and time.monotonic() - self.awaiting_since > self.sync_timeout
        if position_error > self.position_tolerance or yaw_error > self.yaw_tolerance:
            if timed_out:
                self.get_logger().error(
                    "Gazebo pose changed but odometry did not synchronize; "
                    f"position_error={position_error:.3f} m, yaw_error={yaw_error:.3f} rad. "
                    "AMCL pose was not changed."
                )
                self.awaiting_odom_pose = None
                self.awaiting_since = None
            return

        msg = self.awaiting_odom_pose
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        self.amcl_publisher.publish(msg)
        self.awaiting_odom_pose = None
        self.awaiting_since = None
        self.get_logger().info("Gazebo, odometry, and AMCL initial pose synchronized successfully")


def main(args=None):
    rclpy.init(args=args)
    node = InitialPoseGazeboSync()
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
