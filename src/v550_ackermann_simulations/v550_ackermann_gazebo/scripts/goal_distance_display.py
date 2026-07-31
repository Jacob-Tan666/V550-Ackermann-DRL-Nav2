#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


def planar_distance_cm(goal_x, goal_y, robot_x, robot_y):
    return 100.0 * math.hypot(goal_x - robot_x, goal_y - robot_y)


def format_distance(distance_cm):
    if distance_cm >= 100.0:
        return f"Goal distance: {distance_cm / 100.0:.2f} m"
    return f"Goal distance: {distance_cm:.1f} cm"


class GoalDistanceDisplay(Node):
    def __init__(self):
        super().__init__("goal_distance_display")
        self.declare_parameter("goal_topic", "goal_pose")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("distance_topic", "goal_distance_cm")
        self.declare_parameter("marker_topic", "goal_distance_marker")
        self.declare_parameter("plan_topic", "plan")
        self.declare_parameter("goal_tolerance", 0.03)
        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("marker_height", 0.65)
        self.declare_parameter("text_height", 0.28)

        goal_topic = str(self.get_parameter("goal_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        distance_topic = str(self.get_parameter("distance_topic").value)
        marker_topic = str(self.get_parameter("marker_topic").value)
        plan_topic = str(self.get_parameter("plan_topic").value)
        self.goal_tolerance_cm = 100.0 * max(float(self.get_parameter("goal_tolerance").value), 0.0)
        self.marker_height = float(self.get_parameter("marker_height").value)
        self.text_height = max(float(self.get_parameter("text_height").value), 0.05)
        publish_rate = max(float(self.get_parameter("publish_rate").value), 1.0)

        self.goal = None
        self.odom = None
        self.last_tf_warning_ns = 0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.distance_publisher = self.create_publisher(Float32, distance_topic, 10)
        self.marker_publisher = self.create_publisher(Marker, marker_topic, 10)
        self.plan_clear_publisher = self.create_publisher(Path, plan_topic, 10)
        self.goal_subscription = self.create_subscription(PoseStamped, goal_topic, self.on_goal, 10)
        self.odom_subscription = self.create_subscription(Odometry, odom_topic, self.on_odom, 50)
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_distance)
        self.get_logger().info(f"Displaying goal distance on {marker_topic} and {distance_topic}")

    def on_goal(self, msg):
        values = (msg.pose.position.x, msg.pose.position.y)
        if not all(math.isfinite(value) for value in values):
            self.get_logger().warn("Ignored goal with non-finite position")
            return
        self.goal = msg

    def on_odom(self, msg):
        values = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if all(math.isfinite(value) for value in values):
            self.odom = msg

    def goal_in_odom_frame(self):
        target_frame = self.odom.header.frame_id or "odom"
        source_frame = self.goal.header.frame_id or target_frame
        if source_frame == target_frame:
            return self.goal.pose

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
            )
            return do_transform_pose(self.goal.pose, transform)
        except TransformException as error:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self.last_tf_warning_ns >= 2_000_000_000:
                self.last_tf_warning_ns = now_ns
                self.get_logger().warn(f"Cannot display goal distance: {source_frame} -> {target_frame}: {error}")
            return None

    def publish_distance(self):
        if self.goal is None or self.odom is None:
            return

        goal_pose = self.goal_in_odom_frame()
        if goal_pose is None:
            return

        robot = self.odom.pose.pose.position
        distance_cm = planar_distance_cm(goal_pose.position.x, goal_pose.position.y, robot.x, robot.y)
        self.distance_publisher.publish(Float32(data=float(distance_cm)))

        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.odom.header.frame_id or "odom"
        marker.ns = "goal_distance"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = robot.x
        marker.pose.position.y = robot.y
        marker.pose.position.z = robot.z + self.marker_height
        marker.pose.orientation.w = 1.0
        marker.scale.z = self.text_height
        marker.color.a = 1.0
        if distance_cm <= self.goal_tolerance_cm:
            marker.color.r = 0.15
            marker.color.g = 1.0
            marker.color.b = 0.25
            empty_path = Path()
            empty_path.header = marker.header
            self.plan_clear_publisher.publish(empty_path)
        else:
            marker.color.r = 1.0
            marker.color.g = 0.95
            marker.color.b = 0.15
        marker.text = format_distance(distance_cm)
        self.marker_publisher.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = GoalDistanceDisplay()
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
