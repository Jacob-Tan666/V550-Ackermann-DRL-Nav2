#!/usr/bin/env python3
import time

import rclpy
from nav2_msgs.srv import ManageLifecycleNodes
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


class Nav2StartupGate(Node):
    def __init__(self):
        super().__init__("nav2_startup_gate")
        self.declare_parameter("startup_service", "/lifecycle_manager_navigation/manage_nodes")
        self.declare_parameter("gazebo_state_service", "auto")
        self.declare_parameter("required_topics", ["odom", "scan"])
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("source_frame", "base_footprint")
        self.declare_parameter("startup_timeout", 90.0)
        self.declare_parameter("check_period", 0.5)

        self.startup_service = str(self.get_parameter("startup_service").value)
        self.requested_gazebo_state_service = str(self.get_parameter("gazebo_state_service").value)
        self.required_topics = list(self.get_parameter("required_topics").value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.source_frame = str(self.get_parameter("source_frame").value)
        self.startup_timeout = max(float(self.get_parameter("startup_timeout").value), 1.0)
        self.check_period = max(float(self.get_parameter("check_period").value), 0.1)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.startup_client = self.create_client(ManageLifecycleNodes, self.startup_service)
        self.start_time = time.monotonic()
        self.started = False
        self.last_status = ""
        self.timer = self.create_timer(self.check_period, self.on_timer)

        self.get_logger().info(
            "Waiting for Gazebo service, odom/scan topics, and odom TF before starting Nav2"
        )

    def on_timer(self):
        if self.started:
            return

        missing = self.missing_readiness()
        if missing:
            status = ", ".join(missing)
            if status != self.last_status:
                self.get_logger().info(f"Nav2 startup gate waiting for: {status}")
                self.last_status = status
            if time.monotonic() - self.start_time > self.startup_timeout:
                self.get_logger().error(
                    "Timed out waiting for Nav2 prerequisites. "
                    "Check that Gazebo server is running and the robot model loaded."
                )
                self.started = True
            return

        request = ManageLifecycleNodes.Request()
        request.command = ManageLifecycleNodes.Request.STARTUP
        future = self.startup_client.call_async(request)
        future.add_done_callback(self.on_startup_response)
        self.started = True
        self.get_logger().info("All prerequisites ready. Starting Nav2 lifecycle nodes.")

    def missing_readiness(self):
        missing = []
        services = {name for name, _ in self.get_service_names_and_types()}
        topics = {name for name, _ in self.get_topic_names_and_types()}

        gazebo_state_service = self.resolve_gazebo_state_service()
        if gazebo_state_service is None:
            missing.append("gazebo_msgs/srv/SetEntityState service")
        elif gazebo_state_service not in services:
            missing.append(gazebo_state_service)

        for topic in self.required_topics:
            normalized = topic if topic.startswith("/") else "/" + topic
            if normalized not in topics:
                missing.append(normalized)

        if self.startup_service not in services:
            missing.append(self.startup_service)

        try:
            self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except TransformException:
            missing.append(f"TF {self.target_frame}->{self.source_frame}")

        return missing

    def resolve_gazebo_state_service(self):
        requested = self.requested_gazebo_state_service.strip()
        if requested and requested.lower() != "auto":
            return requested

        matches = sorted(
            name
            for name, types in self.get_service_names_and_types()
            if "gazebo_msgs/srv/SetEntityState" in types
        )
        if not matches:
            return None
        for preferred_name in ("/gazebo/set_entity_state", "/set_entity_state"):
            if preferred_name in matches:
                return preferred_name
        return matches[0]

    def on_startup_response(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to call Nav2 startup service: {exc}")
            return
        if response.success:
            self.get_logger().info("Nav2 lifecycle startup accepted.")
        else:
            self.get_logger().error("Nav2 lifecycle startup service returned success=false.")


def main(args=None):
    rclpy.init(args=args)
    node = Nav2StartupGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
