#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateSanitizer(Node):
    def __init__(self):
        super().__init__("joint_state_sanitizer")
        self.declare_parameter("input_topic", "joint_states_raw")
        self.declare_parameter("output_topic", "joint_states")

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)

        self.publisher = self.create_publisher(JointState, output_topic, 10)
        self.subscription = self.create_subscription(JointState, input_topic, self.on_joint_state, 10)
        self.get_logger().info(f"Sanitizing JointState {input_topic} -> {output_topic}")

    @staticmethod
    def _finite_sequence(values, fallback=0.0):
        return [float(value) if math.isfinite(float(value)) else fallback for value in values]

    def on_joint_state(self, msg):
        out = JointState()
        out.header = msg.header
        out.name = list(msg.name)
        out.position = self._finite_sequence(msg.position)
        out.velocity = self._finite_sequence(msg.velocity)
        out.effort = self._finite_sequence(msg.effort)
        self.publisher.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateSanitizer()
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
