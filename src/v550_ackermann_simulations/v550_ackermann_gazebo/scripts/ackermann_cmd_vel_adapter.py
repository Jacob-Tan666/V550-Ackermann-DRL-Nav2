#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node


class AckermannCmdVelAdapter(Node):
    def __init__(self):
        super().__init__("ackermann_cmd_vel_adapter")
        self.declare_parameter("input_topic", "cmd_vel_nav")
        self.declare_parameter("output_topic", "cmd_vel")
        self.declare_parameter("wheel_base", 0.1432)
        self.declare_parameter("max_speed", 1.00)
        self.declare_parameter("max_reverse_speed", 0.35)
        self.declare_parameter("max_steer", 0.32)
        self.declare_parameter("min_turning_speed", 0.08)
        self.declare_parameter("max_steer_rate", 1.5)
        self.declare_parameter("stop_angular_threshold", 1.0e-3)
        self.declare_parameter("command_timeout", 0.35)
        self.declare_parameter("output_rate", 30.0)
        self.declare_parameter("plugin_flips_reverse_steering", True)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.output_rate = self._positive_float("output_rate", self.get_parameter("output_rate").value)
        self._apply_tunable_parameters(
            {
                "wheel_base": self.get_parameter("wheel_base").value,
                "max_speed": self.get_parameter("max_speed").value,
                "max_reverse_speed": self.get_parameter("max_reverse_speed").value,
                "max_steer": self.get_parameter("max_steer").value,
                "min_turning_speed": self.get_parameter("min_turning_speed").value,
                "max_steer_rate": self.get_parameter("max_steer_rate").value,
                "stop_angular_threshold": self.get_parameter("stop_angular_threshold").value,
                "command_timeout": self.get_parameter("command_timeout").value,
                "plugin_flips_reverse_steering": self.get_parameter("plugin_flips_reverse_steering").value,
            }
        )
        self.last_steering = 0.0
        self.last_stamp = None
        self.last_input_stamp = None
        self.target_linear = 0.0
        self.target_angular = 0.0

        self.publisher = self.create_publisher(Twist, output_topic, 10)
        self.subscription = self.create_subscription(Twist, input_topic, self.on_cmd_vel, 10)
        self.output_timer = self.create_timer(1.0 / self.output_rate, self.publish_command)
        self.add_on_set_parameters_callback(self.on_parameters_changed)
        self.get_logger().info(
            f"Adapting Nav2 Twist {input_topic} to Ackermann steering command "
            f"{output_topic}; wheel_base={self.wheel_base:.4f} m, "
            f"max_speed={self.max_speed:.2f} m/s, "
            f"max_reverse_speed={self.max_reverse_speed:.2f} m/s, "
            f"max_steer={self.max_steer:.3f} rad, "
            f"command_timeout={self.command_timeout:.2f} s"
        )

    @staticmethod
    def _clamp(value, lower, upper):
        return max(lower, min(upper, value))

    @staticmethod
    def _positive_float(name, value):
        number = abs(float(value))
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be a positive finite number")
        return number

    @staticmethod
    def _nonnegative_float(name, value):
        number = abs(float(value))
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        return number

    def _apply_tunable_parameters(self, values):
        self.wheel_base = self._positive_float("wheel_base", values["wheel_base"])
        self.max_speed = self._nonnegative_float("max_speed", values["max_speed"])
        self.max_reverse_speed = self._nonnegative_float("max_reverse_speed", values["max_reverse_speed"])
        self.max_steer = self._positive_float("max_steer", values["max_steer"])
        self.min_turning_speed = self._positive_float("min_turning_speed", values["min_turning_speed"])
        self.max_steer_rate = self._nonnegative_float("max_steer_rate", values["max_steer_rate"])
        self.stop_angular_threshold = self._nonnegative_float(
            "stop_angular_threshold", values["stop_angular_threshold"]
        )
        self.command_timeout = self._positive_float("command_timeout", values["command_timeout"])
        self.plugin_flips_reverse_steering = bool(values["plugin_flips_reverse_steering"])

    def on_parameters_changed(self, parameters):
        restart_only = {"input_topic", "output_topic", "output_rate"}
        changed_restart_only = restart_only.intersection(parameter.name for parameter in parameters)
        if changed_restart_only:
            names = ", ".join(sorted(changed_restart_only))
            return SetParametersResult(
                successful=False,
                reason=f"{names} requires restarting the adapter node",
            )

        values = {
            "wheel_base": self.wheel_base,
            "max_speed": self.max_speed,
            "max_reverse_speed": self.max_reverse_speed,
            "max_steer": self.max_steer,
            "min_turning_speed": self.min_turning_speed,
            "max_steer_rate": self.max_steer_rate,
            "stop_angular_threshold": self.stop_angular_threshold,
            "command_timeout": self.command_timeout,
            "plugin_flips_reverse_steering": self.plugin_flips_reverse_steering,
        }
        for parameter in parameters:
            if parameter.name in values:
                values[parameter.name] = parameter.value

        try:
            self._apply_tunable_parameters(values)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        return SetParametersResult(successful=True)

    def _rate_limit_steering(self, target):
        now = self.get_clock().now()
        if self.last_stamp is None:
            self.last_stamp = now
            self.last_steering = target
            return target

        dt = (now - self.last_stamp).nanoseconds * 1.0e-9
        self.last_stamp = now
        if dt <= 0.0 or dt > 1.0 or self.max_steer_rate <= 0.0:
            self.last_steering = target
            return target

        max_delta = self.max_steer_rate * dt
        target = self._clamp(
            target,
            self.last_steering - max_delta,
            self.last_steering + max_delta,
        )
        self.last_steering = target
        return target

    def _input_is_fresh(self):
        if self.last_input_stamp is None:
            return False

        age = (self.get_clock().now() - self.last_input_stamp).nanoseconds * 1.0e-9
        return 0.0 <= age <= self.command_timeout

    def _make_ackermann_command(self, linear, angular):
        if linear >= 0.0:
            linear = min(linear, self.max_speed)
        else:
            linear = max(linear, -self.max_reverse_speed)

        if abs(linear) < 1.0e-6:
            steering = self._rate_limit_steering(0.0)
        else:
            if abs(angular) < self.stop_angular_threshold:
                steering = 0.0
            else:
                effective_speed = max(abs(linear), self.min_turning_speed)
                if self.plugin_flips_reverse_steering:
                    steering = math.atan((self.wheel_base * angular) / effective_speed)
                else:
                    effective_speed = math.copysign(effective_speed, linear)
                    steering = math.atan((self.wheel_base * angular) / effective_speed)
                steering = self._clamp(steering, -self.max_steer, self.max_steer)
            steering = self._rate_limit_steering(steering)

        out = Twist()
        out.linear.x = linear
        out.angular.z = steering
        return out

    def publish_command(self):
        if self._input_is_fresh():
            out = self._make_ackermann_command(self.target_linear, self.target_angular)
        else:
            out = self._make_ackermann_command(0.0, 0.0)
        self.publisher.publish(out)

    def on_cmd_vel(self, msg):
        self.target_linear = float(msg.linear.x)
        self.target_angular = float(msg.angular.z)
        self.last_input_stamp = self.get_clock().now()


def main(args=None):
    rclpy.init(args=args)
    node = AckermannCmdVelAdapter()
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
